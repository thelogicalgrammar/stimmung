"""Render the generated score with the vocal-tract model.

usage: python render.py [--seed 7] [--preview SECONDS] [--out ../out/stimmung_vt]
"""
import argparse
import os
import pickle
import subprocess
import tempfile
import time
import wave
from multiprocessing import Pool

import numpy as np

import vocaltract as vt
from score import OT_VOWELS, ScoreGen

HOP = 256
FR = vt.FS / HOP             # control frame rate (~344 Hz)
SR_OUT = 44100
LIB = None

CONS_DUR = dict(m=0.11, n=0.10, l=0.08, r=0.045, w=0.09, y=0.09, h=0.13)
PAN = {"tenor1": -0.75, "soprano1": -0.45, "bass": -0.12, "alto": 0.15,
       "soprano2": 0.45, "tenor2": 0.75}


def library():
    global LIB
    if LIB is None:
        with open(os.path.join(vt.HERE, "library.pkl"), "rb") as f:
            LIB = pickle.load(f)
    return LIB


def resolve(si, shape):
    """Map a score shape onto what the library has: an overtone tuning keeps the
    requested vowel if possible, else the closest vowel that can carry harmonic n."""
    if shape[0] != "ot":
        return shape
    O = library()["overtones"]
    si_, n, v = si, shape[1], shape[2]
    if (si, n, v) in O:
        return shape
    cands = [k[2] for k in O if k[0] == si and k[1] == n]
    s = vt.SINGERS[si]
    if not cands:
        return ("v", v)
    ref = np.log(vt.vowel_targets(s, v)[:2])
    best = min(cands, key=lambda c: np.sum((np.log(vt.vowel_targets(s, c)[:2]) - ref) ** 2))
    return ("ot", n, best)


def shape_params(si, shape):
    L = library()
    shape = resolve(si, shape)
    if shape[0] == "ot":
        return L["overtones"][(si, shape[1], shape[2])][0]
    return L["vowels"][(si, shape[1])]


_GAIN = {}
HUM_DB = -8.0       # hums sit this far below open vowels
COMPENSATE = 0.8    # fraction of the shape-to-shape loudness difference removed


def shape_level(si, shape, tense):
    """Predicted output level (dB) of a sustained oral shape."""
    s = vt.SINGERS[si]
    A = vt.area_from_params(shape_params(si, shape), s["N"])
    _, lev = vt.harmonic_levels(A, s["f0"], tense)
    return 10 * np.log10(np.sum(10 ** (lev / 10)))


def hum_offset(si):
    """Measured level drop (dB) of humming relative to the open 'u' vowel."""
    s = vt.SINGERS[si]
    N, nf = s["N"], int(1.0 * FR)
    A = vt.area_from_params(shape_params(si, ("v", "u")), N)
    Ah = A.copy(); Ah[N - 3:] *= np.exp(-7.0)
    out = []
    for a, vel in ((A, 0.0), (Ah, 0.55)):
        y = vt.synth(np.tile(a, (nf, 1)), np.full(nf, vel), np.full(nf, s["f0"]),
                     np.full(nf, 0.5), np.full(nf, 0.012), np.full(nf, 0.45), HOP, s["vel"])
        out.append(10 * np.log10(np.mean(y[len(y) // 3:] ** 2)))
    return out[1] - out[0]


def shape_gain(si, shape, tense):
    key = (si, shape, round(tense, 2))
    if key not in _GAIN:
        ref = shape_level(si, ("v", "a"), 0.6)
        if shape[0] == "hum":
            if (si, "hum") not in _GAIN:
                _GAIN[(si, "hum")] = hum_offset(si)
            lev = shape_level(si, ("v", shape[1]), tense) + _GAIN[(si, "hum")] - HUM_DB
        else:
            lev = shape_level(si, shape, tense)
        _GAIN[key] = 10 ** (-COMPENSATE * (lev - ref) / 20)
    return _GAIN[key]


def smooth(x, width):
    """Two passes of a centred box filter along axis 0 (triangular kernel)."""
    w = max(1, int(width))
    if w == 1:
        return x
    for _ in range(2):
        pad = np.concatenate([np.repeat(x[:1], w, 0), x, np.repeat(x[-1:], w, 0)], 0)
        c = np.cumsum(pad, 0, dtype=np.float64)
        y = (c[w:] - c[:-w]) / w
        x = y[w // 2: w // 2 + len(x)]
    return x


def tracks_for_singer(si, events, total):
    s = vt.SINGERS[si]
    N = s["N"]
    nfr = int(total * FR) + 2
    keys = [("v", "schwa")]
    table = [np.log(vt.area_from_params(shape_params(si, ("v", "schwa")), N))]
    kidx = {keys[0]: 0}

    owner = np.full(nfr, -1, int)
    amp = np.zeros(nfr)
    tense = np.full(nfr, 0.55)
    velum = np.zeros(nfr)
    breath = np.zeros(nfr)
    bscale = (s["f0"] / 116.5) ** 2      # equal breath-to-voice ratio across registers
    scoop = np.zeros(nfr)
    cons = np.zeros((nfr, N))          # log-area offsets from consonants / lip closure

    lips = slice(N - 3, N)
    tip = slice(N - 6, N - 3)
    pal = slice(int(0.6 * N), int(0.8 * N))

    for ev in events:
        base = ("v", ev.shape[1]) if ev.shape[0] == "hum" else ev.shape
        if base not in kidx:
            kidx[base] = len(table)
            table.append(np.log(vt.area_from_params(shape_params(si, base), N)))
        i0, i1 = int(ev.t * FR), int((ev.t + ev.dur) * FR)
        i1 = min(max(i1, i0 + 2), nfr)
        owner[i0:i1] = kidx[base]
        amp[i0:i1] = ev.amp * shape_gain(si, ev.shape, ev.tense)
        breath[i0:i1] = 0.016 * bscale * amp[i0:i1]
        if not ev.legato:
            k = np.arange(i1 - i0)
            scoop[i0:i1] = -np.random.default_rng(i0 + si).uniform(10, 30) * np.exp(-k / (0.07 * FR))
        tense[i0:i1] = ev.tense
        if ev.shape[0] == "hum":
            cons[i0:i1, lips] -= 7.0
            velum[i0:i1] = 0.55
        c = ev.onset
        if c:
            j1 = min(i0 + int(CONS_DUR[c] * FR), i1)
            if c == "m":
                cons[i0:j1, lips] -= 7.0; velum[i0:j1] = 0.5
            elif c == "n":
                cons[i0:j1, tip] -= 7.0; velum[i0:j1] = 0.45
            elif c == "l":
                cons[i0:j1, tip] -= 2.2
            elif c == "r":
                cons[i0:j1, tip] -= 3.5
            elif c == "w":
                cons[i0:j1, lips] -= 2.6
            elif c == "y":
                cons[i0:j1, pal] -= 1.6
            elif c == "h":
                amp[i0:j1] *= 0.08; breath[i0:j1] = 0.5 * bscale * ev.amp * shape_gain(si, ev.shape, ev.tense)

    # rests keep the last articulation
    idx = np.where(owner >= 0, np.arange(nfr), 0)
    np.maximum.accumulate(idx, out=idx)
    owner = np.where(owner[idx] >= 0, owner[idx], 0)

    T = np.array(table)
    logA = smooth(T[owner], 0.06 * FR) + smooth(cons, 0.012 * FR)
    area = np.clip(np.exp(logA), 1e-4, 12.0)
    velum = smooth(velum, 0.02 * FR)
    amp = smooth(amp, 0.035 * FR)
    breath = smooth(breath, 0.02 * FR)
    tense = smooth(tense, 0.1 * FR)
    rng = np.random.default_rng(1000 + si)
    cents = smooth(rng.normal(0, 1, nfr), 2.0 * FR)
    cents *= 1.2 / (np.std(cents) + 1e-9)
    f0 = s["f0"] * 2 ** ((cents + smooth(scoop, 0.015 * FR)) / 1200)
    return area, velum, f0, amp, breath, tense


def fftconv(x, h):
    n = len(x) + len(h) - 1
    nfft = 1 << int(np.ceil(np.log2(max(4 * len(h), 1 << 16))))
    L = nfft - len(h) + 1
    H = np.fft.rfft(h, nfft)
    y = np.zeros(n)
    for s0 in range(0, len(x), L):
        blk = np.fft.irfft(np.fft.rfft(x[s0:s0 + L], nfft) * H, nfft)
        e = min(n, s0 + nfft)
        y[s0:e] += blk[:e - s0]
    return y


def decimate2(x):
    taps = 201
    k = np.arange(taps) - taps // 2
    fc = 20500.0 / vt.FS
    h = 2 * fc * np.sinc(2 * fc * k) * np.blackman(taps)
    h /= h.sum()
    return fftconv(x, h)[taps // 2::2][: len(x) // 2]


def render_singer(args):
    si, events, total, outdir = args
    t = time.time()
    s = vt.SINGERS[si]
    area, velum, f0, amp, breath, tense = tracks_for_singer(si, events, total)
    nsamp = int(total * vt.FS)
    y = vt.synth(area, velum, f0, amp, breath, tense, HOP, s["vel"], seed=77 + si, nsamp=nsamp)
    y = decimate2(y)
    # loudness reference: steady 'a' at amp 0.5
    nf = int(1.5 * FR)
    A = vt.area_from_params(shape_params(si, ("v", "a")), s["N"])
    ref = vt.synth(np.tile(A, (nf, 1)), np.zeros(nf), np.full(nf, s["f0"]), np.full(nf, 0.5),
                   np.full(nf, 0.012), np.full(nf, 0.6), HOP, s["vel"])
    g = 1.0 / np.sqrt(np.mean(ref[len(ref) // 3:] ** 2))
    path = os.path.join(outdir, f"{s['name']}.npy")
    np.save(path, (y * g).astype(np.float32))
    print(f"  {s['name']:9s} rendered in {time.time() - t:.0f}s", flush=True)
    return path


def reverb_ir(seconds=3.2, seed=3, sr=SR_OUT, rt_scale=1.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = []
    for ch in range(2):
        noise = rng.normal(0, 1, n)
        X = np.fft.rfft(noise)
        f = np.fft.rfftfreq(n, 1 / sr)
        bands = [(0, 250, 3.0), (250, 1500, 2.6), (1500, 5000, 1.9), (5000, 9000, 1.2),
                 (9000, sr / 2 + 1, 0.6)]
        ir = np.zeros(n)
        for lo, hi, rt in bands:
            b = np.fft.irfft(np.where((f >= lo) & (f < hi), X, 0), n)
            ir += b * np.exp(-6.9 * t / (rt * rt_scale))
        pre = int(0.022 * sr)
        ir = np.concatenate([np.zeros(pre), ir * np.minimum(1, t / 0.03)])
        out.append(ir / np.sqrt(np.sum(ir ** 2)))
    return out


def write_wav(path, stereo, sr=SR_OUT):
    rng = np.random.default_rng(0)
    x = stereo * 32767 + rng.uniform(-0.5, 0.5, stereo.shape) + rng.uniform(-0.5, 0.5, stereo.shape)
    x = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.ascontiguousarray(x.T).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", type=float, default=0)
    ap.add_argument("--out", default=os.path.join(vt.HERE, "..", "out", "stimmung_vt"))
    a = ap.parse_args()

    sc = ScoreGen(a.seed).build()
    total = a.preview or sc.total
    desc = sc.describe()
    print(desc)
    print(f"total {sc.total / 60:.1f} min; rendering {total:.0f}s")
    with open(a.out + "_score.txt", "w") as f:
        f.write(desc + "\n")

    tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
    jobs = [(si, [e for e in sc.events[si] if e.t < total], total, tmp)
            for si in range(len(vt.SINGERS))]
    library(); vt.lib()
    with Pool(len(jobs)) as pool:
        paths = pool.map(render_singer, jobs)

    n = int(total * SR_OUT)
    dry = np.zeros((2, n))
    for si, p in enumerate(paths):
        y = np.load(p).astype(np.float64)[:n]
        th = (PAN[vt.SINGERS[si]["name"]] + 1) * np.pi / 4
        dry[0, :len(y)] += np.cos(th) * y
        dry[1, :len(y)] += np.sin(th) * y
        os.remove(p)
    irL, irR = reverb_ir()
    wet = np.stack([fftconv(dry[0] + 0.3 * dry[1], irL), fftconv(dry[1] + 0.3 * dry[0], irR)])
    mix = np.zeros_like(wet)
    mix[:, :n] = dry
    mix += 0.42 * wet
    # fade tail, normalise to -1 dBFS
    mix = mix[:, : n + int(2.5 * SR_OUT)]
    fl = int(2.0 * SR_OUT)
    mix[:, -fl:] *= np.linspace(1, 0, fl) ** 2
    mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))
    wav = a.out + ".wav"
    write_wav(wav, mix)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "256k",
                    a.out + ".mp3"], check=True)
    print("wrote", wav, "and", a.out + ".mp3")


if __name__ == "__main__":
    main()
