"""Videos of the single-voice DSL examples: the bass's vocal tract and an overtone spiral.

Renders (or reuses from the segment cache) the audio of every example in dsl_examples.py
and draws, per frame, the tract outline, the harmonics of B-flat sounding in the audio,
the program, the step of it being performed and the syllable being sung.

usage: python dsl_examples_video.py [--seed 23] [--only NAME ...] [--out ../out/dsl_examples]
"""
import argparse
import os
import re
import subprocess
import tempfile
import textwrap
import wave

import numpy as np
from PIL import Image, ImageDraw

import dsl_examples as DX
import render_vtl as rv
import video as V
import vtl
from score import ScoreGen
from singers import F0_BASE, SINGERS

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
SS = 2
FPS = V.FPS
COL = V.COLORS[DX.SINGER]
BG = V.BG
TRACT_C, TRACT_SCALE = (330, 405), 1.0
TRACT_CLIP = 652            # the neck is cut off here, above the footer
SPIRAL_C = (940, 400)
TILT_DB = 6.0               # display emphasis per octave, so high overtones show against the fundamental
N_HARM = 40
IPA = {"oe": "ø", "schwa": "ə", "E": "ɛ", "O": "ɔ"}


def font(size, bold=False):
    return V.font(size, bold)      # V.font is sized for SS=2 as well


def mixc(c, k):
    return V.mix(c, k, BG)


def syllables(events, nframes):
    txt, n_ot = [""] * nframes, [0] * nframes
    for e in events:
        shape = rv.resolve(DX.SINGER, e.shape) if e.shape[0] == "ot" else e.shape
        v = shape[2] if shape[0] == "ot" else shape[-1]
        s = "/m/  hum" if shape[0] == "hum" else f"/{e.onset.replace('sh', 'ʃ')}{IPA.get(v, v)}/"
        if e.whisper:
            s += "  whispered"
        for i in range(int(e.t * FPS), min(nframes, int((e.t + e.dur) * FPS))):
            txt[i] = s
            n_ot[i] = shape[1] if shape[0] == "ot" else 0
    return txt, n_ot


def tract_frames(events, total, nframes, svg):
    """Tract outlines and loudness per video frame."""
    vtl.init(SINGERS[DX.SINGER]["speaker"])
    tract, glot, gain = rv.tracks_for_singer(DX.SINGER, events, total)
    step = int(rv.FR / FPS)
    num = re.compile(r'points="([^"]+)"')
    out, last, polys = [], None, None
    for i in range(nframes):
        j = min(i * step + 1, len(tract) - 1)
        if last is None or np.max(np.abs(tract[j] - last)) > 1e-3:
            vtl._lib.vtlExportTractSvg(vtl._ptr(np.ascontiguousarray(tract[j])), svg.encode())
            with open(svg) as f:
                polys = [np.array(m.split(), np.float32).reshape(-1, 2) for m in num.findall(f.read())]
            last = tract[j].copy()
        out.append(polys)
    G = vtl.constants()["glottis_info"]["names"].index("pressure")
    loud = np.clip(glot[::step, G][:nframes] / 8000.0, 0, 1.2) * np.clip(gain[::step][:nframes], 0.3, 2.0)
    return out, np.pad(loud, (0, max(0, nframes - len(loud))))[:nframes]


def harmonic_levels(wav, nframes):
    """Level (0..1) of harmonics 1..N_HARM of B-flat 1 in the audio, per video frame."""
    with wave.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).mean(1) / 32768.0
    win = int(0.12 * sr)
    hann = np.hanning(win)
    nfft = 1 << int(np.ceil(np.log2(win * 4)))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    bins = [int(np.argmin(np.abs(f - k * F0_BASE))) for k in range(1, N_HARM + 1)]
    lev = np.zeros((nframes, N_HARM))
    for i in range(nframes):
        c = int(i / FPS * sr)
        seg = x[max(0, c - win // 2): c + win // 2]
        seg = np.pad(seg, (0, win - len(seg)))
        X = np.abs(np.fft.rfft(seg * hann, nfft))
        lev[i] = [X[b - 1:b + 2].max() for b in bins]
    db = 20 * np.log10(lev + 1e-9) + TILT_DB * np.log2(np.arange(1, N_HARM + 1))
    ref = np.percentile(db[db > -150], 99.5)
    return np.clip((db - (ref - 45)) / 45, 0, 1)


def draw(i, d_, num, name, prog_txt, desc):
    tracts, loud, harm, syl, n_ot, phase, cyc = d_
    S = SS
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    lv = float(np.clip(loud[i], 0, 1))
    h0 = SINGERS[DX.SINGER]["h"]

    # header: program and what to listen for
    d.text((36 * S, 26 * S), f"{num}  ·  bass alone  ·  {name}", fill=mixc(COL, 0.95), font=font(20, True))
    y = 58
    for line in textwrap.wrap(prog_txt, 92):
        d.text((36 * S, y * S), line, fill=(222, 220, 212), font=font(17))
        y += 24
    for line in textwrap.wrap(desc, 120):
        d.text((36 * S, (y + 4) * S), line, fill=(130, 132, 140), font=font(14))
        y += 20

    # the vocal tract
    cx, cy = TRACT_C
    for poly in tracts[i]:
        xs = (poly[:, 0] - (V.TX0 + V.TX1) / 2) * TRACT_SCALE
        ys = (poly[:, 1] - (V.TY0 + V.TY1) / 2) * TRACT_SCALE
        pts = [((cx + a) * S, (cy + b) * S) for a, b in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=mixc(COL, 0.3 + 0.7 * lv), width=int((2.5 + 2.5 * lv) * S), joint="curve")
    d.rectangle([0, TRACT_CLIP * S, 700 * S, H * S], fill=BG)
    if syl[i] and lv > 0.05:
        d.text((cx * S, (cy - 228) * S), syl[i], fill=mixc((240, 238, 230), 0.45 + 0.55 * lv),
               font=font(26, True), anchor="mm")
    if n_ot[i] and lv > 0.05:
        d.text((cx * S, (cy - 200) * S), f"overtone {n_ot[i]}  =  harmonic {n_ot[i] * h0} of B♭",
               fill=mixc(COL, 0.8), font=font(15), anchor="mm")

    # overtone spiral: harmonic k of B-flat 1 at angle log2(k), one turn per octave
    sx, sy = SPIRAL_C
    for k in range(1, N_HARM + 1):
        lk = np.log2(k)
        r = 22 + 34 * lk
        a = 2 * np.pi * lk - np.pi / 2
        x, yk = sx + r * np.cos(a), sy + r * np.sin(a)
        v = harm[i, k - 1]
        own = k % h0 == 0                     # a partial of the bass's note
        col = COL if k == h0 else ((235, 232, 222) if own else (110, 112, 120))
        if not own:                           # not a partial of the bass's note: only noise, drawn faint
            v = 0.0
        rad = 2.5 + 12 * v
        d.ellipse([(x - rad) * S, (yk - rad) * S, (x + rad) * S, (yk + rad) * S], fill=mixc(col, 0.15 + 0.85 * v))
        if k == h0:
            d.ellipse([(x - 16) * S, (yk - 16) * S, (x + 16) * S, (yk + 16) * S], outline=mixc(COL, 0.7), width=2 * S)
        if n_ot[i] and k == n_ot[i] * h0 and lv > 0.05:
            d.ellipse([(x - 19) * S, (yk - 19) * S, (x + 19) * S, (yk + 19) * S], outline=mixc(COL, 0.9),
                      width=3 * S)
        if own and (k <= 32 or v > 0.5):
            d.text(((x + rad + 4) * S, (yk - 8) * S), str(k), fill=mixc((200, 200, 200), 0.3 + 0.6 * v),
                   font=font(12))
    d.text((sx * S, (sy + 240) * S), "harmonics of B♭ (58.3 Hz), one turn per octave; ring = overtone being sung",
           fill=(120, 122, 130), font=font(13), anchor="mm")

    # footer: cycle and the step of the program being performed
    t = i / FPS
    if cyc[i] >= 0:
        d.text((36 * S, (H - 30) * S), f"cycle {cyc[i] + 1} of {cyc[-1] + 1}" if cyc[-1] >= 0 else "",
               fill=(160, 160, 166), font=font(16), anchor="ls")
    if phase[i] and phase[i] != prog_txt:
        d.text((190 * S, (H - 30) * S), "now:  " + textwrap.shorten(phase[i], 110), fill=mixc(COL, 0.8),
               font=font(16), anchor="ls")
    d.text(((W - 36) * S, (H - 30) * S), f"{t:4.1f}s", fill=(130, 132, 140), font=font(16), anchor="rs")
    return img.reduce(S)


def job(args):
    i, seed, outdir = args
    name, desc, prog = DX.EXAMPLES[i]
    rv.library()
    gen = ScoreGen(seed)
    cell, period, label = gen.pulse_cell(DX.SINGER, 0.4)
    pulse = period / sum(map(int, label.split("+")))
    nc = max(4, int(round(DX.DUR / period)))
    events, ctx = DX.events_for(prog, cell, period, pulse, nc, seed + i)
    total = DX.T0 + nc * period + 1.0
    base = os.path.join(outdir, f"{i + 1:02d}_{name}")
    tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
    paths = rv.render_all(events, total, tmp)          # cache hits after dsl_examples.py
    rv.mix(paths, total, base)
    with wave.open(base + ".wav") as w:
        nframes = int(w.getnframes() / w.getframerate() * FPS)
    tracts, loud = tract_frames(events[DX.SINGER], total, nframes, os.path.join(tmp, "tract.svg"))
    harm = harmonic_levels(base + ".wav", nframes)
    syl, n_ot = syllables(events[DX.SINGER], nframes)
    log = ctx.phase_log[DX.SINGER]
    cyc = [int((k / FPS - DX.T0) // period) if DX.T0 <= k / FPS < DX.T0 + nc * period else -1
           for k in range(nframes)]
    cyc[-1] = nc - 1          # lets the footer know the number of cycles
    phase = [log[c] if 0 <= c < len(log) else "" for c in cyc]
    phase[-1] = ""
    data = (tracts, loud, harm, syl, n_ot, phase, cyc)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", base + ".wav",
                          "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                          "-movflags", "+faststart", base + ".mp4"], stdin=subprocess.PIPE)
    for k in range(nframes):
        p.stdin.write(draw(k, data, f"{i + 1:02d}", name, str(prog), desc).tobytes())
    p.stdin.close()
    p.wait()
    os.remove(base + ".wav")
    os.remove(base + ".mp3")
    return base + ".mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "dsl_examples"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = [i for i, (n, _, _) in enumerate(DX.EXAMPLES) if not a.only or n in a.only]
    for i in idx:             # one at a time: each job renders its audio with a pool of its own
        print("wrote", job((i, a.seed, a.out)), flush=True)


if __name__ == "__main__":
    main()
