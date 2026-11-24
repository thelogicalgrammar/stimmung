"""Articulatory shape library for the VocalTractLab voices.

For every singer: the speaker's preset vowels (with F1 raised above f0 for high
voices, as singers do), and overtone tunings that move one formant of a vowel
exactly onto a harmonic of the sung pitch while keeping the other formants.
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

import vtl
from score import ot_range
from singers import SINGERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vtl_library.pkl")

# score vowel name -> VTL shape name
VOWELS = dict(i="i", e="e", E="E", a="a", O="O", o="o", u="u", y="y", oe="2", schwa="@")
# vocal-fold settings for every voice: full closure and a slightly skewed pulse give
# the smoothest source spectrum (fewest deep dips between harmonics)
GLOTTIS_MODS = dict(chink_area=0.0, pulse_shape=0.3, flutter=12.0)
PRESSED_X = 0.01        # vocal-fold rest position: pressed phonation (VTL "pressed" shape) keeps the
                        # resonances sharp, as overtone singers do (+2.3 dB overtone prominence vs 0.02)
MIN_MARGIN_DB = 8.0          # overtone tunings must stand out this much in synthesis
OPT_PARAMS = ["HY", "JA", "LP", "LD", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY"]

_speaker = None


def use_speaker(name):
    global _speaker
    if _speaker != name:
        vtl.init(name)
        _speaker = name


def formants(p, fmax=4500.0):
    f, m = vtl.transfer_function(p, 4096)
    H = 20 * np.log10(m + 1e-12)
    df = f[1] - f[0]
    out = []
    for k in range(1, len(f) - 1):
        if 150 < f[k] < fmax and H[k] > H[k - 1] and H[k] >= H[k + 1]:
            y0, y1, y2 = H[k - 1], H[k], H[k + 1]
            out.append(f[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12) * df)
    return np.array(out)


def harmonic_levels(p, f0, fmax=5000.0, source=None):
    """Level of each harmonic at the lips (dB): tract transfer function plus source+radiation
    (a measured per-singer spectrum if given, else a smooth -6 dB/oct)."""
    f, m = vtl.transfer_function(p, 8192)
    k = np.arange(1, int(fmax / f0) + 1)
    H = 20 * np.log10(np.interp(k * f0, f, m) + 1e-12)
    if source is None:
        return H - 6.0 * np.log2(k)
    return H + np.interp(k, np.arange(1, len(source) + 1), source)


def glottis_params(f0, pressure=8000.0):
    info = vtl.constants()["glottis_info"]
    g = info["neutral"].copy()
    G = {n: i for i, n in enumerate(info["names"])}
    g[G["f0"]], g[G["pressure"]] = f0, pressure
    g[G["x_bottom"]] = g[G["x_top"]] = PRESSED_X
    for k, v in GLOTTIS_MODS.items():
        g[G[k]] = v
    return g


def measure_source(si, V, fmax=5000.0):
    """Source+radiation spectrum at the singer's pitch: synthesised minus transfer function,
    averaged over several vowels (pitch flutter off)."""
    s = SINGERS[si]
    use_speaker(s["speaker"])
    g = glottis_params(s["f0"])
    g[vtl.constants()["glottis_info"]["names"].index("flutter")] = 0.0
    K = int(fmax / s["f0"])
    k = np.arange(1, K + 1)
    rows = []
    for v in ("schwa", "a", "i", "u", "o"):
        p = V[(si, v)][0] if isinstance(V[(si, v)], tuple) else V[(si, v)]
        nf = int(1.3 * 400)
        y = vtl.synth(np.tile(p, (nf, 1)), np.tile(g, (nf, 1)))[int(0.4 * vtl.constants()["sr"]):]
        X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / vtl.constants()["sr"])
        meas = np.array([20 * np.log10(X[np.argmin(np.abs(f - kk * s["f0"]))] + 1e-12) for kk in k])
        rows.append(meas - harmonic_levels(p, s["f0"], fmax, source=np.zeros(K)))
    return np.median(rows, axis=0)


def margins(p, f0, n, source=None):
    L = harmonic_levels(p, f0, source=source)
    lo, hi = max(0, n - 3), min(len(L), n + 2)
    near = np.delete(L[lo:hi], n - 1 - lo)
    return L[n - 1] - np.max(near), L[n - 1] - np.max(np.delete(L, n - 1))


def nelder_mead(f, x0, step=0.08, iters=250, tol=1e-7):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if vals[-1] - vals[0] < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = 2 * cen - pts[-1]; fr = f(xr)
        if fr < vals[0]:
            xe = 3 * cen - 2 * pts[-1]; fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = 0.5 * (cen + pts[-1]); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_formants(p0, tgt, w, prior=0.05):
    """Adjust the articulators of p0 so formants F1-F3 approach tgt."""
    info = vtl.constants()["tract_info"]
    idx = [info["names"].index(n) for n in OPT_PARAMS]
    rng = (info["max"] - info["min"])[idx]
    tgt, w = np.asarray(tgt, float), np.asarray(w, float)

    def make(x):
        p = p0.copy()
        p[idx] = p0[idx] + x * rng
        return vtl.clip_tract(p)

    def cost(x):
        F = formants(make(x))
        if len(F) < 3:
            return 10.0
        return float(np.sum(w * np.log(F[:3] / tgt) ** 2) + prior * np.sum(x ** 2))

    x, c = nelder_mead(cost, np.zeros(len(idx)))
    return make(x), c


def vowel_job(args):
    si, v = args
    s = SINGERS[si]
    use_speaker(s["speaker"])
    p = vtl.shape(VOWELS[v])
    F = formants(p)[:3]
    if F[0] < 1.1 * s["f0"]:
        # high voice: open up so F1 sits above the sung pitch, keep F2/F3
        tgt = F.copy(); tgt[0] = 1.15 * s["f0"]
        tgt[1] = max(tgt[1], 1.5 * tgt[0])
        p, _ = fit_formants(p, tgt, [3.0, 1.0, 0.25])
        F = formants(p)[:3]
    return (si, v), p, F


def overtone_job(args):
    si, v, n, p_v, F_v, src = args
    s = SINGERS[si]
    use_speaker(s["speaker"])
    ft = n * s["f0"]
    k = int(np.argmin(np.abs(np.log(F_v / ft))))
    tgt = F_v.copy(); tgt[k] = ft
    w = np.array([1.0, 1.0, 0.25]); w[k] = 10.0
    p, _ = fit_formants(p_v, tgt, w)
    F = formants(p)[:3]
    return (si, n, v), p, margins(p, s["f0"], n, src), abs(np.log(F[k] / ft))


def level_job(args):
    si, key, p, src = args
    s = SINGERS[si]
    use_speaker(s["speaker"])
    L = harmonic_levels(p, s["f0"], source=src)
    return (si,) + key, 10 * np.log10(np.sum(10 ** (L / 10)))


def measured_margin(args):
    """Synthesise a tuning as a steady sound and measure how far harmonic n stands out."""
    si, key, p = args
    s = SINGERS[si]
    use_speaker(s["speaker"])
    sr = vtl.constants()["sr"]
    nf = int(1.3 * 400)
    y = vtl.synth(np.tile(p, (nf, 1)), np.tile(glottis_params(s["f0"], 7000.0), (nf, 1)))[int(0.4 * sr):]
    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    n = key[1]
    lev = np.array([20 * np.log10(X[np.argmin(np.abs(f - k * s["f0"]))] + 1e-12) for k in range(1, n + 3)])
    lo = max(0, n - 3)
    return (si,) + key, lev[n - 1] - np.max(np.delete(lev[lo:n + 2], n - 1 - lo))


def validate():
    """Keep only overtone tunings that stand out in actual synthesis."""
    with open(OUT, "rb") as f:
        lib = pickle.load(f)
    O = lib["overtones"]
    with Pool(os.cpu_count()) as pool:
        res = dict(pool.map(measured_margin, [(k[0], ("ot", k[1], k[2]), p) for k, (p, m) in O.items()]))
    keep = {k: (p, (res[(k[0], "ot", k[1], k[2])],) + tuple(m[1:]))
            for k, (p, m) in O.items() if res[(k[0], "ot", k[1], k[2])] >= MIN_MARGIN_DB}
    print(f"validated: {len(keep)}/{len(O)} tunings stand out by >= {MIN_MARGIN_DB} dB in synthesis")
    for si, s in enumerate(SINGERS):
        print(s["name"], " ".join(f"{n}:{'/'.join(v for v in VOWELS if (si, n, v) in keep) or '-'}"
                                  for n in ot_range(s)))
    lib["overtones"] = keep
    with open(OUT, "wb") as f:
        pickle.dump(lib, f)


def main():
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        V = {}
        for key, p, F in pool.map(vowel_job, [(si, v) for si in range(len(SINGERS)) for v in VOWELS]):
            V[key] = (p, F)
        print(f"vowels done {time.time() - t:.0f}s", flush=True)
        src = dict(enumerate(pool.starmap(measure_source, [(si, V) for si in range(len(SINGERS))])))
        print(f"source spectra measured {time.time() - t:.0f}s", flush=True)
        jobs = []
        for si, s in enumerate(SINGERS):
            for v in VOWELS:
                p_v, F_v = V[(si, v)]
                for n in ot_range(s):
                    if np.min(np.abs(np.log(F_v / (n * s["f0"])))) <= np.log(1.35):
                        jobs.append((si, v, n, p_v, F_v, src[si]))
        res = pool.map(overtone_job, jobs, chunksize=1)
        O = {k: (p, m) for k, p, m, err in res if m[0] >= MIN_MARGIN_DB and err < 0.03}
        print(f"overtones done {time.time() - t:.0f}s ({len(O)}/{len(jobs)} kept)", flush=True)
        lv = pool.map(level_job, [(k[0], ("v", k[1]), p, src[k[0]]) for k, (p, F) in V.items()] +
                      [(k[0], ("ot", k[1], k[2]), p, src[k[0]]) for k, (p, m) in O.items()])
    levels = dict(lv)
    for si, s in enumerate(SINGERS):
        print(s["name"], s["speaker"], " ".join(
            f"{n}:{'/'.join(v for v in VOWELS if (si, n, v) in O) or '-'}" for n in ot_range(s)))
    with open(OUT, "wb") as f:
        pickle.dump(dict(vowels={k: p for k, (p, F) in V.items()},
                         formants={k: F for k, (p, F) in V.items()},
                         overtones=O, levels=levels, source=src), f)


if __name__ == "__main__":
    sys.exit(validate() if "--validate" in sys.argv else main())
