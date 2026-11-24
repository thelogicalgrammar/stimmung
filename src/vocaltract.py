"""Vocal tract model: C waveguide wrapper, frequency response, shape optimisation."""
import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 88200.0          # internal rate; one tube section = c/FS = 0.397 cm
C_SOUND = 35000.0     # cm/s
GLOT_REFL = 0.75       # cycle-average, used by the frequency-domain model
GLOT_REFL_OPEN = 0.45  # C model: reflection with folds fully open (0.97 when closed)
LIP_A = 0.92
LIP_B = 0.55
DAMP = 0.9993

from singers import F0_BASE, SINGERS

# old Kelly-Lochbaum model: tube sections and velum junction per singer
_TUBE = dict(bass=(46, 25), tenor1=(44, 24), tenor2=(43, 23), alto=(39, 21),
             soprano1=(37, 20), soprano2=(36, 19))
for s in SINGERS:
    s["N"], s["vel"] = _TUBE[s["name"]]

NASAL_LEN = 28
NASAL_AREA = np.interp(np.linspace(0, 1, NASAL_LEN),
                       [0, 0.2, 0.45, 0.7, 1.0], [1.0, 3.5, 4.5, 2.0, 0.9])

# --------------------------------------------------------------------------
# C core
_lib = None


def lib():
    global _lib
    if _lib is None:
        so = os.path.join(HERE, "vt_core.so")
        srcf = os.path.join(HERE, "vt_core.c")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(srcf):
            subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared",
                                   "-fPIC", "-o", so, srcf, "-lm"])
        _lib = ctypes.CDLL(so)
        d = ctypes.POINTER(ctypes.c_double)
        _lib.synth_voice.argtypes = [ctypes.c_int] * 6 + [d] * 7 + [ctypes.c_double] * 6 + \
            [ctypes.c_ulonglong, d]
        _lib.synth_voice.restype = None
    return _lib


def _p(a):
    return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def synth(area, velum, f0, amp, breath, tense, hop, vel, seed=1, jitter=1.0, nsamp=None):
    """area: (frames, N). Others: (frames,). Returns signal at FS."""
    area = np.ascontiguousarray(area, dtype=np.float64)
    nfr, N = area.shape
    nsamp = nsamp or nfr * hop
    out = np.zeros(nsamp)
    arrs = [np.ascontiguousarray(x, dtype=np.float64) for x in (velum, f0, amp, breath, tense)]
    lib().synth_voice(nsamp, hop, nfr, N, NASAL_LEN, vel, _p(area), _p(NASAL_AREA),
                      *[_p(x) for x in arrs], FS, GLOT_REFL_OPEN, LIP_A, LIP_B, DAMP, jitter,
                      seed, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out


# --------------------------------------------------------------------------
# frequency response of exactly the discrete waveguide (velum closed)

def tract_response(A, freqs):
    """Volume-velocity transfer U_lip/U_glottis of the KL tube at freqs (Hz)."""
    A = np.asarray(A, float)
    z = np.exp(1j * 2 * np.pi * np.asarray(freqs, float) / FS)
    zi = 1.0 / z
    g = DAMP
    # state at start of section 0 as linear combos of x (= p-_s0) and u (= Ug)
    ppx = GLOT_REFL * np.ones_like(z); ppu = np.full_like(z, 1.0 / A[0])
    pmx = np.ones_like(z); pmu = np.zeros_like(z)
    for j in range(len(A)):
        # propagate to end of section j
        ppx, ppu = ppx * g * zi, ppu * g * zi
        pmx, pmu = pmx * z / g, pmu * z / g
        if j + 1 < len(A):
            r = A[j] / A[j + 1]
            a, b = 0.5 * (1 + r), 0.5 * (1 - r)
            ppx, pmx = a * ppx + b * pmx, b * ppx + a * pmx
            ppu, pmu = a * ppu + b * pmu, b * ppu + a * pmu
    RL = -LIP_A * (1 - LIP_B) / (1 - LIP_B * zi)
    # pm_end = RL * pp_end  ->  solve x
    x = -(pmu - RL * ppu) / (pmx - RL * ppx)
    pp = ppx * x + ppu
    pm = pmx * x + pmu
    return A[-1] * (pp - pm)


def radiation(freqs):
    return np.abs(1 - np.exp(-1j * 2 * np.pi * np.asarray(freqs) / FS))


def glottal_spectrum(tense, nharm):
    t = np.arange(4096) / 4096
    oq = 0.78 - 0.33 * tense
    tp = oq * (0.62 + 0.12 * tense)
    g = np.where(t < tp, 0.5 * (1 - np.cos(np.pi * t / np.maximum(tp, 1e-9))),
                 np.where(t < oq, np.cos(0.5 * np.pi * (t - tp) / (oq - tp)), 0.0))
    G = np.abs(np.fft.rfft(g))[1:nharm + 1]
    return G


def formants(A, fmax=4500.0):
    f = np.arange(80.0, fmax, 6.0)
    H = 20 * np.log10(np.abs(tract_response(A, f)) + 1e-12)
    pk = np.where((H[1:-1] > H[:-2]) & (H[1:-1] >= H[2:]))[0] + 1
    out = []
    for k in pk:
        y0, y1, y2 = H[k - 1], H[k], H[k + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        out.append(f[k] + d * 6.0)
    return np.array(out)


# --------------------------------------------------------------------------
# area-function parameterisation

NMODES = 8
NPAR = NMODES + 1


def area_from_params(p, N):
    """p[:NMODES] cosine modes of log-area, p[6] log lip-opening offset (bounded via tanh)."""
    x = (np.arange(N) + 0.5) / N
    c = 1.4 * np.tanh(np.asarray(p[:NMODES]))
    logA = np.log(2.6) + sum(c[k] * np.cos(np.pi * (k + 1) * x) for k in range(NMODES))
    A = np.exp(logA)
    # epilarynx tube: narrow first ~2 cm
    lar = x < 0.11
    A[lar] = np.minimum(A[lar], 0.9 + 6 * x[lar])
    # lips: last ~1.2 cm scaled by lip parameter
    lip = 1.6 * np.tanh(p[NMODES])
    A[-3:] *= np.exp(lip)
    return np.clip(A, 0.12, 10.0)


def nelder_mead(f, x0, step=0.4, iters=400, tol=1e-6, rng=None):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        y = np.array(x0, float)
        y[i] += step
        pts.append(y)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = f(xe)
            if fe < fr: pts[-1], vals[-1] = xe, fe
            else: pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# vowel formant targets (F1, F2, F3), male / female, roughly Peterson & Barney
VOWELS_M = dict(i=(280, 2250, 2950), e=(400, 2000, 2600), E=(540, 1800, 2500),
                a=(720, 1150, 2450), O=(580, 880, 2450), o=(430, 780, 2450),
                u=(300, 750, 2300), y=(280, 1750, 2200), oe=(420, 1450, 2350),
                schwa=(500, 1400, 2450))
VOWELS_F = dict(i=(320, 2750, 3300), e=(450, 2400, 3000), E=(620, 2200, 2950),
                a=(880, 1300, 2800), O=(620, 950, 2750), o=(480, 880, 2700),
                u=(360, 850, 2650), y=(330, 2050, 2650), oe=(470, 1650, 2650),
                schwa=(560, 1650, 2800))


def vowel_targets(singer, vowel):
    tgt = np.array((VOWELS_M if singer["male"] else VOWELS_F)[vowel], float)
    # high voices cannot keep F1 below f0: singers raise it
    tgt[0] = max(tgt[0], 1.08 * singer["f0"])
    return tgt


def fit_formants(singer, tgt, w, p_init=None, seed=0):
    N = singer["N"]
    tgt, w = np.asarray(tgt, float), np.asarray(w, float)

    def cost(p):
        F = formants(area_from_params(p, N))
        if len(F) < 3:
            return 10.0 + 0.01 * np.sum(p ** 2)
        return float(np.sum(w * np.log(F[:3] / tgt) ** 2) + 2e-4 * np.sum(p ** 2))

    if p_init is None:
        rng = np.random.default_rng(seed)
        cands = [rng.normal(0, 0.8, NPAR) for _ in range(60)] + [np.zeros(NPAR)]
        starts = sorted(cands, key=cost)[:4]
    else:
        starts = [np.asarray(p_init, float)]
    best = min((nelder_mead(cost, c, iters=600) for c in starts), key=lambda r: r[1])
    return best[0], best[1]


def fit_vowel(singer, vowel, seed=0):
    return fit_formants(singer, vowel_targets(singer, vowel), [1.0, 1.0, 0.25], seed=seed)


def overtone_formant(singer, vowel, n, max_ratio=1.35):
    """Which formant of this vowel a singer would retune onto harmonic n (or None)."""
    tgt = vowel_targets(singer, vowel)
    ft = n * singer["f0"]
    k = int(np.argmin(np.abs(np.log(tgt / ft))))
    return k if abs(np.log(tgt[k] / ft)) <= np.log(max_ratio) else None


def fit_overtone_vowel(singer, vowel, n, p_vowel):
    """Keep the vowel, move its nearest formant exactly onto harmonic n."""
    k = overtone_formant(singer, vowel, n)
    tgt = vowel_targets(singer, vowel)
    tgt[k] = n * singer["f0"]
    w = np.array([1.0, 1.0, 0.25])
    w[k] = 10.0
    p, c = fit_formants(singer, tgt, w, p_init=p_vowel)
    A = area_from_params(p, singer["N"])
    return p, overtone_margins(A, singer["f0"], n), formants(A)[:3]


def harmonic_levels(A, f0, tense=0.6, fmax=5000.0):
    m = np.arange(1, int(fmax / f0) + 1)
    f = m * f0
    lev = np.abs(tract_response(A, f)) * radiation(f) * glottal_spectrum(tense, len(m))
    return m, 20 * np.log10(lev + 1e-15)


def overtone_margins(A, f0, n, tense=0.75):
    """(local, global) prominence in dB of harmonic n: vs n+-2 neighbours, vs all."""
    m, L = harmonic_levels(A, f0, tense)
    lo, hi = max(0, n - 3), min(len(L), n + 2)
    near = np.delete(L[lo:hi], n - 1 - lo)
    return L[n - 1] - np.max(near), L[n - 1] - np.max(np.delete(L, n - 1))


def fit_overtone(singer, n, p_init, prior=0.15, tense=0.75):
    """Shape the tract so harmonic n stands out from its neighbours (and overall)."""
    N, f0 = singer["N"], singer["f0"]
    p_init = np.zeros(NPAR) + np.resize(np.asarray(p_init, float), NPAR)

    def score(p):
        loc, glob = overtone_margins(area_from_params(p, N), f0, n, tense)
        return min(loc, 25.0) + 0.5 * min(glob, 10.0)

    def cost(p):
        return -score(p) + prior * np.sum((p - p_init) ** 2)

    p = nelder_mead(cost, p_init, step=0.35, iters=900)[0]
    p = nelder_mead(cost, p, step=0.15, iters=500)[0]
    return p, overtone_margins(area_from_params(p, N), f0, n, tense)
