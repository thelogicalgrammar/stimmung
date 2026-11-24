"""Stronger overtone tunings, optimised against actual synthesis.

The tract's transfer function predicts sharp resonances, but during phonation the
open glottis damps them: a tuning that looks 15 dB prominent on paper measures ~6 dB.
So this searches with VTL's time-domain synthesis in the loop (pressed phonation, as
overtone singers use): a pattern search over lips, jaw, tongue body/tip/sides and
larynx height, accepting a step only if the sung harmonic stands out more from its
neighbours while the vowel's F1 stays close and the overtone does not get quieter.
Per (singer, overtone, vowel) the stronger of the old and new tuning is kept.

usage: python overtone_strong.py [--out vtl_library_strong.pkl] [--workers N]
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import vtl
import vtl_library as VL
from score import ot_range
from singers import SINGERS

PARAMS = ["JA", "LP", "LD", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY", "HY", "TS3"]
STEP0 = 0.06            # first step, as a fraction of each parameter's range
ROUNDS = 3
VOWEL_PULL = 10.0       # penalty per (log F1 ratio)^2
LOUD_DROP_DB = 6.0      # the overtone may get this much quieter than at the start before penalty
SR = 48000


def synth_harmonics(p, si, n, settle=0.25, window=0.2):
    """Levels (dB) of harmonics 1..n+2 in a short steady synthesis with pressed phonation."""
    s = SINGERS[si]
    nf = int((settle + window) * 400) + 2
    y = vtl.synth(np.tile(p, (nf, 1)), np.tile(VL.glottis_params(s["f0"], 7000.0), (nf, 1)))
    y = y[int(settle * SR):]
    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return np.array([20 * np.log10(X[np.argmin(np.abs(f - k * s["f0"]))] + 1e-12) for k in range(1, n + 3)])


def prominence(L, n):
    lo = max(0, n - 3)
    return L[n - 1] - np.max(np.delete(L[lo:n + 2], n - 1 - lo))


def job(args):
    si, v, n, starts, F_v = args
    VL.use_speaker(SINGERS[si]["speaker"])
    info = vtl.constants()["tract_info"]
    idx = [info["names"].index(x) for x in PARAMS]
    rng = (info["max"] - info["min"])[idx]
    best = None
    for p0 in starts:
        L0 = synth_harmonics(p0, si, n)
        ref = L0[n - 1]

        def score(p):
            L = synth_harmonics(p, si, n)
            F = VL.formants(p)
            pull = VOWEL_PULL * np.log(F[0] / F_v[0]) ** 2 if len(F) else 5.0
            quiet = 0.5 * max(0.0, (ref - LOUD_DROP_DB) - L[n - 1])
            return prominence(L, n) - pull - quiet, prominence(L, n)

        p, (val, pr) = p0.copy(), score(p0)
        step = STEP0
        for _ in range(ROUNDS):
            for j, i in enumerate(idx):
                for sign in (1, -1):
                    q = p.copy()
                    q[i] += sign * step * rng[j]
                    q = vtl.clip_tract(q)
                    v2, pr2 = score(q)
                    if v2 > val + 0.2:
                        p, val, pr = q, v2, pr2
                        break
            step /= 2
        if best is None or val > best[1]:
            best = (p, val, pr)
    return (si, n, v), best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "vtl_library_strong.pkl"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    with open(VL.OUT, "rb") as f:
        lib = pickle.load(f)
    t = time.time()
    O_old = lib["overtones"]
    # starting points: the existing tuning (if any) and a fresh formant-placement tuning
    pre = []
    for si, s in enumerate(SINGERS):
        for v in VL.VOWELS:
            p_v, F_v = lib["vowels"][(si, v)], lib["formants"][(si, v)]
            for n in ot_range(s):
                if np.min(np.abs(np.log(F_v / (n * s["f0"])))) <= np.log(1.35):
                    pre.append((si, v, n, p_v, F_v, lib["source"][si]))
    with Pool(a.workers) as pool:
        fresh = {k: p for k, p, m, err in pool.map(VL.overtone_job, pre, chunksize=1) if err < 0.05}
        jobs = []
        for si, v, n, p_v, F_v, _ in pre:
            k = (si, n, v)
            starts = [x for x in (O_old.get(k, (None,))[0], fresh.get(k)) if x is not None]
            if starts:
                jobs.append((si, v, n, starts, F_v))
        print(f"{len(jobs)} tunings to optimise against synthesis on {a.workers} workers", flush=True)
        done, cand = 0, {}
        for k, p, pr in pool.imap_unordered(job, jobs, chunksize=1):
            cand[k] = p
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)} ({time.time() - t:.0f}s)", flush=True)
        # final measurement, same as the library's validation (pressed phonation)
        meas_new = dict(pool.map(VL.measured_margin, [(k[0], ("ot", k[1], k[2]), p) for k, p in cand.items()]))
        meas_old = dict(pool.map(VL.measured_margin, [(k[0], ("ot", k[1], k[2]), p) for k, (p, m) in O_old.items()]))
    O = {}
    for k in set(cand) | set(O_old):
        mk = (k[0], "ot", k[1], k[2])
        options = []
        if k in cand:
            options.append((meas_new[mk], cand[k]))
        if k in O_old:
            options.append((meas_old[mk], O_old[k][0]))
        m, p = max(options, key=lambda x: x[0])
        if m >= VL.MIN_MARGIN_DB:
            O[k] = (p, (m, 0.0))
    old_m = np.array(list(meas_old.values()))
    new_m = np.array([v[1][0] for v in O.values()])
    print(f"tunings: {len(O_old)} -> {len(O)};  prominence (pressed phonation) median "
          f"{np.median(old_m):.1f} -> {np.median(new_m):.1f} dB, 90th pct {np.percentile(old_m, 90):.1f} -> "
          f"{np.percentile(new_m, 90):.1f}")
    for si, s in enumerate(SINGERS):
        mo = [meas_old[(si, 'ot', k[1], k[2])] for k in O_old if k[0] == si]
        mn = [v[1][0] for k, v in O.items() if k[0] == si]
        print(f"  {s['name']:9s} {len(mo):3d} -> {len(mn):3d} tunings, median {np.median(mo):.1f} -> {np.median(mn):.1f} dB")
    with Pool(a.workers) as pool:
        lv = pool.map(VL.level_job, [(k[0], ("ot", k[1], k[2]), p, lib["source"][k[0]]) for k, (p, m) in O.items()])
    lib["levels"].update(dict(lv))
    lib["overtones"] = O
    with open(a.out, "wb") as f:
        pickle.dump(lib, f)
    print("wrote", a.out, f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
