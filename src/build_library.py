"""Optimise vocal-tract shapes for every singer: vowels and overtone tunings."""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

import vocaltract as vt
from score import ot_range

OUT = os.path.join(vt.HERE, "library.pkl")


def _vowel(job):
    si, v = job
    p, c = vt.fit_vowel(vt.SINGERS[si], v, seed=si * 31 + len(v))
    return (si, v), p, c


def _ot(job):
    si, n, v, p0 = job
    p, marg, F = vt.fit_overtone_vowel(vt.SINGERS[si], v, n, p0)
    return (si, n, v), p, marg


def main():
    t = time.time()
    vowels = list(vt.VOWELS_M)
    with Pool(os.cpu_count()) as pool:
        vres = pool.map(_vowel, [(si, v) for si in range(len(vt.SINGERS)) for v in vowels])
        V = {k: p for k, p, c in vres}
        for k, p, c in vres:
            if c > 0.01:
                print("vowel fit weak", k, round(c, 4))
        print(f"vowels done {time.time() - t:.0f}s", flush=True)
        jobs = [(si, n, v, V[(si, v)]) for si, s in enumerate(vt.SINGERS)
                for n in ot_range(s) for v in vowels
                if vt.overtone_formant(s, v, n) is not None]
        ores = pool.map(_ot, jobs, chunksize=1)
    # keep only tunings where the overtone actually stands out from its neighbours
    O = {k: (p, m) for k, p, m in ores if m[0] >= 3.0}
    print(f"overtones done {time.time() - t:.0f}s")
    for si, s in enumerate(vt.SINGERS):
        row = []
        for n in ot_range(s):
            vs = [v for v in vowels if (si, n, v) in O]
            row.append(f"{n}:{'/'.join(vs) or '-'}")
        print(s["name"], " ".join(row))
    with open(OUT, "wb") as f:
        pickle.dump(dict(vowels=V, overtones=O), f)


if __name__ == "__main__":
    sys.exit(main())
