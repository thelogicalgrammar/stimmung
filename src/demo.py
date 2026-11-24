"""Dry test of the voice model: each singer alone, plain syllables then an overtone sweep.

usage: python demo.py [out.wav]
"""
import os
import sys

import numpy as np

import vocaltract as vt
from render import SR_OUT, HOP, decimate2, library, resolve, tracks_for_singer, write_wav
from score import Ev, ot_range

SYLLABLES = [("m", "a"), ("n", "e"), ("l", "i"), ("n", "o"), ("w", "a"), ("y", "u"), ("m", "oe")]


def singer_demo(si):
    s = vt.SINGERS[si]
    ev, t = [], 0.3
    for onset, v in SYLLABLES:
        ev.append(Ev(t, 0.75, ("v", v), onset, 0.55, 0.6))
        t += 0.85
    t += 0.5
    for n in ot_range(s):
        sh = resolve(si, ("ot", n, "a"))
        if sh[0] == "ot":
            ev.append(Ev(t, 1.1, sh, "", 0.5, 0.75, legato=True))
            t += 1.1
    total = t + 0.6
    tr = tracks_for_singer(si, ev, total)
    y = vt.synth(*tr, HOP, s["vel"], seed=5 + si, nsamp=int(total * vt.FS))
    y = decimate2(y)
    return y / np.max(np.abs(y))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(vt.HERE, "..", "out", "voice_demo.wav")
    library()
    parts = []
    for si in range(len(vt.SINGERS)):
        parts += [0.7 * singer_demo(si), np.zeros(int(0.8 * SR_OUT))]
    y = np.concatenate(parts)
    write_wav(out, np.stack([y, y]) * 0.9)
    print("wrote", out, f"{len(y) / SR_OUT:.0f}s")


if __name__ == "__main__":
    main()
