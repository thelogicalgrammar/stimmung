"""Sketch of three Stimmung-like processes, about 80 s:

1. speech becomes song: a spoken line, whose words turn into a sung pattern that the
   others identify with;
2. a magic name is called and absorbed into the running rhythm, singer by singer;
3. transformation: tempo glides and each singer crosses over to a new pattern at its
   own moment, arriving at a tutti unison; cadence, silence, a last spoken line.

usage: python sketch_processes.py [--out ../out/sketch_processes]
"""
import argparse
import os
import tempfile
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import render_vtl as rv
import vtl
from score import Ev, Model, ScoreGen, map_overtone, ot_range
from singers import SINGERS

HERE = os.path.dirname(os.path.abspath(__file__))
BASS, TEN1, TEN2, ALTO, SOP1, SOP2 = range(6)

# invented text, in rough German SAMPA (VTL's phoneme set): (phonemes with durations), accented index
LINE1 = [[("d", .05), ("@", .07)], [("m", .07), ("u:", .22), ("n", .1)],
         [("o:", .16), ("p", .07), ("@", .06), ("n", .09), ("s", .08)], [("I", .06), ("t", .05), ("s", .1)],
         [("m", .08), ("aU", .26), ("T", .14)]]
LINE2 = [[("E", .07), ("n", .05), ("d", .04)], [("d", .04), ("@", .06)],
         [("v", .06), ("O:", .18), ("t", .05), ("6", .12)], [("z", .09), ("I", .1), ("N", .1), ("s", .1)],
         [("b", .06), ("E", .18), ("k", .16)]]
ACCENTS = {1: [1, 2, 4], 2: [2, 3, 4]}

# the line's words as singable syllables: moo-n o-pen mouth
WORD_SYLLABLES = [("m", "u", 1), ("n", "hum", 0), ("", "o", 1), ("p", "E", 0), ("n", "hum", 0),
                  ("m", "a", 1), ("w", "u", 0)]
NAME = [("k", "a", 1), ("l", "i", 0)]


def speech_segments(line, f0, which):
    segs = [("", 0.12, None, 0)]
    for wi, w in enumerate(line):
        for j, (ph, d) in enumerate(w):
            # speech near the singer's own pitch, with a gentle phrase declination
            segs.append((ph, d * 1.25, f0 * (1.1 - 0.22 * wi / len(line)), j == 1 and wi in ACCENTS[which]))
    return segs + [("", 0.25, None, 0)]


class Sketch(ScoreGen):
    def word_cell(self, lead, syllables, pulse, ots):
        """Syllables on a pulse grid: stressed ones get two pulses and an overtone."""
        cell, t, k = [], 0.0, 0
        for c, v, stress in syllables:
            d = (2 if stress else 1) * pulse
            if v == "hum":
                shape = ("hum", "u")
            elif stress:
                shape = ("ot", ots[k % len(ots)], v if v in ("u", "o", "a", "i", "y") else "a")
                k += 1
            else:
                shape = ("v", v)
            cell.append(Ev(t, d * 0.8, shape, c, 1.0 if stress else 0.7, 0.7))
            t += d
        return cell, t + pulse

    def build_sketch(self):
        r = self.rng
        self.speech = []                       # (singer, start, line number)
        # 0-10 s: low hum; tenor 1 speaks the first line
        for si, t0 in [(BASS, 0.5), (ALTO, 2.0)]:
            self.add(si, Ev(t0, 9.5 - t0, ("hum", "u"), "m", 0.15, 0.5, amp1=0.22))
        self.speech.append((TEN1, 3.2, 1))

        # 10-28 s: the words become a pattern; soprano 1 and alto identify, bass hums on
        pulse = 0.3
        ots = ot_range(SINGERS[TEN1])
        m1 = Model(0, "words", TEN1, 10.0, 28.0)
        m1.cycle, m1.period = self.word_cell(TEN1, WORD_SYLLABLES, pulse, [ots[1], ots[3], ots[2]])
        m1.t1 = m1.t0 + np.ceil((28.0 - m1.t0) / m1.period) * m1.period    # end on a cycle boundary
        self.place(TEN1, m1, m1.t0, m1.t1, lambda t: 0.45 + 0.01 * (t - 10))
        k = lambda n: m1.t0 + n * m1.period
        self.place(SOP1, m1, k(2), m1.t1, lambda t: 0.4, jitter=0.012)
        self.place(ALTO, m1, k(3), m1.t1, lambda t: 0.4, shift=pulse, jitter=0.012)   # canon
        self.add(BASS, Ev(11.0, 16.0, ("hum", "o"), "m", 0.25, 0.5, amp1=0.35))

        # ~24 s: soprano 2 calls KALI; 28-44 s the name replaces the pattern singer by singer
        tcall = self.call(SOP2, 24.0, "KALI", 0.75)
        name_syl = [NAME[i % 2] for i in range(len(WORD_SYLLABLES))]
        named = [(c2, v2, s1) for (c2, v2, _), (_, _, s1) in zip(name_syl, WORD_SYLLABLES)]
        m2 = Model(1, "name", TEN1, 28.0, 44.0)
        m2.t0 = m1.t1
        m2.cycle, m2.period = self.word_cell(TEN1, named, pulse, [ots[1], ots[3], ots[2]])
        m2.t1 = m2.t0 + np.floor((44.0 - m2.t0) / m2.period) * m2.period   # end on a cycle boundary
        # first the caller sings the name in the pattern's rhythm, then the others convert
        order = [(SOP2, 0), (TEN1, 1), (SOP1, 1), (ALTO, 2), (TEN2, 3), (BASS, 3)]
        for si, kk in order:
            start = m2.t0 + kk * m2.period
            if si in (TEN1, SOP1, ALTO):       # until converting, keep singing the words
                self.place(si, m1, m1.t1, start, lambda t: 0.45)
            self.place(si, m2, start, m2.t1, lambda t: 0.5 + 0.012 * (t - 28), jitter=0.01,
                       shift=pulse if si == ALTO else 0.0)

        # 44-60 s: transformation; tempo glides 0.3 -> 0.19 s per pulse while each singer
        # crosses to a new 3+3+2 pattern at its own moment, arriving together
        t = m2.t1
        newc = [("b", "a", 1), ("d", "a", 0), ("d", "a", 0), ("b", "i", 1), ("d", "a", 0), ("d", "a", 0),
                ("t", "u", 1), ("k", "a", 0)]
        cross = {SOP2: 1, TEN1: 2, SOP1: 2, ALTO: 3, TEN2: 4, BASS: 4}
        ncyc = 6
        for c in range(ncyc):
            x = c / (ncyc - 1)
            p = 0.3 + (0.19 - 0.3) * x
            for si in range(6):
                new = c >= cross[si]
                if new:
                    syl = newc
                    cell, per = [], 0.0
                    for cc, v, acc in syl:     # one pulse each; accents on 3+3+2
                        sh = ("ot", ots[[1, 2, 3][len(cell) % 3]], v) if acc else ("v", v)
                        cell.append(Ev(per, p * 0.6, sh, cc, 1.0 if acc else 0.65, 0.7))
                        per += p
                else:
                    cell, per = self.word_cell(TEN1, named, p, [ots[1], ots[3], ots[2]])
                amp = 0.55 + 0.35 * x
                for e in cell:
                    ev = self.mapped(e, TEN1, si)
                    self.add(si, replace(ev, t=t + e.t, amp=e.amp * amp))
            t += per
        # arrival: two cycles of tutti unison at full strength
        for _ in range(2):
            per = 0.0
            for cc, v, acc in newc:
                for si in range(6):
                    sh = ("ot", ots[[1, 2, 3][int(per / 0.19) % 3]], v) if acc else ("v", v)
                    self.add(si, self.mapped(Ev(t + per, 0.19 * 0.6, sh, cc, 0.95 if acc else 0.7, 0.75),
                                             TEN1, si))
                per += 0.19
            t += per
        # cadence on the first accent, then silence
        for si in range(6):
            sh = self.mapped(Ev(0, 1, ("ot", ots[1], "a"), "b", 1.0), TEN1, si).shape
            self.add(si, Ev(t, 1.3, sh, "b", 0.95, 0.75, bend=(0, -150), amp1=0.1))
        t += 1.3 + 2.5
        # coda: alto speaks the second line over a fading hum
        for si in (BASS, TEN2):
            self.add(si, Ev(t, 9.0, ("hum", "u"), "m", 0.2, 0.5, amp1=0.04))
        self.speech.append((ALTO, t + 1.0, 2))
        self.total = t + 9.5
        for ev in self.events:
            ev.sort(key=lambda e: e.t)
            keep = []
            for e in ev:
                if keep and keep[-1].t + keep[-1].dur > e.t - 0.04:
                    if e.t - keep[-1].t < 0.06:
                        continue
                    keep[-1].dur = e.t - 0.04 - keep[-1].t
                keep.append(e)
            ev[:] = keep
        return self


def render_job(args):
    si, events, speech, total, outdir = args
    s = SINGERS[si]
    vtl.init(s["speaker"])
    y = np.zeros(int(total * rv.SR))
    if events:
        tract, glot, gain = rv.tracks_for_singer(si, events, total)
        sung = vtl.synth(tract, glot) * np.repeat(gain[1:], vtl.STATE_SAMPLES)
        y[: len(sung)] = sung[: len(y)]
    voiced = np.abs(y) > 0.1 * np.max(np.abs(y)) if np.any(y) else None
    ref = np.sqrt(np.mean(y[voiced] ** 2)) if voiced is not None and voiced.any() else 0.02
    for t0, line in speech:
        # speak at the bottom of the singer's register: an octave below its chord tone, for high voices
        f0 = s["f0"] if s["f0"] < 260 else s["f0"] / 2
        work = os.path.join(outdir, s["name"])      # VTL writes its score files here
        os.makedirs(work, exist_ok=True)
        sp = vtl.speak(speech_segments(LINE1 if line == 1 else LINE2, f0, line), work)
        sp *= 0.9 * ref / (np.sqrt(np.mean(sp[np.abs(sp) > 0.1 * np.abs(sp).max()] ** 2)) + 1e-12)
        i0 = int(t0 * rv.SR)
        n = min(len(sp), len(y) - i0)
        y[i0:i0 + n] += sp[:n]
    path = os.path.join(outdir, f"{s['name']}.npy")
    np.save(path, y.astype(np.float32))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "sketch_processes"))
    a = ap.parse_args()
    rv.library()
    sk = Sketch(a.seed).build_sketch()
    tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
    jobs = [(si, sk.events[si], [(t, l) for s_, t, l in sk.speech if s_ == si], sk.total, tmp)
            for si in range(len(SINGERS))]
    with Pool(len(jobs)) as pool:
        paths = pool.map(render_job, jobs)
    rv.mix(paths, sk.total, a.out)
    print(f"{sk.total:.0f}s")


if __name__ == "__main__":
    main()
