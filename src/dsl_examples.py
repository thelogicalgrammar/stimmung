"""Single-voice examples of strategy programs, for the README.

One leader pattern; the bass, alone, performs a series of hand-written programs in the
strategy DSL (src/dsl.py), each rendered to a short mp3.

usage: python dsl_examples.py [--seed 23] [--dry] [--out ../docs/audio]
"""
import argparse
import os
import subprocess
import tempfile

import dsl
import render_vtl as rv
from dsl import (Alternate, Blend, Call, Converge, Cycle, Diverge, Drone, Dyn, End, Leader, Name, Own, Play,
                 Seq, Sustain, T)
from score import ScoreGen
from singers import SINGERS

HERE = os.path.dirname(os.path.abspath(__file__))
SINGER = 0               # bass: centre of the stereo image, widest overtone range
T0 = 0.4                 # lead-in before the first cycle
DUR = 14.0               # seconds per example (rounded to whole cycles)


def then(op, arg=None, at=0.34):
    """The pattern as given, then transformed: the transformation is heard against the original."""
    return Seq(Play(Leader(), Cycle(at)), Play(T(op, arg, Leader()), End()))


EXAMPLES = [
    ("pattern", "The leader's pattern, as everyone first hears it.",
     Play(Leader(), End())),
    ("retro", "Played backwards: the accents now close each group.",
     then("retro")),
    ("augment", "At half speed (the pattern now spans two cycles).",
     then("augment")),
    ("invert", "Overtone contour turned upside down; the rhythm stays.",
     then("invert")),
    ("melisma", "A three-note overtone turn on every accent.",
     then("melisma")),
    ("whisper", "Whispered: no pitch, no overtones, only the tract's noise.",
     then("whisper")),
    ("converge", "Identification as a process: starts heavily transformed and sheds one "
                 "transformation every cycle until it is the leader's pattern.",
     Converge([("hum", None), ("retro", None), ("staccato", None)], 1, End())),
    ("diverge", "The reverse: drifts away from the pattern one transformation at a time.",
     Diverge([("syncopate", None), ("vowels", 2), ("leap", None)], 1)),
    ("blend", "A hummed version dissolves note by note into the sung one.",
     Blend(T("hum", None, Leader()), Leader(), 5, End())),
    ("name", "Calls a magic name, then alternates the pattern with the name's syllables "
             "laid on its rhythm.",
     Seq(Call("YEMAYA"), Alternate(Leader(), Name("YEMAYA"), End()))),
    ("own", "A counter-proposal: a motif of the singer's own on the leader's pulse grid, "
            "in a swell.",
     Dyn("swell", Play(Own(42), End()))),
    ("sustain", "Holds the pattern's first overtone as one long tone per cycle, over a "
                "closing drone.",
     Seq(Sustain(Leader(), Cycle(0.6)), Play(Drone("hum"), End()))),
]


def events_for(prog, cell, period, pulse, nc, seed):
    import numpy as np
    ctx = dsl.Ctx(SINGER, cell, period, pulse, nc, np.random.default_rng(seed))
    progs = [None] * len(SINGERS)
    progs[SINGER] = prog
    out = dsl.interpret(progs, ctx)
    return dsl.to_events(out, ctx, T0, dyn=0.6), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--dry", action="store_true", help="print the pattern and programs, render nothing")
    ap.add_argument("--only", nargs="*", help="render only these examples")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "audio"))
    a = ap.parse_args()
    rv.library()
    gen = ScoreGen(a.seed)
    cell, period, label = gen.pulse_cell(SINGER, 0.4)
    pulse = period / sum(map(int, label.split("+")))
    nc = max(4, int(round(DUR / period)))
    print(f"pattern {label}, cycle {period:.2f}s, {nc} cycles")
    for e in cell:
        print(f"  {e.t:5.2f}s {e.onset or '-':2s} {e.shape}  amp {e.amp:.2f}")
    os.makedirs(a.out, exist_ok=True)
    for i, (name, text, prog) in enumerate(EXAMPLES):
        if a.only and name not in a.only:
            continue
        events, _ = events_for(prog, cell, period, pulse, nc, a.seed + i)
        total = T0 + nc * period + 1.0
        print(f"\n{i + 1:02d} {name}: {prog}\n   {len(events[SINGER])} notes, {total:.1f}s")
        if a.dry:
            continue
        tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
        paths = rv.render_all(events, total, tmp)
        base = os.path.join(tmp, name)
        rv.mix(paths, total, base)
        mp3 = os.path.join(a.out, f"{i + 1:02d}_{name}.mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".wav", "-b:a", "128k", mp3],
                       check=True)
        print("   ->", mp3)


if __name__ == "__main__":
    main()
