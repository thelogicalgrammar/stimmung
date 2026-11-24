"""One leader pattern, four models, each with freshly sampled singer programs.

usage: python dsl_sketch.py [--seed 11] [--render]
"""
import argparse
import os
import tempfile
from multiprocessing import Pool

import numpy as np

import dsl
import render_vtl as rv
from score import NAMES, ScoreGen
from singers import SINGERS

HERE = os.path.dirname(os.path.abspath(__file__))


def build(seed, n_models=4, model_dur=20.0, gap=2.0, leader=1):
    rng = np.random.default_rng(seed)
    gen = ScoreGen(seed)
    cell, period, label = gen.pulse_cell(leader, 0.55)
    pulse = period / sum(map(int, label.split("+")))
    tempers = [dsl.sample_temperament(rng) for _ in SINGERS]
    names = list(rng.choice(list(NAMES), 4, replace=False))
    events = [[] for _ in SINGERS]
    history, report, prev, prev_period = [], [], {}, {}
    t0 = 1.0
    report.append(f"leader {SINGERS[leader]['name']}, pattern {label}, cycle {period:.2f}s\n")
    report.append("temperaments:")
    for s, tp in enumerate(tempers):
        report.append(f"  {SINGERS[s]['name']:9s} " + "  ".join(f"{k} {v:.1f}" for k, v in tp.items()))
    for m in range(n_models):
        nc = max(4, int(round(model_dur / period)))

        def factory(r, nc=nc):
            return dsl.Ctx(leader, cell, period, pulse, nc, r, dict(prev), dict(prev_period))
        score, progs, ctx, out, d, st = dsl.choose(factory, leader, tempers, names, history, rng)
        history.append((d, st))
        ev = dsl.to_events(out, ctx, t0, dyn=0.55)
        for s in range(len(SINGERS)):
            events[s] += ev[s]
            last = [n for _, n, _ in out[s] if n]
            prev[s], prev_period[s] = (last[-1] if last else []), period
        report.append(f"\nmodel {m + 1}  {t0:5.1f}s-{t0 + nc * period:5.1f}s  "
                      + "  ".join(f"{k} {v:.2f}" for k, v in zip(dsl.DESC_NAMES, d)))
        for s, p in enumerate(progs):
            report.append(f"  {SINGERS[s]['name']:9s}{'*' if s == leader else ' '} {p}")
        t0 += nc * period + gap
    for ev in events:
        ev.sort(key=lambda e: e.t)
        keep = []
        for e in ev:
            if keep and keep[-1].t + keep[-1].dur > e.t - 0.04:
                if e.t - keep[-1].t < 0.06:
                    continue
                keep[-1].dur = e.t - 0.04 - keep[-1].t
            keep.append(e)
        ev[:] = keep
    return events, t0 + 1.0, "\n".join(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "dsl_sketch"))
    a = ap.parse_args()
    rv.library()
    events, total, report = build(a.seed)
    print(report)
    with open(a.out + "_programs.txt", "w") as f:
        f.write(report + "\n")
    if a.render:
        tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
        with Pool(len(SINGERS)) as pool:
            paths = pool.map(rv.render_singer, [(s, events[s], total, tmp) for s in range(len(SINGERS))])
        rv.mix(paths, total, a.out)


if __name__ == "__main__":
    main()
