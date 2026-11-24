"""A small language for singer strategies, sampled per model.

Each singer's behaviour in a model is a program in a typed DSL:

  Pattern  (one or more cycles of notes, overtones in the leader's harmonic numbers)
    Leader | Prev | Name(N) | Drone(v) | Copy(singer) | Own(seed) | Recall(model)
    | T(op, Pattern)
    ops: rotate(k) retro augment diminish fragment(a,b) accents thin(p) invert
         shift(d) vowels(k) cons(c) whisper legato staccato hum
         mirror densify syncopate accent(k) warp(k) sigh monovowel(v) leap melisma
         tremolo complement
  Stream   (behaviour through time, one step per leader cycle)
    Play(P, until) | Wait(until) | Seq(S, ...) | Canon(shift, P, until)
    | Alternate(P, P, until) | Hocket(i, n, until) | Echo(singer, lag, until)
    | Converge(ops, per) | Diverge(ops, per) | Call(N) | Dyn(curve, S)
    | Blend(P, P, cycles, until) | Sustain(P, until) | Breathe(k, S)
  Cond     Cycle(x) | Joined(n) | After(singer) | Heard | Prob(p) | End
           | Not(c) | Any(c, c) | Quiet(n) | Silent(singer)
  Curve    const | cresc | dim | swell | terrace | pulse | follow(singer)
  Leader   the same streams, plus moves that change the model while it runs:
           cue(op) transforms the shared pattern (everyone following follows the new
           version); adopt(singer) makes a follower's variant the pattern
  Ensemble roles give a group coordinated programs: pair hocket, chain of entries,
           unison block, mirror pair

Programs are sampled from a probabilistic grammar whose weights depend on each
singer's temperament.  An interpreter runs all six programs together, cycle by
cycle, against shared state (who sings, who has joined the leader, which names
were called), so conditions can refer to other singers.  For each model many
candidate program sets are sampled; incoherent ones are rejected and the one
whose behaviour (and structure) is most unlike the recent models is kept.

Strategies have a history: each singer keeps a lineage (one program for leading, one
for following) and usually mutates it rather than starting afresh, and each section of
the piece has a dialect (production weights and favourite transformations).
"""
import copy
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from score import NAMES, OT_VOWELS, Ev, map_overtone, ot_range, vowel_class
from singers import SINGERS

NS = len(SINGERS)


# ---------------------------------------------------------------- patterns
@dataclass
class PVal:
    notes: list          # Ev templates, t relative to the pattern start
    cycles: int = 1


def _sorted(notes):
    return sorted(notes, key=lambda e: e.t)


class Pattern:
    def tag(self):
        return type(self).__name__


class Leader(Pattern):
    def eval(self, ctx, me):
        return PVal(list(ctx.lead_notes))

    def __str__(self):
        return "leader"


class Prev(Pattern):
    """What this singer sang at the end of the previous model (else the leader)."""
    def eval(self, ctx, me):
        prev = ctx.prev.get(me)
        if not prev:
            return PVal(list(ctx.lead_notes))
        L = ctx.period
        return PVal([replace(e, t=e.t * L / max(1e-6, ctx.prev_period[me])) for e in prev
                     if e.t * L / max(1e-6, ctx.prev_period[me]) < L])

    def __str__(self):
        return "my-previous"


class Name(Pattern):
    def __init__(self, name):
        self.name = name

    def eval(self, ctx, me):
        """The name's syllables laid on the leader's accents and pulses."""
        syl = NAMES[self.name]
        notes = []
        for j, e in enumerate(ctx.lead_notes):
            c, v, stress = syl[j % len(syl)]
            shape = ("hum", "u") if v == "hum" else \
                (("ot", e.shape[1], vowel_class(v)) if e.shape[0] == "ot" else ("v", v))
            notes.append(replace(e, shape=shape, onset=c))
        return PVal(notes)

    def __str__(self):
        return f"name({self.name})"


class Drone(Pattern):
    def __init__(self, vowel):
        self.vowel = vowel

    def eval(self, ctx, me):
        shape = ("hum", "u") if self.vowel == "hum" else ("v", self.vowel)
        return PVal([Ev(0.05, ctx.period - 0.15, shape, "m", 0.55, 0.5, legato=False)])

    def __str__(self):
        return f"drone({self.vowel})"


class Copy(Pattern):
    """Another singer's most recent material (not necessarily the leader's)."""
    def __init__(self, s):
        self.s = s

    def eval(self, ctx, me):
        for notes in reversed(ctx.history[self.s]):
            if notes:
                return PVal([e for e in notes if e.t < ctx.period])
        return PVal(list(ctx.lead_notes))

    def __str__(self):
        return f"copy({SINGERS[self.s]['name']})"


class Own(Pattern):
    """A motif of the singer's own on the leader's pulse grid: a counter-proposal."""
    def __init__(self, seed):
        self.seed = seed

    def eval(self, ctx, me):
        r = np.random.default_rng(self.seed)
        rng_n = ot_range(SINGERS[ctx.leader])
        npulse = max(2, int(round(ctx.period / ctx.pulse)))
        cons = list(r.choice(["b", "d", "m", "n", "l", "y", "w", "t", "k"], 2, replace=False))
        vow = list(r.choice(OT_VOWELS, 2, replace=False))
        c = int(r.integers(len(rng_n)))
        notes, p = [], 0
        while p < npulse:
            g = int(r.choice([2, 3]))
            notes.append(Ev(p * ctx.pulse, ctx.pulse * min(g, npulse - p) * 0.7, ("ot", rng_n[c], vow[0]),
                            cons[0], 1.0, 0.75))
            if g == 3 and p + 2 < npulse and r.random() < 0.7:
                notes.append(Ev((p + 2) * ctx.pulse, ctx.pulse * 0.6, ("v", vow[1]), cons[1], 0.65, 0.6))
            c = int(np.clip(c + r.choice([-1, 1, 2]), 0, len(rng_n) - 1))
            p += g
        return PVal(notes)

    def __str__(self):
        return f"own#{self.seed % 1000}"


class Recall(Pattern):
    """A quotation: the leader's pattern of an earlier (not the previous) model, moved into
    this leader's harmonics and fitted to this model's cycle."""
    def __init__(self, j):
        self.j = j

    def eval(self, ctx, me):
        mem = ctx.memory.get(self.j)
        if not mem:
            return PVal(list(ctx.lead_notes))
        notes, period, lead = mem
        sc = ctx.period / max(1e-6, period)
        out = []
        for e in notes:
            if e.t * sc >= ctx.period:
                continue
            shape = ("ot", map_overtone(e.shape[1], lead, SINGERS[ctx.leader]), e.shape[2]) \
                if e.shape[0] == "ot" else e.shape
            out.append(replace(e, t=e.t * sc, dur=e.dur * min(1.0, sc), shape=shape))
        return PVal(out)

    def __str__(self):
        return f"recall(model {self.j + 1})"


class T(Pattern):
    """A transformation applied to a pattern."""
    def __init__(self, op, arg, p):
        self.op, self.arg, self.p = op, arg, p

    def tag(self):
        return "T:" + self.op

    def __str__(self):
        a = "" if self.arg is None else (f"{self.arg:g}" if isinstance(self.arg, float) else str(self.arg))
        a = ", ".join(map(str, self.arg)) if isinstance(self.arg, tuple) else a
        return f"{self.op}({a + ', ' if a else ''}{self.p})"

    def eval(self, ctx, me):
        pv = self.p.eval(ctx, me)
        return apply_op(self.op, self.arg, pv, ctx)


def apply_op(op, arg, pv, ctx):
    L = ctx.period * pv.cycles
    n = list(pv.notes)
    pulse = ctx.pulse
    rng_n = ot_range(SINGERS[ctx.leader])
    if not n:
        return pv
    if op == "rotate":
        n = [replace(e, t=(e.t + arg * pulse) % L) for e in n]
    elif op == "retro":
        n = [replace(e, t=max(0.0, L - e.t - e.dur - pulse * 0.2)) for e in n]
    elif op == "augment":
        return PVal(_sorted([replace(e, t=e.t * 2, dur=e.dur * 2) for e in n]), pv.cycles * 2)
    elif op == "diminish":
        h = [replace(e, t=e.t / 2, dur=max(0.07, e.dur / 2)) for e in n]
        n = h + [replace(e, t=e.t + L / 2) for e in h]
    elif op == "fragment":
        a, b = arg
        n = [e for e in n if a <= e.t / L < b] or n[:1]
    elif op == "accents":
        n = [replace(e, dur=e.dur * 1.4) for e in n if e.amp >= 0.99] or n[:1]
    elif op == "thin":
        keep = np.random.default_rng(int(arg * 1000)).random(len(n)) > arg
        n = [e for e, k in zip(n, keep) if k] or n[:1]
    elif op in ("invert", "shift"):
        idx = [rng_n.index(e.shape[1]) for e in n if e.shape[0] == "ot" and e.shape[1] in rng_n]
        lo, hi = (min(idx), max(idx)) if idx else (0, 0)
        out = []
        for e in n:
            if e.shape[0] == "ot" and e.shape[1] in rng_n:
                i = rng_n.index(e.shape[1])
                j = lo + hi - i if op == "invert" else i + arg
                e = replace(e, shape=("ot", rng_n[int(np.clip(j, 0, len(rng_n) - 1))], e.shape[2]))
            out.append(e)
        n = out
    elif op == "vowels":
        def rot(v):
            return OT_VOWELS[(OT_VOWELS.index(v) + arg) % len(OT_VOWELS)] if v in OT_VOWELS else v
        n = [replace(e, shape=(e.shape[0], e.shape[1], rot(e.shape[2])) if e.shape[0] == "ot"
                     else (e.shape[0], rot(e.shape[1]))) for e in n]
    elif op == "cons":
        n = [replace(e, onset=arg) for e in n]
    elif op == "whisper":
        n = [replace(e, whisper=True, shape=("v", e.shape[-1]) if e.shape[0] == "ot" else e.shape,
                     onset=e.onset if e.onset not in ("m", "n") else "h") for e in n]
    elif op == "hum":
        n = [replace(e, shape=("hum", "u"), onset="m", dur=e.dur * 1.2) for e in n]
    elif op == "legato":
        n = _sorted(n)
        n = [replace(e, dur=max(e.dur, ((n[i + 1].t if i + 1 < len(n) else L) - e.t) * 0.95), legato=i > 0)
             for i, e in enumerate(n)]
    elif op == "staccato":
        n = [replace(e, dur=max(0.07, e.dur * 0.45)) for e in n]
    elif op == "mirror":                 # first half, then its reflection
        h = [e for e in n if e.t < L / 2] or n[:1]
        n = h + [replace(e, t=max(L / 2, L - e.t - e.dur)) for e in h]
    elif op == "densify":                # fill long rests with light pulses
        n = _sorted(n)
        extra = []
        for a, b in zip(n, n[1:] + [replace(n[0], t=n[0].t + L)]):
            t = a.t + 2 * pulse
            while t <= b.t - pulse * 0.9:
                extra.append(Ev(t % L, pulse * 0.5, ("v", a.shape[-1] if a.shape[0] != "hum" else "u"),
                                str(a.onset or "n"), 0.55, 0.55))
                t += pulse
        n = n + extra
    elif op == "syncopate":              # weak notes half a pulse late
        n = [e if e.amp >= 0.99 else replace(e, t=(e.t + pulse / 2) % L) for e in n]
    elif op == "accent":                 # move the accents arg notes along
        n = _sorted(n)
        amps = [e.amp for e in n]
        n = [replace(e, amp=amps[(i - arg) % len(n)]) for i, e in enumerate(n)]
    elif op == "warp":                   # accelerate (arg<1) or hold back (arg>1) within the cycle
        n = [replace(e, t=L * (e.t / L) ** arg,
                     dur=max(0.07, e.dur * arg * max(1e-3, e.t / L) ** (arg - 1) if e.t > 0 else e.dur))
             for e in n]
        n = [replace(e, dur=min(e.dur, 3 * pulse)) for e in n]
    elif op == "sigh":                   # accents fall away
        n = [replace(e, bend=(0, -220), amp1=e.amp * 0.55, dur=e.dur * 1.3) if e.amp >= 0.99 else e
             for e in n]
    elif op == "monovowel":
        n = [replace(e, shape=("ot", e.shape[1], arg) if e.shape[0] == "ot" else
                     (("v", arg) if e.shape[0] == "v" else e.shape)) for e in n]
    elif op == "leap":                   # accents jump to the top of the range
        n = [replace(e, shape=("ot", rng_n[-1 - (i % 2)], e.shape[2]))
             if e.amp >= 0.99 and e.shape[0] == "ot" else e for i, e in enumerate(n)]
    elif op == "melisma":                # an overtone turn on each accent
        out = []
        for e in n:
            if e.amp >= 0.99 and e.shape[0] == "ot" and e.shape[1] in rng_n and e.dur > 0.2:
                i = rng_n.index(e.shape[1])
                d = e.dur / 3
                for j, di in enumerate([0, 1, 0]):
                    out.append(replace(e, t=e.t + j * d, dur=d * 0.98, legato=j > 0, onset=e.onset if j == 0 else "",
                                       shape=("ot", rng_n[min(len(rng_n) - 1, i + di)], e.shape[2])))
            else:
                out.append(e)
        n = out
    elif op == "tremolo":                # long notes re-articulated on half pulses
        out = []
        for e in n:
            if e.dur >= 1.4 * pulse:
                k = int(e.dur / (pulse / 2))
                out += [replace(e, t=e.t + j * pulse / 2, dur=pulse * 0.35, onset=e.onset if j == 0 else "h",
                                amp=e.amp * (1 if j == 0 else 0.75)) for j in range(k)]
            else:
                out.append(e)
        n = out
    elif op == "complement":             # sing only on pulses where the leader is silent
        on = {int(round(e.t / pulse)) for e in ctx.lead_notes}
        grid = int(round(ctx.period / pulse))
        v = next((e.shape[-1] for e in n if e.shape[0] != "hum"), "a")
        n = [Ev(k * pulse, pulse * 0.55, ("v", v), "l" if k % 2 else "n", 0.7, 0.6)
             for k in range(grid) if k not in on] or n[:1]
    return PVal(_sorted(n), pv.cycles)


# ---------------------------------------------------------------- conditions
class Cond:
    pass


class Cycle(Cond):
    def __init__(self, x):
        self.x = x

    def __call__(self, ctx, me):
        return ctx.k >= self.x * ctx.nc

    def __str__(self):
        return f"cycle≥{self.x:.0%}"


class Joined(Cond):
    def __init__(self, n):
        self.n = n

    def __call__(self, ctx, me):
        return len(ctx.joined - {me}) >= self.n

    def __str__(self):
        return f"joined≥{self.n}"


class After(Cond):
    def __init__(self, s):
        self.s = s

    def __call__(self, ctx, me):
        return self.s in ctx.joined_ever

    def __str__(self):
        return f"after({SINGERS[self.s]['name']})"


class Heard(Cond):
    def __call__(self, ctx, me):
        return bool(ctx.called)

    def __str__(self):
        return "name-heard"


class Prob(Cond):
    def __init__(self, p):
        self.p = p

    def __call__(self, ctx, me):
        return ctx.rng.random() < self.p

    def __str__(self):
        return f"chance({self.p:g})"


class Not(Cond):
    def __init__(self, c):
        self.c = c

    def __call__(self, ctx, me):
        return not self.c(ctx, me)

    def __str__(self):
        return f"not {self.c}"


class Any(Cond):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, ctx, me):
        return self.a(ctx, me) or self.b(ctx, me)

    def __str__(self):
        return f"({self.a} or {self.b})"


class Quiet(Cond):
    """Fewer than n others sang in the last cycle."""
    def __init__(self, n):
        self.n = n

    def __call__(self, ctx, me):
        return len(ctx.singing - {me}) < self.n

    def __str__(self):
        return f"fewer-than-{self.n}-singing"


class Silent(Cond):
    def __init__(self, s):
        self.s = s

    def __call__(self, ctx, me):
        return self.s not in ctx.singing

    def __str__(self):
        return f"silent({SINGERS[self.s]['name']})"


class End(Cond):
    def __call__(self, ctx, me):
        return False

    def __str__(self):
        return "end"


# ---------------------------------------------------------------- streams
# each run() is a generator yielding, per leader cycle, (notes, identical_to_leader)

CURVES = {"const": lambda x, ctx: 1.0, "cresc": lambda x, ctx: 0.6 + 0.6 * x,
          "dim": lambda x, ctx: 1.15 - 0.6 * x, "swell": lambda x, ctx: 0.6 + 0.55 * np.sin(np.pi * x),
          "terrace": lambda x, ctx: (0.65, 0.9, 1.15)[min(2, int(3 * x))],
          "pulse": lambda x, ctx: 1.1 if ctx.k % 2 == 0 else 0.7}


def curve_gain(curve, ctx):
    x = ctx.k / max(1, ctx.nc - 1)
    if curve.startswith("follow:"):      # match another singer's loudness in the last cycle
        s = int(curve.split(":")[1])
        h = ctx.history[s][-1] if ctx.history[s] else []
        return float(np.clip(np.mean([e.amp for e in h]) / 0.8, 0.5, 1.3)) if h else 0.8
    return CURVES[curve](x, ctx)


class Stream:
    def tag(self):
        return type(self).__name__


def _play(ctx, me, pv, c, shift=0.0):
    """The slice of a (possibly multi-cycle) pattern that falls in its c-th cycle."""
    L = ctx.period
    k = c % pv.cycles
    return [replace(e, t=e.t - k * L + shift) for e in pv.notes if k * L <= e.t < (k + 1) * L]


def _timeout(ctx):
    return max(2, int(0.65 * ctx.nc))


class Play(Stream):
    def __init__(self, p, until):
        self.p, self.until = p, until

    def run(self, ctx, me):
        c = 0
        while not self.until(ctx, me):
            pv = self.p.eval(ctx, me)
            yield _play(ctx, me, pv, c), isinstance(self.p, Leader)
            c += 1

    def __str__(self):
        return f"play({self.p}) until {self.until}"


class Wait(Stream):
    def __init__(self, until):
        self.until = until

    def run(self, ctx, me):
        c = 0
        while not self.until(ctx, me) and c < _timeout(ctx):
            yield [], False
            c += 1

    def __str__(self):
        return f"wait until {self.until}"


class Seq(Stream):
    def __init__(self, *parts):
        self.parts = parts

    def run(self, ctx, me):
        for p in self.parts:
            ctx.phase[me] = str(p)
            yield from p.run(ctx, me)

    def __str__(self):
        return "; ".join(map(str, self.parts))


class Canon(Stream):
    def __init__(self, shift, p, until):
        self.shift, self.p, self.until = shift, p, until

    def run(self, ctx, me):
        c = 0
        while not self.until(ctx, me):
            yield _play(ctx, me, self.p.eval(ctx, me), c, self.shift * ctx.pulse), isinstance(self.p, Leader)
            c += 1

    def __str__(self):
        return f"canon(+{self.shift} pulses, {self.p}) until {self.until}"


class Alternate(Stream):
    def __init__(self, a, b, until):
        self.a, self.b, self.until = a, b, until

    def run(self, ctx, me):
        c = 0
        while not self.until(ctx, me):
            p = (self.a, self.b)[c % 2]
            yield _play(ctx, me, p.eval(ctx, me), c), isinstance(p, Leader)
            c += 1

    def __str__(self):
        return f"alternate({self.a} | {self.b}) until {self.until}"


class Hocket(Stream):
    def __init__(self, i, n, until):
        self.i, self.n, self.until = i, n, until

    def run(self, ctx, me):
        while not self.until(ctx, me):
            yield [e for j, e in enumerate(ctx.lead_notes) if j % self.n == self.i], True

    def __str__(self):
        return f"hocket(notes {self.i} mod {self.n}) until {self.until}"


class Echo(Stream):
    def __init__(self, s, lag, until):
        self.s, self.lag, self.until = s, lag, until

    def run(self, ctx, me):
        while not self.until(ctx, me):
            h = ctx.history[self.s]
            src = h[ctx.k - self.lag] if ctx.k - self.lag >= 0 and ctx.k - self.lag < len(h) else []
            yield [replace(e, amp=e.amp * 0.8) for e in src], False

    def __str__(self):
        return f"echo({SINGERS[self.s]['name']}, {self.lag} cycle{'s' * (self.lag > 1)} late) until {self.until}"


class Converge(Stream):
    """Start from the leader's pattern under a stack of transformations; drop one every
    `per` cycles until identical (identification as a process)."""
    def __init__(self, ops, per, hold=None):
        self.ops, self.per, self.hold = ops, per, hold or Cycle(0.0)

    def run(self, ctx, me):
        for i in range(len(self.ops), 0, -1):
            p = Leader()
            for op, arg in self.ops[:i]:
                p = T(op, arg, p)
            pv = p.eval(ctx, me)
            for c in range(self.per * pv.cycles):
                yield _play(ctx, me, pv, c), False
        yield list(ctx.lead_notes), True        # identical; then hold the identity
        while not self.hold(ctx, me):
            yield list(ctx.lead_notes), True

    def __str__(self):
        chain = " ∘ ".join(f"{o}" + (f"({a})" if a is not None else "") for o, a in self.ops)
        hold = "" if isinstance(self.hold, Cycle) and self.hold.x == 0 else f", then hold until {self.hold}"
        return f"converge[{chain}] every {self.per}{hold}"


class Diverge(Converge):
    """The reverse: from the leader's pattern, add one transformation every `per` cycles."""
    def run(self, ctx, me):
        for i in range(0, len(self.ops) + 1):
            p = Leader()
            for op, arg in self.ops[:i]:
                p = T(op, arg, p)
            pv = p.eval(ctx, me)
            for c in range(self.per * pv.cycles):
                yield _play(ctx, me, pv, c), i == 0
        while True:                      # hold the last state
            yield _play(ctx, me, pv, 0), False

    def __str__(self):
        chain = " ∘ ".join(f"{o}" + (f"({a})" if a is not None else "") for o, a in self.ops)
        return f"diverge[{chain}] every {self.per}"


class Call(Stream):
    def __init__(self, name):
        self.name = name

    def run(self, ctx, me):
        notes, t = [], 0.05
        for i, (c, v, stress) in enumerate(NAMES[self.name]):
            last = i == len(NAMES[self.name]) - 1
            d = (0.6 if stress else 0.22) + (0.4 if last else 0.0)
            shape = ("hum", "u") if v == "hum" else ("v", v)
            notes.append(Ev(t, d, shape, c, 1.1 if stress else 0.85, 0.8,
                            bend=(0, -450) if last else ((-150, 0) if stress else None)))
            t += d + 0.03
        ctx.called.add(self.name)
        yield notes, False

    def __str__(self):
        return f"call {self.name}"


class Blend(Stream):
    """Pattern a dissolves into pattern b note by note over `cycles`, then b holds."""
    def __init__(self, a, b, cycles, until):
        self.a, self.b, self.cycles, self.until = a, b, cycles, until

    def run(self, ctx, me):
        r = np.random.default_rng(self.cycles * 7919 + me)
        c = 0
        while c <= self.cycles or not self.until(ctx, me):
            x = min(1.0, (c + 1) / (self.cycles + 1))
            A = _play(ctx, me, self.a.eval(ctx, me), c)
            B = _play(ctx, me, self.b.eval(ctx, me), c)
            notes = [e for e in A if r.random() > x] + [e for e in B if r.random() < x]
            yield _sorted(notes), x >= 1.0 and isinstance(self.b, Leader)
            c += 1

    def __str__(self):
        return f"blend({self.a} → {self.b}, {self.cycles} cycles) until {self.until}"


class Sustain(Stream):
    """Hold the pattern's first overtone as one long tone per cycle."""
    def __init__(self, p, until):
        self.p, self.until = p, until

    def run(self, ctx, me):
        while not self.until(ctx, me):
            pv = self.p.eval(ctx, me)
            src = next((e for e in pv.notes if e.shape[0] == "ot"), pv.notes[0] if pv.notes else None)
            shape = src.shape if src is not None else ("hum", "u")
            yield [Ev(0.03, ctx.period - 0.08, shape, "m", 0.7, 0.6, legato=True)], False

    def __str__(self):
        return f"sustain({self.p}) until {self.until}"


class Breathe(Stream):
    """Drop out every k-th cycle (the program keeps running underneath)."""
    def __init__(self, k, s):
        self.k, self.s = k, s

    def run(self, ctx, me):
        for i, (notes, ident) in enumerate(self.s.run(ctx, me)):
            yield ([] if i % self.k == self.k - 1 else notes), ident and i % self.k != self.k - 1

    def __str__(self):
        return f"breathe(every {self.k}, {self.s})"


class Dyn(Stream):
    def __init__(self, curve, s):
        self.curve, self.s = curve, s

    def run(self, ctx, me):
        for notes, ident in self.s.run(ctx, me):
            g = curve_gain(self.curve, ctx)
            yield [replace(e, amp=e.amp * g) for e in notes], ident

    def __str__(self):
        c = self.curve if not self.curve.startswith("follow:") else \
            f"follow({SINGERS[int(self.curve.split(':')[1])]['name']})"
        return f"{c}{{ {self.s} }}"


class Cue(Stream):
    """The leader transforms the model's pattern; from now on everyone who follows the
    leader follows the new version."""
    def __init__(self, op, arg):
        self.op, self.arg = op, arg

    def run(self, ctx, me):
        pv = apply_op(self.op, self.arg, PVal(list(ctx.lead_notes)), ctx)
        notes = _sorted(_play(ctx, me, pv, 0))
        if notes:
            ctx.lead_notes = notes
            ctx.cues.append((ctx.k, str(self)))
        yield list(ctx.lead_notes), True

    def __str__(self):
        a = "" if self.arg is None else f"({self.arg})"
        return f"cue {self.op}{a}"


class Adopt(Stream):
    """The leader takes a follower's current variant as the model's pattern."""
    def __init__(self, s):
        self.s = s

    def run(self, ctx, me):
        h = next((n for n in reversed(ctx.history[self.s]) if n), None)
        if h and h != ctx.lead_notes:
            ctx.lead_notes = _sorted([replace(e, t=e.t % ctx.period, amp=min(1.0, e.amp / 0.8))
                                      for e in h])
            ctx.cues.append((ctx.k, str(self)))
        yield list(ctx.lead_notes), True

    def __str__(self):
        return f"adopt {SINGERS[self.s]['name']}'s variant"


class Role(Stream):
    """A program given as part of an ensemble role (for display)."""
    def __init__(self, name, s):
        self.name, self.s = name, s

    def run(self, ctx, me):
        yield from self.s.run(ctx, me)

    def tag(self):
        return "Role:" + self.name.split()[0]

    def __str__(self):
        return f"[{self.name}] {self.s}"


def walk(node):
    """All nodes of a program (for structural comparison)."""
    yield node
    for v in vars(node).values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, (Pattern, Stream)):
                yield from walk(x)
            elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
                yield ("op", x[0])


def tags(node):
    return Counter(n.tag() if hasattr(n, "tag") else f"T:{n[1]}" for n in walk(node))


# ---------------------------------------------------------------- sampling
OPS = ["rotate", "retro", "augment", "diminish", "fragment", "accents", "thin", "invert", "shift",
       "vowels", "cons", "whisper", "legato", "staccato", "hum", "mirror", "densify", "syncopate",
       "accent", "warp", "sigh", "monovowel", "leap", "melisma", "tremolo", "complement"]
TRAITS = ["eager", "stubborn", "ornate", "imitative", "caller", "quiet"]
# what a leader's cue may do to the pattern (ops that keep it a one-cycle pattern)
LEAD_OPS = ["rotate", "retro", "invert", "shift", "vowels", "syncopate", "accent", "leap", "melisma",
            "sigh", "monovowel", "densify", "diminish", "thin", "cons", "mirror", "tremolo"]
PRODUCTIONS = ["leader", "prev", "name", "drone", "copy", "own", "recall", "T", "play", "converge",
               "diverge", "canon", "alternate", "hocket", "echo", "call", "blend", "sustain", "cue",
               "adopt", "ensemble", "inherit"]


def sample_dialect(rng):
    """A section's way of behaving: production weights, and a handful of favourite
    transformations that come up far more often than the rest."""
    d = {k: float(np.exp(rng.normal(0, 0.7))) for k in PRODUCTIONS}
    fav = set(rng.choice(OPS, 5, replace=False))
    d.update({"op:" + o: (4.0 if o in fav else 0.6) for o in OPS})
    d["favourites"] = sorted(fav)
    return d


def ensemble(rng, leader, parts, sampler):
    """A coordinated group among the followers: (description, {singer: program})."""
    r = rng
    fol = [s for s in parts if s != leader]
    if len(fol) < 2:
        return None, {}
    kinds, w = ["pair hocket", "chain", "unison", "mirror pair"], np.array([1.0, 1.3, 0.6, 0.9])
    kind = str(r.choice(kinds, p=w / w.sum()))
    k = 2 if kind in ("pair hocket", "mirror pair") else \
        int(r.integers(2, min(4 if kind == "chain" else 3, len(fol)) + 1))
    g = [int(x) for x in r.choice(fol, k, replace=False)]
    names = "+".join(SINGERS[s]["name"] for s in g)
    start = Cycle(round(float(r.uniform(0.1, 0.4)), 2))
    if kind == "pair hocket":
        progs = {s: Seq(Wait(start), Hocket(i, 2, End())) for i, s in enumerate(g)}
    elif kind == "chain":
        style = str(r.choice(["unison", "converge", "canon"]))
        ops = [sampler.op() for _ in range(int(r.integers(1, 3)))]
        progs = {}
        for i, s in enumerate(g):
            w = Wait(start if i == 0 else After(g[i - 1]))
            body = {"unison": lambda: Play(Leader(), End()),
                    "converge": lambda: Converge(list(ops), 1),
                    "canon": lambda: Seq(Canon(i + 1, Leader(), Cycle(0.8)), Play(Leader(), End()))}[style]()
            progs[s] = Seq(w, body)
        kind = f"chain ({style})"
    elif kind == "unison":
        one = sampler.program()
        progs = {s: copy.deepcopy(one) for s in g}
    else:
        p = sampler.non_leader()
        x = Cycle(round(float(r.uniform(0.45, 0.7)), 2))
        progs = {g[0]: Seq(Wait(start), Play(p, x), Blend(p, Leader(), 3, End())),
                 g[1]: Seq(Wait(start), Play(T("invert", None, p), x), Blend(T("invert", None, p), Leader(), 3, End()))}
    desc = f"{kind}: {names}"
    return desc, {s: Role(desc.split(":")[0], pr) for s, pr in progs.items()}


def sample_temperament(rng):
    return {t: float(rng.uniform(0.2, 1.0)) for t in TRAITS}


class Sampler:
    def __init__(self, rng, temper, me, leader, names, parts, bias=None, lead_call=None, memory=(),
                 dialect=None):
        self.r, self.t, self.me, self.leader, self.names = rng, temper, me, leader, names
        self.dialect = dialect or {}     # the section's production weights ("op:x" for transformations)
        self.memory = list(memory)       # earlier models that may be quoted (most salient first)
        self.parts = parts
        self.bias = bias or {}           # production name -> weight multiplier (model type)
        self.lead_call = lead_call       # name the leader opens with (name models)

    def d(self, name):
        return self.dialect.get(name, 1.0)

    def pick(self, options):
        """options: [(weight, thunk)]"""
        w = np.array([max(1e-6, o[0]) for o in options])
        return options[int(self.r.choice(len(options), p=w / w.sum()))][1]()

    def op(self):
        r, t = self.r, self.t
        w = np.array([self.d("op:" + o) for o in OPS])
        op = str(r.choice(OPS, p=w / w.sum()))
        arg = {"rotate": int(r.integers(1, 4)), "fragment": tuple(float(x) for x in sorted(np.round(r.uniform(0, 1, 2), 2))),
               "thin": float(np.round(r.uniform(0.3, 0.6), 2)), "shift": int(r.choice([-2, -1, 1, 2])),
               "vowels": int(r.integers(1, 4)), "cons": str(r.choice(["b", "d", "l", "n", "t", "k", "y", "w"])),
               "accent": int(r.integers(1, 3)), "warp": float(r.choice([0.7, 1.4])),
               "monovowel": str(r.choice(OT_VOWELS)),
               }.get(op)
        if op == "fragment" and arg[1] - arg[0] < 0.3:
            arg = (round(max(0.0, arg[1] - 0.45), 2), arg[1])
        return op, arg

    def pattern(self, depth=0):
        r, t = self.r, self.t
        d = self.d
        opts = [(3.0 * d("leader"), Leader), (0.8 * t["imitative"] * d("prev"), Prev),
                (0.5 * t["caller"] * d("name"), lambda: Name(str(r.choice(self.names)))),
                (0.6 * t["quiet"] * d("drone"), lambda: Drone(str(r.choice(["hum", "u", "o", "schwa"])))),
                (0.6 * t["imitative"] * (len(self.parts) > 2) * d("copy"),
                 lambda: Copy(int(r.choice([s for s in self.parts if s not in (self.me, self.leader)]
                                           or [self.leader])))),
                ((0.5 * (1 - t["imitative"]) + 0.2 * t["ornate"]) * d("own"), lambda: Own(int(r.integers(1 << 20)))),
                (0.9 * (0.3 + t["imitative"]) * bool(self.memory) * d("recall"),
                 lambda: Recall(int(self.memory[min(len(self.memory) - 1, int(r.geometric(0.45)) - 1)])))]
        if depth < 2:
            opts.append((2.5 * t["ornate"] * d("T"), lambda: T(*self.op(), self.pattern(depth + 1))))
        return self.pick(opts)

    def cond(self, late=False):
        r, t = self.r, self.t
        others = [s for s in self.parts if s != self.me]
        x0 = 0.15 + 0.5 * (1 - t["eager"]) + (0.25 if late else 0.0)
        return self.pick([(2.0, lambda: Cycle(round(float(np.clip(x0 + r.normal(0, 0.1), 0.05, 0.9)), 2))),
                          (1.0, lambda: Joined(int(r.integers(1, max(2, min(4, len(self.parts) - 1)))))),
                          (1.0 * t["imitative"], lambda: After(int(r.choice(others)))),
                          (0.5 * t["caller"], Heard),
                          (0.6, lambda: Prob(round(float(r.uniform(0.15, 0.5)), 2))),
                          (0.5 * t["quiet"], lambda: Quiet(int(r.integers(1, 3)))),
                          (0.4 * t["imitative"] * bool(others), lambda: Silent(int(r.choice(others)))),
                          (0.3, lambda: Not(Joined(int(r.integers(1, 3))))),
                          (0.3, lambda: Any(self.cond(late), self.cond(late)) if r.random() < 0.5
                           else Cycle(round(float(np.clip(x0, 0.05, 0.9)), 2)))])

    def phase(self, final):
        r, t = self.r, self.t
        until = End() if final else self.cond(late=True)
        others = [s for s in self.parts if s != self.me]
        b = lambda name: self.bias.get(name, 1.0) * self.d(name)
        return self.pick([
            (2.0 * b("play"), lambda: Play(self.pattern(), until)),
            ((2.2 * t["stubborn"] + 0.6) * b("converge"), lambda: Converge([self.op() for _ in range(int(r.integers(1, 4)))],
                                                         int(r.integers(1, 3)), until)),
            ((1.0 * (1 - t["stubborn"]) * (not final) + 0.3) * b("diverge"), lambda: Diverge([self.op() for _ in range(
                int(r.integers(1, 3)))], int(r.integers(1, 3)))),
            (1.2 * b("canon"), lambda: Canon(int(r.integers(1, 4)), self.pattern(1), until)),
            (0.8 * b("alternate"), lambda: self.alternate(until)),
            (0.8 * b("hocket"), lambda: self.hocket(until)),
            (1.0 * t["imitative"] * b("echo") * bool(others),
             lambda: Echo(int(r.choice(others)), int(r.integers(1, 3)), until)),
            (0.8 * t["caller"] * (not final) * b("call"), lambda: Call(str(r.choice(self.names)))),
            (1.0 * b("blend"), lambda: Blend(self.non_leader(), Leader(), int(r.integers(2, 5)), until)),
            (0.6 * t["quiet"] * b("sustain"), lambda: Sustain(self.pattern(1), until)),
        ])

    def lead_op(self):
        for _ in range(20):
            op, arg = self.op()
            if op in LEAD_OPS:
                return op, arg
        return "rotate", 1

    def trigger(self):
        """When the leader makes its next move: once others have joined, at a point in time,
        or once a particular singer has joined."""
        r = self.r
        others = [s for s in self.parts if s != self.me]
        return self.pick([
            (2.0 * bool(others), lambda: Joined(int(r.integers(1, max(2, len(others) + 1))))),
            (1.5, lambda: Cycle(round(float(r.uniform(0.25, 0.7)), 2))),
            (0.8 * bool(others), lambda: After(int(r.choice(others))) if others else Cycle(0.5))])

    def lead_program(self):
        """The leader presents the pattern and may change the model as it goes: a cue
        transforms the pattern for everyone, adopt takes over a follower's variant."""
        r, b = self.r, lambda name: self.bias.get(name, 1.0) * self.d(name)
        others = [s for s in self.parts if s != self.me]
        phases = [Call(self.lead_call)] if self.lead_call else []
        w = np.array([3.0, 2.0 * b("cue"), 1.0 * b("cue")])
        for _ in range(int(r.choice(3, p=w / w.sum()))):
            phases.append(Play(Leader(), self.trigger()))
            phases.append(self.pick([
                (2.0 * b("cue"), lambda: Cue(*self.lead_op())),
                (1.0 * b("adopt") * bool(others), lambda: Adopt(int(r.choice(others)))),
                (0.5 * b("call") * (not self.lead_call), lambda: Call(str(r.choice(self.names))))]))
        if r.random() < 0.15:
            phases += [Play(Leader(), Cycle(round(float(r.uniform(0.6, 0.85)), 2))),
                       Diverge([self.op() for _ in range(2)], 1)]
        else:
            phases.append(Play(Leader(), End()))
        return Dyn(str(r.choice(list(CURVES))), Seq(*phases))

    # -- evolving a program instead of sampling afresh ------------------------------
    def repair(self, prog):
        """Re-point references to singers who are not taking part (or to oneself)."""
        others = [s for s in self.parts if s != self.me] or [self.leader]
        for node in walk(prog):
            if isinstance(node, (Copy, After, Silent, Echo, Adopt)) and (node.s not in self.parts or node.s == self.me):
                node.s = int(self.r.choice(others))
            if isinstance(node, Dyn) and node.curve.startswith("follow:") and \
                    int(node.curve.split(":")[1]) not in others:
                node.curve = "follow:" + str(int(self.r.choice(others)))
            if isinstance(node, Blend) and str(node.a) == str(node.b):
                node.a = self.non_leader()
            if isinstance(node, Recall) and node.j not in self.memory:
                node.j = int(self.r.choice(self.memory)) if self.memory else node.j
        return prog

    def _slots(self, node, out):
        for k, v in vars(node).items():
            if isinstance(v, Pattern):
                out.append(("pat", node, k))
                self._slots(v, out)
            elif isinstance(v, Cond):
                if not isinstance(v, End):
                    out.append(("cond", node, k))
            elif isinstance(v, Stream):
                self._slots(v, out)
            elif k == "parts":
                node.parts = list(v)
                for i, x in enumerate(node.parts):
                    out.append(("phase", node, i))
                    self._slots(x, out)
            elif k == "ops":
                out.append(("ops", node, k))
            elif k in ("shift", "lag", "cycles", "k", "per") and isinstance(v, int):
                out.append(("int", node, k))
            elif k == "curve" and not v.startswith("follow:"):
                out.append(("curve", node, k))
        if isinstance(node, (T, Cue)):
            out.append(("op", node, None))
        return out

    def mutate(self, prog, n=1):
        r = self.r
        prog = self.repair(copy.deepcopy(prog))
        for _ in range(n):
            slots = self._slots(prog, [])
            if self.me == self.leader:          # the leader keeps presenting the pattern
                slots = [x for x in slots if x[0] not in ("pat",)]
            if not slots:
                break
            kind, node, k = slots[int(r.integers(len(slots)))]
            if kind == "pat":
                setattr(node, k, self.pattern(1))
            elif kind == "cond":
                setattr(node, k, self.trigger() if self.me == self.leader else self.cond(late=True))
            elif kind == "phase":
                old = node.parts[k]
                last = k == len(node.parts) - 1
                if isinstance(old, Wait):
                    node.parts[k] = Wait(self.cond())
                elif isinstance(old, (Cue, Adopt)):
                    node.parts[k] = Cue(*self.lead_op())
                elif self.me != self.leader:
                    if not last and r.random() < 0.3:
                        node.parts.pop(k)               # drop a phase
                    elif r.random() < 0.3 and len(node.parts) < 4:
                        node.parts.insert(k, self.phase(final=False))   # insert one before it
                    else:
                        node.parts[k] = self.phase(final=last)
            elif kind == "ops":
                ops = list(getattr(node, k))
                j = int(r.integers(len(ops)))
                if r.random() < 0.3 and len(ops) < 4:
                    ops.insert(j, self.op())
                elif r.random() < 0.3 and len(ops) > 1:
                    ops.pop(j)
                else:
                    ops[j] = self.op()
                setattr(node, k, ops)
            elif kind == "int":
                lo = 2 if isinstance(node, Breathe) else 1
                setattr(node, k, max(lo, getattr(node, k) + int(r.choice([-1, 1]))))
            elif kind == "curve":
                setattr(node, k, str(r.choice(list(CURVES))))
            elif kind == "op":
                node.op, node.arg = self.lead_op() if isinstance(node, Cue) else self.op()
        return self.repair(prog)

    def non_leader(self):
        """A pattern that is not plainly the leader's (for blends toward it)."""
        for _ in range(10):
            p = self.pattern()
            if not isinstance(p, Leader):
                return p
        return T(*self.op(), Leader())

    def hocket(self, until):
        n = int(self.r.choice([2, 3])) if len(self.parts) > 3 else 2
        return Hocket(int(self.r.integers(0, n)), n, until)

    def alternate(self, until):
        a = self.pattern()
        for _ in range(10):
            b = self.pattern()
            if str(b) != str(a):
                break
        return Alternate(a, b, until)

    def program(self):
        r, t = self.r, self.t
        if self.me == self.leader:
            return self.lead_program()
        n = int(r.choice([1, 2, 3], p=[0.25, 0.5, 0.25]))
        parts = [self.phase(final=False) for _ in range(n - 1)] + [self.phase(final=True)]
        # a phase that only repeats the next one's material is redundant
        same = lambda a, b: type(a) is type(b) and str(a).split(" until ")[0] == str(b).split(" until ")[0]
        parts = [p for i, p in enumerate(parts) if i + 1 == len(parts) or not same(p, parts[i + 1])]
        if r.random() < 0.6 + 0.3 * (1 - t["eager"]):
            parts = [Wait(self.cond())] + parts
        prog = parts[0] if len(parts) == 1 else Seq(*parts)
        if r.random() < 0.25 * (1 - t["eager"]) + 0.1:
            prog = Breathe(int(r.integers(3, 6)), prog)
        if r.random() < 0.45:
            others = [s for s in self.parts if s != self.me]
            curve = f"follow:{int(r.choice(others))}" if others and r.random() < 0.2 * t["imitative"] / 0.6 \
                else str(r.choice(list(CURVES)))
            prog = Dyn(curve, prog)
        return prog


# ---------------------------------------------------------------- interpreter
@dataclass
class Ctx:
    leader: int
    lead_notes: list
    period: float
    pulse: float
    nc: int
    rng: np.random.Generator
    prev: dict = field(default_factory=dict)
    prev_period: dict = field(default_factory=dict)
    k: int = 0
    joined: set = field(default_factory=set)
    joined_ever: set = field(default_factory=set)
    called: set = field(default_factory=set)
    singing: set = field(default_factory=set)                  # who sang in the last cycle
    history: list = field(default_factory=lambda: [[] for _ in range(NS)])
    phase: dict = field(default_factory=dict)                  # singer -> phrase being performed
    phase_log: list = field(default_factory=lambda: [[] for _ in range(NS)])
    memory: dict = field(default_factory=dict)                 # model -> (lead notes, period, leader)
    cues: list = field(default_factory=list)                   # (cycle, leader's move)
    origin: dict = field(default_factory=dict)                 # singer -> new | evolved | role | returned
    role: str = ""                                             # ensemble role, if any
    base: list = None                                          # programs before a morph's lead-in


def interpret(programs, ctx):
    """Run all programs together; returns per-singer [(cycle, notes, ident)]."""
    gens = [p.run(ctx, s) if p is not None else None for s, p in enumerate(programs)]
    for s, p in enumerate(programs):
        ctx.phase[s] = str(p.s if isinstance(p, Dyn) else p) if p is not None else ""
    out = [[] for _ in range(NS)]
    order = [ctx.leader] + [s for s in range(NS) if s != ctx.leader]   # the leader's cues come first
    for k in range(ctx.nc):
        ctx.k = k
        joined_now = set()
        for s in order:
            g = gens[s]
            notes, ident = [], False
            if g is not None:
                try:
                    notes, ident = next(g)
                except StopIteration:
                    gens[s] = None
                    ctx.phase[s] = ""
            ctx.history[s].append(notes)
            ctx.phase_log[s].append(ctx.phase.get(s, "") if notes or ctx.phase.get(s, "").startswith("wait")
                                    else "")
            out[s].append((k, notes, ident))
            if ident and notes:
                joined_now.add(s)
        ctx.joined = joined_now
        ctx.joined_ever |= joined_now
        ctx.singing = {s for s in range(NS) if out[s][-1][1]}
    return out


def to_events(out, ctx, t0, dyn, me_list=range(NS)):
    """Absolute events, overtones mapped to each singer's own harmonic nearest the leader's."""
    events = [[] for _ in range(NS)]
    for s in me_list:
        for k, notes, _ in out[s]:
            base = t0 + k * ctx.period
            for e in notes:
                if e.shape[0] == "ot":
                    e = replace(e, shape=("ot", map_overtone(e.shape[1], ctx.leader, SINGERS[s]), e.shape[2]))
                a = float(np.clip(e.amp * dyn, 0.08, 1.0))
                events[s].append(replace(e, t=base + e.t, amp=a,
                                         amp1=None if e.amp1 is None else float(np.clip(e.amp1 * dyn, 0.05, 1.0))))
    return events


# ---------------------------------------------------------------- evaluation
def descriptors(out, ctx):
    """What a model sounds like, as numbers: texture, synchrony, convergence, colour."""
    nc = ctx.nc
    active = np.array([[bool(out[s][k][1]) for k in range(nc)] for s in range(NS)])
    lead_on = {round(e.t, 2) for e in ctx.lead_notes}
    notes = [e for s in range(NS) if s != ctx.leader for _, ns, _ in out[s] for e in ns]
    sync = np.mean([round(e.t % ctx.period, 2) in lead_on for e in notes]) if notes else 0.0
    first = [next((k for k, ns, i in out[s] if i and ns), nc) / nc for s in range(NS) if s != ctx.leader]
    vowels = Counter(e.shape[-1] for e in notes)
    p = np.array(list(vowels.values()), float) / max(1, sum(vowels.values()))
    return np.array([
        active.sum(0).mean() / NS,                                   # voices sounding
        np.std(active.sum(0)) / NS,                                  # texture change
        len(notes) / (nc * ctx.period * (NS - 1)),                   # note density
        sync,                                                        # rhythmic agreement
        np.mean(first),                                              # time to identity
        sum(out[s][-1][2] for s in range(NS)) / NS,                  # identity at the end
        -(p * np.log(p + 1e-12)).sum() / np.log(6),                  # vowel variety
        np.mean([e.whisper or e.shape[0] == "hum" for e in notes]) if notes else 0.0,
    ])


DESC_NAMES = ["voices", "texture-change", "density", "sync", "identity-time", "identity-end",
              "vowel-variety", "whisper/hum"]


def coherent(out, ctx, min_end=2, absent=0):
    lead_cycles = sum(bool(n) for _, n, _ in out[ctx.leader])
    end_ident = sum(out[s][-1][2] for s in range(NS))
    silent = sum(not any(n for _, n, _ in out[s]) for s in range(NS)) - absent
    return lead_cycles >= 0.75 * ctx.nc and end_ident >= min_end and silent <= 1


INHERIT_BONUS = 0.35


def choose(ctx_factory, leader, tempers, names, history, rng, n_cand=60, parts=None, bias=None,
           lead_call=None, carry=(), memory=(), fixed=None, lineage=None, p_inherit=0.0,
           dialect=None, p_role=0.0):
    """Sample candidate program sets; keep the coherent one most unlike recent models.
    memory: earlier models the singers may quote; fixed: singer -> program kept as is (a
    returning model brings back some of its old strategies); lineage: (singer, "lead"|"follow")
    -> that singer's last program, mutated instead of resampled with probability p_inherit;
    p_role: chance of an ensemble role among the followers."""
    parts = list(range(NS)) if parts is None else list(parts)
    min_end = 2 if len(parts) > 2 else 1
    best = None
    for c in range(n_cand):
        progs, origin = [None] * NS, {}
        for s in parts:
            smp = Sampler(rng, tempers[s], s, leader, names, parts, bias, lead_call, memory, dialect)
            old = (lineage or {}).get((s, "lead" if s == leader else "follow"))
            if old is not None and rng.random() < p_inherit:
                progs[s], origin[s] = smp.mutate(old, int(rng.integers(1, 3))), "evolved"
            else:
                progs[s], origin[s] = smp.program(), "new"
        role = ""
        if rng.random() < p_role:
            smp = Sampler(rng, tempers[parts[-1]], parts[-1], leader, names, parts, bias, lead_call, memory, dialect)
            role, assigned = ensemble(rng, leader, parts, smp)
            for s, p in assigned.items():
                progs[s], origin[s] = p, "role"
        for s, p in (fixed or {}).items():
            if s in parts:
                progs[s], origin[s] = p, "returned"
        base = list(progs)
        for s in carry:      # morph from the previous model: old material dissolves into the new
            k = int(rng.integers(2, 5))
            progs[s] = Seq(Blend(Prev(), Leader(), k, Cycle(0.0)), progs[s])
        ctx = ctx_factory(np.random.default_rng(int(rng.integers(1 << 30))))
        out = interpret(progs, ctx)
        if not coherent(out, ctx, min_end, NS - len(parts)):
            continue
        ctx.origin, ctx.role, ctx.base = origin, role or "", base
        d = descriptors(out, ctx)
        st = sum((tags(p) for p in progs if p is not None), Counter())
        if history:
            dn = min(np.linalg.norm(d - h[0]) for h in history[-3:])
            keys = set(st) | set(history[-1][1])
            a = np.array([st[k] for k in keys], float)
            b = np.array([history[-1][1][k] for k in keys], float)
            sn = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            score = dn + 0.5 * sn
            # continuity of character: evolved strategies are worth keeping
            score += INHERIT_BONUS * np.mean([o == "evolved" for o in origin.values()])
        else:
            score = rng.random()
        if best is None or score > best[0]:
            best = (score, progs, ctx, out, d, st)
    return best
