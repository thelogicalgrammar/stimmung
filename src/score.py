"""Generative score in the manner of Stimmung.

Every pitch is a harmonic of B-flat 1 (58.27 Hz); each singer keeps one
fundamental and moves only the overtones of it.  The piece is a chain of
"models".  A leader presents a pattern alone; the others "identify" with it
one at a time, copying it in unison, in canon, or by taking over some of its
notes, retuning their overtones to the leader's sounding frequency.  Voices
not taking part are silent.

Large-scale form comes from an intensity curve (quiet emergence, a first
build, a calm valley, a main climax, dissolution).  Intensity sets each
model's type, tempo, number of voices and dynamics.

The form has memory.  The models are grouped into sections, each with its own
vowel colour, a protagonist who leads more often, and a home tempo; within a
section successive tempos are related by simple ratios (3:2, 4:3, 2:1...).
Earlier models return later, transformed (moved to another voice, at a related
tempo, reversed, inverted, whispered or sung), and section openings are the
likeliest to return; the last section begins with the first model's material
in the first section's colour.  Singers may also quote distant models.

Singers' strategies have memory too: each singer's program usually evolves from its
last one, each section has its own dialect of the strategy language, and the last
section brings back the strategies of the first.
"""
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from singers import SINGERS

OT_VOWELS = ["u", "o", "a", "i", "y"]

# magic names: (onset consonant, vowel, stressed) per syllable
NAMES = {
    "INANNA": [("", "i", 0), ("n", "a", 1), ("n", "a", 0)],
    "YEMAYA": [("y", "e", 0), ("m", "a", 1), ("y", "a", 0)],
    "MAWU":   [("m", "a", 1), ("w", "u", 0)],
    "NOMMO":  [("n", "o", 1), ("m", "o", 0)],
    "AMUN":   [("", "a", 0), ("m", "u", 1), ("n", "hum", 0)],
    "LONO":   [("l", "o", 1), ("n", "o", 0)],
    "INARI":  [("", "i", 0), ("n", "a", 1), ("r", "i", 0)],
    "OLORUN": [("", "o", 0), ("l", "o", 1), ("r", "u", 0), ("n", "hum", 0)],
    "NAMMU":  [("n", "a", 1), ("m", "u", 0)],
    "TIAMAT": [("t", "i", 1), ("", "a", 0), ("m", "a", 1), ("t", "schwa", 0)],
    "BALDUR": [("b", "a", 1), ("d", "u", 0)],
    "KALI":   [("k", "a", 1), ("l", "i", 0)],
    "DAGDA":  [("d", "a", 1), ("d", "a", 0)],
    "SHIVA":  [("sh", "i", 1), ("w", "a", 0)],
}

# rhythmic groupings of pulses; the first pulse of each group is accented
GROUPINGS = [[3, 3, 2], [2, 2, 3], [3, 2, 2], [3, 3, 2, 2], [2, 3], [3, 4], [2, 2, 2, 3],
             [3, 3, 3, 2], [5], [4, 4], [3, 2, 3]]
VOICED = ["b", "d", "m", "n", "l", "y", "w", "g"]
UNVOICED = ["t", "k", "p", "h", "s", "sh"]


@dataclass
class Ev:
    t: float
    dur: float
    shape: tuple            # ("v", vowel) | ("ot", n, vowel) | ("hum", vowel)
    onset: str = ""
    amp: float = 0.5
    tense: float = 0.6
    legato: bool = False
    whisper: bool = False
    bend: tuple = None      # (start, end) pitch offset in cents across the note
    amp1: float = None      # amplitude at the end of the note (linear ramp), if different


@dataclass
class Model:
    idx: int
    kind: str
    leader: int
    t0: float
    t1: float
    intensity: float = 0.5
    label: str = ""
    name: str = ""
    period: float = 0.0
    cycle: list = field(default_factory=list)   # Ev templates, t relative to the cycle start
    ident: dict = field(default_factory=dict)   # singer -> identification time
    cut: bool = True        # ends in a cadence and silence (else overlaps into the next)
    exit: str = ""          # transition into the next model (see TRANSITIONS)
    programs: dict = field(default_factory=dict)  # singer -> strategy program (DSL text)
    phases: dict = field(default_factory=dict)    # singer -> [(t0, t1, phrase being performed)]
    section: str = ""
    ret: int = None         # the earlier model this one brings back, if any
    pulse: float = 0.0
    tempo: str = ""         # tempo relation to the previous pulsed model (or to the returning one)
    origin: dict = field(default_factory=dict)    # singer -> new | evolved | role | returned
    role: str = ""          # ensemble role among the followers, if any
    cues: list = field(default_factory=list)      # [(t, leader's move)]


# optional {singer name: set of overtone numbers the voice model can actually produce}
AVAILABLE = None


def ot_range(s, fmin=380.0, fmax=2300.0):
    ns = [n for n in range(2, 30) if fmin <= n * s["f0"] <= fmax]
    if AVAILABLE and AVAILABLE.get(s["name"]):
        ns = [n for n in ns if n in AVAILABLE[s["name"]]] or ns
    return ns


def map_overtone(n_lead, lead, s):
    """Overtone of singer s closest to the leader's sounding overtone frequency."""
    f = n_lead * SINGERS[lead]["f0"]
    rng = ot_range(s)
    return min(rng, key=lambda n: abs(np.log(n * s["f0"] / f)))


def vowel_class(v):
    return {"e": "i", "E": "i", "oe": "y", "O": "o", "schwa": "o"}.get(v, v)


TRANSITIONS = ["cut", "breath", "overlap", "thin", "dissolve", "linger", "morph", "signal"]
DSL_KINDS = ("pulse", "hocket", "whisper", "drone", "name")


def transition_weights(x, dI, prev_dsl, next_dsl, boundary=False):
    """How suitable each transition is at relative position x, for an intensity change dI.
    Early on models stand apart (cuts, breaths, signals); later they flow into each other
    (morph, linger, overlap); falls in intensity dissolve or thin out; the end dissolves.
    Section boundaries are marked (cut, dissolve, signal); inside a section models flow."""
    continuous = prev_dsl and next_dsl
    w = {
        "cut": 1.2 * (1 - x) + 0.8 * (dI < -0.3),
        "breath": 0.6,
        "overlap": 0.3 + 0.7 * x * (dI > -0.15),
        "thin": 0.3 + 0.9 * (dI < 0),
        "dissolve": 0.4 + 1.2 * (dI < -0.2) + 1.5 * (x > 0.85),
        "linger": (0.2 + 1.3 * x) * continuous,
        "morph": (0.1 + 1.8 * x) * continuous * (dI > -0.25),
        "signal": 0.6 + 0.8 * (dI > 0.15) * (x < 0.85),
    }
    marked = {"cut": 2.5, "dissolve": 1.8, "signal": 1.8, "breath": 0.6}
    return {k: v * (marked.get(k, 0.25) if boundary else (0.5 if k == "cut" else 1.0)) for k, v in w.items()}


# sections: (title, start as a fraction of the piece); each gets a vowel colour, a
# protagonist and a home tempo when the score is built
SECTIONS = [("emergence", 0.0), ("first build", 0.2), ("valley", 0.42), ("ascent", 0.58),
            ("climax", 0.74), ("return", 0.9)]
COLOURS = {"dark": ["u", "o", "y"], "open": ["o", "a", "u"], "bright": ["i", "y", "a"], "round": ["y", "u", "o"]}
# tempo relations (new tempo : old), and how often a fresh model picks each
RATIOS = [(1, 1), (3, 2), (2, 3), (4, 3), (3, 4), (2, 1), (1, 2), (5, 4), (4, 5)]
# what a returning model may undergo
RETURN_OPS = ["retro", "rotate", "invert", "shift", "vowels", "mirror", "syncopate", "leap", "sigh",
              "melisma", "monovowel", "augment"]


def section_of(x):
    return max(i for i, (_, x0) in enumerate(SECTIONS) if x >= x0 - 1e-9)


def intensity_curve(x):
    """Planned intensity at relative position x in the piece."""
    pts = [(0.0, 0.1), (0.12, 0.3), (0.3, 0.7), (0.4, 0.55), (0.5, 0.2), (0.6, 0.45),
           (0.78, 1.0), (0.86, 0.8), (0.94, 0.35), (1.0, 0.1)]
    xs, ys = zip(*pts)
    return float(np.interp(x, xs, ys))


class SavedScore:
    """What later stages (video) need from a ScoreGen, without its generator state."""
    def __init__(self, sc):
        self.models, self.events, self.total = sc.models, sc.events, sc.total


class ScoreGen:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.events = [[] for _ in SINGERS]
        self.models = []
        import dsl
        self.dsl = dsl
        self.tempers = [dsl.sample_temperament(self.rng) for _ in SINGERS]
        self.dsl_history = []
        self.dsl_prev, self.dsl_prev_period = {}, {}
        self.prog_objs = {}             # model -> {singer: program}, for returns
        self.section_heads = set()      # models that open a section
        self.lineage = {}               # (singer, "lead"|"follow") -> last program
        self.lineage_snap = {}          # section -> lineage at its end

    # -- helpers ------------------------------------------------------------
    def add(self, si, ev):
        self.events[si].append(ev)

    def mapped(self, ev, lead, si):
        if si != lead and ev.shape[0] == "ot":
            return replace(ev, shape=("ot", map_overtone(ev.shape[1], lead, SINGERS[si]), ev.shape[2]))
        return ev

    def place(self, si, m, t_from, t_to, amp, shift=0.0, keep=None, jitter=0.0):
        """Sing the model's cycle in time with the leader's grid, from t_from to t_to.

        amp(t) scales the template amplitudes, shift delays the grid (canon),
        keep(j, k) selects which notes this singer takes (hocket).
        """
        r = self.rng
        k = max(0, int(np.ceil((t_from - m.t0 - shift) / m.period - 1e-9)))
        while True:
            base = m.t0 + shift + k * m.period
            if base >= t_to:
                break
            for j, e in enumerate(m.cycle):
                t = base + e.t + (r.normal(0, jitter) if jitter else 0.0)
                if t < t_from - 1e-6 or t + e.dur > t_to + 0.3:
                    continue
                if keep is not None and not keep(j, k):
                    continue
                a = amp(t)
                ev = replace(self.mapped(e, m.leader, si), t=t, amp=e.amp * a,
                             amp1=None if e.amp1 is None else e.amp1 * a)
                self.add(si, ev)
            k += 1
        return m.t0 + shift + k * m.period

    # -- material -------------------------------------------------------------
    def motif(self, lead, k):
        """k overtone numbers forming a small contour in the leader's range."""
        r = self.rng
        rng_n = ot_range(SINGERS[lead])
        c = int(r.integers(0, len(rng_n)))
        out = []
        for _ in range(k):
            out.append(rng_n[c])
            c = int(np.clip(c + r.choice([-2, -1, 1, 1, 2]), 0, len(rng_n) - 1))
        return out

    def pulse_cell(self, lead, I, whisper=False, pulse=None, colour=None):
        """A metric pattern: grouped pulses, accented group starts, staccato syllables."""
        r = self.rng
        if pulse is None:
            pulse = float(np.interp(I, [0, 1], [0.36, 0.15]) * r.uniform(0.9, 1.12))
        groups = list(GROUPINGS[int(r.integers(len(GROUPINGS)))])
        if sum(groups) < 6:
            groups = groups * 2
        cons = list(r.choice(UNVOICED if whisper else VOICED, size=2, replace=False))
        if not whisper and r.random() < 0.5:
            cons.append(str(r.choice(UNVOICED[:4])))
        acc_cons = cons[0]
        notes = self.motif(lead, len(groups))
        vow = list(r.choice(colour or OT_VOWELS, size=2, replace=False))
        cell, t = [], 0.0
        stacc = float(np.interp(I, [0, 1], [0.75, 0.5]))
        for g, n_acc in zip(groups, notes):
            for p in range(g):
                if p == 0:
                    shape = ("v", str(r.choice(["a", "i", "u", "e", "o"]))) if whisper \
                        else ("ot", n_acc, vow[0])
                    cell.append(Ev(t, pulse * (0.85 if g > 2 else stacc), shape, acc_cons, 1.0,
                                   0.75, whisper=whisper))
                elif r.random() < 0.72:
                    v = str(r.choice(["a", "i", "u", "schwa", "e"]))
                    shape = ("v", v) if (whisper or r.random() < 0.5) else ("ot", n_acc, vow[1])
                    cell.append(Ev(t, pulse * stacc, shape, str(r.choice(cons)), 0.62, 0.6,
                                   whisper=whisper))
                t += pulse
        label = "+".join(map(str, groups))
        return cell, t, label

    def glide_cell(self, lead):
        """Slow overtone melody on one breath per note."""
        r = self.rng
        rng_n = ot_range(SINGERS[lead])
        lo = int(r.integers(0, max(1, len(rng_n) - 3)))
        path = rng_n[lo:lo + 4]
        seq = path + path[-2:0:-1] if len(path) > 2 else path + path[::-1]
        cell, t = [], 0.0
        for i, n in enumerate(seq):
            d = float(r.uniform(1.6, 3.2))
            v = OT_VOWELS[min(4, rng_n.index(n) * 5 // max(1, len(rng_n)))]
            cell.append(Ev(t, d, ("ot", n, v), "m" if i == 0 else "", 0.85, 0.7, legato=i > 0,
                           amp1=1.0))
            t += d
        return cell, t + 1.2

    def name_cell(self, lead, name, pulse):
        """The name chanted in rhythm (used once a call has been answered)."""
        r = self.rng
        rng_n = ot_range(SINGERS[lead])
        cell, t = [], 0.0
        for onset, v, stress in NAMES[name]:
            d = (2 if stress else 1) * pulse
            if v == "hum":
                shape = ("hum", "u")
            elif stress:
                shape = ("ot", int(r.choice(rng_n)), vowel_class(v))
            else:
                shape = ("v", v)
            cell.append(Ev(t, d * 0.8, shape, onset, 1.0 if stress else 0.7, 0.7))
            t += d
        return cell, t + 2 * pulse

    def call(self, si, t, name, amp, fall=True):
        """Declaim a name: free rhythm, stressed syllable stretched, falling inflection."""
        r = self.rng
        syl = NAMES[name]
        rng_n = ot_range(SINGERS[si])
        for i, (onset, v, stress) in enumerate(syl):
            last = i == len(syl) - 1
            d = (r.uniform(0.55, 0.8) if stress else r.uniform(0.18, 0.28)) + (0.5 if last else 0.0)
            shape = ("hum", "u") if v == "hum" else \
                ("ot", int(r.choice(rng_n)), vowel_class(v)) if stress else ("v", v)
            bend = (0, -r.uniform(200, 700)) if (last and fall) else \
                (-r.uniform(80, 200), 0) if stress else None
            self.add(si, Ev(t, d, shape, onset, amp * (1.0 if stress else 0.8), 0.8,
                            bend=bend, amp1=amp * 0.5 if last else None))
            t += d + 0.03
        return t

    # -- model types ------------------------------------------------------------
    def join_times(self, m, parts, first_after):
        """Identification: participants other than the leader join one by one on cycle starts."""
        r = self.rng
        others = [p for p in parts if p != m.leader]
        r.shuffle(others)
        span = m.t1 - first_after
        out = {}
        for i, si in enumerate(others):
            frac = (i + r.uniform(0.2, 0.8)) / max(1, len(others)) * 0.75
            t = first_after + frac * span
            k = np.ceil((t - m.t0) / m.period)
            out[si] = m.t0 + k * m.period
        return out

    def do_pulse(self, m, parts, whisper=False):
        r = self.rng
        I = m.intensity
        m.cycle, m.period, groups = self.pulse_cell(m.leader, I, whisper)
        reps = max(2, int(np.ceil(2.0 / m.period)))
        m.ident = self.join_times(m, parts, m.t0 + reps * m.period)
        dyn = 0.18 + 0.72 * I if not whisper else 0.45
        grow = lambda t: dyn * (0.7 + 0.3 * (t - m.t0) / (m.t1 - m.t0))
        end = self.place(m.leader, m, m.t0, m.t1, grow)
        pulse = m.period / max(1, len(m.cycle))
        for i, si in enumerate(sorted(m.ident, key=m.ident.get)):
            mode = r.choice(["unison", "unison", "canon"]) if not whisper else "canon"
            shift = 0.0
            if mode == "canon":
                shift = float(r.choice([1, 2, 3])) * m.period / sum(map(int, groups.split("+")))
            self.place(si, m, m.ident[si], m.t1, lambda t: 0.85 * grow(t), shift=shift, jitter=0.012)
        m.label = ("whispered rhythm " if whisper else "rhythm ") + groups
        return end

    def do_hocket(self, m, parts):
        r = self.rng
        I = m.intensity
        m.cycle, m.period, groups = self.pulse_cell(m.leader, I)
        reps = 2
        m.ident = self.join_times(m, parts, m.t0 + reps * m.period)
        dyn = 0.2 + 0.7 * I
        order = sorted(m.ident, key=m.ident.get)
        # the cycle's notes are dealt out among the leader and those who have joined
        def owner(j, k, t):
            group = [m.leader] + [s for s in order if m.ident[s] <= t + 1e-6]
            if m.cycle[j].amp >= 1.0 and len(group) > 3:
                return None     # accents sung by everyone
            return group[(j + k) % len(group)]
        for si in [m.leader] + order:
            start = m.t0 if si == m.leader else m.ident[si]
            def keep(j, k, si=si):
                t = m.t0 + k * m.period + m.cycle[j].t
                o = owner(j, k, t)
                return o is None or o == si
            self.place(si, m, start, m.t1, lambda t: dyn, keep=keep)
        m.label = "hocket " + groups
        return m.t0 + np.ceil((m.t1 - m.t0) / m.period) * m.period

    def do_drone(self, m, parts):
        """Overtone melody by the leader over humming; partners take up the melody late."""
        r = self.rng
        m.cycle, m.period = self.glide_cell(m.leader)
        dyn = 0.18 + 0.5 * m.intensity
        others = [p for p in parts if p != m.leader]
        hummers = others[: max(1, len(others) // 2)]
        for si in hummers:
            t = m.t0 + r.uniform(0, 3)
            while t < m.t1 - 2:
                d = min(r.uniform(6, 11), m.t1 - t)
                self.add(si, Ev(t, d, ("hum", str(r.choice(["u", "o", "schwa"]))), "m",
                                dyn * 0.7, 0.5, amp1=dyn * r.uniform(0.5, 0.9)))
                t += d + r.uniform(0.4, 1.0)
        self.place(m.leader, m, m.t0, m.t1, lambda t: dyn)
        late = [p for p in others if p not in hummers]
        m.ident = {si: m.t0 + np.ceil((0.5 + 0.2 * i) * (m.t1 - m.t0) / m.period) * m.period
                   for i, si in enumerate(late)}
        for si, t in m.ident.items():
            self.place(si, m, t, m.t1, lambda t: dyn * 0.8, jitter=0.05)
        m.label = "overtone melody"
        return m.t1

    def do_name(self, m, parts):
        """Leader calls a magic name; others echo; finally all chant it in rhythm."""
        r = self.rng
        dyn = 0.35 + 0.6 * m.intensity
        t = m.t0
        others = [p for p in parts if p != m.leader]
        r.shuffle(others)
        t = self.call(m.leader, t, m.name, dyn) + r.uniform(1.0, 2.2)
        answered = []
        for si in others:
            if t > m.t0 + 0.55 * (m.t1 - m.t0):
                break
            t = self.call(si, t, m.name, dyn * 0.8, fall=r.random() < 0.6) + r.uniform(0.5, 1.6)
            answered.append(si)
            m.ident[si] = t
            if r.random() < 0.4:
                t = self.call(m.leader, t, m.name, dyn) + r.uniform(0.6, 1.4)
        # the chant: everyone who answered, in unison rhythm
        pulse = float(np.interp(m.intensity, [0, 1], [0.32, 0.17]))
        m.cycle, m.period = self.name_cell(m.leader, m.name, pulse)
        start = m.t0
        m.t0, t1 = t, max(t + 3 * m.period, m.t1)
        end = t
        for si in [m.leader] + answered:
            end = max(end, self.place(si, m, t, t1, lambda x: dyn, jitter=0.01))
        m.t0, m.t1 = start, t1
        m.label = f"name {m.name}"
        return end

    def do_swell(self, m, parts):
        """Tutti sustained chord; the vowel opens step by step (rising overtones), swelling."""
        r = self.rng
        vowels = ["u", "o", "a", "E", "i"] if r.random() < 0.5 else ["i", "y", "oe", "o", "u"]
        n = len(vowels)
        seg = (m.t1 - m.t0) / n
        peak = 0.25 + 0.7 * m.intensity
        env = lambda x: 0.12 + (peak - 0.12) * np.sin(np.pi * min(1, max(0, x))) ** 1.5
        for si in parts:
            off = r.uniform(0, 1.2)
            for i, v in enumerate(vowels):
                a0 = (i * seg + (off if i == 0 else 0)) / (m.t1 - m.t0)
                a1 = (i + 1) * seg / (m.t1 - m.t0)
                t = m.t0 + a0 * (m.t1 - m.t0)
                d = (a1 - a0) * (m.t1 - m.t0) - 0.02
                self.add(si, Ev(t, d, ("v", v), "m" if i == 0 else "", float(env(a0)), 0.7,
                                legato=i > 0, amp1=float(env(a1))))
        m.label = "swell " + "–".join(vowels)
        return m.t1

    def do_duet(self, m, parts):
        """Two singers in question and answer, converging into one line."""
        r = self.rng
        a = m.leader
        b = next((p for p in parts if p != a), (a + 3) % len(SINGERS))
        dyn = 0.25 + 0.5 * m.intensity
        pulse = float(np.interp(m.intensity, [0, 1], [0.45, 0.25]))
        t = m.t0
        turn = 0
        while t < m.t1 - 3:
            si = (a, b)[turn % 2]
            notes = self.motif(si, int(r.integers(3, 6)))
            together = t > m.t0 + 0.6 * (m.t1 - m.t0)
            singers = (a, b) if together else (si,)
            for j, n in enumerate(notes):
                d = pulse * float(r.choice([1, 1, 2, 3]))
                for s_ in singers:
                    shape = ("ot", n if s_ == si else map_overtone(n, si, SINGERS[s_]),
                             str(r.choice(OT_VOWELS)))
                    self.add(s_, Ev(t, d * 0.9, shape, str(r.choice(["l", "n", "m", "d", ""])),
                                    dyn * (1.0 if j == 0 else 0.8), 0.7,
                                    bend=(0, -250) if j == len(notes) - 1 else None))
                t += d
            t += pulse * float(r.choice([1, 2, 3]))
            turn += 1
        m.ident = {b: m.t0 + 0.6 * (m.t1 - m.t0)}
        m.label = "duet"
        return t

    def recast(self, q, lead, pl):
        """An earlier model's pattern brought back: in the new leader's harmonics, at the
        planned tempo, under a few transformations.  Returns (cycle, period, pulse, how)."""
        r, dsl = self.rng, self.dsl
        sc = pl["pulse"] / q.pulse if pl.get("pulse") and q.pulse else pl.get("scale", 1.0)
        cyc = [replace(self.mapped(e, q.leader, lead) if q.leader != lead else e, t=e.t * sc,
                       dur=max(0.07, e.dur * sc)) for e in q.cycle]
        period, pulse = q.period * sc, (q.pulse or q.period / max(1, len(q.cycle))) * sc
        how = []
        if q.leader != lead:
            how.append(f"now {SINGERS[lead]['name']}")
        if pl.get("flip"):
            cyc = [replace(e, whisper=pl["flip"] == "whispered") for e in cyc]
            how.append(pl["flip"])
        for op in pl.get("ops", ()):
            arg = {"rotate": int(r.integers(1, 4)), "shift": int(r.choice([-2, -1, 1, 2])),
                   "vowels": int(r.integers(1, 4)), "monovowel": str(r.choice(pl["colour"]))}.get(op)
            ctx = dsl.Ctx(lead, cyc, period, pulse, 1, r)
            pv = dsl.apply_op(op, arg, dsl.PVal(cyc), ctx)
            cyc, period = sorted(pv.notes, key=lambda e: e.t), period * pv.cycles
            how.append(op)
        return cyc, period, pulse, ", ".join(how)

    def do_dsl(self, m, parts, kind, carry=(), pl=None):
        """The leader's pattern comes from the model type (or from an earlier model that
        returns); every participant's strategy is a program sampled from the DSL (see dsl.py),
        chosen for novelty against recent models."""
        r, dsl = self.rng, self.dsl
        pl = pl or {}
        I = m.intensity
        bias, lead_call, fixed = {}, None, {}
        colour = pl.get("colour")
        if pl.get("ret") is not None:
            q = self.models[pl["ret"]]
            m.cycle, m.period, pulse, how = self.recast(q, m.leader, pl)
            m.ret = q.idx
            base = q.label.split(" ← ")[0].split("  ")[0]
            m.label = f"{base} ← model {q.idx + 1}" + (f" ({how})" if how else "")
            if q.kind == "name":
                m.name, lead_call = q.name, q.name
                bias = {"echo": 2.0, "call": 2.0}
            elif kind == "hocket":
                bias = {"hocket": 4.0}
            elif kind == "drone":
                bias = {"converge": 1.5, "canon": 0.5, "hocket": 0.3}
            # some singers bring back their old strategies
            for s, prog in self.prog_objs.get(q.idx, {}).items():
                if s in parts and s != m.leader and r.random() < 0.5:
                    fixed[s] = prog
        elif kind in ("pulse", "hocket", "whisper"):
            m.cycle, m.period, groups = self.pulse_cell(m.leader, I, whisper=kind == "whisper",
                                                        pulse=pl.get("pulse"), colour=colour)
            pulse = m.period / sum(map(int, groups.split("+")))
            m.label = {"pulse": "rhythm ", "hocket": "hocket ", "whisper": "whispered rhythm "}[kind] + groups
            if kind == "hocket":
                bias = {"hocket": 4.0}
        elif kind == "drone":
            m.cycle, m.period = self.glide_cell(m.leader)
            pulse = m.period / max(1, len(m.cycle))
            bias = {"converge": 1.5, "canon": 0.5, "hocket": 0.3}
            m.label = "overtone melody"
        else:   # name: the leader calls it, then the name becomes the pattern
            pulse = pl.get("pulse") or float(np.interp(I, [0, 1], [0.32, 0.17]))
            m.cycle, m.period = self.name_cell(m.leader, m.name, pulse)
            lead_call = m.name
            bias = {"echo": 2.0, "call": 2.0}
            m.label = f"name {m.name}"
        m.pulse = pulse
        if m.tempo and kind != "drone":
            m.label += f"  {m.tempo}"
        # what the singers may quote: section openings first, then the more recent models
        older = [q for q in self.models if q.idx < m.idx - 1 and q.cycle and q.kind in DSL_KINDS]
        heads = [q.idx for q in older if q.idx in self.section_heads]
        memory = heads[::-1] + [q.idx for q in reversed(older) if q.idx not in heads]
        memory = memory[:6]
        nc = max(3, int(round((m.t1 - m.t0) / m.period)))

        def factory(rr):
            return dsl.Ctx(m.leader, m.cycle, m.period, pulse, nc, rr, dict(self.dsl_prev),
                           dict(self.dsl_prev_period),
                           memory={q.idx: (q.cycle, q.period, q.leader) for q in older})
        best = dsl.choose(factory, m.leader, self.tempers, list(NAMES), self.dsl_history, r,
                          n_cand=50, parts=parts, bias=bias, lead_call=lead_call,
                          carry=[s for s in carry if s in parts and s != m.leader], memory=memory,
                          fixed=fixed, lineage=self.lineage, dialect=pl.get("dialect"),
                          p_inherit=pl.get("p_inherit", 0.0), p_role=pl.get("p_role", 0.0))
        if best is None:        # nothing coherent: everyone simply sings the model
            progs = [dsl.Play(dsl.Leader(), dsl.End()) if s in parts else None for s in range(len(SINGERS))]
            ctx = factory(np.random.default_rng(0))
            out = dsl.interpret(progs, ctx)
            d, st = dsl.descriptors(out, ctx), sum((dsl.tags(p) for p in progs if p), Counter())
        else:
            _, progs, ctx, out, d, st = best
        self.dsl_history.append((d, st))
        self.prog_objs[m.idx] = {s: p for s, p in enumerate(ctx.base or progs) if p is not None}
        for s, p in enumerate(ctx.base or progs):
            if p is not None:
                self.lineage[(s, "lead" if s == m.leader else "follow")] = p.s if isinstance(p, dsl.Role) else p
        m.origin, m.role = dict(ctx.origin), ctx.role
        m.cues = [(m.t0 + k * m.period, txt) for k, txt in ctx.cues]
        if m.role:
            m.label += "  · " + m.role.split(":")[0]
        dyn = 0.18 + 0.72 * I if kind != "whisper" else 0.45
        ev = dsl.to_events(out, ctx, m.t0, dyn)
        for s in range(len(SINGERS)):
            for e in ev[s]:
                self.add(s, e)
            sung = [n for _, n, _ in out[s] if n]
            if sung:
                self.dsl_prev[s], self.dsl_prev_period[s] = sung[-1], m.period
            if progs[s] is not None:
                m.programs[s] = str(progs[s])
                first = next((k for k, n, i in out[s] if i and n), None)
                if s != m.leader and first is not None:
                    m.ident[s] = m.t0 + first * m.period
                spans, cur = [], None
                for k, ph in enumerate(ctx.phase_log[s]):
                    if cur and cur[2] == ph:
                        cur[1] = m.t0 + (k + 1) * m.period
                    else:
                        cur = [m.t0 + k * m.period, m.t0 + (k + 1) * m.period, ph]
                        spans.append(cur)
                m.phases[s] = [tuple(x) for x in spans if x[2]]
        return m.t0 + nc * m.period

    # -- transitions ---------------------------------------------------------
    def _drop_after(self, s, t_from, t_to):
        self.events[s] = [e for e in self.events[s] if not (t_from <= e.t < t_to)]

    def thin(self, m, parts, end):
        """Voices drop out one by one over the model's last stretch; the leader stays."""
        r = self.rng
        others = [s for s in parts if s != m.leader]
        r.shuffle(others)
        span = min(0.35 * (end - m.t0), 14.0)
        for k, s in enumerate(others):
            self._drop_after(s, end - span * (1 - k / max(1, len(others))), end + 1.0)

    def dissolve(self, m, parts, end):
        """Everyone holds their last note and lets it fade, staggered; returns its length."""
        r = self.rng
        d = float(r.uniform(4.0, 8.0))
        for s in parts:
            last = [e for e in self.events[s] if m.t0 <= e.t < end + 0.5]
            if not last:
                continue
            e = max(last, key=lambda e: e.t)
            shape = e.shape if not e.whisper else ("v", e.shape[-1])
            hold = d * r.uniform(0.6, 1.0)
            self.add(s, Ev(e.t + e.dur + 0.02, hold, shape, "", max(0.15, e.amp * 0.8), 0.6, legato=True,
                           amp1=0.04))
        return d

    def linger(self, prev, t_new):
        """Old voices keep repeating their old material, fading, until their first note in the
        new model (or for up to three old cycles)."""
        m, parts, end = prev
        r = self.rng
        for s in parts:
            src = self.dsl_prev.get(s)
            if not src or not m.period:
                continue
            nxt = [e.t for e in self.events[s] if e.t >= t_new]
            stop = min(min(nxt) - 0.3 if nxt else 1e9, end + m.period * r.uniform(1.5, 3.0))
            k, base = 0, end
            while base < stop:
                g = 0.8 * (1 - (base - end) / max(1e-3, stop - end)) + 0.15
                for e in src:
                    if base + e.t + e.dur < stop:
                        self.add(s, replace(e, t=base + e.t, amp=float(np.clip(e.amp * g, 0.08, 1.0)),
                                            amp1=None))
                base += m.period
                k += 1

    def signal(self, m, parts, end, caller, names):
        """A singer calls a magic name over the last bars; everyone breaks off at the call."""
        r = self.rng
        name = str(r.choice(names))
        tc = max(m.t0 + 2.0, end - r.uniform(1.0, 2.5))
        for s in range(len(SINGERS)):
            self._drop_after(s, tc - 0.05, end + 1.0)
        return self.call(caller, tc, name, float(np.clip(0.5 + 0.5 * m.intensity, 0.4, 0.95)))

    def cadence(self, m, parts, t):
        """Everyone taking part lands together on the model's first note, then silence."""
        e = m.cycle[0] if m.cycle else Ev(0, 1, ("v", "a"), "", 1.0)
        d = float(self.rng.uniform(0.7, 1.4))
        for si in parts:
            ev = self.mapped(e, m.leader, si)
            self.add(si, replace(ev, t=t, dur=d, amp=min(0.95, 0.3 + 0.7 * m.intensity),
                                 bend=(0, -120), amp1=0.15, whisper=False))
        return t + d

    # -- form ------------------------------------------------------------------
    def build(self, n_models=24, intro=28.0):
        r = self.rng
        nS = len(SINGERS)
        names = list(r.permutation(list(NAMES)))

        # intro: voices enter one by one from the bass upward, humming, then opening to 'u'
        for si in range(nS):
            ts = 1.0 + si * 3.8
            self.add(si, Ev(ts, intro - ts - 4.0, ("hum", "u"), "m", 0.12, 0.45, amp1=0.3))
            self.add(si, Ev(intro - 4.0 + si * 0.15, 3.0, ("v", "u"), "", 0.3, 0.5, legato=True,
                            amp1=0.08))

        # sections: a vowel colour, a protagonist and a home tempo each; the last section
        # recalls the first one's colour and protagonist
        cols = list(r.permutation(list(COLOURS)))
        prots = list(r.permutation(nS))
        sect = [dict(title=title, colour=COLOURS[cols[k % len(cols)]], prot=int(prots[k % nS]),
                     dialect=self.dsl.sample_dialect(r))
                for k, (title, _) in enumerate(SECTIONS)]
        sect[-1].update(colour=sect[0]["colour"], prot=sect[0]["prot"], dialect=sect[0]["dialect"])

        # plan the models (type, intensity, voices, tempo, returns) before generating, so each
        # transition can take the next model into account
        plans, prev_kind, prev_lead, ref_pulse = [], None, -1, None
        for i in range(n_models):
            x = i / (n_models - 1)
            k_sec = section_of(x)
            head = not plans or plans[-1]["sec"] != k_sec
            sc_ = sect[k_sec]
            I = float(np.clip(intensity_curve(x) + r.normal(0, 0.06), 0.05, 1.0))
            # a return of an earlier model?  likelier later, and at section openings; the last
            # section opens with the first model's material
            ret, flip = None, None
            cands = [j for j, q in enumerate(plans) if q["sec"] < k_sec and q["kind"] in DSL_KINDS]
            p_ret = 0.08 + 0.3 * x + (0.35 if head and k_sec >= 2 else 0.0)
            if cands and head and k_sec == len(SECTIONS) - 1:
                ret = cands[0]
            elif r.random() < p_ret:
                # each model returns at most once; the first is kept for the last section
                times = Counter(q["ret"] for q in plans)
                cands = [j for j in cands if j != cands[0] and not times[j]]
                w = np.array([(2.5 if plans[j]["head"] else 1.0)
                              * (0.5 if plans[j]["ret"] is not None else 1.0)
                              * (1 + (i - j) / n_models) for j in cands])
                ret = int(r.choice(cands, p=w / w.sum())) if cands else None
            if ret is not None:
                kind = plans[ret]["kind"]
                if kind == "whisper" and I > 0.45:
                    kind, flip = "pulse", "sung"
                elif kind == "pulse" and I < 0.3:
                    kind, flip = "whisper", "whispered"
            else:
                # which model types suit this intensity, and how strongly
                fit = {"drone": 1 - I, "whisper": 0.6 * (I < 0.35), "duet": 0.5 * (I < 0.55),
                       "name": 0.8, "pulse": 0.3 + I, "hocket": 0.1 + I, "swell": 1.2 * (I > 0.75)}
                used = {k: sum(p["kind"] == k for p in plans) for k in fit}
                w = np.array([fit[k] / (1 + used[k]) * (k != prev_kind) for k in fit]) + 1e-9
                kind = str(r.choice(list(fit), p=w / w.sum()))
            nvoices = int(np.clip(round(2 + 4.5 * I + r.normal(0, 0.6)), 2, nS))
            if kind == "swell":
                nvoices = max(nvoices, 5)
            if kind == "duet":
                nvoices = 2
            # the section's protagonist leads more often; a return may keep its old leader
            cand_lead = [s for s in range(nS) if s != prev_lead]
            w = np.array([3.0 if s == sc_["prot"] else 1.0 for s in cand_lead])
            lead = int(r.choice(cand_lead, p=w / w.sum()))
            if ret is not None and plans[ret]["lead"] != prev_lead and r.random() < 0.5:
                lead = plans[ret]["lead"]
            pool = [s for s in range(nS) if s != lead]
            r.shuffle(pool)
            if ret is not None:     # the old company comes back first
                pool.sort(key=lambda s: s not in plans[ret]["parts"])
            parts = [lead] + [int(p) for p in pool[: nvoices - 1]]
            dur = float(np.interp(I, [0, 1], [40, 24]) * r.uniform(0.75, 1.25))
            if kind == "swell":
                dur = r.uniform(16, 24)
            dia = sc_["dialect"]
            last = k_sec == len(SECTIONS) - 1
            # strategies evolve from the singer's last program, less so at a fresh section
            # (but the last section resumes the first section's strategies)
            p_inh = (0.3 if head and not last else 0.65) * np.sqrt(dia["inherit"])
            pl = dict(kind=kind, I=I, lead=lead, parts=parts, dur=dur, sec=k_sec, head=head, ret=ret,
                      flip=flip, colour=sc_["colour"], pulse=None, tempo="", dialect=dia,
                      p_inherit=float(np.clip(p_inh if not (head and last) else 0.85, 0.15, 0.85)),
                      p_role=float(np.clip(0.35 * dia["ensemble"], 0.1, 0.7)))
            # tempo: related by a simple ratio to the previous pulse (inside a section) or to
            # the returning model's; free at a section opening
            if kind in ("pulse", "hocket", "whisper", "name"):
                want = float(np.interp(I, [0, 1], [0.36, 0.15] if kind != "name" else [0.32, 0.17]))
                ref = plans[ret]["pulse"] if ret is not None else (None if head else ref_pulse)
                if ref is None:
                    pl["pulse"] = want * float(r.uniform(0.9, 1.12))
                else:
                    opts = [(a_, b_) for a_, b_ in RATIOS if 0.13 <= ref * b_ / a_ <= 0.45]
                    sc = np.array([-4 * abs(np.log(ref * b_ / a_ / want)) - 0.3 * np.log2(a_ * b_)
                                   - 0.4 * (a_ == b_) for a_, b_ in opts])
                    w = np.exp(2 * (sc - sc.max()))
                    a_, b_ = opts[int(r.choice(len(opts), p=w / w.sum()))]
                    pl["pulse"] = ref * b_ / a_
                    if ret is None:
                        pl["tempo"] = "same tempo" if a_ == b_ else f"tempo ×{a_}/{b_}"
                    else:
                        pl["tempo"] = "original tempo" if a_ == b_ else f"×{a_}/{b_} original tempo"
                ref_pulse = pl["pulse"]
            elif ret is not None:   # a returning melody, slower or faster
                pl["scale"] = float(r.choice([1.0, 1.5, 2 / 3]))
            if ret is not None:
                n_ops = int(r.choice([0, 1, 1, 2]))
                pl["ops"] = [str(o) for o in r.choice(RETURN_OPS, n_ops, replace=False)]
                if plans[ret]["kind"] == "drone":
                    pl["ops"] = [o for o in pl["ops"] if o not in ("augment", "syncopate", "melisma")]
            plans.append(pl)
            prev_kind, prev_lead = kind, lead
        self.sections = sect

        t = intro + 1.5
        carry, linger_from = (), None
        for i, pl in enumerate(plans):
            kind, I, lead, parts = pl["kind"], pl["I"], pl["lead"], list(pl["parts"])
            if carry and kind != "duet":            # morph: continuing singers take part
                parts += [s for s in carry if s not in parts][: max(0, nS - len(parts))]
            m = Model(i, kind, lead, t, t + pl["dur"], intensity=I, section=sect[pl["sec"]]["title"],
                      tempo=pl["tempo"])
            if pl["head"]:
                self.section_heads.add(i)
                if pl["sec"] > 0:
                    self.lineage_snap[plans[i - 1]["sec"]] = dict(self.lineage)
                if pl["sec"] == len(SECTIONS) - 1 and 0 in self.lineage_snap:
                    self.lineage = dict(self.lineage_snap[0])   # the first section's strategies return
            if kind == "name" and pl["ret"] is None:
                m.name = names[sum(1 for q in self.models if q.kind == "name") % len(names)]
            if kind in ("swell", "duet"):
                end = {"swell": self.do_swell, "duet": self.do_duet}[kind](m, parts)
            else:
                end = self.do_dsl(m, parts, kind, carry=carry, pl=pl)
            if linger_from is not None:
                self.linger(linger_from, m.t0)
            carry, linger_from = (), None
            # how to leave this model
            nxt = plans[i + 1] if i + 1 < len(plans) else None
            if nxt is None:
                exit_ = "dissolve"
            else:
                boundary = nxt["sec"] != pl["sec"]
                wts = transition_weights(i / (n_models - 1), nxt["I"] - I, kind in DSL_KINDS,
                                         nxt["kind"] in DSL_KINDS, boundary)
                used = Counter(q.exit for q in self.models)
                prev_exit = self.models[-1].exit if self.models else None
                w = np.array([wts[k] / (1 + 0.7 * used[k]) * (k != prev_exit) for k in TRANSITIONS]) + 1e-9
                exit_ = str(r.choice(TRANSITIONS, p=w / w.sum()))
            m.exit = exit_
            m.cut = exit_ in ("cut", "breath", "signal")
            if exit_ == "cut":
                if kind in ("pulse", "hocket", "name"):
                    end = self.cadence(m, parts, end)
                t = end + (r.uniform(3.0, 5.5) if boundary else r.uniform(1.5, 4.0))
            elif exit_ == "breath":
                t = end + r.uniform(0.4, 1.0)
            elif exit_ == "overlap":
                t = end - r.uniform(1.0, 3.0)
            elif exit_ == "thin":
                self.thin(m, parts, end)
                t = end - r.uniform(0.0, 1.5)
            elif exit_ == "dissolve":
                d = self.dissolve(m, parts, end)
                t = end + d * r.uniform(0.3, 0.5)
            elif exit_ == "linger":
                linger_from = (m, parts, end)
                t = end
            elif exit_ == "morph":
                carry = tuple(parts)
                t = end
            elif exit_ == "signal":
                end = self.signal(m, parts, end, nxt["lead"], names)
                t = end + r.uniform(0.3, 0.8)
            m.t1 = end
            self.models.append(m)

        # coda: everyone returns, humming the chord softly, and the voices leave from the top
        end = t + 1.0
        for si in range(nS):
            stop = end + 30.0 - (nS - 1 - si) * 3.5 if si else end + 34.0
            ts = end + si * 0.7
            n = ot_range(SINGERS[si])[0]
            self.add(si, Ev(ts, 9.0, ("ot", n, "u"), "m", 0.1, 0.5, amp1=0.3))
            self.add(si, Ev(ts + 9.0, stop - ts - 9.0, ("hum", "u"), "", 0.3, 0.5, legato=True,
                            amp1=0.05))
        self.total = end + 40.0
        for ev in self.events:
            ev.sort(key=lambda e: e.t)
            # resolve overlaps: an event cuts the previous one short
            keep = []
            for e in ev:
                if keep and keep[-1].t + keep[-1].dur > e.t - 0.04:
                    if e.t - keep[-1].t < 0.06:
                        continue            # simultaneous duplicate: keep the first
                    keep[-1].dur = e.t - 0.04 - keep[-1].t
                keep.append(e)
            ev[:] = keep
        return self

    def describe(self):
        lines = []
        for m in self.models:
            if m.idx in self.section_heads:
                sc_ = next(x for x in self.sections if x["title"] == m.section)
                lines.append(f"── {m.section}  (colour {'/'.join(sc_['colour'])}, protagonist "
                             f"{SINGERS[sc_['prot']]['name']}, favourite ops {' '.join(sc_['dialect']['favourites'])})")
            mm, ss = divmod(m.t0, 60)
            ids = ", ".join(SINGERS[s]["name"] for s in sorted(m.ident, key=m.ident.get))
            lines.append(f"{int(mm):2d}:{ss:04.1f}  model {m.idx + 1:2d}  I={m.intensity:.2f}  "
                         f"{m.label:44s} leader {SINGERS[m.leader]['name']:9s} "
                         f"→ {m.exit:8s} joins: {ids}")
            if m.origin:
                c = Counter(m.origin.values())
                lines.append("                 strategies: " + ", ".join(f"{v} {k}" for k, v in c.items())
                             + "".join(f";  {int((t - m.t0))}s {txt}" for t, txt in m.cues))
        return "\n".join(lines)
