"""Render the generated score with VocalTractLab voices.

usage: python render_vtl.py [--seed 7] [--preview SECONDS] [--demo] [--out ../out/stimmung_vtl]
"""
import argparse
import os
import pickle
import subprocess
import tempfile
import time
from multiprocessing import Pool

from dataclasses import replace

import numpy as np

import score
import vtl
from articulation import CONSONANTS, TAU_CONS, TAU_LEGATO, TAU_RELEASE, TAU_VOWEL, Articulation, tam
from mixing import master, mix  # noqa: F401  (mixing lives apart so it never invalidates the cache)
from render import smooth, write_wav
from score import Ev, ScoreGen, ot_range
from singers import SINGERS
from vtl_library import GLOTTIS_MODS, OUT as LIB_PATH, PRESSED_X, VOWELS

HERE = os.path.dirname(os.path.abspath(__file__))
FR = 1.0 / 0.0025            # control frames per second
SR = 48000
LIB = None

COMPENSATE = 0.6             # fraction of shape-to-shape loudness differences removed
HUM_BOOST_DB = 6.0           # humming is quieter; let it sit a little closer to open vowels
VELUM_OPEN = 0.6
NASAL_MAX_PRESS = 7000.0     # VTL's solver diverges for nasals (open velum, closed mouth) above this
CONS_DUR = dict(m=0.11, n=0.10, l=0.08, r=0.035, w=0.09, y=0.09, h=0.12,
                b=0.07, d=0.06, g=0.07, p=0.09, t=0.08, k=0.09, s=0.13, sh=0.13)
VOICELESS = {"p", "t", "k", "s", "sh", "h"}
ASPIRATION = 0.055           # glottis stays open this long after a voiceless release
WHISPER_ABDUCT = 0.2
WHISPER_BOOST_DB = 20.0
TONGUE = ["JA", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY"]
# which articulators each consonant takes over from the vowel
CONS_PARAMS = dict(
    m=("ll-labial-closure", ["JA", "LP", "LD"]),
    n=("tt-alveolar-closure", ["JA", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY"]),
    l=("tt-alveolar-lateral", ["JA", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY", "TS3"]),
    r=("tt-alveolar-closure", ["TTX", "TTY", "TBX", "TBY"]),
    b=("ll-labial-closure", ["JA", "LP", "LD"]),
    p=("ll-labial-closure", ["JA", "LP", "LD"]),
    d=("tt-alveolar-closure", TONGUE),
    t=("tt-alveolar-closure", TONGUE),
    g=("tb-velar-closure", ["JA", "TCX", "TCY", "TBX", "TBY"]),
    k=("tb-velar-closure", ["JA", "TCX", "TCY", "TBX", "TBY"]),
    s=("tt-alveolar-fricative", TONGUE + ["TS1", "TS2", "TS3"]),
    sh=("tt-postalveolar-fricative", TONGUE + ["LP", "TS1", "TS2", "TS3"]),
)


def library():
    global LIB
    if LIB is None:
        with open(LIB_PATH, "rb") as f:
            LIB = pickle.load(f)
        # the score only asks each voice for overtones it can make stand out
        score.AVAILABLE = {s["name"]: {k[1] for k in LIB["overtones"] if k[0] == si}
                           for si, s in enumerate(SINGERS)}
    return LIB


def resolve(si, shape):
    """Overtone tuning for (n, vowel); else the closest vowel that can carry n; else the vowel."""
    if shape[0] != "ot":
        return shape
    O = library()["overtones"]
    n, v = shape[1], shape[2]
    if (si, n, v) in O:
        return shape
    cands = [k[2] for k in O if k[0] == si and k[1] == n]
    if not cands:
        return ("v", v)
    F = library()["formants"]
    ref = np.log(F[(si, v)][:2])
    return ("ot", n, min(cands, key=lambda c: np.sum((np.log(F[(si, c)][:2]) - ref) ** 2)))


def params_for(si, shape):
    L = library()
    shape = resolve(si, shape)
    if shape[0] == "ot":
        return L["overtones"][(si, shape[1], shape[2])][0], shape
    return L["vowels"][(si, shape[1])], ("v", shape[1])


def gain_for(si, shape):
    L = library()
    _, key = params_for(si, shape if shape[0] != "hum" else ("v", shape[1]))
    ref = L["levels"][(si, "v", "a")]
    g = 10 ** (-COMPENSATE * (L["levels"][(si,) + key] - ref) / 20)
    return g * (10 ** (HUM_BOOST_DB / 20) if shape[0] == "hum" else 1.0)


def context(v):
    if v in ("i", "e", "E", "y", "oe"):
        return "i"
    if v in ("u", "o", "O"):
        return "u"
    return "a"


def tracks_for_singer(si, events, total):
    if NATURAL:
        return tracks_natural(si, events, total)
    s = SINGERS[si]
    info = vtl.constants()
    names = info["tract_info"]["names"]
    gnames = info["glottis_info"]["names"]
    P = {n: i for i, n in enumerate(names)}
    G = {n: i for i, n in enumerate(gnames)}
    nfr = int(total * FR) + 2

    table = [library()["vowels"][(si, "schwa")]]
    kidx = {("v", "schwa"): 0}
    owner = np.full(nfr, -1, int)
    over_val = np.zeros((nfr, len(names)))
    over_w = np.zeros((nfr, len(names)))
    velum = np.full(nfr, -0.1)
    press = np.zeros(nfr)
    abduct = np.full(nfr, 0.12)          # vocal folds apart when not singing
    gain = np.full(nfr, np.nan)          # rests keep the last note's gain
    scoop = np.zeros(nfr)
    bend = np.zeros(nfr)
    nasal = np.zeros(nfr, bool)
    rng = np.random.default_rng(500 + si)

    def overlay(i0, i1, shape_params, which):
        for n in which:
            over_val[i0:i1, P[n]] = shape_params[P[n]]
            over_w[i0:i1, P[n]] = 1.0

    for ev in events:
        base = ("v", ev.shape[1]) if ev.shape[0] == "hum" else ev.shape
        p, key = params_for(si, base)
        if key not in kidx:
            kidx[key] = len(table)
            table.append(p)
        i0 = int(ev.t * FR)
        i1 = min(max(int((ev.t + ev.dur) * FR), i0 + 2), nfr)
        owner[i0:i1] = kidx[key]
        amp = np.linspace(ev.amp, ev.amp if ev.amp1 is None else ev.amp1, i1 - i0)
        press[i0:i1] = 1500 + 11000 * amp
        abduct[i0:i1] = WHISPER_ABDUCT if ev.whisper else 0.0
        # dynamics act on loudness beyond what breath pressure alone gives
        gain[i0:i1] = gain_for(si, ev.shape) * np.sqrt(amp / 0.6) * \
            (10 ** (WHISPER_BOOST_DB / 20) if ev.whisper else 1.0)
        if ev.bend:
            b0, b1 = ev.bend
            bend[i0:i1] = b0 + (b1 - b0) * np.linspace(0, 1, i1 - i0) ** 2
        vow = ev.shape[1] if ev.shape[0] != "ot" else ev.shape[2]
        ctx = context(vow)
        if ev.shape[0] == "hum":
            overlay(i0, i1, vtl.shape(f"ll-labial-closure({ctx})"), CONS_PARAMS["m"][1])
            velum[i0:i1] = VELUM_OPEN
            nasal[i0:i1] = True
        if not ev.legato:
            k = np.arange(i1 - i0)
            scoop[i0:i1] = -rng.uniform(8, 25) * np.exp(-k / (0.07 * FR))
        c = ev.onset
        if not c:
            continue
        j1 = min(i0 + int(CONS_DUR[c] * FR), i1)
        if c in CONS_PARAMS:
            name, which = CONS_PARAMS[c]
            overlay(i0, j1, vtl.shape(f"{name}({ctx})"), which)
            if c in ("m", "n"):
                velum[i0:j1] = VELUM_OPEN
                nasal[i0:j1] = True
            if c in VOICELESS:
                abduct[i0:min(j1 + int(ASPIRATION * FR), i1)] = 0.25
        elif c == "w":
            u = vtl.shape("u").copy()
            u[P["LD"]] -= 0.5
            overlay(i0, j1, u, ["LP", "LD"])
        elif c == "y":
            iy = vtl.shape("i").copy()
            iy[P["TCY"]] += 0.3
            overlay(i0, j1, iy, ["TCX", "TCY", "TTX", "TTY", "TBX", "TBY"])
        elif c == "h":
            abduct[i0:j1] = 0.25

    # rests keep the last articulation
    idx = np.where(owner >= 0, np.arange(nfr), 0)
    np.maximum.accumulate(idx, out=idx)
    owner = np.where(owner[idx] >= 0, owner[idx], 0)

    T = np.array(table)
    tract = smooth(T[owner], 0.04 * FR)
    w = smooth(over_w, 0.014 * FR)
    vals = np.where(over_w > 0, over_val, np.nan)
    # spread consonant targets into their ramps so blending has a value to blend toward
    vals = _ffill_bfill(vals)
    tract = (1 - w) * tract + w * np.nan_to_num(vals)
    tract[:, P["VO"]] = smooth(velum, 0.025 * FR)
    tract = vtl.clip_tract(tract)

    glot = np.tile(info["glottis_info"]["neutral"], (nfr, 1))
    cents = smooth(rng.normal(0, 1, nfr), 1.5 * FR)
    cents *= 1.5 / (np.std(cents) + 1e-9)
    glot[:, G["f0"]] = s["f0"] * 2 ** ((cents + smooth(scoop + bend, 0.015 * FR)) / 1200)
    pad = int(0.02 * FR)
    near_nasal = smooth(nasal.astype(float), 2 * pad) > 0
    press = np.where(near_nasal, np.minimum(press, NASAL_MAX_PRESS), press)
    glot[:, G["pressure"]] = smooth(press, 0.015 * FR)
    ab = smooth(abduct, 0.02 * FR)
    glot[:, G["x_bottom"]] += ab
    glot[:, G["x_top"]] += ab
    for k, v in GLOTTIS_MODS.items():
        glot[:, G[k]] = v
    gain = _ffill_bfill(gain[:, None])[:, 0]
    return tract, glot, smooth(np.nan_to_num(gain, nan=1.0), 0.015 * FR)


# consonant clarity (tuned by a consonant-identification test under sung, reverberant conditions)
PRESS_BUILDUP = True         # breath pressure rises during the closure before a syllable: stronger bursts
APPROX_V2 = True             # w / y as full gestures (rounded lips; palatal tongue) rather than nudges
STOP_SCALE = 1.3             # closure durations of stops
ANTICIPATE = 0.8             # fraction of the consonant sung before the beat, so the vowel lands on it
DICTION_DB = 0.0             # optional boost of bursts / frication (half for nasals and approximants)
NATURAL = True               # VTL-style gestures and target approximation (else box smoothing)
PHRASE_GAP = 0.3             # shorter rests keep some breath pressure (staccato within a phrase)
GAP_PRESSURE = 0.35          # fraction of the note's pressure kept in a short rest
GAP_ABDUCT = 0.08
L_BLEND = 1.0                # how far /l/ goes toward its full lateral target
VSTOP_PRESS = 1.0            # pressure factor during voiced stop closures
_ART = {}


def tracks_natural(si, events, total):
    """Control tracks built like VTL's gestural model: vowel and context-dependent consonant
    targets, each parameter following them as a 5th-order critically damped system."""
    s = SINGERS[si]
    info = vtl.constants()
    P = {n: i for i, n in enumerate(info["tract_info"]["names"])}
    G = {n: i for i, n in enumerate(info["glottis_info"]["names"])}
    art = _ART.setdefault(s["speaker"], Articulation())
    nfr = int(total * FR) + 2
    nP = len(P)
    rng = np.random.default_rng(500 + si)
    fr = lambda sec: int(round(sec * FR))

    target = np.full((nfr, nP), np.nan)
    tau = np.full((nfr, nP), TAU_VOWEL)
    vel = np.full(nfr, np.nan)
    press = np.zeros(nfr)
    p_tau_up = np.full(nfr, 0.010)       # VTL: lung pressure onset/offset time constants
    p_tau_dn = np.full(nfr, 0.020)
    abduct = np.full(nfr, 0.12)
    gain = np.full(nfr, np.nan)
    scoop = np.zeros(nfr)
    bend = np.zeros(nfr)
    nasal = np.zeros(nfr, bool)

    events = sorted(events, key=lambda e: e.t)
    if ANTICIPATE:
        moved, prev_end = [], -1.0
        for e in events:
            d = 0.0
            if e.onset:
                d = min(ANTICIPATE * CONS_DUR[e.onset], max(0.0, e.t - prev_end - 0.03))
            moved.append(replace(e, t=e.t - d, dur=e.dur + d))
            prev_end = e.t + e.dur
        events = moved
    diction = np.zeros(nfr)
    vowel_of = []
    for ev in events:                    # vowels first; rests keep the last articulation
        base = ("v", ev.shape[1]) if ev.shape[0] == "hum" else ev.shape
        p, _ = params_for(si, base)
        vowel_of.append(p)
        i0, i1 = fr(ev.t), min(max(fr(ev.t + ev.dur), fr(ev.t) + 2), nfr)
        target[i0:i1] = p
        tau[i0:i1] = TAU_LEGATO if ev.legato else TAU_VOWEL
        vel[i0:i1] = p[P["VO"]]
    target = _ffill_bfill(target)
    vel = _ffill_bfill(vel[:, None])[:, 0]

    for k, ev in enumerate(events):
        vp = vowel_of[k]
        i0, i1 = fr(ev.t), min(max(fr(ev.t + ev.dur), fr(ev.t) + 2), nfr)
        amp = np.linspace(ev.amp, ev.amp if ev.amp1 is None else ev.amp1, i1 - i0)
        press[i0:i1] = 1500 + 11000 * amp
        abduct[i0:i1] = WHISPER_ABDUCT if ev.whisper else 0.0
        gain[i0:i1] = gain_for(si, ev.shape) * np.sqrt(amp / 0.6) * \
            (10 ** (WHISPER_BOOST_DB / 20) if ev.whisper else 1.0)
        if ev.bend:
            b0, b1 = ev.bend
            bend[i0:i1] = b0 + (b1 - b0) * np.linspace(0, 1, i1 - i0) ** 2
        if not ev.legato:             # small scoop into the note, less on quick notes
            depth = rng.uniform(8, 25) * min(1.0, ev.dur / 0.4)
            scoop[i0:i1] = -depth * np.exp(-np.arange(i1 - i0) / (0.05 * FR))
        # a short rest before the next note keeps some pressure: staccato inside a phrase
        nxt = events[k + 1].t if k + 1 < len(events) else None
        if nxt is not None and 0 < nxt - (ev.t + ev.dur) < PHRASE_GAP:
            j1 = min(fr(nxt), nfr)
            press[i1:j1] = GAP_PRESSURE * press[i1 - 1]
            abduct[i1:j1] = GAP_ABDUCT
            p_tau_up[i1:min(j1 + fr(0.05), nfr)] = 0.006
            p_tau_dn[i1:j1] = 0.012
        if ev.shape[0] == "hum":
            target[i0:i1] = art.consonant("m", vp)
            tau[i0:i1] = TAU_CONS
            vel[i0:i1] = VELUM_OPEN
            nasal[i0:i1] = True
        c = ev.onset
        if not c:
            continue
        dc = CONS_DUR[c] * (STOP_SCALE if c in ("b", "d", "g", "p", "t", "k") else 1.0)
        if APPROX_V2 and c in ("w", "y"):
            dc = 0.11
            if c == "y":
                q = art.consonant_shape("tb-palatal-fricative", vp)
                q = vp + 0.8 * (q - vp)
            else:
                q = vtl.shape("u").copy()
                q[P["LD"]] = min(q[P["LD"]], 0.12)
                q[P["LP"]] = max(q[P["LP"]], vtl.shape("u")[P["LP"]])
                q[P["VO"]] = vp[P["VO"]]
            g0, g1 = max(0, fr(ev.t - 0.035)), fr(ev.t + dc - 0.035)
            target[g0:g1] = q
            tau[g0:g1] = TAU_CONS
            tau[g1:min(nfr, g1 + fr(0.06))] = 0.014
        if PRESS_BUILDUP and c in CONSONANTS and c not in ("m", "n"):
            lead_ = art.lead_lag(c)[0]
            j0 = max(0, fr(ev.t - lead_))
            press[j0:i0] = np.maximum(press[j0:i0], press[i0])
            p_tau_up[j0:i0] = 0.008
        if c in CONSONANTS:
            lead, lag = art.lead_lag(c)
            g0 = max(0, fr(ev.t - lead))
            g1 = min(nfr, max(fr(ev.t + dc - lag), g0 + fr(lead + 0.01)))
            ct = art.consonant(c, vp)
            if c == "l":
                ct = vp + L_BLEND * (ct - vp)
            target[g0:g1] = ct
            if c in ("b", "d", "g"):
                press[max(0, fr(ev.t - 0.02)):fr(ev.t + dc)] *= VSTOP_PRESS
            tau[g0:g1] = TAU_CONS
            tau[g1:min(nfr, g1 + fr(0.06))] = TAU_RELEASE
            if c in ("m", "n"):
                vel[g0:g1] = VELUM_OPEN
                nasal[g0:fr(ev.t + dc)] = True
        elif c in ("w", "y") and not APPROX_V2:
            q = (vtl.shape("u") if c == "w" else vtl.shape("i")).copy()
            cols = ["LP", "LD"] if c == "w" else ["TCX", "TCY", "TTX", "TTY", "TBX", "TBY"]
            q[P["LD"]] -= 0.5
            q[P["TCY"]] += 0.3
            g0, g1 = max(0, fr(ev.t - 0.03)), fr(ev.t + dc - 0.03)
            for n in cols:
                target[g0:g1, P[n]] = q[P[n]]
        if c in VOICELESS:
            abduct[max(0, fr(ev.t - 0.03)):min(fr(ev.t + dc + ASPIRATION), i1)] = 0.25
        if DICTION_DB:
            if c in ("b", "d", "g", "p", "t", "k"):          # the release burst and transition
                diction[fr(ev.t + dc - 0.01):fr(ev.t + dc + 0.04)] = DICTION_DB
            elif c in ("s", "sh", "h"):
                diction[fr(ev.t):fr(ev.t + dc)] = DICTION_DB
            else:
                diction[fr(ev.t):fr(ev.t + dc)] = DICTION_DB / 2

    tract = tam(target, tau)
    tract[:, P["VO"]] = tam(vel, 0.012, 0.010)          # VTL velic opening/closing
    tract = vtl.clip_tract(tract)

    near_nasal = smooth(nasal.astype(float), fr(0.04)) > 0
    press = np.where(near_nasal, np.minimum(press, NASAL_MAX_PRESS), press)
    glot = np.tile(info["glottis_info"]["neutral"], (nfr, 1))
    cents = smooth(rng.normal(0, 1, nfr), 1.5 * FR)
    cents *= 1.5 / (np.std(cents) + 1e-9)
    glot[:, G["f0"]] = s["f0"] * 2 ** ((cents + tam(scoop + bend, 0.005)) / 1200)
    glot[:, G["pressure"]] = np.maximum(tam(press, p_tau_up, p_tau_dn), 0.0)
    ab = tam(abduct, 0.010)
    glot[:, G["x_bottom"]] = PRESSED_X + ab          # pressed phonation: sharper resonances
    glot[:, G["x_top"]] = PRESSED_X + ab
    for k, v in GLOTTIS_MODS.items():
        glot[:, G[k]] = v
    # loudness compensation moves slowly, so it never makes syllable-rate jumps
    gain = np.nan_to_num(_ffill_bfill(gain[:, None])[:, 0], nan=1.0)
    return tract, glot, tam(gain, 0.04) * 10 ** (tam(diction, 0.004) / 20)


def _ffill_bfill(a):
    """Fill NaNs along time with the nearest defined value (forward, then backward)."""
    out = a.copy()
    for arr in (out, out[::-1]):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr))[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        filled = np.take_along_axis(arr, idx, axis=0)
        arr[mask] = filled[mask]
    return out


def render_singer(args):
    si, events, total, outdir = args
    t = time.time()
    s = SINGERS[si]
    vtl.init(s["speaker"])
    tract, glot, gain = tracks_for_singer(si, events, total)
    log = []
    y = vtl.synth(tract, glot, log)
    y *= np.repeat(gain[1:], vtl.STATE_SAMPLES)[: len(y)]
    path = os.path.join(outdir, f"{s['name']}.npy")
    np.save(path, y.astype(np.float32))
    with open(os.path.join(outdir, f"{s['name']}_glitches.pkl"), "wb") as f:
        pickle.dump(log, f)
    times = ", ".join(f"{i / FR:.2f}s" for i, _, _ in log[:8])
    print(f"  {s['name']:9s} ({s['speaker']}) rendered in {time.time() - t:.0f}s; "
          f"solver restarts: {len(log)}{' at ' + times if log else ''}", flush=True)
    return path


# -- segmented, cached rendering ---------------------------------------------------
SEG_GAP = 0.5                # rests at least this long split a voice into separate segments
SEG_LEAD, SEG_TAIL = 0.15, 0.6
CACHE_DIR = os.path.join(HERE, "..", "cache", "segments")
_SPEAKER = None
_CODE_KEY = None


def code_key():
    """Fingerprint of everything that shapes the sound, so cached segments go stale with it."""
    global _CODE_KEY
    if _CODE_KEY is None:
        import hashlib
        h = hashlib.sha1()
        for f in ("render_vtl.py", "articulation.py", "vtl.py", "vtl_library.pkl"):
            with open(os.path.join(HERE, f), "rb") as fh:
                h.update(fh.read())
        _CODE_KEY = h.hexdigest()[:16]
    return _CODE_KEY


def segments(events, total):
    """Split a voice at long rests: [(start, end, events shifted to start)]."""
    segs = []
    for e in sorted(events, key=lambda e: e.t):
        a, b = max(0.0, e.t - SEG_LEAD), min(total, e.t + e.dur + SEG_TAIL)
        if segs and a <= segs[-1][1] + SEG_GAP:
            segs[-1][1] = max(segs[-1][1], b)
            segs[-1][2].append(e)
        else:
            segs.append([a, b, [e]])
    return [(a, b, [replace(e, t=e.t - a) for e in evs]) for a, b, evs in segs]


def render_segment(args):
    """One stretch of one voice; cached on disk by its content and the code version."""
    global _SPEAKER
    si, a, b, evs, use_cache = args
    import hashlib
    key = hashlib.sha1(repr((si, SINGERS[si]["speaker"], round(b - a, 4), evs, code_key())).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".npy")
    if use_cache and os.path.exists(path):
        return si, a, np.load(path), [], True
    if _SPEAKER != SINGERS[si]["speaker"]:
        vtl.init(SINGERS[si]["speaker"])
        _SPEAKER = SINGERS[si]["speaker"]
    tract, glot, gain = tracks_for_singer(si, evs, b - a)
    log = []
    y = vtl.synth(tract, glot, log)
    y = (y * np.repeat(gain[1:], vtl.STATE_SAMPLES)[: len(y)]).astype(np.float32)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npy"
        np.save(tmp, y)
        os.replace(tmp, path)
    return si, a, y, [(a + i / FR, tr, gl) for i, tr, gl in log], False


def render_all(events, total, outdir, use_cache=True, workers=None):
    """All voices, split into segments rendered in parallel; returns per-voice stem paths."""
    tasks = [(si, a, b, evs, use_cache) for si in range(len(SINGERS)) for a, b, evs in segments(events[si], total)]
    tasks.sort(key=lambda x: -(x[2] - x[1]))            # longest first, for load balance
    workers = workers or max(1, min(len(tasks), (os.cpu_count() or 4) - 2))
    stems = [np.zeros(int(total * SR) + SR, np.float32) for _ in SINGERS]
    logs = [[] for _ in SINGERS]
    t0, hits, done = time.time(), 0, 0
    with Pool(workers) as pool:
        for si, a, y, log, hit in pool.imap_unordered(render_segment, tasks):
            i0 = int(round(a * SR))
            n = min(len(y), len(stems[si]) - i0)
            stems[si][i0:i0 + n] += y[:n]
            logs[si] += log
            hits += hit
            done += 1
    sung = sum(b - a for _, a, b, _, _ in tasks)
    print(f"  {len(tasks)} segments ({sung:.0f}s of singing, {hits} from cache) on {workers} cores "
          f"in {time.time() - t0:.0f}s", flush=True)
    paths = []
    for si, s in enumerate(SINGERS):
        if logs[si]:
            print(f"  {s['name']:9s} solver restarts: {len(logs[si])} at "
                  + ", ".join(f"{t:.2f}s" for t, _, _ in sorted(logs[si])[:8]), flush=True)
        with open(os.path.join(outdir, f"{s['name']}_glitches.pkl"), "wb") as f:
            pickle.dump(logs[si], f)
        p = os.path.join(outdir, f"{s['name']}.npy")
        np.save(p, stems[si][: int(total * SR)])
        paths.append(p)
    return paths


def demo_events(si):
    """Plain syllables, then an overtone sweep on 'a'-like vowels."""
    ev, t = [], 0.3
    for onset, v in [("m", "a"), ("n", "e"), ("l", "i"), ("n", "o"), ("w", "a"), ("y", "u"), ("m", "oe")]:
        ev.append(Ev(t, 0.75, ("v", v), onset, 0.5, 0.6))
        t += 0.85
    ev.append(Ev(t + 0.2, 1.8, ("hum", "u"), "m", 0.45, 0.6))
    t += 2.6
    for n in ot_range(SINGERS[si]):
        sh = resolve(si, ("ot", n, "a"))
        if sh[0] == "ot":
            ev.append(Ev(t, 1.2, sh, "", 0.5, 0.75, legato=True))
            t += 1.2
    return ev, t + 0.5


def demo_job(si):
    s = SINGERS[si]
    vtl.init(s["speaker"])
    ev, total = demo_events(si)
    tract, glot, gain = tracks_for_singer(si, ev, total)
    y = vtl.synth(tract, glot) * np.repeat(gain[1:], vtl.STATE_SAMPLES)
    return y / (np.max(np.abs(y)) + 1e-12)


def save_score(sc, path):
    with open(path, "wb") as f:
        pickle.dump(score.SavedScore(sc), f)


class _ScoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SavedScore":           # also reads files saved when the class lived in __main__
            return score.SavedScore
        return super().find_class(module, name)


def load_score(path):
    with open(path, "rb") as f:
        return _ScoreUnpickler(f).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", type=float, default=0, help="render only this many seconds")
    ap.add_argument("--start", type=float, default=0, help="with --preview: start here")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the segment cache")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "stimmung_vtl"))
    a = ap.parse_args()
    library()

    if a.demo:
        with Pool(len(SINGERS)) as pool:
            parts = pool.map(demo_job, range(len(SINGERS)))
        y = np.concatenate([np.concatenate([0.7 * p, np.zeros(int(0.8 * SR))]) for p in parts])
        out = os.path.join(HERE, "..", "out", "voice_demo_vtl.wav")
        write_wav(out, np.stack([y, y]) * 0.9, SR)
        print("wrote", out, f"{len(y) / SR:.0f}s")
        return

    sc = ScoreGen(a.seed).build()
    total = a.preview or sc.total
    t0 = a.start if a.preview else 0.0
    with open(a.out + "_score.txt", "w") as f:
        f.write(sc.describe() + "\n")
    if not a.preview:       # the video reads exactly this score (no regeneration)
        save_score(sc, a.out + "_score.pkl")
    print(f"total {sc.total / 60:.1f} min; rendering {total:.0f}s", flush=True)
    tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
    events = [[replace(e, t=e.t - t0) for e in sc.events[si] if t0 <= e.t < t0 + total]
              for si in range(len(SINGERS))]
    paths = render_all(events, total, tmp, use_cache=not a.no_cache)
    mix(paths, total, a.out)


if __name__ == "__main__":
    main()
