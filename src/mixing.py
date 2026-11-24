"""Spatial mix and mastering of the six voice stems.

Kept apart from render_vtl.py so that changing the mix never invalidates the cached
voice segments (their cache key hashes the synthesis code only).

The singers sit on the video's hexagon (bass at the top, clockwise); the listener sits
just in front of it, below the alto.  Each voice gets a direction and a distance from
that geometry and is placed with simple binaural cues: an interaural time difference,
a level difference with head shadow at the far ear, its own early reflections, and a
reverb send that grows with distance into decorrelated left/right tails.
"""
import os
import subprocess

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import lfilter

from render import fftconv, reverb_ir, write_wav
from singers import SINGERS

SR = 48000
SEAT_ANGLES = np.deg2rad([-90 + 60 * k for k in range(6)])   # as in video.py
LISTENER_Y = 1.7            # in hexagon radii, below the centre (screen y points down)
WIDTH = 1.8                 # azimuth exaggeration, so the image fills the stereo field
MAX_AZ = np.deg2rad(80)
MAX_ITD = 0.00038           # s; kept below the full head's 0.66 ms for mono compatibility
SHADOW_HZ = 1800.0          # far-ear head-shadow low-pass
REVERB = 0.15


def seat(si):
    """(azimuth in radians, positive = right; distance in hexagon radii)"""
    a = SEAT_ANGLES[si]
    x, y = np.cos(a), np.sin(a)
    dx, dy = x, LISTENER_Y - y
    return float(np.clip(np.arctan2(dx, dy) * WIDTH, -MAX_AZ, MAX_AZ)), float(np.hypot(dx, dy))


def _lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    return lfilter([1 - a], [1, -a], x)


def _delay(x, d):
    d = int(round(d))
    return x if d <= 0 else np.concatenate([np.zeros(d), x[:-d]])


def spatialise(y, az, dist, seed):
    """Mono voice -> (2, n) direct sound and early reflections."""
    s = np.sin(az)
    near_is_right = s >= 0
    p = 0.85 * s                                              # constant-power pan on top of the cues
    g_l, g_r = np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)
    shade = 0.7 * abs(s)
    far = (1 - shade) * y + shade * _lowpass(y, SHADOW_HZ)
    far = _delay(far, MAX_ITD * abs(s) * SR)
    left, right = (far * g_l, y * g_r) if near_is_right else (y * g_l, far * g_r)
    g = 1.0 / (0.7 + 0.3 * dist)                              # gentle distance attenuation
    out = np.stack([left, right]) * g
    # early reflections from the walls: a few delayed, duller copies, each from its own side
    rng = np.random.default_rng(seed)
    base = _lowpass(y, 4000.0) * g
    for k in range(6):
        d = rng.uniform(0.006, 0.034) * (0.8 + 0.2 * dist) * SR
        side = rng.integers(2)
        amp = 0.32 * (0.8 ** k)
        refl = _delay(base, d) * amp
        out[side] += refl
        out[1 - side] += _delay(refl, rng.uniform(0.1, 0.4) * MAX_ITD * SR) * 0.6
    return out


def mix(paths, total, out, reverb=REVERB):
    n = int(total * SR)
    dry = np.zeros((2, n))
    send = np.zeros(n)
    for si, p in enumerate(paths):
        y = np.load(p).astype(np.float64)[:n]
        y /= 0.05          # common scale; voices are balanced by pressure, not normalised
        az, dist = seat(si)
        dry[:, :len(y)] += spatialise(y, az, dist, seed=100 + si)
        send[:len(y)] += y * (0.6 + 0.25 * dist) / 1.2       # farther voices are wetter
    irL, irR = reverb_ir(sr=SR, rt_scale=0.5)                # a drier room: consonants stay clear
    wet = np.stack([fftconv(send, irL), fftconv(send, irR)])  # independent tails: decorrelated
    m = np.zeros_like(wet)
    m[:, :n] = dry
    m += reverb * wet
    m = m[:, : n + int(2.5 * SR)]
    fl = int(2.0 * SR)
    m[:, -fl:] *= np.linspace(1, 0, fl) ** 2
    if not np.all(np.isfinite(m)):
        raise RuntimeError("mix contains non-finite samples; stems kept in " + os.path.dirname(paths[0]))
    m = master(m)
    write_wav(out + ".wav", m, SR)
    for p in paths:
        os.remove(p)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".wav", "-b:a", "256k",
                    out + ".mp3"], check=True)
    print("wrote", out + ".wav", "and", out + ".mp3")
    return m


def master(m, sr=SR, threshold_db=-40.0, ratio=2.0, attack=0.02, release=0.4):
    """Gentle compression of the stereo mix (levels relative to its peak), then -1 dBFS peak.
    Keeps the dynamic contrasts but lifts quiet passages to a listenable level."""
    m = m / (np.max(np.abs(m)) + 1e-12)
    h = int(0.01 * sr)
    n = m.shape[1] // h
    lev = 10 * np.log10(np.mean(m[:, : n * h].reshape(2, n, h) ** 2, axis=(0, 2)) + 1e-12)
    a_att, a_rel = np.exp(-0.01 / attack), np.exp(-0.01 / release)
    env = np.empty(n)
    e = lev[0]
    for i in range(n):
        a = a_att if lev[i] > e else a_rel
        e = a * e + (1 - a) * lev[i]
        env[i] = e
    gain_db = -np.maximum(env - threshold_db, 0.0) * (1 - 1 / ratio)
    g = np.interp(np.arange(m.shape[1]), np.arange(n) * h + h / 2, 10 ** (gain_db / 20))
    m = m * g
    # loudness: the loud passages (99th percentile of 10 ms levels) sit at -12 dBFS ...
    lev = 10 * np.log10(np.mean(m[:, : n * h].reshape(2, n, h) ** 2, axis=(0, 2)) + 1e-12)
    m = m * 10 ** ((-12.0 - np.percentile(lev, 99)) / 20)
    # ... and a look-ahead limiter keeps the few transients above that under -1 dBFS
    ceiling = 10 ** (-1 / 20)
    pk = maximum_filter1d(np.max(np.abs(m), 0), int(0.005 * sr))
    need = np.minimum(1.0, ceiling / (pk + 1e-12))
    w = int(0.01 * sr)
    need = minimum_filter1d(need, 2 * w)
    gl = np.convolve(need, np.hanning(2 * w) / np.hanning(2 * w).sum(), "same")
    m = m * np.minimum(gl, need)
    return np.clip(m, -ceiling, ceiling)
