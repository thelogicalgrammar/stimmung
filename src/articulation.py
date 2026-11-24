"""Natural articulator trajectories, following VocalTractLab's own gestural model
(Birkholz, GesturalScore.cpp):

- a consonant's target is interpolated from its (a)/(i)/(u) variants according to
  where the surrounding vowel lies in the a-i-u subspace (coarticulation);
- every control parameter approaches its succession of targets as a 5th-order
  critically damped system with time constant tau (target approximation);
- consonant gestures start early enough that the constriction is actually formed
  when the consonant is due, using lead/lag times measured on the tube model.
"""
import numpy as np

import vtl

FR = 400.0
ORDER = 5
TAU_VOWEL = 0.012            # VTL: tau = 1/10 of a ~100 ms transition
TAU_LEGATO = 0.02            # slower, sung vowel changes
TAU_CONS = 0.008
TAU_RELEASE = 0.010

TONGUE_SUBSPACE = ["HX", "HY", "JX", "JA", "VS", "VO", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY",
                   "TRX", "TRY", "TS1", "TS2", "TS3"]
# consonant -> VTL shape family
CONSONANTS = {
    "b": "ll-labial-closure", "p": "ll-labial-closure", "m": "ll-labial-closure",
    "d": "tt-alveolar-closure", "t": "tt-alveolar-closure", "n": "tt-alveolar-closure",
    "r": "tt-alveolar-closure", "l": "tt-alveolar-lateral",
    "s": "tt-alveolar-fricative", "sh": "tt-postalveolar-fricative",
    "g": "tb-velar-closure", "k": "tb-velar-closure",
}


def tam(target, tau_up, tau_down=None, dt=1.0 / FR):
    """Target approximation: ORDER cascaded one-pole stages (impulse response ~ t^4 e^-t/tau).

    target: (n, k); tau_up/tau_down: (n, k) or (n,) time constants in seconds, used when the
    target lies above/below the current value.
    """
    target = np.asarray(target, float)
    squeeze = target.ndim == 1
    if squeeze:
        target = target[:, None]
    n, k = target.shape
    def alpha(tau):
        tau = np.asarray(tau, float)
        tau = tau.reshape(n, -1) if tau.ndim else np.full((n, 1), float(tau))
        return 1.0 - np.exp(-dt / np.broadcast_to(tau, (n, k)))
    au = alpha(tau_up)
    ad = au if tau_down is None else alpha(tau_down)
    y = np.repeat(target[:1], ORDER, 0)
    out = np.empty_like(target)
    for i in range(n):
        x = target[i]
        a = au[i] if tau_down is None else np.where(x >= y[-1], au[i], ad[i])
        for s in range(ORDER):
            y[s] += a * (x - y[s])
            x = y[s]
        out[i] = y[-1]
    return out[:, 0] if squeeze else out


class Articulation:
    """Per-speaker consonant targets and timing; call after vtl.init(speaker)."""

    def __init__(self):
        names = vtl.constants()["tract_info"]["names"]
        self.P = {n: i for i, n in enumerate(names)}
        self.A, self.I, self.U = (vtl.shape(v) for v in ("a", "i", "u"))
        self.shapes = {fam: [vtl.shape(f"{fam}({v})") for v in "aiu"] for fam in set(CONSONANTS.values())}
        self.timing = {}

    def subspace(self, p):
        """VTL's mapToVowelSubspace + limitVowelSubspaceCoord."""
        idx = [self.P[n] for n in TONGUE_SUBSPACE]
        M = np.stack([self.I[idx] - self.A[idx], self.U[idx] - self.A[idx]], 1)
        at, bt = np.linalg.lstsq(M, p[idx] - self.A[idx], rcond=None)[0]
        lp, ld = self.P["LP"], self.P["LD"]
        L = np.array([[self.I[lp] - self.A[lp], self.U[lp] - self.A[lp]],
                      [self.I[ld] - self.A[ld], self.U[ld] - self.A[ld]]])
        try:
            al, bl = np.linalg.solve(L, [p[lp] - self.A[lp], p[ld] - self.A[ld]])
        except np.linalg.LinAlgError:
            al, bl = 0.0, 0.0

        def lim(a, b):
            a, b = np.clip(a, 0, 1), np.clip(b, 0, 1)
            if a + b > 1:
                d = a + b - 1
                a, b = a - d / 2, b - d / 2
            return a, b
        return lim(at, bt) + lim(al, bl)

    def consonant(self, c, vowel):
        """Context-dependent consonant target (all parameters, like VTL)."""
        sa, si, su = self.shapes[CONSONANTS[c]]
        at, bt, al, bl = self.subspace(vowel)
        out = sa + at * (si - sa) + bt * (su - sa)
        for n in ("LP", "LD"):
            j = self.P[n]
            out[j] = sa[j] + al * (si[j] - sa[j]) + bl * (su[j] - sa[j])
        out[self.P["VO"]] = vowel[self.P["VO"]]
        return out

    def consonant_shape(self, family, vowel):
        """Context-dependent target for any VTL shape family with (a)/(i)/(u) variants."""
        if family not in self.shapes:
            self.shapes[family] = [vtl.shape(f"{family}({v})") for v in "aiu"]
        sa, si, su = self.shapes[family]
        at, bt, al, bl = self.subspace(vowel)
        out = sa + at * (si - sa) + bt * (su - sa)
        out[self.P["VO"]] = vowel[self.P["VO"]]
        return out

    def lead_lag(self, c):
        """Seconds a gesture must start before its constriction forms (lead) and end before
        it releases (lag), measured on the tube model in /a/ context."""
        if c not in self.timing:
            v = self.A
            cons = self.consonant(c, v)
            amin = lambda p: max(np.min(vtl.tube(p)[1]), 1e-4)
            thr = np.sqrt(amin(v) * amin(cons))         # halfway, on a log scale
            n = int(0.15 * FR)
            on = tam(np.vstack([v[None]] * 2 + [cons[None]] * n), TAU_CONS)
            off = tam(np.vstack([cons[None]] * 2 + [v[None]] * n), TAU_RELEASE)
            t_on = next((i for i in range(2, n) if amin(on[i]) <= thr), n) - 2
            t_off = next((i for i in range(2, n) if amin(off[i]) > thr), n) - 2
            self.timing[c] = (t_on / FR, t_off / FR)
        return self.timing[c]
