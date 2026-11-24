"""ctypes wrapper for the VocalTractLab 2.4 API (Birkholz), built from source.

The API keeps one global synthesiser, so use one speaker per process.
"""
import ctypes
import os

import numpy as np

LIB_PATH = os.environ.get("VTL_LIB", os.path.expanduser("~/.venvs/stimmung/vtl24/libVocalTractLabApi24.so"))
SPEAKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")

_lib = None
_c = {}
STATE_SAMPLES = 120      # 2.5 ms per control frame at 48 kHz

_dp = ctypes.POINTER(ctypes.c_double)


def _ptr(a):
    return a.ctypes.data_as(_dp)


def init(speaker):
    """Load a speaker ('W02', 'M01', 'JD2' or a path)."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB_PATH)
    else:
        _lib.vtlClose()
    path = speaker if os.path.exists(speaker) else os.path.join(SPEAKER_DIR, speaker + ".speaker")
    rc = _lib.vtlInitialize(path.encode())
    if rc != 0:
        raise RuntimeError(f"vtlInitialize({path}) failed: {rc}")
    sr, nt, ntp, ngp = (ctypes.c_int() for _ in range(4))
    _lib.vtlGetConstants(ctypes.byref(sr), ctypes.byref(nt), ctypes.byref(ntp), ctypes.byref(ngp))
    _c.update(sr=sr.value, n_tube=nt.value, n_tract=ntp.value, n_glottis=ngp.value)
    _c["tract_info"] = _param_info(_lib.vtlGetTractParamInfo, ntp.value)
    _c["glottis_info"] = _param_info(_lib.vtlGetGlottisParamInfo, ngp.value)
    return dict(_c)


def _param_info(fn, n):
    names = ctypes.create_string_buffer(32 * n)
    lo, hi, neu = np.zeros(n), np.zeros(n), np.zeros(n)
    fn(names, _ptr(lo), _ptr(hi), _ptr(neu))
    return dict(names=names.value.decode().split(), min=lo, max=hi, neutral=neu)


def constants():
    return dict(_c)


def shape(name):
    p = np.zeros(_c["n_tract"])
    if _lib.vtlGetTractParams(name.encode(), _ptr(p)) != 0:
        raise KeyError(name)
    return p


def clip_tract(p):
    i = _c["tract_info"]
    return np.clip(p, i["min"], i["max"])


def tube(p):
    n = _c["n_tube"]
    L, A = np.zeros(n), np.zeros(n)
    art = (ctypes.c_int * n)()
    inc, tts, vel = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    _lib.vtlTractToTube(_ptr(np.ascontiguousarray(p, float)), _ptr(L), _ptr(A), art,
                        ctypes.byref(inc), ctypes.byref(tts), ctypes.byref(vel))
    return L, A


def transfer_function(p, nfft=4096):
    """Magnitude of the glottis->lips volume-velocity transfer function, freqs in Hz."""
    mag, ph = np.zeros(nfft), np.zeros(nfft)
    _lib.vtlGetTransferFunction(_ptr(np.ascontiguousarray(p, float)), nfft, _ptr(mag), _ptr(ph))
    f = np.arange(nfft // 2) * _c["sr"] / nfft
    return f, mag[: nfft // 2]


def synth(tract, glottis, log=None):
    """tract: (frames, n_tract), glottis: (frames, n_glottis); 2.5 ms per frame.

    If VTL's solver produces non-finite samples, the block is silenced, the synthesiser
    is restarted from the current state, and (frame, tract, glottis) is appended to log.
    """
    tract = np.ascontiguousarray(tract, float)
    glottis = np.ascontiguousarray(glottis, float)
    nfr = len(tract)
    out = np.zeros((nfr - 1) * STATE_SAMPLES)
    buf = np.zeros(STATE_SAMPLES)
    _lib.vtlResetTractSynthesis()
    _lib.vtlSynthesisAddTract(0, _ptr(buf), _ptr(tract[0]), _ptr(glottis[0]))
    for i in range(1, nfr):
        _lib.vtlSynthesisAddTract(STATE_SAMPLES, _ptr(buf), _ptr(tract[i]), _ptr(glottis[i]))
        if not np.all(np.isfinite(buf)) or np.max(np.abs(buf)) > 10.0:
            if log is not None:
                log.append((i, tract[i].copy(), glottis[i].copy()))
            buf[:] = 0.0
            _lib.vtlResetTractSynthesis()
            _lib.vtlSynthesisAddTract(0, _ptr(buf), _ptr(tract[i]), _ptr(glottis[i]))
        out[(i - 1) * STATE_SAMPLES: i * STATE_SAMPLES] = buf
    return out


def speak(segments, workdir, f0_default=110.0):
    """Speech through VTL's own segment -> gestural score -> audio pipeline.

    segments: list of (sampa, duration_s, f0_Hz or None, accent); "" is a pause.
    Returns float audio at 48 kHz.
    """
    seg = os.path.join(workdir, "speech.seg")
    ges = os.path.join(workdir, "speech.ges")
    lines, word_start = [], True
    for name, dur, f0, accent in segments:
        f = 0.0 if not name else (f0 or f0_default)
        lines.append(f"name = {name if name else ' '}; start_of_word = {int(word_start and bool(name))}; "
                     f"word_accent = {int(accent)}; start_of_phrase = 0; duration_s = {dur:.6f}; "
                     f"f0_20%_Hz = {f:.2f}; f0_50%_Hz = {f:.2f}; f0_80%_Hz = {f:.2f}; ")
        word_start = not name
    with open(seg, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if _lib.vtlSegmentSequenceToGesturalScore(seg.encode(), ges.encode()) != 0:
        raise RuntimeError("segment sequence -> gestural score failed")
    total = sum(d for _, d, _, _ in segments)
    audio = np.zeros(int((total + 1.0) * _c["sr"]))
    n = ctypes.c_int()
    if _lib.vtlGesturalScoreToAudio(ges.encode(), b"", b"", _ptr(audio), ctypes.byref(n), 0, 0) != 0:
        raise RuntimeError("gestural score -> audio failed")
    return audio[: n.value]
