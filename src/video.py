"""Video of the VocalTractLab rendering: six singing vocal tracts in a circle, and an
overtone spiral in the centre showing which harmonics of B-flat are sounding.

usage: python video.py [--seed 7] [--audio ../out/stimmung_vtl.wav] [--out ../out/stimmung_vtl.mp4]
"""
import argparse
import functools
import os
import pickle
import re
import subprocess
import tempfile
import textwrap
import time
import wave
from multiprocessing import Pool, get_context

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_vtl as rv
import vtl
from score import ScoreGen
from singers import F0_BASE, SINGERS

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 25
W, H = 1920, 1080
SS = 2                                   # supersampling for antialiasing
BG = (11, 13, 19)
COLORS = [(232, 168, 76), (214, 104, 84), (204, 96, 156), (132, 118, 220), (84, 164, 222), (96, 202, 168)]
N_HARM = 48
FONT_DIR = os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf")
SCRATCH = os.environ.get("SCRATCH", tempfile.gettempdir())


@functools.lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                              size * SS)


# ---------------------------------------------------------------------------
# data

def get_score(score_path, seed):
    """The score the audio was rendered from, if saved; else regenerate it from the seed."""
    if score_path and os.path.exists(score_path):
        return rv.load_score(score_path)
    return ScoreGen(seed).build()


def tract_outlines(args):
    """Midsagittal outlines of one singer's tract at every video frame (VTL SVG export)."""
    si, score_path, seed, nframes = args
    s = SINGERS[si]
    rv.library()
    sc = get_score(score_path, seed)
    vtl.init(s["speaker"])
    tract, glot, gain = rv.tracks_for_singer(si, sc.events[si], sc.total)
    step = int(rv.FR / FPS)
    svg = os.path.join(SCRATCH, f"tract_{si}.svg")
    num = re.compile(r'points="([^"]+)"')
    out, last, polys = [], None, None
    for i in range(nframes):
        j = min(i * step + 1, len(tract) - 1)
        if last is None or np.max(np.abs(tract[j] - last)) > 1e-3:   # still tracts reuse the outline
            vtl._lib.vtlExportTractSvg(vtl._ptr(np.ascontiguousarray(tract[j])), svg.encode())
            with open(svg) as f:
                polys = [np.array(m.split(), np.float32).reshape(-1, 2) for m in num.findall(f.read())]
            last = tract[j].copy()
        out.append(polys)
    G = vtl.constants()["glottis_info"]["names"].index("pressure")
    loud = np.clip(glot[:: step, G][:nframes] / 8000.0, 0, 1.2) * np.clip(gain[:: step][:nframes], 0.3, 2.0)
    return si, out, loud


def harmonic_levels(audio_path, nframes):
    """Level (0..1) of harmonics 1..N_HARM of B-flat 1 in the mix, per video frame."""
    with wave.open(audio_path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).mean(1) / 32768.0
    win = int(0.16 * sr)
    hann = np.hanning(win)
    nfft = 1 << int(np.ceil(np.log2(win * 4)))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    bins = [np.argmin(np.abs(f - k * F0_BASE)) for k in range(1, N_HARM + 1)]
    lev = np.zeros((nframes, N_HARM))
    for i in range(nframes):
        c = int(i / FPS * sr)
        seg = x[max(0, c - win // 2): c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        X = np.abs(np.fft.rfft(seg * hann, nfft))
        lev[i] = [X[b - 1:b + 2].max() for b in bins]
    db = 20 * np.log10(lev + 1e-9)
    ref = np.percentile(db[db > -150], 99.5)
    return np.clip((db - (ref - 50)) / 50, 0, 1)


def captions(sc, nframes):
    """Per frame: current model line, magic name being sung, and each singer's syllable."""
    model_txt, name_txt, leader = [""] * nframes, [""] * nframes, [-1] * nframes
    for m in sc.models:
        lab = m.label
        for i in range(int(m.t0 * FPS), min(nframes, int(m.t1 * FPS))):
            nxt = f"  ·  then {m.exit}" if getattr(m, "exit", "") else ""
            sec = f"{m.section}  ·  " if getattr(m, "section", "") else ""
            model_txt[i] = f"{sec}model {m.idx + 1} of {len(sc.models)}  ·  {lab}  ·  led by {SINGERS[m.leader]['name']}{nxt}"
            leader[i] = m.leader
            if m.kind == "name":
                name_txt[i] = m.name
    syl = [[""] * nframes for _ in SINGERS]
    for si, evs in enumerate(sc.events):
        for e in evs:
            shape = rv.resolve(si, e.shape) if e.shape[0] == "ot" else e.shape
            v = {"oe": "ø", "schwa": "ə", "E": "ɛ", "O": "ɔ"}.get(shape[-1] if shape[0] != "ot" else shape[2],
                                                               shape[-1] if shape[0] != "ot" else shape[2])
            txt = "/m/  hum" if shape[0] == "hum" else f"/{e.onset.replace('sh', 'ʃ') + v}/"
            if e.whisper:
                txt += "  whispered"
            if shape[0] == "ot":
                txt += f"   overtone {shape[1]}"
            for i in range(int(e.t * FPS), min(nframes, int((e.t + e.dur) * FPS))):
                syl[si][i] = txt
    # each singer's strategy: the phrase of its program being performed
    strat = [[""] * nframes for _ in SINGERS]
    for m in sc.models:
        for si, spans in getattr(m, "phases", {}).items():
            for t0, t1, txt in spans:
                lines = textwrap.wrap(txt, 46)
                txt = "\n".join(lines[:2]) + (" …" if len(lines) > 2 else "")
                for i in range(int(t0 * FPS), min(nframes, int(t1 * FPS))):
                    strat[si][i] = txt
    return model_txt, name_txt, leader, syl, strat


# ---------------------------------------------------------------------------
# drawing

CX, CY = W / 2, H / 2 + 10
RX, RY = 540, 340
ANGLES = np.deg2rad([-90 + 60 * k for k in range(6)])      # bass at the top, clockwise
# tract coordinates (VTL svg units) -> box
TX0, TX1, TY0, TY1 = -140.0, 240.0, -70.0, 290.0
TSCALE = 0.5


def place(si):
    a = ANGLES[si]
    return CX + RX * np.cos(a), CY + RY * np.sin(a)


def mix(c, k, bg=BG):
    return tuple(int(bg[j] + (c[j] - bg[j]) * k) for j in range(3))


def draw_frame(i, data, first=0):
    """Frame i of data; first is the absolute index of data's frame 0 (for the clock)."""
    tracts, louds, harm, model_txt, name_txt, leader, syl, strat = data
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    S = SS

    # overtone spiral: one turn per octave, harmonic k at angle log2(k)
    for k in range(1, N_HARM + 1):
        lk = np.log2(k)
        r = 30 + 28 * lk
        a = 2 * np.pi * lk - np.pi / 2
        x, y = CX + r * np.cos(a), CY + r * np.sin(a)
        v = harm[i, k - 1]
        sung = [si for si, s in enumerate(SINGERS) if s["h"] == k]
        col = COLORS[sung[0]] if sung else (235, 232, 222)
        rad = 2.5 + 11 * v
        d.ellipse([(x - rad) * S, (y - rad) * S, (x + rad) * S, (y + rad) * S], fill=mix(col, 0.15 + 0.85 * v))
        if sung:
            d.ellipse([(x - 15) * S, (y - 15) * S, (x + 15) * S, (y + 15) * S], outline=mix(col, 0.6), width=2 * S)
        if k <= 24 or v > 0.55:
            d.text(((x + rad + 4) * S, (y - 8) * S), str(k), fill=mix((200, 200, 200), 0.25 + 0.6 * v),
                   font=font(12))

    # the six vocal tracts, facing the centre
    for si, s in enumerate(SINGERS):
        px, py = place(si)
        lv = float(np.clip(louds[si][i], 0, 1))
        col = COLORS[si]
        face_left = px > CX + 1
        if leader[i] == si:
            d.ellipse([(px - 125) * S, (py - 112) * S, (px + 125) * S, (py + 112) * S],
                      outline=mix(col, 0.55), width=3 * S)
        for poly in tracts[si][i]:
            xs = (poly[:, 0] - (TX0 + TX1) / 2) * TSCALE
            ys = (poly[:, 1] - (TY0 + TY1) / 2) * TSCALE
            if face_left:
                xs = -xs
            pts = [((px + a) * S, (py + b) * S) for a, b in zip(xs, ys)]
            if len(pts) > 1:
                d.line(pts, fill=mix(col, 0.25 + 0.75 * lv), width=int((2 + 2 * lv) * S), joint="curve")
        d.text((px * S, (py + 104) * S), s["name"], fill=mix(col, 0.85), font=font(20, True), anchor="mm")
        d.text((px * S, (py + 128) * S), f"harmonic {s['h']}  ·  {s['f0']:.0f} Hz",
               fill=mix((190, 190, 190), 0.6), font=font(14), anchor="mm")
        if strat[si][i]:
            d.multiline_text((px * S, (py + 146) * S), strat[si][i], fill=mix(col, 0.55),
                             font=font(13), anchor="ma", align="center", spacing=3 * S)
        if syl[si][i] and lv > 0.05:
            d.text((px * S, (py - 104) * S), syl[si][i], fill=mix((240, 238, 230), 0.4 + 0.6 * lv),
                   font=font(18), anchor="mm")

    # titles and captions
    d.text((40 * S, 34 * S), "STIMMUNG  —  six simulated vocal tracts on the overtones of B♭",
           fill=(220, 218, 210), font=font(22, True))
    d.text((40 * S, 66 * S), "VocalTractLab articulatory synthesis  ·  centre: harmonics of B♭ (58.3 Hz), "
           "one turn per octave", fill=(130, 132, 140), font=font(15))
    t = (first + i) / FPS
    d.text(((W - 40) * S, 34 * S), f"{int(t // 60)}:{int(t % 60):02d}", fill=(160, 160, 166),
           font=font(22), anchor="ra")
    if model_txt[i]:
        d.text((40 * S, (H - 40) * S), model_txt[i], fill=(200, 198, 190), font=font(19), anchor="ls")
    if name_txt[i]:
        d.text((40 * S, (H - 76) * S), name_txt[i], fill=(240, 220, 170), font=font(34, True), anchor="ls")
    return img.reduce(SS)     # box-filter downsampling; LANCZOS took ~80% of the frame time


def slice_data(data, lo, hi):
    tracts, louds, harm, model_txt, name_txt, leader, syl, strat = data
    return ([t[lo:hi] for t in tracts], [l[lo:hi] for l in louds], harm[lo:hi], model_txt[lo:hi],
            name_txt[lo:hi], leader[lo:hi], [s[lo:hi] for s in syl], [s[lo:hi] for s in strat])


def render_chunk(args):
    first, n, chunk_path, seg = args
    with open(chunk_path, "rb") as f:
        data = pickle.load(f)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "20", "-threads", "3", "-pix_fmt", "yuv420p", seg], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(draw_frame(i, data, first).tobytes())
    p.stdin.close()
    p.wait()
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audio", default=os.path.join(HERE, "..", "out", "stimmung_vtl.wav"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "stimmung_vtl.mp4"))
    ap.add_argument("--frames", type=int, default=0, help="only the first N frames (test)")
    ap.add_argument("--reuse", action="store_true", help="reuse the saved analysis")
    a = ap.parse_args()

    t0 = time.time()
    data_path = os.path.join(SCRATCH, "video_data.pkl")
    if not (a.reuse and os.path.exists(data_path)):
        analyse(a, data_path)

    nproc = max(1, min(6, os.cpu_count() - 2))
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    nframes = len(data[2])
    edges = np.linspace(0, nframes, nproc + 1).astype(int)
    segs = []
    for k in range(nproc):   # each worker loads only its own frames
        chunk_path = os.path.join(SCRATCH, f"chunk{k:02d}.pkl")
        with open(chunk_path, "wb") as f:
            pickle.dump(slice_data(data, edges[k], edges[k + 1]), f)
        segs.append((edges[k], edges[k + 1] - edges[k], chunk_path, os.path.join(SCRATCH, f"seg{k:02d}.mp4")))
    del data
    # spawn, not fork: forked workers deadlocked on locks inherited from the parent
    with get_context("spawn").Pool(nproc) as pool:
        files = pool.map(render_chunk, segs)
    print(f"frames rendered {time.time() - t0:.0f}s", flush=True)
    lst = os.path.join(SCRATCH, "segs.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in files)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", a.audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", a.out], check=True)
    for _, _, chunk_path, seg in segs:
        os.remove(chunk_path)
        os.remove(seg)
    print("wrote", a.out, f"({time.time() - t0:.0f}s)")


def analyse(a, data_path):
    t0 = time.time()
    rv.library()
    score_path = os.path.splitext(a.audio)[0] + "_score.pkl"
    sc = get_score(score_path, a.seed)
    print("score:", score_path if os.path.exists(score_path) else f"regenerated from seed {a.seed}", flush=True)
    with wave.open(a.audio) as w:
        dur = w.getnframes() / w.getframerate()
    nframes = a.frames or int(dur * FPS)

    with Pool(len(SINGERS)) as pool:
        res = sorted(pool.map(tract_outlines, [(si, score_path, a.seed, nframes) for si in range(len(SINGERS))]))
    tracts = [r[1] for r in res]
    louds = [np.pad(r[2], (0, max(0, nframes - len(r[2]))))[:nframes] for r in res]
    print(f"tract outlines {time.time() - t0:.0f}s", flush=True)
    harm = harmonic_levels(a.audio, nframes)
    data = (tracts, louds, harm) + captions(sc, nframes)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    print(f"analysis done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
