# stimmung

A generative piece for six synthetic voices in the manner of Stockhausen's *Stimmung*,
sung by articulatory vocal-tract models ([VocalTractLab](https://www.vocaltractlab.de/), Birkholz).

Every pitch is a harmonic of B-flat 1 (58.27 Hz). Each singer holds one fundamental
(harmonics 2, 3, 4, 5, 7, 9) and moves only its overtones, by retuning the shape of its
vocal tract so that a formant sits exactly on a chosen harmonic.

The piece is a chain of *models*: a leader presents a pattern alone and the others
"identify" with it one at a time — copying it in unison, in canon, by taking over some of
its notes, or transforming it. How each singer behaves in each model is a small program
sampled from a probabilistic grammar (`src/dsl.py`), so the strategies differ from model
to model and from singer to singer, and evolve over the piece.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Form | `src/score.py` | Intensity arc, sections (vowel colour, protagonist, home tempo), model types, returns of earlier models |
| Strategies | `src/dsl.py` | Typed DSL for singer behaviour; grammar sampling, interpreter, novelty selection |
| Shapes | `src/vtl_library.py`, `src/overtone_strong.py` | Optimise vowel and overtone-tuning tract shapes per singer → `vtl_library.pkl` |
| Articulation | `src/articulation.py` | VTL-style target-approximation trajectories and coarticulated consonants |
| Synthesis | `src/render_vtl.py`, `src/vtl.py` | Render each voice with the VocalTractLab API (segment-cached, parallel) |
| Mix | `src/mixing.py` | Binaural placement on a hexagon, reverb, mastering |
| Video | `src/video.py` | Six animated vocal tracts and an overtone spiral |

Other scripts: `dsl_sketch.py` (four models with freshly sampled programs) and
`sketch_processes.py` (short sketches of speech-to-song, name calling, transformation).

`render.py`, `vocaltract.py`, `vt_core.c`, `build_library.py`, `demo.py` and `library.pkl`
are the first version, a self-written tube model; `render_vtl.py` still borrows a few
helpers from `render.py`.

## Setup

Python 3.12, plus `ffmpeg` on the path.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build the shared library of the **VocalTractLab 2.4 API** from its source
(vocaltractlab.de) and point `VTL_LIB` at it:

```bash
export VTL_LIB=/path/to/libVocalTractLabApi24.so
```

The speaker files in `src/speakers/` (`M01`, `JD2`, `W02`) come from VocalTractLab.

## Usage

Run from `src/`:

```bash
python render_vtl.py --seed 7                   # full piece -> ../out/stimmung_vtl.{wav,mp3}, score text
python render_vtl.py --seed 7 --preview 60      # first minute only
python render_vtl.py --demo                     # each voice alone
python video.py --audio ../out/stimmung_vtl.wav --out ../out/stimmung_vtl.mp4
python dsl_sketch.py --seed 11 --render
```

Rendering a full piece takes a while; rendered voice segments are cached in `cache/segments/`
keyed by the synthesis code, so re-renders after score or mix changes are fast.
The video renderer is memory-hungry; set `SCRATCH` to a disk with room for its temp files.

The shape library `src/vtl_library.pkl` is included, so you don't need to rebuild it.
To rebuild it anyway (slow), run `python vtl_library.py` and then
`python overtone_strong.py --out vtl_library.pkl`.
