# Deploying Vision AI at the Edge — Helmet Detection Workshop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmertxeInfoTech/edge-ai-helmet-detection-workshop/blob/main/workshop/Helmet_Detection_Participant_Colab.ipynb)

A hands-on workshop that builds a helmet-violation detection system from
dashcam video, and then prepares it to run on a small edge device.

Website: **https://emertxeinfotech.github.io/edge-ai-helmet-detection-workshop/**

A detection model finds motorcycles, riders, heads and number plates in every
frame. A tracker gives each object an ID that stays with it. A matching stage
works out which rider is on which motorcycle, and which head belongs to which
rider. Finally, a decision stage collects evidence across many frames and
reports a violation only when that evidence is strong and consistent.

The interesting part is not the model. A detector reports **objects**; it
never reports **relationships**. Everything that turns "a helmet is present"
into "that rider has no helmet" is work you write yourself, and that is what
this workshop is about.

## Start here

Click the badge above, or open
`workshop/Helmet_Detection_Participant_Colab.ipynb` in Google Colab and run it
from top to bottom. Nothing has to be installed on your machine.

In Colab, first select **Runtime → Change runtime type → T4 GPU**. The
notebook works on CPU too, but a GPU is roughly ten times faster: on a Colab
CPU session the model runs at about 3 frames per second.

The setup cell downloads about 53 MB: the trained model (22.6 MB) and a
20-second sample clip (30.8 MB).

| Notebook | What it is for |
|---|---|
| `workshop/Helmet_Detection_Participant_Colab.ipynb` | The workshop itself. Start here. |
| `workshop/Helmet_Detection_Participant_Colab_BACKUP_executed.ipynb` | The same notebook with every output already filled in. Use it if you lose your Colab session, or to read later. |
| `workshop/full_pipeline_executed.ipynb` | A longer walkthrough of all eight modules. Good reading after the workshop. |

New to machine learning? **[GLOSSARY.md](GLOSSARY.md)** explains every term used
in the workshop, in plain words.

### What you should see

On the supplied clip, with the thresholds as shipped:

- 500 frames processed (20 seconds of video)
- about 16 riders with enough head evidence to judge
- **one** confirmed violation

Only one violation is the correct answer here. Several other riders were
mostly detected without a helmet and were still not reported, because the
evidence did not clear the thresholds. The notebook prints each of those
riders and the reason. That gap between "probably" and "reported" is the most
important idea in the workshop.

## How the decision works

A rider is reported only when all three conditions hold:

| Setting | Default | Meaning |
|---|---:|---|
| `MIN_OBSERVATIONS` | 3 | seen in at least this many frames |
| `MIN_AGREEMENT` | 0.67 | this fraction of those frames must agree |
| `MIN_CONFIDENCE` | 0.70 | average model confidence must reach this |

This gives four possible answers, not two: `pending` (not enough frames yet),
`uncertain` (evidence is weak or mixed), `compliant`, and `violation`. A
system that is forced to choose between only *helmet* and *no helmet* will be
confidently wrong quite often.

## Repository layout

| Path | Contents |
|---|---|
| `workshop/` | The Colab notebooks used in the session |
| `final/` | Current pipeline: `src/` modules, `main.ipynb`, model and sample clip |
| `notebooks/` | One teaching notebook per stage of the pipeline |
| `project/pipeline/` | An earlier version of the same modules, kept for the stage notebooks |
| `project/webapp/` | FastAPI and SQLite dashboard for reviewing reported violations |
| `*.qmd`, `_quarto.yml` | Sources for the website |

Two things to know before editing:

- `final/src/*.py` is **generated** by the `%%writefile` cells inside
  `final/main.ipynb`. Change the notebook cell, not the `.py` file, or your
  edit disappears the next time somebody runs the notebook.
- `project/pipeline/` and `final/src/` are separate copies that have drifted
  apart. New work belongs in `final/`.

## The eight modules

| Module | What it does |
|---|---|
| `config.py` | Every path, class name and threshold, in one place |
| `detection.py` | Turns raw model output into per-class lookup tables |
| `geometry.py` | Reduces boxes to comparable points, and measures overlap |
| `association.py` | Matches riders to motorcycles, heads to riders, plates to motorcycles |
| `rm_instance.py` | Stores what has been learned about each rider so far |
| `temporal.py` | Turns many frames into a single answer |
| `visualization.py` | Draws the current state so a human can check it |
| `main.py` | Runs every stage, in order, over a whole video |

Class IDs in `models/best.pt`: `0 rider, 1 motorcycle, 2 helmet, 3 no_helmet,
4 license_plate`.

## Running it on your own machine

```bash
pip install "ultralytics>=8.3,<9" "onnx>=1.17,<2" "onnxruntime>=1.20,<2" \
            "scipy>=1.13,<2" "lap>=0.5.12" "onnxslim>=0.1.82"

cd final
python -m src.main      # writes a report and violation images into output/
```

Export the model for an edge device:

```python
from ultralytics import YOLO
YOLO("final/models/best.pt").export(
    format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17
)
```

Exporting to ONNX makes the model portable — it no longer needs Python or
PyTorch. It does not automatically make it faster. On a plain CPU, ONNX is
often a little *slower* than PyTorch at batch size one. Real speed-ups come
from the next step: lower precision (FP16 or INT8), or a runtime built for one
specific chip, such as TensorRT on a Jetson.

TensorRT `.engine` files only work on the device and TensorRT version that
built them, so they are generated on the target device and not committed here.

## Third-party licensing

This project uses [Ultralytics](https://github.com/ultralytics/ultralytics)
YOLO, which is distributed under **AGPL-3.0**, and the trained weights in this
repository carry that licence. Read the Ultralytics licensing terms before any
commercial use, redistribution, or deployment as a hosted service.

© Emertxe Information Technologies
