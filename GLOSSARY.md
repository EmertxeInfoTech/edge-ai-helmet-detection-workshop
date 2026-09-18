# Workshop Glossary

**Deploying Vision AI at the Edge — Helmet Detection Workshop**

Every term you will hear today, in plain words. Nothing here assumes you have
done machine learning before. Skim it now; come back to it whenever a word on
a slide does not land.

---

## 1. The basics

**Model** — A file holding everything a computer learned during training. Ours
is `best.pt`, about 22.6 MB.

**Training** — Showing a computer thousands of labelled examples until it can
recognise those things itself. Done once, before today, on powerful hardware.

**Inference** — Running a finished model on new data to get an answer. This is
what we do all day. Training makes the model; inference uses it.

**Weights** — The numbers inside the model that training adjusted. "Loading the
weights" just means loading the model.

**Class** — One kind of object the model was taught to find. Ours knows five:
rider, motorcycle, helmet, no_helmet, license_plate.

**Dataset** — The collection of labelled images used for training. A person drew
every single box in it by hand.

**YOLO** — "You Only Look Once", the family of detection models we use. One pass
over an image finds every object at once, which is what makes it quick enough
for video.

**Ultralytics** — The library that gives us YOLO, tracking and model export
behind a single, simple interface.

---

## 2. What the model gives back

**Detection** — One object the model found. Three parts: a class, a box, and a
confidence.

**Bounding box** — Where the object is, as four numbers `[x1, y1, x2, y2]`: the
top-left and bottom-right corners, measured in pixels.

**Confidence** — How sure the model is, from 0 to 1. It is the model's own
score, not a guarantee of being right.

**Confidence threshold (`conf`)** — The cut-off below which we ignore a
detection. We use 0.25. Raise it for fewer but safer detections; lower it to
catch more and admit more noise.

**NMS (Non-Maximum Suppression)** — Tidying up. The model fires several
overlapping boxes for the same object; NMS keeps the strongest and deletes the
rest. It is why you usually see one box per rider and not five.

**Preprocessing** — Preparing a frame before the model sees it: resizing it to
640x640 and rescaling the pixel values.

**Postprocessing** — Turning the model's raw output back into readable boxes.
NMS happens here.

---

## 3. Video and tracking

**Frame** — One still image from a video. Our clip is 500 frames: 25 frames a
second for 20 seconds.

**Tracker** — Software that follows the same object from frame to frame and
gives it a number that stays with it.

**Track ID** — That number. Rider 27 in this frame should still be rider 27 in
the next. It is useful to us, but it is **not** a person's identity and carries
no legal meaning.

**ByteTrack** — The particular tracking method we use. One line of
configuration; the hard part is understanding what it gives you.

**`persist=True`** — Tells the tracker to remember the previous frame. Without
it, IDs restart constantly.

**`stream=True`** — Hands us one frame at a time instead of holding every result
in memory. This matters a great deal on a small device.

---

## 4. Putting objects together

**Association** — Working out which objects belong together: which rider is on
which motorcycle, which head belongs to which rider. The model never does this
for you. This is the heart of the workshop.

**IoU (Intersection over Union)** — How much two boxes overlap: the shared area
divided by the total area they cover. 0 means no overlap, 1 means identical. A
rider sitting on a motorcycle usually scores between 0.3 and 0.6.

**Cost** — A number saying how *bad* a pairing is. Low cost means a good match.
We use `1 - IoU`, so heavy overlap becomes a low cost.

**Cost matrix** — A table with one row per rider and one column per motorcycle,
holding the cost of every possible pairing.

**Hungarian algorithm** — A method that looks at the whole cost table at once
and picks the combination of pairs with the lowest total cost. Better than
grabbing the best-looking pair first and leaving someone with a bad option.

**Rejection threshold (`max_cost`)** — Our veto. The algorithm always returns
*some* answer, even a silly one; this throws away pairings that remain
implausible. We use 0.90 for rider-to-motorcycle.

**`max_distance`** — The same idea for heads: a helmet more than 60 pixels from
a rider's head position is not matched to that rider.

---

## 5. Deciding over time

**Observation** — One frame's opinion about one rider, for example
`("no_helmet", 0.81)`.

**Temporal decision** — Combining many frames into a single answer, instead of
trusting one frame that might be blurred or half hidden.

**Agreement** — What fraction of a rider's frames say the same thing. 8 frames
out of 10 is 0.80.

**The four answers** — Not two:

| Answer | Meaning |
|---|---|
| `pending` | Not enough frames yet. Ask me later. |
| `uncertain` | Enough frames, but they disagree or the model was unsure. |
| `compliant` | Good evidence of a helmet. |
| `violation` | Good evidence of no helmet. |

**Evidence** — The saved image and the reasoning behind a reported violation.
What a human reviewer would actually look at before acting.

**Assisted review** — The honest description of a system like ours. It puts a
case in front of a person; it does not issue a fine by itself.

---

## 6. Getting it onto a device

**Edge / edge AI** — Running the model on a device near the camera, instead of
sending video away to a server.

**Cloud** — Sending frames to a remote server to be processed there.

**Hybrid** — The device decides locally and sends only confirmed cases to a
server for human review. This is what most real systems actually do.

**Export** — Converting the model into a different file format so other software
can run it.

**ONNX** — A common, portable model format that many runtimes can load. It frees
the model from Python and PyTorch. It makes the model portable, **not
automatically faster**.

**Runtime** — The software that actually executes a model. ONNX Runtime is one.
TensorRT is another.

**TensorRT** — NVIDIA's runtime. It compiles a model for one specific chip, so
it is fast but tied to that device and version.

**FP32 / FP16 / INT8** — How precisely numbers are stored: 32-bit, 16-bit or
8-bit. Fewer bits means smaller and often faster, with some risk to accuracy.

**Quantization** — Converting a model to lower precision, such as FP16 or INT8.

**Calibration** — Feeding a tool sample images so it can quantize to INT8
sensibly. Skipping this is how accuracy quietly collapses.

---

## 7. Measuring speed honestly

**Latency** — Time to process one frame, in milliseconds. Lower is better.

**Throughput / FPS** — Frames handled per second. Higher is better.

**Median** — The middle value across many runs. More honest than the average,
and far more honest than the best run.

**p95** — 95 runs out of 100 were faster than this. It shows the slow cases,
which are what drop frames in a live system.

**Warm-up** — The first few runs, always slower because of setup and memory
allocation. Thrown away before measuring.

**Batch size** — How many images are processed together. A live camera has a
batch size of 1, so that is what we measure.

**TOPS** — Trillions of operations per second: a chip's *peak theoretical*
capability, usually quoted as INT8 with sparsity. A ceiling, not a promise.

**GFLOPs** — Billions of operations needed for one pass of a model. Ours is
about 28.4 per frame.

---

## 8. Hardware

**CPU** — General-purpose processor. Runs everything, slowly for AI work.

**GPU** — Parallel processor, far faster for models. Colab's T4 is one.

**Jetson Orin Nano** — A small NVIDIA computer built for edge AI: roughly 40
TOPS, 1024 GPU cores, 8 GB shared memory, 7 to 25 watts. Figures vary by
variant and power mode.

**Colab** — Google's free notebook service that runs in a browser. Our shared
lab for the day.

---

## 9. What goes wrong

**Occlusion** — Something blocking the view: a bus, a pole, another rider.

**Domain shift** — A model trained in one setting performing worse somewhere
else: a different city, camera or light.

**False positive** — Reporting a violation that did not happen. In enforcement,
this is the expensive mistake — a real person is wrongly accused.

**False negative** — Missing a real violation.

**OCR** — Reading text from an image. We crop the number plate; reading the
characters on it is a separate job, outside this workshop.

---

## The numbers we use today

| Setting | Value | What it controls |
|---|---:|---|
| `conf` | 0.25 | minimum confidence for a detection to count |
| `imgsz` | 640 | every frame is resized to 640x640 |
| `max_cost` | 0.90 | rejects an implausible rider-to-motorcycle pair |
| `max_distance` | 60 | rejects a head too far from a rider, in pixels |
| `MIN_OBSERVATIONS` | 3 | frames needed before any verdict at all |
| `MIN_AGREEMENT` | 0.67 | fraction of frames that must agree |
| `MIN_CONFIDENCE` | 0.70 | average confidence needed to accept a verdict |
| `FAST_PREVIEW` | False | True caps the pipeline at 150 frames |
| `EDGE_FRAME_LIMIT` | 100 | frames pushed through ONNX Runtime |

**Class IDs in the model:** `0 rider` · `1 motorcycle` · `2 helmet` ·
`3 no_helmet` · `4 license_plate`

Every one of these numbers was chosen by a person, not discovered by the model.
Choosing them, and being able to defend the choice, is the engineering.

---

*Workshop repository:*
<https://github.com/EmertxeInfoTech/edge-ai-helmet-detection-workshop>
