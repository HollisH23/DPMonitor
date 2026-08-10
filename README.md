# DPMonitor

**AI-powered physiotherapy coaching that runs entirely on your own device.**

DPMonitor watches a patient perform a rehabilitation exercise through an
ordinary camera, counts their repetitions, measures how far each joint moves,
and tells them when their form is drifting — in real time, with no video ever
leaving the device.

There are two applications in this repository, built on the same analysis
pipeline:

| | Platform | Where processing happens |
|---|---|---|
| **iOS app** | iPhone (iOS 18+) | 100% on-device. No network code at all. |
| **Web app** | Browser + local server | On the user's own machine. No cloud service. |

---

## Why this exists

Physiotherapy works when patients do their exercises correctly at home,
between appointments. In practice most of that happens unsupervised, so:

- Patients don't know whether they're compensating with the wrong muscles.
- Clinicians get no objective record of what actually happened at home —
  only "I think I did about ten."
- Commercial motion-tracking apps typically stream video to a server, which
  is a hard sell for medical use and a privacy risk for the patient.

DPMonitor addresses all three. It gives live corrective feedback, produces an
objective session record (reps, range of motion, movement stability, fatigue),
and does it without a single byte of video leaving the device.

---

## What it measures

For every session the app produces:

- **Repetition count** — from the vertical travel of the hips, using a
  hysteresis detector that ignores postural sway.
- **Range of motion (ROM)** — per-joint minimum, maximum and total swing in
  degrees, for both knees, both hips and both elbows.
- **Movement stability** — how smooth the motion was, from the RMS of joint
  velocity and acceleration. Shaky movement scores lower.
- **Fatigue index** — whether the later repetitions in a set were measurably
  shakier than the earlier ones.
- **Form quality** — a learned score from a graph neural network (see the
  status note below).
- **Framing assistance** — before recording starts, the app checks the patient
  is properly positioned and tells them how to fix it.

---

> ### ⚠️ Project status: research prototype
>
> **The form-quality score is not active in this build.** The repository ships
> no trained model checkpoint, so the network is built from deterministic
> seeded weights. Its output is reproducible but not clinically meaningful, and
> the app detects this and shows the form score as `—` rather than displaying a
> confident, meaningless number.
>
> **Everything geometric works fully**: repetition counting, range of motion,
> stability, fatigue and the centering assistant are all measured directly from
> the skeleton and never touch the model.
>
> To enable form scoring, supply a trained checkpoint:
> `python scripts/export_coreml.py --weights path/to/checkpoint.pt`.
> No code changes are needed.
>
> This is a university research project and is **not a medical device**. It
> must not be used to diagnose, treat or make clinical decisions.

---

## How it works

A camera frame goes in; a set of measurements comes out.

```
   Camera frame (30 per second)
              │
              ▼
   ┌──────────────────────┐
   │  MediaPipe Pose      │  Finds 33 body landmarks
   │  (BlazePose Full)    │  (nose, shoulders, hips, knees, ankles …)
   └──────────┬───────────┘
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
 SCREEN space        WORLD space
 (x, y in 0–1)       (x, y, z in metres)
    │                    │
    │                    ├─► Smoothing      Removes camera jitter
    │                    ├─► Normalisation  Cancels out where the patient
    │                    │                  stands and how far away they are
    │                    ├─► Occlusion fix  Holds a joint's last good position
    │                    │                  when it's briefly hidden
    │                    ▼
    │            ┌───────────────────┐
    │            │  64-frame window  │  ~2 seconds of movement
    │            └─────────┬─────────┘
    │                      ▼
    │            ┌───────────────────┐
    │            │  CTR-GCN model    │  Graph neural network over the
    │            │  (Core ML / ANE)  │  skeleton → form quality
    │            └───────────────────┘
    │
    ├─► Rep counting      Hip travel over time
    ├─► Joint angles      ROM, tremor, fatigue
    └─► Centering check   "Step back", "Move right", "Head cut off"
```

**Why two coordinate spaces?** Screen coordinates tell you *where in the frame*
the patient is, which is what the rep counter and the framing assistant need.
World coordinates are measured in metres and centred on the hips, which is what
the neural network was configured for. Using the wrong one for either job
produces plausible-looking numbers that are quietly wrong.

**The model.** CTR-GCN (Channel-wise Topology Refinement Graph Convolutional
Network) treats the skeleton as a graph — joints are nodes, bones are edges —
and learns how that graph deforms over time. It's a standard architecture for
skeleton-based action recognition. This project uses a custom 33-node graph
matching MediaPipe's landmark layout, rather than the 25-node NTU-RGB+D layout
the original paper used.

---

## Repository layout

```
DPMonitor/
├── ios/               iPhone app (Swift / SwiftUI)      — 27 source files
│   ├── DPMonitor/
│   │   ├── Core/      Pose capture, maths, Core ML inference
│   │   ├── Views/     Camera, skeleton overlay, HUD, results, history
│   │   └── Storage/   Local Core Data session history
│   └── DPMonitorTests/  Parity tests against the Python reference
│
├── backend/           Django server for the web app        — 69 tests
│   ├── api/           REST + WebSocket endpoints
│   ├── analyzer/      The analysis pipeline (Python reference implementation)
│   └── clinical_sessions/  Session storage
│
├── frontend/          Web app (React + Vite)
│
├── ctrgcn/            The CTR-GCN model definition
│
└── scripts/
    ├── export_coreml.py           PyTorch → ONNX → Core ML conversion
    └── gen_reference_fixtures.py  Generates test fixtures for the iOS app
```

`backend/analyzer/` is the **reference implementation**. The iOS app
re-implements the same maths in Swift, and the test suite pins the two together
so they can't drift apart.

---

## Getting started

### Web app

You need Python 3.10+ and Node 18+.

```bash
# 1. Backend
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo         # optional: creates a demo account
python manage.py runserver
```

```bash
# 2. Frontend (in a second terminal)
cd frontend
npm install
npm run dev                        # opens http://localhost:5173
```

After `seed_demo` you can sign in as **`demo`** / **`demopass`**.

### iOS app

You need a Mac with Xcode 16+, and a **physical iPhone** — the simulator has no
camera for the pose tracker to read.

```bash
brew install xcodegen cocoapods

# 1. Download the pose model (~9 MB)
curl -L -o ios/DPMonitor/Models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# 2. Build the Core ML model from the PyTorch source
python scripts/export_coreml.py

# 3. Generate and open the Xcode project
cd ios
xcodegen generate
pod install
open DPMonitor.xcworkspace
```

Then set your development team in Xcode and run on your device.

> **Note:** the Xcode project is generated from `ios/project.yml` and is not
> checked into git. Re-run `xcodegen generate && pod install` whenever files are
> added or removed, or Xcode won't see them.

Full iOS documentation — architecture, threading model, parity testing and the
NPU profiling procedure — is in **[`ios/README.md`](ios/README.md)**.

---

## Testing

```bash
cd backend && python manage.py test        # 69 tests
```

```bash
cd ios && xcodebuild test -workspace DPMonitor.xcworkspace -scheme DPMonitor \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro Max'
```

The interesting part of the test strategy is **cross-language parity**. Six
numeric components exist in both Python and Swift, and a divergence between
them wouldn't crash anything — it would silently produce different clinical
numbers on different platforms. So `scripts/gen_reference_fixtures.py` runs the
Python implementations over hundreds of inputs and writes the results to JSON;
the Swift tests load that file and assert agreement to within `1e-4`.

The framing assistant is verified the same way against its original desktop
implementation, matching exactly across 30,000 randomised inputs.

---

## Privacy

This is the design constraint the whole project is built around.

- **No video is ever recorded, saved, or transmitted.** Camera frames are
  processed in memory and discarded.
- **The iOS app has no networking code.** Not "network access disabled" — the
  target links no networking framework and makes no requests. You can verify
  this by running it in Airplane Mode; every feature works.
- **The web app talks only to a server you run yourself**, on your own machine.
  There is no hosted service.
- **Session history is stored locally** — Core Data on iOS (encrypted at rest),
  SQLite on the desktop. No cloud sync, no analytics, no telemetry.
- The app requests **camera access only**. No photo library, microphone, or
  location.

---

## Known limitations

- **Form scoring is inactive** without a trained checkpoint — see the status
  note above.
- **One person at a time.** The pipeline assumes a single patient in frame.
- **Range-of-motion figures differ slightly between the iOS and web apps.** iOS
  measures angles from true 3D world coordinates; the web app uses screen
  coordinates, whose depth channel is on an arbitrary scale. The iOS numbers
  are the more accurate ones. Compare rep counts across platforms, not absolute
  ROM degrees.
- **iOS 18+ and a recent iPhone.** Developed and profiled on iPhone 16 Pro Max.
- **The rep counter is tuned for vertical movements** such as squats and
  lunges, since it tracks hip height. Exercises without vertical hip travel
  need a different heuristic.

---

## Licence

Released under the **[MIT Licence](LICENSE)** — you are free to use, modify and
distribute this code, including commercially, provided the copyright notice and
licence text are retained. The software is provided as-is, without warranty.

> **Note:** this is a university research project, and confirmation that the
> code may be released under MIT is still pending with the institution. If that
> turns out otherwise the licence may change; the position will be updated here.

Third-party components carry their own licences — see
[Acknowledgements](#acknowledgements). Note in particular that the CTR-GCN
reference implementation this project builds on is separately licensed.

---

## Acknowledgements

Built as a research project at the **University of Sydney**, supervised by
Prof. Jinman Kim.

This work builds on:

- **[CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)** — Chen et al.,
  *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action
  Recognition*, ICCV 2021.
- **[MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)**
  (BlazePose) — Google's on-device pose landmark detection.
