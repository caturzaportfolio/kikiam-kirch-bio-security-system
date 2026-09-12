# Kikiam Bio Security System

> Python-first biometric security research and benchmarking laboratory.

A local computer-vision system for experimenting with camera-based face detection, biometric representation, identity matching, authentication decision policies, and measurable security/performance benchmarks.

**Project status:** 🧪 Research / Benchmark — not production access control

## Windows Local Test Deployment — BIO-001

BIO-001 is deployable on a Windows laptop for local camera and face-detection testing. This deployment proves only:

```text
Camera -> Frame Capture -> Face Detection -> Local Visualization -> Timing Metrics
```

It does **not** perform identity recognition, enrollment, authentication, liveness/PAD, cloud processing, or physical access control.

### Requirements

- Windows 10 or Windows 11
- Git for Windows
- Python 3.12
- A working webcam or built-in laptop camera

Do **not** deploy from `C:\Windows\System32`. Use a normal user-owned working directory such as `%USERPROFILE%\Projects`.

### Install and verify Git first

Check whether Git is already available:

```powershell
git --version
```

If PowerShell reports that `git` is not recognized, install Git with Windows Package Manager:

```powershell
winget install --id Git.Git -e --source winget
```

After installation, either reopen PowerShell or refresh the current process PATH:

```powershell
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
git --version
```

Do not continue until `git --version` succeeds.

### PowerShell deployment

Create a normal working directory, then clone and run the BIO-001 branch:

```powershell
New-Item -ItemType Directory -Force "$HOME\Projects" | Out-Null
Set-Location "$HOME\Projects"

git clone https://github.com/caturzaportfolio/kikiam-kirch-bio-security-system.git
Set-Location .\kikiam-kirch-bio-security-system
git checkout KIKIAM-BIO-001-CAMERA-BASELINE

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\bio-security.exe detect --camera 0
```

The direct `.venv\Scripts\...` commands intentionally do not require PowerShell virtual-environment activation, so they also work on machines where script activation is restricted.

To request a specific capture size:

```powershell
.\.venv\Scripts\bio-security.exe detect --camera 0 --width 1280 --height 720
```

Press `Q` or `ESC` to stop the camera session. `Ctrl+C` also performs cleanup. Frames are not saved by default.

### Cleanup after an accidental System32 attempt

If an earlier failed deployment created `C:\Windows\System32\.venv`, remove only that accidental virtual environment before retrying:

```powershell
if (Test-Path "C:\Windows\System32\.venv") {
    Remove-Item -Recurse -Force "C:\Windows\System32\.venv"
}
```

### Optional verification before camera testing

```powershell
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\pytest.exe
```

The repository CI already runs these checks. Hardware acceptance still requires execution on the actual Windows test laptop so the camera, real resolution, FPS, and observed detection/total latency can be recorded.

### Expected behavior

When the test starts, the application should open the selected camera, display the local video stream, draw face-detection boxes when faces are detected, and show timing information. If the camera cannot be opened, a frame cannot be read, or the detector cannot initialize, the program exits with an explicit error instead of silently continuing.

---

## 1. Purpose

The purpose of this project is to build and benchmark a laptop-based biometric pipeline that can:

1. Capture frames from a camera.
2. Detect faces.
3. Generate biometric representations (embeddings).
4. Compare observations against enrolled identities.
5. Produce explicit authentication decisions.
6. Measure accuracy, false accepts/rejects, latency, resource usage, reliability, and presentation-attack behavior.

The project follows the **Kikiam-Harness** engineering model: evidence before assumptions, explicit scope, measurable verification, security-by-design, and human authority over consequential security decisions.

---

## 2. Important Security Boundary

This repository is initially a **research and engineering benchmark**.

It is **not** currently intended to:

- control a real door lock;
- grant access to a real facility;
- store biometric data in the cloud;
- expose biometric services to the public internet;
- claim that face recognition alone provides secure authentication.

Facial biometric data and embeddings are sensitive. Only authorized and consented test data should be used.

**Do not commit raw biometric images, embeddings, camera captures, credentials, or other sensitive data to Git.**

Any future transition from benchmark software to physical access control requires a new threat model, security review, liveness/presentation-attack strategy, recovery policy, and explicit human approval.

---

## 3. Architecture

The system is designed as a **local modular monolith**. The complete biometric pipeline runs on the machine connected to the camera.

```text
                    ┌──────────────────────┐
                    │   Local Operator UI  │
                    │       CLI first      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Application Layer     │
                    │ Pipeline Orchestrator │
                    └──────────┬───────────┘
                               │
Camera → Capture → Detection → Preprocessing → Embedding
                                                    │
                                                    ▼
                                                Matching
                                                    │
                                                    ▼
                                            Decision Engine
                                              │          │
                                              ▼          ▼
                                            Audit    Benchmark
                                              │          │
                                              └────┬─────┘
                                                   ▼
                                            Local Results/DB
```

### Core pipeline

```text
Camera
  ↓
Frame Capture
  ↓
Face Detection
  ↓
Face Count / Quality Policy
  ↓
Preprocess + Align
  ↓
Embedding Model
  ↓
Template Retrieval
  ↓
Similarity Calculation
  ↓
Decision Policy
  ↓
Decision Event
  ├── Local Result
  ├── Audit Record
  └── Benchmark Metrics
```

### Critical design principle

> **The model produces measurements. The decision engine produces security decisions.**

A model similarity/confidence score must not automatically become an access decision. Thresholds, quality checks, multiple-face behavior, unknown-person behavior, and error handling belong to explicit policy logic.

---

## 4. Architecture Layers

| Layer | Responsibility |
|---|---|
| `presentation` | Local operator interaction |
| `application` | Coordinates recognition and benchmark workflows |
| `domain` | Decisions, thresholds, identities, and result contracts |
| `ports` | Stable interfaces between domain and infrastructure |
| `infrastructure` | Camera, computer vision, models, storage, and metrics |
| `benchmark` | Experiments, threshold sweeps, and evaluation metrics |
| `audit` | Security-relevant event recording |

The domain and decision logic should remain testable without a physical camera or neural-network model.

---

## 5. Development Stack

| Area | Technology / Direction |
|---|---|
| Language | **Python 3.12** — verified BIO-001 runtime |
| Environment | Python `venv` |
| Project configuration | `pyproject.toml` |
| Camera / computer vision | **OpenCV** |
| Numerical computing | **NumPy** |
| Inference boundary | **ONNX Runtime** — future recognition work |
| Face model | Model-agnostic; recognition model not yet selected |
| Benchmark analysis | Pandas where justified |
| Visualization | Local OpenCV display for BIO-001; Matplotlib where justified later |
| Configuration / validation | Pydantic where justified |
| Testing | pytest |
| Lint / formatting | Ruff |
| Type checking | mypy |
| Logging | Python `logging` with structured event fields |
| Local metadata | SQLite when structured persistence becomes necessary |
| API | FastAPI — future, not required for the first slice |
| Web UI | React + TypeScript — future presentation layer only |

Dependencies must be selected based on hardware compatibility, maintenance, performance, security, and licensing rather than popularity alone.

---

## 6. Face Model Strategy

The embedding/detection model is intentionally **not hard-coded as an architectural dependency**.

The intended boundary is:

```text
Model Adapter
     ↓
DetectionResult / EmbeddingResult
     ↓
Domain Contracts
```

This allows different models to be benchmarked without rewriting the matching and decision layers.

### InsightFace

InsightFace is a candidate for research benchmarking because of its face-analysis capabilities and ONNX-based workflows. However, model licensing must be reviewed separately from the code/package license before any use beyond permitted research contexts.

The exact model, version, license, and benchmark configuration must be recorded with every meaningful benchmark run.

---

## 7. Recognition Concepts

This project explicitly distinguishes four concepts:

- **Detection:** Is a face present?
- **Identification:** Which enrolled identity is most similar?
- **Verification:** Does the observation match the claimed identity?
- **Authentication:** Should policy grant access under the current conditions?

These are related but are not interchangeable.

### Decision flow

```text
Frame
  ↓
Face detected?
  ├── No → NO_FACE
  └── Yes
       ↓
Embedding generated?
  ├── No → PROCESSING_ERROR
  └── Yes
       ↓
Compare against enrolled templates
       ↓
Best similarity score
       ↓
Threshold + quality checks + policy
       ├── ACCEPTED
       ├── REJECTED
       └── UNKNOWN
```

---

## 8. Enrollment

Enrollment is a controlled workflow rather than simply placing an image in a directory.

A test identity should have multiple observations covering controlled variation such as:

- frontal view;
- small pose changes;
- different distances;
- normal indoor lighting variation;
- optional normal appearance variation.

Enrollment observations must be separated from final evaluation observations.

Conceptual template metadata:

```text
subject_id
embedding_version
model_version
enrollment_timestamp
source_dataset
quality_metadata
```

A stable internal `subject_id` should be used rather than making a person's name the primary biometric key.

---

## 9. Benchmark Design

The benchmark is designed to answer measurable questions rather than merely demonstrate that the camera works.

### Accuracy

- How often are genuine users correctly accepted?
- How often are unknown users correctly rejected?
- How does performance change with lighting, pose, distance, and camera quality?

### Security

- Can a printed photograph fool the system?
- Can a phone/tablet replay fool the system?
- Can replayed video fool the system?
- What happens with multiple visible faces?
- What happens under partial occlusion?

### Performance

- Detection latency
- Embedding latency
- Matching latency
- End-to-end latency
- FPS
- CPU usage
- RAM usage
- Startup time
- Enrollment time

### Reliability

- Camera disconnect/reconnect
- No-face conditions
- Multiple-face conditions
- Poor lighting
- Corrupt enrollment data
- Model loading failure
- Storage failure

---

## 10. Core Metrics

At minimum, benchmark runs should measure:

| Metric | Meaning |
|---|---|
| TAR | Genuine users correctly accepted |
| FRR | Genuine users incorrectly rejected |
| FAR | Unknown users incorrectly accepted |
| TRR | Unknown users correctly rejected |
| Detection latency | Time spent locating a face |
| Embedding latency | Time spent generating the biometric representation |
| Matching latency | Time spent comparing templates |
| End-to-end latency | Total time from usable frame to decision |
| FPS | Frames processed per second |
| CPU | Processor utilization |
| RAM | Memory utilization |

A benchmark must never report a single “accuracy” number without describing its dataset, split, threshold, conditions, and evaluation protocol.

---

## 11. Threshold Benchmarking

The matching threshold is treated as a **security parameter** rather than an arbitrary constant.

The benchmark should sweep multiple thresholds and record:

- FAR
- FRR
- TAR
- TRR

The objective is not simply to maximize recognition accuracy. For an access-control use case, false acceptance can be substantially more consequential than false rejection.

The selected threshold must therefore be supported by benchmark evidence and documented as an engineering decision.

---

## 12. Dataset Structure

The planned benchmark dataset is divided into separate partitions:

```text
data/
├── enrollment/
├── genuine/
├── impostor/
└── spoof/
```

- `enrollment/` — reference observations used to create templates.
- `genuine/` — later observations from enrolled subjects.
- `impostor/` — observations from subjects who are not the claimed identity.
- `spoof/` — presentation-attack material where ethically and legally appropriate.

Never evaluate against the exact observations used to create an enrollment template.

Dataset metadata should record dataset version, subject/sample counts, camera, resolution, lighting, pose, capture date, model version, and benchmark configuration.

---

## 13. Presentation Attack Testing

Basic face recognition is **not liveness detection**.

The benchmark should explicitly evaluate presentation attacks such as:

1. Printed photograph.
2. Phone/tablet image replay.
3. Video replay.
4. Different lighting conditions.
5. Partial face occlusion.

Dedicated liveness / presentation-attack detection may be researched later if benchmark evidence justifies it.

---

## 14. Current Repository Structure

```text
kikiam-kirch-bio-security-system/
├── .github/workflows/ci.yml
├── docs/BIO-001.md
├── src/bio_security/
│   ├── infrastructure/
│   │   └── opencv_runtime.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── application.py
│   ├── cli.py
│   ├── domain.py
│   └── ports.py
├── tests/
│   ├── test_detection_session.py
│   └── test_metrics.py
├── pyproject.toml
├── README.md
└── .gitignore
```

The current structure is intentionally narrow. Embedding, matching, enrollment, storage, audit, API, and web layers are not created until their behavior is earned by later milestones.

---

## 15. Development Roadmap

### Phase 0 — Discover

- Inspect the local repository.
- Confirm repository state.
- Identify OS, Python version, CPU, RAM, GPU, and camera.
- Verify model/library hardware compatibility.

### Phase 1 — Define

- Establish objective and non-goals.
- Define threat model.
- Define dataset protocol.
- Define metrics.
- Define acceptance criteria.

### Phase 2 — Camera Baseline

```text
Camera
  ↓
Frame Capture
  ↓
Face Detection
  ↓
Local Visualization
  ↓
Timing Metrics
```

### Phase 3 — Enrollment

- Capture controlled test observations.
- Generate templates.
- Persist templates locally.
- Record model/template versions.

### Phase 4 — Recognition

- Generate embeddings from observations.
- Compare against enrolled templates.
- Apply explicit decision policy.
- Record structured decision events.

### Phase 5 — Benchmark Harness

- Automate repeatable runs.
- Sweep thresholds.
- Calculate FAR/FRR/TAR/TRR.
- Measure latency and resources.
- Produce machine-readable result artifacts.

### Phase 6 — Security Evaluation

- Unknown-person tests.
- Presentation-attack tests.
- Multiple-face tests.
- Camera failure tests.
- Corrupt-data tests.
- Local data-exposure review.

### Phase 7 — Verification Gate

Do not move toward physical access control until benchmark evidence, security limitations, failure behavior, sensitive-data handling, and the broader security boundary have been reviewed and explicitly approved.

---

## 16. Testing Strategy

### Unit tests

BIO-001 currently contains deterministic tests for its pure detection-session and timing behavior. Later milestones add tests for matching, threshold policy, FAR/FRR, enrollment, and other recognition behavior when those features actually exist.

### Integration tests

Test the actual boundaries between camera adapters, detector/model adapters, storage, and later recognition pipeline components as they are introduced.

Where possible, use prerecorded controlled test images instead of requiring a physical camera for every test.

### Benchmark tests

Benchmark runs are evidence-producing experiments rather than ordinary pass/fail tests. Every run should capture its environment and configuration.

---

## 17. Benchmark Reproducibility

Every run should have a unique `run_id` and record enough information to reproduce or interpret the result:

```text
run_id
timestamp
commit_sha
python_version
os
cpu
ram
gpu
camera
resolution
model_version
embedding_version
dataset_version
threshold
sample_count
TAR
FRR
FAR
TRR
median_latency_ms
p95_latency_ms
FPS
CPU
RAM
spoof_result
notes
```

The benchmark result is only meaningful together with its configuration and environment.

---

## 18. Kikiam-Harness Engineering Rules

This project follows the Kikiam-Harness operating model:

```text
DISCOVER
   ↓
DEFINE
   ↓
SPECIFY
   ↓
MODEL
   ↓
ARCHITECT
   ↓
GOVERN
   ↓
PLAN
   ↓
CONTEXTUALIZE
   ↓
IMPLEMENT
   ↓
VERIFY
   ↓
REVIEW
   ↓
SECURE
   ↓
DELIVER
   ↓
DEPLOY
   ↓
OBSERVE
   ↓
OPERATE
   ↓
LEARN
   ↓
EVOLVE
```

### Engineering principles

- Evidence outranks assumption.
- Human remains final authority for consequential security decisions.
- Generated code is not verified software.
- Preserve unrelated behavior and structure.
- Make small, reversible changes.
- Meaningful changes require reason, scope, and verification.
- Prefer existing patterns before introducing new abstractions.
- Security, privacy, reliability, and operability are design concerns from the beginning.
- Stop when acceptance criteria are satisfied rather than expanding scope.

### Agent contract

AI coding agents working in this repository must:

1. Inspect before modifying.
2. Understand the existing architecture before proposing changes.
3. Identify ambiguity, risks, and missing evidence.
4. Avoid inventing requirements.
5. Plan meaningful cross-cutting changes.
6. Modify only the requested scope.
7. Avoid unrelated refactoring, redesign, cleanup, dependency upgrades, or architecture replacement.
8. Verify every meaningful change.
9. Report actual changes and verification evidence.
10. Stop when acceptance criteria are satisfied.

---

## 19. Security Classification

**Current risk:** HIGH

**Potential future risk:** CRITICAL if integrated with physical access control.

### Assets

- biometric images;
- biometric embeddings/templates;
- subject identifiers;
- authentication decisions;
- benchmark results;
- local system access.

### Initial threats

- unauthorized enrollment;
- template theft or replacement;
- presentation/replay attacks;
- false acceptance/rejection;
- camera tampering;
- file modification;
- compromised dependencies/models.

### Initial security boundary

The benchmark assumes a trusted local development laptop. It is not designed to withstand a determined attacker with control of the host operating system.

---

## 20. Current Status

**Stage:** IMPLEMENT / VERIFY

**Current milestone:** BIO-001 — Camera → Face Detection → Visualization → Timing Metrics

**Deployment:** Windows/local laptop test deployment available on `KIKIAM-BIO-001-CAMERA-BASELINE`

**Recognition model:** Not implemented in BIO-001

**Web UI:** Not required yet

**Physical access control:** Explicitly out of scope

---

## 21. First Milestone Acceptance Criteria

- [ ] Camera initializes and operates reliably on the target Windows laptop.
- [ ] Face detection works on the target laptop.
- [ ] Detection and total latency are observed on real hardware.
- [x] Output is locally observable by implementation design.
- [x] Camera/frame/detector failures are handled explicitly.
- [x] Testable architecture boundaries are established.
- [x] Biometric/capture/result/model artifacts are excluded from Git by repository policy.
- [x] Repository CI passes Ruff, mypy, and pytest.
- [ ] Real hardware evidence is recorded and README/benchmark protocol are synchronized with the result.

---

## 22. License / Model Notice

This repository's application code and any third-party models or pretrained weights may have different licensing terms.

Before distributing, commercializing, or deploying the system, review the license for every dependency and pretrained model used by the implementation.

---

## 23. Next Engineering Action

Run BIO-001 on the target Windows laptop and record the actual camera, resolution, FPS, detection latency, and total latency. Use that evidence to accept or reject the camera baseline before starting enrollment or recognition work.
