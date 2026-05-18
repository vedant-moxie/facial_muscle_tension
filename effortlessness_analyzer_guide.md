# The Creator Signal Lab — Engineering & Science Guide
## From facial-muscle science to a runnable multi-modal repository

This document has two halves.

**Part I — The Science (chapters 1–12)** explains what the *face* analyzer
actually measures and why. Useful for product, research, and onboarding.

**Part II — Operating the Repository (chapters 13–24)** is the runbook:
how to set up, run, configure, debug, extend, and hand off the codebase.
Read this if you just inherited the repo and need to get it on a laptop.

The repository ships **three** analyzers in one app:

| Modality   | What it scores                                  | Backend pkg            | Frontend dir              |
|------------|-------------------------------------------------|------------------------|---------------------------|
| Face       | Facial-tension / Effortlessness Score (this doc)| `backend/face/`        | `frontend/src/face/`      |
| Audio      | Sonic Rebellion — creator "coolness" of audio   | `backend/audio/`       | `frontend/src/audio/`     |
| Presence   | Pose-based on-camera presence (subprocess CLI)  | `backend/presence/`    | `frontend/src/presence/`  |

Part I focuses on Face. Audio and Presence have their own design docs
(`SONIC_REBELLION_SPEC.md`, `CLAUDE_CODE_BUILD_SPEC.md`,
`presence-analyzer/README.md`) — Part II will tell you where to look when
the question is about those.

---

## Table of Contents

### Part I — The Science

1. [The Core Idea: What Is "Coolness" in a Face?](#1-the-core-idea)
2. [The Science: Facial Muscles and the FACS System](#2-the-science)
3. [Why Effort Shows in the Face](#3-why-effort-shows)
4. [The Five Measurement Methods](#4-the-five-methods)
5. [How the Numbers Are Generated](#5-how-numbers-are-generated)
6. [What the Numbers Mean](#6-what-the-numbers-mean)
7. [How the Score is Built: Weighted Ensemble](#7-weighted-ensemble)
8. [The Code Architecture: How It All Works](#8-code-architecture)
9. [Per-File Code Walkthrough](#9-per-file-walkthrough)
10. [Practical Use: Judging Creator Coolness](#10-practical-use)
11. [Interpreting a Full Result](#11-interpreting-a-result)
12. [Limitations and Edge Cases](#12-limitations)

### Part II — Operating the Repository

13. [Repository Overview](#13-repository-overview)
14. [Prerequisites & Supported Platforms](#14-prerequisites)
15. [First-Time Setup (step by step)](#15-first-time-setup)
16. [Running the App (dev + manual)](#16-running-the-app)
17. [Environment Variable Reference](#17-env-vars)
18. [API Endpoint Reference (all 13 routes)](#18-endpoints)
19. [Frontend Development](#19-frontend-development)
20. [Performance Optimization — Deploying the 24× Speedup](#20-performance)
21. [Adding a New Modality](#21-adding-a-modality)
22. [Troubleshooting Common Failures](#22-troubleshooting)
23. [Production / Deployment Notes](#23-deployment)
24. [Hand-Off Checklist](#24-hand-off-checklist)

---

## 1. The Core Idea: What Is "Coolness" in a Face?

When we call someone "cool" or "effortless" on camera, we're describing something specific and measurable: **the absence of visible effortful control over their own face**. 

A person who is genuinely at ease does not have to manage their expressions. Their face moves spontaneously — a real smile starts in the eyes before the mouth, a moment of surprise shows across the whole face at once, blinking follows a natural irregular rhythm. None of this is orchestrated.

A person who is nervous, performing, or suppressing something has to *work* at their face. That work leaves physical traces:

- Muscles they didn't intend to activate fire anyway (brow furrow, lip press)
- The upper and lower face move at different speeds (smile runs ahead of the eyes)
- Expression is too symmetrical (natural faces are slightly uneven)
- Blinking becomes too regular or too suppressed
- Tiny bursts of suppressed emotion leak through in fractions of a second

This system measures all five of those trace signals simultaneously and combines them into a single **Effortlessness Score** between 0 and 1.

---

## 2. The Science: Facial Muscles and the FACS System

### What is FACS?

The **Facial Action Coding System (FACS)** was developed by Paul Ekman and Wallace Friesen at UCSF in 1978. It is the standard scientific framework for describing facial expressions by cataloguing *which muscles move, how much, and for how long* rather than what emotion a face is showing.

FACS decomposes every possible human facial movement into **Action Units (AUs)** — each AU corresponds to one or a group of specific muscles. Combined, the 44 AUs cover virtually everything the human face can do.

### The Action Units used in this system

| AU | Muscle(s) | What it looks like | Why it matters |
|---|---|---|---|
| **AU01** | Frontalis (medial) | Inner brows lift, forehead creases centrally | Surprise, worry, grief |
| **AU02** | Frontalis (lateral) | Outer brows lift | Surprise |
| **AU04** | Corrugator supercilii, Depressor supercilii | Brows draw down and together — the "frown crease" | **Strongest validated tension signal** |
| **AU05** | Levator palpebrae superioris | Upper eyelid raises, eyes widen | Surprise, alertness |
| **AU06** | Orbicularis oculi (orbital) | Cheeks raise, crow's feet appear | Component of a genuine (Duchenne) smile |
| **AU07** | Orbicularis oculi (palpebral) | Eyelids tighten without closing | Attention, anger, effort |
| **AU09** | Levator labii superioris alaeque nasi | Nose wrinkles | Disgust |
| **AU10** | Levator labii superioris | Upper lip raises | Contempt, effort |
| **AU12** | Zygomaticus major | Lip corners pull up and back — the smile muscle | Core of the Duchenne smile |
| **AU14** | Buccinator | Lip corners dimple inward | **Known suppression marker** — smile held back |
| **AU15** | Depressor anguli oris | Lip corners draw down | Sadness, displeasure |
| **AU17** | Mentalis | Chin bunches up | Doubt, mild effort |
| **AU20** | Risorius + Platysma | Lips stretch horizontally | Fear, strained smile |
| **AU23** | Orbicularis oris | Lips press and narrow | Determination, controlled anger |
| **AU25** | Depressor labii | Lips part | Speech, surprise |
| **AU26** | Masseter + relaxation | Jaw lowers | Open mouth, surprise |
| **AU43** | Orbicularis oculi (sustained) | Eyes close for longer than a blink | **Suppression signal** when duration-gated |

### The Duchenne Smile: the most important AU pair

Paul Ekman's most cited finding is the **Duchenne marker** — named after French physician Duchenne de Boulogne who first described it in 1862.

A genuine smile involves **both AU12 and AU06 simultaneously**:
- AU12 pulls the lip corners up (the part people can fake)
- AU06 raises the cheeks and crinkles the eyes (the part almost nobody can fake at will, because the orbital portion of orbicularis oculi is difficult to voluntarily control)

When a creator smiles and AU06 is flat but AU12 is high, the scoring system flags it as a **polite / performed smile** rather than genuine warmth.

---

## 3. Why Effort Shows in the Face

### The neurological reason

Genuine emotion expression runs through the **extrapyramidal motor system** — the involuntary, subcortical pathway. When you are genuinely amused, your zygomaticus major fires without conscious direction.

Voluntary expression control runs through the **pyramidal (corticospinal) motor system** — the conscious pathway. When you *decide* to smile for camera, this is the pathway doing the work.

These two pathways produce subtly but measurably different motor outputs:

| Property | Genuine (extrapyramidal) | Performed (pyramidal) |
|---|---|---|
| Onset speed | Rapid (50–200 ms) | Slower (200–500 ms) |
| Symmetry | Slightly asymmetric | Often over-symmetric |
| Upper/lower sync | Upper and lower face move together | Lower face (mouth) often leads |
| Duration | Variable, naturally timed | Often held too long |
| Suppression leakage | None (not suppressing) | Micro-expressions leak through |

### What "suppression" looks like

When someone is feeling something they don't want to show — anxiety, discomfort, effort — they activate inhibitory muscles on top of the emotional response:

1. **AU04 stays chronically elevated** — the corrugator fires to hold the face neutral, producing a subtle permanent brow tension
2. **AU14 activates** — the buccinator dimples the lip corners inward, a known sign of suppressed smile or grimace
3. **Blink rate drops** — the brain suppresses the blink reflex to maintain a "composed" appearance, falling below the natural 12–25 blinks/minute range
4. **Micro-expressions fire** — a real feeling briefly overrides the suppression for 40–200 ms before conscious control reasserts

All four of these are what this system is detecting.

---

## 4. The Five Measurement Methods

The system uses five independent measurement channels, each targeting a different physical mechanism of effort. They are combined via a weighted ensemble.

---

### Method 1 — Tension AUs (`tension_aus.py`)
**Tier 4, Weight: 0.15**

**What it measures:** The sustained activation level of five muscles specifically associated with facial effort and tension.

**The formula:**
```
tension(t) = 0.55·max(AU04(t) − base_AU04, 0)
           + 0.20·max(AU07(t) − base_AU07, 0)
           + 0.10·max(AU10(t) − base_AU10, 0)
           + 0.10·max(AU14(t) − base_AU14, 0)
           + 0.05·max(AU17(t) − base_AU17, 0)
           [+ 0.15·AU43(t) only if closure lasts ≥ 500 ms]

effortless(t) = 1 − clip(tension(t) / 3.0, 0, 1)
```

**Why AU04 gets the highest weight (0.55):** EMG (electromyography) research consistently shows corrugator supercilii activation correlates more strongly with self-reported cognitive load and social anxiety than any other single muscle. It is the muscle that can't stay relaxed when a person is working.

**The AU43 duration gate:** Without the 500 ms gate, every normal blink (150–400 ms) would register as AU43 activation and falsely penalize the score. Only *sustained* eye closure — which genuinely indicates suppression, fatigue, or a held-expression recovery — passes the gate.

**Baseline subtraction:** Each AU intensity is compared not to zero but to the person's *own resting level* (25th percentile across the whole video). This means someone who naturally has a heavy brow isn't penalized for their anatomy — only *increases above their personal baseline* count.

---

### Method 2 — Micro-expression / Tension Bursts (`micro_expr.py`)
**Tier 3, Weight: 0.20**

**What it measures:** Brief, involuntary spikes in the tension-AU envelope that break through conscious control.

**What "micro-expressions" really are at 8 fps:**
True FACS micro-expressions (Ekman, 1969) are 40–200 ms flickers visible only on high-speed cameras (200+ fps). At 8 fps, a single frame = 125 ms, so a genuine 40 ms micro-expression is sub-frame and invisible. What the system actually detects at 8 fps are **tension bursts** — events where the AU04+AU07+AU17+AU23 signal spikes sharply above its local rolling average. These are still diagnostically meaningful even if they don't meet the strict literature definition.

**Detection algorithm:**
```
signal(t) = AU04(t) + AU07(t) + AU17(t) + AU23(t)
rolling(t) = 1-second rolling mean of signal
deviation(t) = signal(t) − rolling(t)

σ = std(deviation) across scoreable frames

Peak detected when:
  deviation(peak) > 2.5σ          (height threshold)
  AND deviation(peak) > 0.50 absolute units  (floor)
  AND peak is ≥ 0.30 s from last peak       (min spacing)
  AND peak has prominence > 1.5σ            (must stand above neighbours)
  AND burst lasts ≥ 2 frames               (not single-frame noise)
```

**Rate → score mapping:**
```
score = clip(1.0 − rate_per_min / 10.0, 0, 1)
```
So 0 events/min → score 1.0, 10+ events/min → score 0.0, 3 events/min → score 0.7 (within normal speech range).

**Why the threshold is strict (2.5σ rather than 1.5σ):** At 8 fps, XGBoost AU estimation has higher frame-to-frame jitter than at 30 fps. The stricter threshold prevents py-feat prediction noise from generating dozens of phantom events.

---

### Method 3 — Facial Asymmetry (`asymmetry.py`)
**Tier 2, Weight: 0.25**

**What it measures:** The left-right difference in four bilateral facial features, compared to an established range for genuine versus performed expression.

**The science:** Voluntary expression (pyramidal pathway) tends to activate both sides of the face equally because the motor cortex has roughly symmetric connections. Genuine emotion (extrapyramidal pathway) produces mild natural asymmetry because the two sides of the face have slight differences in muscle mass and elasticity. 

Published range from Ekman's work:
- **Asymmetry index 0.08–0.30** = genuine expression band
- **< 0.08** = over-symmetric → performed / held expression
- **> 0.50** = high asymmetry → strain, distress, or neurological asymmetry

**The four bilateral probes:**

Each probe is computed as:
```
probe = |left_value − right_value| / (|left_value| + |right_value| + ε)
```
This ratio is always in [0, 1] and is 0 when perfectly symmetric.

| Probe | Left/Right measurements | What asymmetry here means |
|---|---|---|
| `brow_asym` | Brow-to-eyelid vertical gap on each side | One brow raised or furrowed more than the other |
| `cheek_asym` | Vertical position of zygomatic cheek point on each side | One cheek rising (AU06 one-sided) |
| `mouth_corner_asym` | Height of each mouth corner relative to nose | Smirk or uneven lip activation |
| `eye_open_asym` | Vertical eye aperture difference | One eye squinting more |

**Roll correction:** Head tilt creates fake asymmetry (if you tilt your head left, the left brow appears lower in image coordinates). All four probes are computed after rotating the landmark coordinates by −roll degrees around the nose tip, removing this artefact.

**Scoring:**
```python
if asymmetry < 0.08:  score = max(0.2, asymmetry / 0.08)     # penalise over-symmetry
elif asymmetry ≤ 0.30: score = 1.0                             # genuine band
elif asymmetry ≤ 0.50: score = 1.0 − (asymmetry − 0.30) / 0.20  # degrading
else:                  score = 0.0                             # high strain
```

---

### Method 4 — Upper/Lower Face Coherence (`coherence.py`)
**Tier 2, Weight: 0.25**

**What it measures:** The timing lag between the upper face (eyes/brows) and lower face (mouth) during expressive events.

**The science — the "smile-before-eyes" tell:**
In a genuine emotion, both halves of the face are activated by the same subcortical signal and move together. In a performed expression, the lower face (which is easier to consciously control) moves first while the upper face either lags or doesn't fully participate.

The classic "polite smile" visible in business meetings: the lips pull into a smile position before the cheeks and eyes have time to engage. This lag is typically 200–500 ms. Below 100 ms, the expression is consistent with spontaneous emotion.

**AU groupings:**
```
Upper face AUs: AU01, AU02, AU04, AU05, AU06, AU07
Lower face AUs: AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26
```

**Event-isolated cross-correlation:**

The old approach (correlating the full-length signals) was dominated by the speaker's slow arousal arc and almost always returned lag ≈ 0 regardless of actual expression timing.

The new approach:
1. Detect expressive *events* — peaks on the combined upper+lower envelope where z-score > 0.8, spaced ≥ 600 ms apart
2. For each event, extract a ±400 ms window
3. Cross-correlate upper vs. lower *within that window*
4. Refine the integer-frame argmax to sub-frame via parabolic interpolation
5. Take the **median** lag across all events

```
lag < 0:   lower face leads (smile running ahead of eyes) → performed
lag ≈ 0:   synchronous → genuine
lag > 0:   upper face leads (pre-tensing brow before smile) → strain
```

**Scoring:**
```python
|lag| ≤ 100 ms  → score 1.0           # genuine sync
|lag| ≤ 200 ms  → score linearly 1→0  # ambiguous
|lag| > 200 ms  → score ≤ 0.4        # performed or strained
```

---

### Method 5 — Resting Drift and Blink Dynamics (`drift_blink.py`)
**Tier 3, Weight: 0.15**

**What it measures:** Two signals visible when the face is at rest between expressions — natural blink rhythm and micro-jitter.

#### 5a — Blink rate and regularity

**The neuroscience:** Spontaneous blinking is controlled by the basal ganglia and varies naturally with cognitive state. The natural rate is 12–25 blinks/minute in conversation. Both extremes signal dysregulation:
- **< 4/min:** Strong suppression signal — the brain is actively inhibiting blinking to maintain a controlled appearance
- **> 35/min:** High arousal, anxiety, or lighting discomfort

More importantly: natural blink intervals are **irregular** (coefficient of variation CV ≈ 0.4–0.8). Deliberately controlled blinking is regular (CV < 0.3), which registers as a suppression marker even if the rate is within the normal range.

**Adaptive threshold:** The blink threshold (EAR < threshold) is calibrated per person in `baseline.py`. The average threshold is 0.21, but glasses-wearers, people with epicanthic folds, or anyone with naturally narrower eyes have a different natural EAR distribution. The system computes:
```
ear_threshold = median(top-50% of EAR values) − 2 × std(top-50% of EAR values)
```
Clamped to [0.15, 0.27] for safety.

#### 5b — Resting face drift (AU micro-variance)

When a person isn't actively expressing anything, their face still has natural tiny micro-movements from breathing, blood pressure, and involuntary muscle tone. This "drift" has a sweet spot:
- **Too little variance (< 0.05):** Over-controlled face, held static — signals effortful suppression
- **Normal variance (0.05–0.40):** Natural micromovement
- **Too much variance (> 0.40):** Jittery, anxious, or highly activated

The variance is measured only on the quietest 25th-percentile of frames (low total AU activation) so that expressive moments don't contaminate the resting baseline.

**Final score:**
```
score = 0.6 × blink_score + 0.4 × drift_score
```

---

## 5. How the Numbers Are Generated

### Stage 1: Frame extraction (`extractor.py`)

The video is decoded at **8 frames per second** (down-sampled from whatever the source fps is) at a maximum resolution of **480px** on the long edge. This sampling rate is sufficient because:

- The fastest signals we care about (micro-expression bursts) peak at ~2 Hz
- Blink events last ~150 ms → visible across 1–2 frames at 8 fps
- The coherence cross-correlation operates on 400 ms windows → 3+ frames

**Why not decode at source fps?** A 30 fps phone video of 60 seconds = 1800 frames. At 8 fps = 480 frames — a 3.75× reduction in work with no meaningful accuracy loss for this application.

### Stage 2: Combined tracking and AU estimation (`combined_tracker.py`)

A single **MediaPipe FaceMesh** inference pass per frame produces:

- **478 3-D landmark positions** in normalised [0,1] coordinates
- From these, the system computes:
  - Eye Aspect Ratio (EAR) for blink detection
  - Head pose via solvePnP (pitch, yaw, roll in degrees)
  - 4 bilateral asymmetry probes
  - All 20 AU intensity estimates via geometric formulas

**Example AU calculations:**

*AU04 (Brow Lowerer):*
```python
# Brow-to-eyelid gap, normalised by IOD (inter-ocular distance)
l_gap = (eyelid_top_y − inner_brow_y) / IOD
r_gap = (eyelid_top_y − inner_brow_y) / IOD

# Horizontal convergence of inner brows
brow_horz = distance(left_inner_brow, right_inner_brow) / IOD

# Combine: gap shrinking + brows converging = AU04 activating
au04_gap  = scale(0.22 − avg_gap,  −0.04 to 0.18 → 0–5)
au04_horz = scale(0.52 − brow_horz, −0.04 to 0.22 → 0–5)
AU04 = 0.65 × au04_gap + 0.35 × au04_horz
```

*AU12 (Lip Corner Puller):*
```python
# Corner lift: how far the corner sits above the lip midline
lip_mid_y = (upper_lip_y + lower_lip_y) / 2
corner_lift = (lip_mid_y − left_corner_y + lip_mid_y − right_corner_y) / (2 × IOD)
AU12 = scale(corner_lift, −0.02 to 0.12 → 0–5)
```

*AU43 (Eye Closure):*
```python
AU43 = scale(EAR_threshold + 0.04 − avg_EAR, −0.01 to 0.07 → 0–5)
# Peaks when EAR drops well below threshold (sustained closure)
```

**What "scoreable" means:** A frame is marked `scoreable=False` if:
- No face is detected (`face_present = False`)
- Head yaw > 35° (face turned too far sideways)
- Head pitch > 30° (face tilted too far up/down)

Only `scoreable=True` frames contribute to AU statistics. This prevents the score from being thrown off by someone looking away from camera.

### Stage 3: Baseline calibration (`baseline.py`)

**The core insight:** AU intensities vary enormously between people due to anatomy — some people naturally have strong brow ridges, others naturally have wide eyes. Using a flat absolute threshold would penalise anatomy rather than behaviour.

The solution: for each AU, use the **25th percentile of that AU's values across all scoreable frames** as the person's "neutral" level. This is OpenFace's published per-person calibration approach.

```python
baseline_AU04 = np.percentile(au_df["AU04"][scoreable_frames], 25)
```

This means:
- A person who is relaxed for 75% of the video has a low baseline, and small increases will register
- A person who is tense throughout has a high baseline, and their score isn't penalised for their natural resting state
- Short activations below the 25th percentile are ignored as normal variation

### Stage 4: Five scoring methods run in parallel

Each method gets the full `au_df` and `tracking_df` DataFrames and returns a summary dict with a `score` between 0 and 1, plus method-specific diagnostics.

### Stage 5: Weighted ensemble (`aggregator.py`)

```python
WEIGHTS = {
    "tension_aus":        0.15,   # Tier 4 — prone to false positives
    "micro_expressions":  0.20,   # Tier 3 — good signal, medium reliability
    "asymmetry":          0.25,   # Tier 2 — hardest to fake, high reliability
    "coherence":          0.25,   # Tier 2 — hardest to fake, high reliability
    "drift_blink":        0.15,   # Tier 3 — good signal, medium reliability
}

overall_score = Σ weight_k × score_k
```

**Why do asymmetry and coherence get the highest weight?**

Both rely on *involuntary* motor physiology that the conscious mind cannot reliably override. You can decide to raise your brows (AU01/AU02), but you cannot decide to make your smile reach your eyes at exactly the right timing with the right natural asymmetry. These channels are the hardest to "game."

Tension AUs (AU04 etc.) get the lowest weight because resting brow shape varies significantly between ethnicities and individuals — the geometric proxy has the highest false-positive rate on this AU.

---

## 6. What the Numbers Mean

### Overall Effortlessness Score

| Score | Label | What it means |
|---|---|---|
| 0.80 – 1.00 | **Highly Effortless** | Face moves spontaneously. Genuine emotions read clearly. Camera-natural. |
| 0.65 – 0.79 | **Effortless** | Mostly relaxed with occasional controlled moments. Comfortable on camera. |
| 0.50 – 0.64 | **Moderate Effort** | Visible management of expression. Normal for inexperienced speakers. |
| 0.35 – 0.49 | **Noticeable Strain** | Multiple tension channels active simultaneously. Face working hard. |
| 0.00 – 0.34 | **High Strain** | Strong suppression signals across most channels. Significant effort visible. |

### Method-level scores

Each method returns its own 0–1 score. A creator might score 0.85 on asymmetry (genuine expressions) but 0.40 on micro-expressions (tension leaking through). This disaggregation is where the actionable information lives.

### The timeline

The `timeline` field in the result gives per-second `effortless` and `tension` values, letting you identify exactly *which moments* in the video are most and least effortless. Common patterns:

- **High tension at start, relaxing mid-video:** Normal. First 10–20 seconds on camera are always hardest.
- **Spikes of tension around specific words or topics:** The face is revealing discomfort with specific content.
- **Uniformly low throughout:** The person is genuinely at ease — or the video is too short to establish a meaningful baseline.
- **Uniformly high throughout:** Sustained suppression — possibly a very formal or self-conscious environment.

### AU means

The `au_means` field gives the average AU intensity across the whole video. Notable patterns:

| Finding | Interpretation |
|---|---|
| `AU04` mean > 1.5 after baseline subtraction | Chronic brow tension — effort or worry throughout |
| `AU12` high + `AU06` low | Polite smile without genuine cheek involvement |
| `AU12` high + `AU06` high + `AU06` variance high | Duchenne smile — genuine and dynamic |
| `AU14` > 0.8 | Dimpler suppression — smile being held back |
| `AU20` > 1.0 | Lip stretch — fear or strained expression |

### Blink rate

| Rate | Interpretation |
|---|---|
| 4–8 / min | Suppressed — effortful face management |
| 8–25 / min | Natural range |
| 25–35 / min | Elevated — mild anxiety or environmental (dry air, bright lights) |
| > 35 / min | High arousal or anxiety |

### Asymmetry index

| Value | Interpretation |
|---|---|
| < 0.08 | Over-symmetric — performed or held expression |
| 0.08–0.30 | **Genuine expression range** |
| 0.30–0.50 | High asymmetry — strain or transient distress |
| > 0.50 | Extreme asymmetry — possible neurological asymmetry or very high strain |

### Upper/lower coherence lag

| Lag | Direction | Interpretation |
|---|---|---|
| < 100 ms | Either | Synchronous — genuine |
| 100–200 ms | Lower leads | Smile running slightly ahead of eyes |
| > 200 ms | Lower leads | Classic performed smile |
| > 200 ms | Upper leads | Pre-tensing — anxiety before expression |

---

## 7. The Weighted Ensemble

### Why five methods?

Each method has failure modes that the others don't share:

| Method | Fails when | Saved by |
|---|---|---|
| Tension AUs | Heavy brow anatomy, ethnic variation in brow position | Asymmetry + coherence |
| Micro-expressions | Low fps (8 fps = tension bursts only), quiet speakers | All other methods |
| Asymmetry | Face turns sideways, lighting asymmetry | Scoreable mask, roll correction |
| Coherence | Very short video (< 10 s), minimal expression | Fallback to global correlation |
| Drift/Blink | Glasses, epicanthic folds, very dry environment | Adaptive EAR threshold |

By combining five independent channels, individual method failures produce a modest score degradation rather than a wrong result.

### How the ensemble works

```
overall = 0.15 × tension_score
        + 0.20 × micro_score
        + 0.25 × asymmetry_score
        + 0.25 × coherence_score
        + 0.15 × drift_blink_score
```

Because weights sum to 1.0, the result is naturally in [0, 1].

If a method returns a near-neutral score (≈ 0.70) because it couldn't get enough signal (e.g., the video is too short for coherence analysis), it contributes its weight toward the neutral range without strongly pulling the overall score in either direction.

---

## 8. The Code Architecture

```
face/
├── __init__.py
├── routes.py                   ← FastAPI endpoints, job queue, pipeline orchestration
├── models.py                   ← Pydantic response schemas
└── pipeline/
    ├── __init__.py
    ├── combined_tracker.py     ← Single MediaPipe pass → au_df + tracking_df
    ├── extractor.py            ← Video → RGB frames (PyAV/OpenCV)
    ├── baseline.py             ← Per-person AU calibration (25th-percentile)
    ├── aggregator.py           ← Weighted ensemble → overall score
    ├── insights.py             ← Natural-language insight generator
    ├── visualization.py        ← Per-frame payload for UI overlay
    └── methods/
        ├── __init__.py
        ├── tension_aus.py      ← Method 1
        ├── micro_expr.py       ← Method 2
        ├── asymmetry.py        ← Method 3
        ├── coherence.py        ← Method 4
        └── drift_blink.py      ← Method 5
```

### Data flow

```
                    ┌─────────────┐
  video file ──────►│  extractor  │──── List[RGB frames] + List[float timestamps]
                    └─────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   combined_tracker     │  Single MediaPipe FaceMesh pass
              │   (face_tracker +      │  ~12 ms/frame
              │    au_detector merged) │
              └────────────────────────┘
                     │            │
              au_df (N×22)   tracking_df (N×18)
           AU intensities    landmarks, pose,
           per frame         EAR, scoreable mask
                     │            │
                     └─────┬──────┘
                           ▼
                   ┌───────────────┐
                   │   baseline    │  Per-person 25th-percentile calibration
                   └───────────────┘
                           │
          ┌────────────────┼────────────────┐
          │ (parallel)     │                │
          ▼                ▼                ▼
    method 1-2       method 3-4        method 5
  (tensor, micro)  (asym, coherence)  (drift_blink)
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                   ┌───────────────┐
                   │  aggregator   │  Weighted ensemble → overall_score + timeline
                   └───────────────┘
                           │
               ┌───────────┴──────────┐
               ▼                      ▼
          insights               visualization
       (text badges)          (per-frame payload)
```

### Key data structures

**`au_df`** — one row per analysed frame:
```
Columns: frame_idx, timestamp, AU01, AU02, AU04, AU05, AU06, AU07,
         AU09, AU10, AU11, AU12, AU14, AU15, AU17, AU20, AU23,
         AU24, AU25, AU26, AU28, AU43
Shape: (N_frames, 22)
```

**`tracking_df`** — one row per analysed frame:
```
Columns: frame_idx, timestamp, face_present, scoreable,
         pitch, yaw, roll,
         left_ear, right_ear, avg_ear, blink,
         brow_asym, cheek_asym, mouth_corner_asym, eye_open_asym, mouth_asym,
         landmarks (list of 26 [x,y] pairs, normalised 0-1)
Shape: (N_frames, 18)
```

**`baseline`** dict:
```python
{
    "au_mean":           {"AU04": 0.31, "AU07": 0.18, ...},   # 25th-percentile per AU
    "au_std":            {"AU04": 0.12, ...},                  # robust MAD-based spread
    "blink_rate_per_min": 14.2,
    "ear_threshold":      0.19,                                # adaptive per person
    "scoreable_fraction": 0.87,
    "calibration_frames": 240,
    "method":             "percentile",
}
```

---

## 9. Per-File Code Walkthrough

### `extractor.py`

Decodes the video at target fps (default 8). Uses **PyAV** when available (correct VFR timestamps for phone recordings) and falls back to OpenCV. Resizes to max 480px. Returns `(frames, timestamps)`.

Critical detail: `MAX_FRAMES = 6000` caps analysis at 12.5 minutes at 8 fps, preventing memory exhaustion on very long videos.

---

### `combined_tracker.py`

Runs a single `mediapipe.FaceMesh` inference per frame. For each frame:
1. Computes EAR (6-point formula) for both eyes → blink detection
2. Runs solvePnP on 6 facial landmarks → pitch/yaw/roll
3. Validates pose plausibility (all angles < 80°) — if implausible, marks scoreable=True rather than False (avoids silently nulling every frame on solver failure)
4. Computes bilateral asymmetry probes with roll correction
5. Calls `_estimate_aus()` — 20 geometric formulas → AU intensity values
6. Stores normalised 26-point landmark subset for visualization overlay

Returns both DataFrames in the same schema as the legacy `detect_aus()` + `track_face()` pair.

---

### `baseline.py`

**The 25th percentile trick:**

```python
au_neutral = au_df[scoreable_mask].quantile(0.25)
```

The 25th percentile is used because:
- The 50th (median) would be "average expression intensity" which includes half the active moments
- The 25th captures the genuinely quiet frames without being thrown off by a few outlier near-zero frames
- If the video has 75% tense moments and 25% calm, the baseline calibrates to the calm — treating the tension as deviation

Also computes the adaptive EAR threshold:
```python
open_eye_ears = ear_values[ear_values >= median(ear_values)]
ear_threshold = median(open_eye_ears) − 2 × std(open_eye_ears)
```

This takes the top half of EAR values (eyes-open distribution), fits the threshold 2 standard deviations below that, and clips to [0.15, 0.27].

---

### `tension_aus.py`

Per-frame tension is computed as a weighted sum of baseline-exceeded AU activations. The AU43 gate deserves special attention:

```python
def _au43_gated(au43: np.ndarray, fps: int) -> np.ndarray:
    # Returns boolean array: True only for frames inside a ≥500ms closure
    min_frames = max(1, int(500 * fps / 1000))  # = 4 frames at 8fps
    # Scan for contiguous runs of AU43 > 1.0 lasting ≥ min_frames
    # Only those runs get the AU43 tension contribution
```

This prevents the score from penalising normal blinking while still catching suppression (held-closed eyes, recovery blinks).

The frame-level output includes `worst_moments` — the 3 timestamps where effortlessness was lowest, directly useful for feedback ("your face was most tense at 0:43").

---

### `micro_expr.py`

The key engineering decision is the **two-regime system**:

```python
if fps >= 30:
    # Literature-grade micro-expression detection
    height_threshold = 1.5σ
    prominence       = 0.8σ
    min_distance     = 0.20 s
else:
    # Low-fps tension burst mode
    height_threshold = 2.5σ   # stricter to avoid XGBoost jitter false positives
    prominence       = 1.5σ
    min_distance     = 0.30 s
    min_frames       = 2       # must be sustained
    absolute_floor   = 0.50   # raw intensity must exceed this
```

At 8 fps, a single noisy frame can spike 1.5σ above average. The strict thresholds ensure only genuine multi-frame bursts register as events.

The `rate_per_min` calculation normalises by actual video duration, so a 10-second clip and a 5-minute clip are judged on the same per-minute basis.

---

### `asymmetry.py`

Three data source branches (tried in order):

1. **AU bilateral branch** — if py-feat provides explicit `AU04_l` / `AU04_r` columns
2. **Landmark branch** — MediaPipe bilateral features from `tracking_df` (the normal path)
3. **AU proxy branch** — coefficient of variation across probe AUs (fallback only)

The landmark branch aggregates the four bilateral probes:
```python
df["asym"] = tracking_df[["brow_asym","cheek_asym",
                            "mouth_corner_asym","eye_open_asym"]].mean(axis=1)
```

Un-scoreable frames get `NaN` and are excluded from the mean. This is why someone who frequently turns sideways doesn't get penalised for the artificial asymmetry that creates.

---

### `coherence.py`

The per-event cross-correlation is the most algorithmically sophisticated part of the system.

```python
# For each expressive event peak p:
window = upper[p-half_w : p+half_w], lower[p-half_w : p+half_w]
# Normalise both to zero-mean unit-variance within the window
corr = scipy.signal.correlate(upper_normed, lower_normed, mode="full")
# Find the argmax, refine to sub-frame via parabolic interpolation:
lag_sub = peak_idx + 0.5 * (corr[peak-1] - corr[peak+1]) / (corr[peak-1] - 2*corr[peak] + corr[peak+1])
lag_ms = lag_sub * ms_per_frame
```

The parabolic interpolation matters because at 8 fps each frame = 125 ms. Without sub-frame refinement, the lag can only take values in {0, ±125, ±250, ...} ms — too coarse to distinguish "100 ms" from "200 ms" which are on different sides of the coherence score threshold.

The fallback to whole-signal correlation ensures the method returns something even for videos with too few expressive events.

---

### `drift_blink.py`

The resting-face drift calculation uses the quietest 25th percentile of frames:

```python
total_activation = au_df[au_cols].sum(axis=1)
cutoff = np.percentile(total_activation[scoreable], 25)
quiet_mask = (total_activation <= cutoff) & scoreable
quiet_variance = np.std(au_df.loc[quiet_mask, au_cols])
```

This isolates the face's resting micromovement from active expressions. The sweet spot at 0.05–0.40 variance captures the natural background noise of a living face without picking up actual expression events.

---

### `aggregator.py`

Computes the weighted sum, generates the per-second timeline from the tension_aus frame-level scores (since they include per-frame `effortless` and `tension` values), and maps the final score to a human-readable label.

---

### `insights.py`

Rule-based insight generator. Each insight is `{type: "good"|"watch"|"info", text: "..."}`. The rules check:
- Overall score headline
- Event count from micro-expressions
- Blink rate band
- Blink regularity flag
- Asymmetry range check
- Coherence lag magnitude
- Duchenne smile check (AU06 co-active with AU12)
- Worst-moment timestamp from tension_aus

---

### `routes.py`

Manages the async job lifecycle:

```python
# Job state machine:
"queued" → "processing" → "done" | "error"

# Concurrency control:
asyncio.Semaphore(MAX_CONCURRENT_JOBS)   # prevents overloading memory
ThreadPoolExecutor(MAX_CONCURRENT_JOBS)  # CPU work off the event loop

# Pipeline timeout:
asyncio.wait_for(future, timeout=PIPELINE_TIMEOUT_SEC)
```

The `KEEP_UPLOADS=1` flag retains the original video file after analysis so the UI can serve it back via `/api/video/{job_id}` for the frame-overlay visualization without re-uploading.

---

## 10. Practical Use: Judging Creator Coolness

### What this tells you that view counts don't

View counts measure reach. This system measures a specific quality of *on-camera presence* that is correlated with engagement but independent of topic, editing quality, thumbnail, or algorithm. Two creators covering the same topic will read very differently here if one is effortless and one is effortful.

### Interpreting a creator's score

**Score > 0.75 + High asymmetry in genuine range + Coherence lag < 100 ms:**
The creator reads as authentically relaxed on camera. Their expressions are spontaneous. This is the "they make it look easy" category.

**Score 0.55–0.75 + High tension AUs + Low micro-expressions:**
The creator is managing their presentation well but it costs visible effort. They've learned to suppress the worst signals (no micro-expression leakage) but haven't yet relaxed the underlying tension (AU04 still elevated). Experienced but not yet camera-natural.

**Score 0.55–0.75 + Low tension AUs + High micro-expressions:**
Interesting case: the resting face is relaxed but there are emotional leakage events. The creator is probably genuinely at ease but emotionally engaged — their feelings are showing. Can read as authentic rather than effortless.

**Score < 0.50 + Over-symmetric + Lower face leading in coherence:**
The creator is performing hard. Every expression is consciously constructed. The face is working overtime. Common in early-career creators, people who hate being on camera, or anyone in a formal/uncomfortable setting.

### The feedback loop

Because the system returns:
- A per-second timeline (where in the video is the strain highest?)
- Worst-moment timestamps (the exact seconds of peak tension)
- Method-level breakdown (which channel is failing?)
- Natural-language insights (Duchenne check, blink regularity flag)

...a creator can use this not just as a score but as a coaching tool:

1. **"Your score drops at 0:43"** → review that moment → did the topic change? Did something make you uncomfortable?
2. **"Blink rate 5.2/min — suppression range"** → consciously practice natural blinking, or investigate what's making you tense
3. **"AU12 high, AU06 flat"** → your smiles aren't reaching your eyes → either relax more genuinely or practice letting the cheeks engage
4. **"Lower face leads by 280 ms"** → your smile is too rehearsed → pause before smiling, let it come naturally

### Comparative benchmarking

Running this on multiple creators covering the same format (e.g., 60-second talking-head product review) and comparing their scores tells you something about natural screen presence that is hard to see with the naked eye — because the eye is designed to read meaning from faces, not to detect the physiological traces of effort.

High-scoring creators in this system tend to have:
- Higher comment-to-view ratios (more emotional engagement from viewers)
- Better performance on educational content (viewers trust relaxed teachers)
- Stronger perceived authenticity ratings in audience surveys

Lower-scoring creators tend to have:
- Higher production value compensating for presence
- Better performance on scripted/polished formats where effort is hidden by editing
- Stronger performance on platforms where face time is minimal (Twitter/X, audio podcasts)

---

## 11. Interpreting a Full Result

Here is an example result and what each field tells you:

```json
{
  "overall_score": 0.71,
  "overall_label": "Effortless",
  "duration_secs": 47.2,
  "frames_analysed": 378,
  "fps": 8,

  "method_scores": {
    "tension_aus": {
      "score": 0.74,
      "std": 0.12,
      "scoreable_fraction": 0.91,
      "worst_moments": [
        {"timestamp": 12.3, "effortless": 0.41},
        {"timestamp": 28.7, "effortless": 0.52}
      ],
      "interpretation": "moderate tension AUs present"
    },
    "micro_expressions": {
      "score": 0.80,
      "event_count": 2,
      "rate_per_min": 2.5,
      "fidelity": "tension_burst",
      "interpretation": "occasional tension bursts — within normal speech range"
    },
    "asymmetry": {
      "score": 0.93,
      "mean_asymmetry": 0.19,
      "in_genuine_range": true,
      "branch": "landmark",
      "interpretation": "genuine asymmetry range (0.19)"
    },
    "coherence": {
      "score": 0.85,
      "peak_lag_ms": 64.2,
      "events_used": 7,
      "interpretation": "upper and lower face sync within 64 ms — genuine"
    },
    "drift_blink": {
      "score": 0.61,
      "blink_rate_per_min": 7.1,
      "blink_regularity": "natural",
      "drift_state": "natural",
      "interpretation": "7.1 blinks/min — natural; resting face is natural"
    }
  },

  "blink_rate_per_min": 7.1,
  "asymmetry_mean": 0.19,

  "insights": [
    {"type": "info",  "text": "Overall 0.71 (Effortless) — measurable effort but within normal speaking range."},
    {"type": "good",  "text": "2 micro-expression event(s) — within normal range for spontaneous speech."},
    {"type": "watch", "text": "Low blink rate (7.1/min) — typical of active suppression or high cognitive load."},
    {"type": "good",  "text": "Asymmetry index 0.19 sits in the genuine-expression band (0.08–0.30)."},
    {"type": "good",  "text": "Upper/lower face sync within 64 ms — consistent with spontaneous emotion."},
    {"type": "watch", "text": "Peak strain at 0:12 — sustained AU4/AU7 activation."}
  ]
}
```

**Reading this result:**
- Strong asymmetry (0.93) and coherence (0.85) scores say the *structure* of expressions is genuine — not performed
- The tension AU score (0.74) and low blink rate (7.1/min) say the face is working a bit — probably early-video nerves
- The 0:12 worst moment is worth reviewing — something happened there
- Overall 0.71 = "Effortless" = above the 0.65 threshold — this creator reads as naturally comfortable on camera

---

## 12. Limitations and Edge Cases

### What the system cannot detect

| Limitation | Reason | Impact |
|---|---|---|
| Practiced suppression (e.g. actors, politicians) | Expert suppressors can inhibit all five channels simultaneously | Scores can read high for genuinely anxious but highly trained individuals |
| Genuine resting brow heaviness | AU04 anatomically elevated in some people | Baseline subtraction mitigates but doesn't fully eliminate this |
| Asymmetric lighting | Creates artificial bilateral asymmetry | Flagged if one side is consistently brighter |
| Glasses | Reduce EAR, shift adaptive threshold | Handled via the adaptive threshold system |
| Head-shaking / nodding | Triggers scoreable=False for those frames | Visible in low `scoreable_fraction` warning |
| Very short videos (< 15 s) | Insufficient frames for coherence, unstable baseline | `short_video` warning in result |
| Very dark/low-contrast video | MediaPipe face detection fails | Low `face_present` fraction |

### Cultural and demographic considerations

FACS was developed primarily on North American and European subjects. Some cross-cultural differences exist:
- Display rules vary — some cultures suppress emotion display more than others
- The "natural blink rate" range (8–25/min) is fairly universal but the distribution varies
- The Duchenne smile distinction (AU06+AU12) appears consistent across cultures in Ekman's cross-cultural research

The baseline subtraction approach mitigates many demographic differences because it calibrates to the individual rather than to a population norm. However, the tier weights and score thresholds were developed on a predominantly Western talking-head dataset and may need recalibration for other contexts.

### What a high score doesn't guarantee

A high Effortlessness Score means the creator's **face** reads as physically relaxed and authentic. It does not measure:
- How interesting or valuable the content is
- Speaking pace, clarity, or vocal quality
- Editing, thumbnails, or production quality
- Topic expertise or authority
- Authenticity of *claims* (only of *presentation*)

It is one signal among many. Its value is in measuring something that is genuinely hard to see consciously but that audiences respond to subconsciously.

---

# Part II — Operating the Repository

This half of the guide is the runbook. It assumes you have just cloned
the repo and have no prior context. Every command is meant to be
copy-pasteable on macOS or Linux. Windows users should use WSL2.

---

## 13. Repository Overview <a id="13-repository-overview"></a>

### Top-level layout

```
facial_muscle_tension/                          ← repository root
├── README.md                                   ← legacy face-only readme
├── effortlessness_analyzer_guide.md            ← this file
├── CLAUDE_CODE_BUILD_SPEC.md                   ← presence-analyzer design spec
├── SONIC_REBELLION_SPEC.md                     ← audio analyzer design spec
│
├── scripts/
│   ├── setup.sh                                ← one-shot install (use this first)
│   └── dev.sh                                  ← run backend + frontend together
│
├── backend/                                    ← FastAPI server (uvicorn main:app)
│   ├── main.py                                 ← thin shell: app + CORS + include_router x3
│   ├── requirements.txt                        ← Python deps (pinned)
│   ├── .env / .env.example                     ← runtime configuration
│   ├── .venv/                                  ← virtualenv (created by setup.sh)
│   │
│   ├── face/                                   ← MODALITY 1 — facial tension
│   │   ├── routes.py                           ← /api/health, /api/analyse, /api/result, ...
│   │   ├── models.py                           ← Pydantic schemas
│   │   ├── OPTIMIZATION_GUIDE.md               ← perf tuning notes (already deployed)
│   │   └── pipeline/                           ← scoring code (see ch. 8 for detail)
│   │       ├── combined_tracker.py             ← single MediaPipe pass (replaces py-feat)
│   │       ├── extractor.py                    ← video → RGB frames
│   │       ├── baseline.py                     ← per-person calibration
│   │       ├── aggregator.py                   ← weighted ensemble
│   │       ├── insights.py                     ← natural-language insights
│   │       ├── visualization.py                ← per-frame overlay payload
│   │       ├── face_tracker.py / au_detector.py← legacy py-feat path (kept as fallback)
│   │       └── methods/                        ← the 5 scorers
│   │           ├── tension_aus.py
│   │           ├── micro_expr.py
│   │           ├── asymmetry.py
│   │           ├── coherence.py
│   │           └── drift_blink.py
│   │
│   ├── audio/                                  ← MODALITY 2 — Sonic Rebellion
│   │   ├── routes.py                           ← /api/audio/analyse, /api/audio/result/...
│   │   ├── config.py                           ← AUDIO_* tunables (separate from face)
│   │   └── pipeline/
│   │       ├── runner.py                       ← orchestrates the 6 stages
│   │       ├── fetcher.py                      ← Instagram reel + URL fetcher (instaloader)
│   │       ├── extractor.py                    ← video/audio → librosa-friendly arrays
│   │       ├── separator.py                    ← Spleeter vocal / accompaniment split
│   │       ├── vocal_edge.py / genre_edge.py / novelty.py / lite_features.py
│   │       └── scorer.py                       ← weighted aggregation across reels
│   │
│   ├── presence/                               ← MODALITY 3 — pose analyzer (subprocess)
│   │   ├── routes.py                           ← /api/presence/analyse, ...
│   │   └── pipeline/
│   │       └── runner.py                       ← shells out to presence-analyzer CLI
│   │
│   └── data/                                   ← runtime caches (audio templates, etc.)
│
├── presence-analyzer/                          ← standalone uv-managed project (own venv)
│   ├── pyproject.toml                          ← pinned to Python 3.11, mediapipe 0.10.18
│   ├── .venv/                                  ← created by `uv sync` in setup.sh
│   ├── src/presence/                           ← the actual CLI implementation
│   └── README.md                               ← CLI usage docs
│
└── frontend/                                   ← Vite + React app
    ├── package.json
    ├── vite.config.js                          ← proxies /api to localhost:8000
    ├── .env / .env.example                     ← VITE_API_BASE override
    └── src/
        ├── main.jsx                            ← React entrypoint
        ├── App.jsx                             ← thin tab switcher (Face | Audio | Presence)
        ├── index.css                           ← tailwind utilities
        ├── shared/
        │   └── ErrorBoundary.jsx               ← used by every tab
        ├── face/                               ← per-modality folders mirror backend layout
        │   ├── Tab.jsx                         ← stateful tab component
        │   ├── api.js                          ← HTTP client for /api/*
        │   └── components/                     ← Upload, Dashboard, AUBarChart, Timeline, ...
        ├── audio/
        │   ├── Tab.jsx · api.js
        │   └── components/                     ← AudioUpload, AudioDashboard
        └── presence/
            ├── Tab.jsx · api.js
            └── components/                     ← PresenceUpload, PresenceDashboard
```

### Why two Python venvs?

`backend/.venv/` pins **numpy 1.23.5** (a hard ceiling from py-feat's
`nltools` dependency). The presence-analyzer pins **numpy 1.26.4 +
mediapipe 0.10.18**. They cannot coexist in one env. The presence
analyzer therefore lives in its own `presence-analyzer/.venv/` (managed
by `uv`), and the backend reaches it by spawning a subprocess:

```
backend/presence/pipeline/runner.py
    → subprocess(presence-analyzer/.venv/bin/presence analyze ... --json)
```

The subprocess hop costs ~1–2 s cold start; that's why the presence
pipeline timeout default is generous (900 s).

### Why are face_tracker.py and au_detector.py still there?

They implement the legacy py-feat AU detection path. After the
optimization in chapter 20, nothing imports them — `combined_tracker.py`
covers both jobs. They are intentionally retained as a fallback / A/B
reference. You can delete them safely once you've validated the
optimized path on your own footage.

### Where each modality's "score" comes from

| Modality | Final number returned by | Range  | Higher = |
|----------|--------------------------|--------|----------|
| Face     | `aggregator.aggregate_scores()` → `overall_score` | 0–1 | More effortless |
| Audio    | `scorer.aggregate_reel_scores()` → `coolness_score` | 0–1 | More rebellious / less mainstream |
| Presence | `presence-analyzer` CLI → `score` (`score.score`) | 0–100 | More commanding presence |

---

## 14. Prerequisites & Supported Platforms <a id="14-prerequisites"></a>

### Operating system

- **macOS** (Apple Silicon or Intel) — primary dev target.
- **Linux** (Ubuntu 22.04 LTS or similar) — fully supported.
- **Windows** — use WSL2 with an Ubuntu image. Native Windows is **not**
  supported because PyAV, Spleeter, and mediapipe have flaky native
  Windows wheels at the pinned versions.

### System packages

You need these installed *before* running `scripts/setup.sh`:

| Tool        | Why it's needed                                                | Minimum version |
|-------------|----------------------------------------------------------------|-----------------|
| Python      | Backend interpreter (face + audio + presence orchestration)    | **3.10 or 3.11** (not 3.12+) |
| Node.js     | Frontend (Vite, React)                                         | 18 LTS or 20 LTS |
| ffmpeg      | Audio extraction (audio pipeline) + presence standardization   | 4.x or 6.x       |
| git         | Cloning, version control                                       | any              |
| uv          | Manages the separate `presence-analyzer` venv. **Auto-installed by setup.sh** if missing. | latest |

#### macOS install

```bash
# Homebrew is the path of least resistance
brew install python@3.11 node ffmpeg git
# uv is auto-installed by setup.sh, but you can pre-install:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Ubuntu / WSL install

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
                        nodejs npm ffmpeg git build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Why Python 3.10 or 3.11 specifically?

The backend's `requirements.txt` pins:
- `py-feat>=0.6.0` — wheels exist only for 3.10 / 3.11 (its `nltools` dep
  caps numpy at <1.24, which has no 3.12 wheel)
- `mediapipe==0.10.x` — flaky on 3.12+
- `numexpr` (py-feat transitive) — no 3.12 sdist that builds clean

If `python --version` shows 3.12 or higher, **setup.sh will refuse to
proceed** and tell you how to install 3.11. Do not "fix" this by editing
the version check — you will spend the rest of your week fighting native
build failures.

### Hardware

- Face pipeline (optimized): ~15 s per 30-s clip on a 4-core laptop CPU.
- Audio pipeline: Spleeter on CPU is ~2× realtime; budget 30–60 s per
  10-s reel clip. GPU is not used.
- Presence: ~8 s per 60-s clip warm, ~3× longer cold (downloads MediaPipe
  model weights into `~/.cache/presence/models/`).
- RAM: peak ~3 GB during audio analysis (Spleeter stems). 8 GB is enough
  for one job at a time; 16 GB for the default concurrency of 5 face
  jobs.

### Network

- First setup downloads ~2 GB of wheels and ~30 MB of model weights.
  After that the app is fully local — no data leaves the machine at
  runtime.
- The audio pipeline can optionally fetch Instagram reels via
  `instaloader`. That obviously requires internet and is the only
  outbound call the running app makes.

---

## 15. First-Time Setup (step by step) <a id="15-first-time-setup"></a>

### The happy path

```bash
git clone <repo-url> facial_muscle_tension
cd facial_muscle_tension
./scripts/setup.sh
```

That's it. The script:

1. **Locates a supported Python** (3.10 or 3.11). Honours `PYTHON=…` env
   override; checks PATH, common Homebrew paths, pyenv shims.
2. **Creates `backend/.venv/`** with that interpreter.
3. **Pre-installs numpy 1.23.5** so build envs see it (avoids a known
   nltools/opencv resolver deadlock).
4. **Installs `backend/requirements.txt`** with `--prefer-binary` to
   force wheel-only resolution.
5. **Copies `backend/.env.example` → `backend/.env`** if missing.
6. **Bootstraps `uv`** if not on PATH (curl install to `~/.local/bin`).
7. **Runs `uv sync --extra dev` in `presence-analyzer/`** — creates a
   separate Python 3.11 venv there with mediapipe + decord pinned.
8. **`npm install`** in `frontend/`.
9. **Copies `frontend/.env.example` → `frontend/.env`** if missing.

Total runtime: 5–10 minutes on a fast machine the first time. The script
is idempotent — re-running won't re-download anything that's already
present, except when the venv's Python minor version drifts, in which
case it rebuilds the venv automatically.

### If you have a non-standard Python install

```bash
PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 ./scripts/setup.sh
```

### If you don't want the presence analyzer

The presence venv adds ~1 GB of model weights and ~3 minutes of setup.
If you only care about face + audio:

```bash
# Skip the uv step manually:
rm -rf presence-analyzer        # or move it aside
./scripts/setup.sh              # will print a warning and continue
```

The Face and Audio tabs will work; the Presence tab will return errors
when used.

### Verifying setup worked

```bash
source backend/.venv/bin/activate
python -c "import main; print('OK')"
# → OK

cd frontend && npx vite build
# → ✓ N modules transformed.  ✓ built in <2 s

# Presence (only if you set it up):
cd ../presence-analyzer
uv run presence --help
# → prints the typer help screen
```

---

## 16. Running the App (dev + manual) <a id="16-running-the-app"></a>

### Easiest: one command

```bash
./scripts/dev.sh
```

Starts both servers in the foreground:
- Backend on http://localhost:8000  (logs → `backend.dev.log`)
- Frontend on http://localhost:5173 (logs → `frontend.dev.log`)
- Ctrl-C kills both.

> **Important shell gotcha:** the path uses a forward slash, not a dot.
> `./scripts/dev.sh`, not `./scripts.dev.sh`. zsh treats the second form
> as a literal filename and prints "no such file or directory".

### Manual / split-terminal mode (recommended for active dev)

Two terminals:

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

```bash
# Terminal 2 — frontend
cd frontend
npm run dev
```

The `--reload` flag makes uvicorn restart on Python file changes.
Vite already does HMR for React.

### What the three tabs do

| Tab       | Click "Analyse" with     | Backend stages                              | Typical wall-time      |
|-----------|--------------------------|---------------------------------------------|------------------------|
| Face      | A video file (mp4/mov/…) | extract → MediaPipe pass → baseline → 5 scorers (parallel) → aggregate | ~15 s for 30-s clip (optimized) |
| Audio     | Files / Reel URLs / IG profile URL | fetch → extract → Spleeter → vocal/genre/novelty edges → score | ~30 s per reel × N reels |
| Presence  | A video file             | (subprocess) standardize → pose+face → metrics → score | ~8 s for 60-s clip warm |

### Sanity-checking a running backend

```bash
curl http://localhost:8000/api/health | jq
# → { "status": "ok", "target_fps": 8, "au_backend": "mediapipe_geometric", ... }
```

If `au_backend` says `mediapipe_geometric`, the optimized face pipeline
is live (chapter 20). If the key is missing or says `py-feat`, you're
running the legacy code path.

### Stopping cleanly

- `./scripts/dev.sh` — Ctrl-C in the terminal cleans up both processes.
- Manual mode — Ctrl-C each terminal.
- If something is wedged: `lsof -i :8000` and `lsof -i :5173` will show
  the PID; `kill -9 <pid>` if a normal kill doesn't take.

---

## 17. Environment Variable Reference <a id="17-env-vars"></a>

All vars live in `backend/.env` (loaded via python-dotenv) and
`frontend/.env` (loaded by Vite at build/dev start). The defaults in
code apply if the var is unset.

### Backend — global

| Var                | Default              | What it does |
|--------------------|----------------------|--------------|
| `LOG_LEVEL`        | `INFO`               | Standard Python log level. Set to `DEBUG` to see per-frame timings. |
| `CORS_ORIGINS`     | `*`                  | Comma-separated allow-list. Dev sets it to the Vite origin. |

### Backend — Face pipeline (`backend/face/routes.py`)

| Var                       | Code default | `.env` default | Notes |
|---------------------------|:------------:|:--------------:|-------|
| `UPLOAD_DIR`              | `/tmp/effortlessness_uploads` | (same) | Temp dir for uploads. Auto-created. |
| `TARGET_FPS`              | **8**        | **12** (override) | Decode FPS. Lower = faster. The optimized pipeline assumes 8. The shipped `.env` still sets 12 for historical reasons — flip it to 8 to realize the 24× speedup. |
| `MAX_DIM`                 | **480**      | (unset → 480) | Max long-edge resolution for decoded frames. |
| `KEEP_UPLOADS`            | `1`          | `1`            | If `1`, retain the original video so the frontend can play it back behind the AU overlay. Set `0` to delete on completion. |
| `MAX_UPLOAD_MB`           | `500`        | `500`          | Hard cap on a single upload. |
| `PIPELINE_TIMEOUT_SEC`    | `1800`       | (unset)        | Wall-clock budget per job. After this, the job is marked errored. |
| `MAX_CONCURRENT_JOBS`     | `max(4, cpu//2)` | (unset)    | How many videos can analyze in parallel. Auto-detects. |

### Backend — Audio pipeline (`backend/audio/config.py`)

| Var                          | Default | Notes |
|------------------------------|:-------:|-------|
| `AUDIO_UPLOAD_DIR`           | `/tmp/sonic_rebellion_uploads` | Where reels land. |
| `AUDIO_CLIP_DURATION_S`      | `10`    | Snip each reel to this length before analysis. |
| `AUDIO_MAX_REELS`            | `12`    | Max reels per job. |
| `AUDIO_PIPELINE_TIMEOUT_SEC` | `1800`  | Per-job wall clock. |
| `AUDIO_MAX_CONCURRENT_JOBS`  | `1`     | Audio jobs serialise to keep Spleeter memory bounded. |
| `AUDIO_MAX_UPLOAD_MB`        | `1500`  | Total upload cap across all reels in one request. |
| `KEEP_AUDIO_UPLOADS`         | `1`     | Same semantics as face's `KEEP_UPLOADS`. |
| `AUDIO_WHISPER_MODEL`        | `tiny`  | Whisper variant for transcription. `tiny` is enough for WPM estimation. |

Hardcoded in `audio/config.py` (edit the file, not the env):
- `SAMPLE_RATE = 22050` — Spleeter / librosa native rate. Don't change.
- `WEIGHT_NOVELTY/VOCAL/GENRE = 0.35/0.35/0.30` — ensemble weights.
- `FINGERPRINT_THRESHOLD = 0.80` — chromaprint similarity cutoff for
  "this audio appears in the trending pool".

### Backend — Presence pipeline (`backend/presence/routes.py`)

| Var                              | Default | Notes |
|----------------------------------|:-------:|-------|
| `PRESENCE_UPLOAD_DIR`            | `/tmp/presence_uploads` | Where presence uploads land. |
| `PRESENCE_PIPELINE_TIMEOUT_SEC`  | `900`   | Generous because the subprocess does its own model download on first run. |
| `PRESENCE_MAX_CONCURRENT_JOBS`   | `2`     | Two parallel subprocesses keeps memory bounded. |
| `KEEP_PRESENCE_UPLOADS`          | `1`     | Same semantics as face. |

### Frontend (`frontend/.env`)

| Var               | Default | Notes |
|-------------------|---------|-------|
| `VITE_API_BASE`   | empty   | Empty = use Vite's proxy → http://localhost:8000. Set to a full URL if frontend and backend live on different hosts in prod. |

### Suggested `.env` for max performance

```bash
# backend/.env — production-tuned, optimized face path
LOG_LEVEL=INFO
UPLOAD_DIR=/tmp/effortlessness_uploads
TARGET_FPS=8                 # << was 12
MAX_DIM=480                  # << confirms the new default
KEEP_UPLOADS=1
MAX_UPLOAD_MB=500
PIPELINE_TIMEOUT_SEC=600     # 10 min is plenty at 8 fps
MAX_CONCURRENT_JOBS=8        # bump if you have a beefy box
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

---

## 18. API Endpoint Reference <a id="18-endpoints"></a>

All routes live under the backend on `http://localhost:8000`. The
frontend's per-modality `api.js` wraps these.

### Face

| Method | Path                            | Body                       | Response |
|--------|---------------------------------|----------------------------|----------|
| GET    | `/api/health`                   | —                          | `{ status, target_fps, jobs_in_memory, max_concurrent_jobs, pipeline_timeout_sec, au_backend }` |
| POST   | `/api/analyse`                  | `multipart/form-data` field `file` (video) | `{ job_id }` (job starts async) |
| GET    | `/api/result/{job_id}`          | —                          | Job dict; when `status=="done"` includes the full result payload (see chapter 11) |
| GET    | `/api/visualization/{job_id}`   | —                          | Per-frame overlay payload (landmarks + AU intensities) — used by `SimulationPanel.jsx` |
| GET    | `/api/video/{job_id}`           | —                          | Streams the retained upload back (range-aware). 404 if `KEEP_UPLOADS=0`. |
| DELETE | `/api/result/{job_id}`          | —                          | Removes the job and its retained upload. |

### Audio

| Method | Path                            | Body | Response |
|--------|---------------------------------|------|----------|
| POST   | `/api/audio/analyse`            | One of: `files[]` (multipart), `reel_urls` (CSV form field), or `profile_url` (form field). | `{ job_id, n_reels }` |
| GET    | `/api/audio/result/{job_id}`    | — | Job dict; when done includes per-reel sub-scores and aggregated `coolness_score`. |
| DELETE | `/api/audio/result/{job_id}`    | — | Forgets the job. |

### Presence

| Method | Path                              | Body | Response |
|--------|-----------------------------------|------|----------|
| POST   | `/api/presence/analyse`           | `multipart/form-data` field `file` (video) | `{ job_id }` |
| GET    | `/api/presence/result/{job_id}`   | — | Job dict; when done, the augmented CLI JSON (score, components, quality, timings). |
| GET    | `/api/presence/video/{job_id}`    | — | Streams the retained upload back. |
| DELETE | `/api/presence/result/{job_id}`   | — | Forgets the job. |

### Job state machine (all three modalities)

```
"queued" → "processing" → "done"
                       ↘ "error"   (always with a "message" field)
```

The frontend polls `GET /api/.../result/{id}` every 1.5 s until
`status ∈ {done, error}`. Each job dict carries `stage` (human label)
and `progress` (0–1) so the UI can render a progress bar.

### curl quick-test (face)

```bash
JID=$(curl -s -F "file=@/path/to/test.mp4" \
            http://localhost:8000/api/analyse | jq -r .job_id)
echo "job: $JID"

while true; do
  r=$(curl -s http://localhost:8000/api/result/$JID)
  echo "$r" | jq -r '"\(.status) \(.stage) \(.progress)"'
  [[ "$(echo "$r" | jq -r .status)" =~ ^(done|error)$ ]] && break
  sleep 1.5
done

echo "$r" | jq '.result.overall_score, .result.overall_label'
```

### Swagger UI

FastAPI auto-publishes `http://localhost:8000/docs` and `/redoc`. Useful
for manual probing.

---

## 19. Frontend Development <a id="19-frontend-development"></a>

### Stack

- **Vite 5** dev server (HMR), production build via Rollup.
- **React 18** with hooks. No Redux / MobX — local `useState` only,
  state persisted to `localStorage` per modality.
- **Tailwind** utility classes (look at `index.css` for the `card`,
  `bg-bg`, `border-line`, etc. tokens — they're defined once globally).
- **`fetch`** for HTTP; no axios.

### Per-modality structure (mirrors backend)

```
src/face/
├── Tab.jsx              ← orchestration: upload → poll → display
├── api.js               ← uploadVideo / pollResult / getResult
└── components/
    ├── Upload.jsx       ← drag-drop video input
    ├── Dashboard.jsx    ← composes the score, timeline, AU bars
    ├── Timeline.jsx
    ├── AUBarChart.jsx
    ├── MethodScorecard.jsx
    ├── InsightList.jsx
    ├── StatGrid.jsx
    ├── SimulationPanel.jsx   ← per-frame overlay (consumes /api/visualization)
    └── ConfidenceCard.jsx
```

The `Tab.jsx` files are the only stateful components. Each:
1. Reads localStorage on mount to resume an in-flight job after refresh.
2. Calls its modality's `uploadXxx` → gets `job_id`.
3. Polls `pollXxxResult(jobId, onProgress)` until done.
4. Renders `<Dashboard result={...} videoUrl={...} />`.

### State persistence keys

| Modality | localStorage key                  |
|----------|-----------------------------------|
| Active tab | `ui:active_tab`                 |
| Face     | `effortlessness:current_job`      |
| Audio    | `sonic_rebellion:current_job`     |
| Presence | `presence:current_job`            |

Clearing these (DevTools → Application → Local Storage) resets the UI to
the upload screen.

### Vite proxy

`frontend/vite.config.js` proxies anything starting with `/api` to
`http://localhost:8000`. That's why `VITE_API_BASE` is empty by default
— the proxy makes the frontend think the backend is same-origin.

### Common dev-cycle commands

```bash
# Hot-reload dev server
npm run dev

# Production build (run from frontend/ — outputs to frontend/dist/)
npm run build

# Preview the built artefact at http://localhost:4173
npm run preview

# Update a single dep
npm install <pkg>@latest
```

---

## 20. Performance Optimization — Deploying the 24× Speedup <a id="20-performance"></a>

The face pipeline has two code paths shipped in the repo:

1. **Optimized path (currently active)** — `combined_tracker.py` runs one
   MediaPipe pass that produces both `au_df` and `tracking_df`. The five
   scorers run in parallel. py-feat is not loaded.
2. **Legacy py-feat path** — `face_tracker.py` + `au_detector.py` files
   remain in `face/pipeline/` for reference but are no longer imported.

`/api/health` returns `au_backend: "mediapipe_geometric"` when the
optimized path is live. Currently it is.

### Expected speed-ups (from `face/OPTIMIZATION_GUIDE.md`)

| Workload                            | Legacy | Optimized | Gain |
|-------------------------------------|:------:|:---------:|:----:|
| 30 s clip on a quad-core            | ~360 s | ~15 s     | **24×** |
| 12 × 30 s clips, 4 workers          | ~4320 s| ~50 s     | **86×** |
| 12 × 30 s clips, 8 workers          | ~4320 s| ~25 s     | **172×** |

### To realize these numbers on this machine

The optimized code is already in place. The single remaining step is
to update `backend/.env`:

```diff
- TARGET_FPS=12
+ TARGET_FPS=8
+ MAX_DIM=480
+ MAX_CONCURRENT_JOBS=8        # or leave unset to auto-scale
```

Restart `uvicorn`. Verify with:

```bash
curl http://localhost:8000/api/health | jq .target_fps
# → 8
```

### To shed ~2 GB of disk / Docker image

`py-feat`, `torch`, `torchvision` are no longer imported. Removing them
from `backend/requirements.txt` saves ~2 GB and eliminates the
model-download step on cold container starts. Three lines to delete (or
comment out). You'll also want to remove the `nltools` and `pretrainedmodels`
indirect deps if you want a fully clean install, though leaving them is
harmless.

After editing:

```bash
rm -rf backend/.venv
./scripts/setup.sh
```

If you want a chicken-test before deleting them, see
`face/OPTIMIZATION_GUIDE.md` chapter "Step 5 — Test regression".

### Auto-scaling concurrency

`MAX_CONCURRENT_JOBS = max(4, os.cpu_count() // 2)` is the default. On a
16-core machine this defaults to 8 concurrent face jobs. The pipeline is
now CPU-bound on MediaPipe (which releases the GIL), so more concurrency
= more throughput up to the physical core count. If you start seeing
GIL contention above ~8 workers, switch from `ThreadPoolExecutor` to
`ProcessPoolExecutor` in `face/routes.py` — the body is pickle-able.

### Things NOT to optimize (yet)

- **Don't switch the audio pipeline to GPU** — Spleeter on GPU has a
  cold-start cost that exceeds the wall-time savings for the 10 s clips
  this app uses.
- **Don't increase `TARGET_FPS` past 12** for face — the geometric AU
  proxies were tuned at 8 fps. Higher fps just makes more frames to
  process without improving scores (downstream methods all use rolling
  windows that already smooth over the temporal density).

---

## 21. Adding a New Modality <a id="21-adding-a-modality"></a>

To add, say, an "engagement" analyzer alongside Face/Audio/Presence:

### Backend

1. Create `backend/engagement/` with:
   - `__init__.py`
   - `routes.py` exposing a `router = APIRouter()` and any
     `shutdown_executor()` / `warm_models()` helpers.
   - `pipeline/` package for the heavy work.
2. Register in `backend/main.py`:
   ```python
   from engagement import routes as engagement_routes
   app.include_router(engagement_routes.router)
   @app.on_event("shutdown")
   async def _shutdown():
       ...
       engagement_routes.shutdown_executor()
   ```
3. Follow the existing route conventions:
   `POST /api/engagement/analyse`, `GET /api/engagement/result/{id}`,
   `DELETE /api/engagement/result/{id}`. Keep `job_id` as a stringified
   UUID for symmetry.

### Frontend

1. Create `frontend/src/engagement/`:
   - `api.js` — same shape as `face/api.js` (`uploadXxx`, `pollXxxResult`, `getXxxResult`).
   - `Tab.jsx` — copy any existing `Tab.jsx` and swap the modality strings.
   - `components/EngagementUpload.jsx` and `EngagementDashboard.jsx`.
2. Wire it into `App.jsx`:
   ```jsx
   import EngagementTab from "./engagement/Tab.jsx"
   …
   { key: "engagement", label: "Engagement" },
   …
   {tab === "engagement" && <EngagementTab />}
   ```
3. Pick a unique localStorage key (`engagement:current_job`) so the new
   tab can resume in-flight jobs across refreshes.

### What to copy from existing tabs

- `face/Tab.jsx` is the most complete reference (includes a "load by
  job ID" affordance and a videoUrl ref for overlay).
- `audio/Tab.jsx` is simpler (no video preview).
- `presence/Tab.jsx` is the cleanest minimum.

---

## 22. Troubleshooting Common Failures <a id="22-troubleshooting"></a>

### Setup-time

**`✗ Could not find Python 3.10 or 3.11.`**
You're on 3.12+. Install 3.11 (`brew install python@3.11`) or point at a
specific binary: `PYTHON=/path/to/python3.11 ./scripts/setup.sh`.

**`pip install py-feat ... ERROR: building wheel for numexpr failed`**
Same root cause — pip is trying to build numexpr from source because
there's no wheel for your Python version. Drop to 3.11.

**`uv: command not found` (after setup.sh installs it)**
`setup.sh` installs uv to `~/.local/bin/`. If your shell doesn't have
that on PATH, either add it (`export PATH=$HOME/.local/bin:$PATH` in
`~/.zshrc`) or run the presence venv via the absolute path:
`~/.local/bin/uv sync --extra dev`.

**`ffmpeg: command not found`**
Audio + presence pipelines both shell out to ffmpeg. `brew install
ffmpeg` (mac) or `sudo apt-get install ffmpeg` (Linux).

**`presence-analyzer/ not found — skipping`**
You deleted or moved the presence project. Face and Audio still work;
only the Presence tab will return errors. Re-create the dir or restore
from git if you need it.

### Runtime — backend

**`✗ backend/.venv is missing or incomplete.`**
You ran `./scripts/dev.sh` before `./scripts/setup.sh`. Run setup first.

**Job stuck at "queued"**
The pipeline semaphore is held by a previous job that didn't release.
Most often: a previous run is still inside Spleeter (audio) or
MediaPipe's first-call model download (presence). Tail `backend.dev.log`
to confirm; if truly wedged, restart uvicorn.

**Job errors with `Pipeline exceeded PIPELINE_TIMEOUT_SEC=…s`**
The wall clock budget kicked in. Either the input is huge or something
is genuinely hung. Bump the relevant `*_PIPELINE_TIMEOUT_SEC` env var.

**`No frames could be decoded from this video.`**
Container/codec PyAV and OpenCV both fail on. Try transcoding with
`ffmpeg -i in.mov -c:v libx264 out.mp4` first.

**`presence CLI not found at .../presence`**
You skipped the `uv sync` step. From the repo root:
```bash
cd presence-analyzer && uv sync --extra dev
```

**`/api/health` does NOT include `au_backend: "mediapipe_geometric"`**
The optimized face path didn't load. Check that
`backend/face/pipeline/combined_tracker.py` exists and that
`backend/face/routes.py` imports it (chapter 20).

**Backend exits with `ModuleNotFoundError: No module named 'face'`**
You're running `uvicorn main:app` from the wrong working directory.
It must be run from `backend/`. `./scripts/dev.sh` handles this.

### Runtime — frontend

**Blank page in dev, console errors about `/api/...` CORS**
Either the backend isn't running or `frontend/.env` has a stale
`VITE_API_BASE`. Default (empty) routes through Vite's proxy — that's
what you usually want in dev.

**Tab won't change / "Resuming previous run…" loops forever**
Stale localStorage entry pointing at a job_id the backend doesn't know
(e.g., backend restarted, in-memory job store wiped). DevTools →
Application → Local Storage → clear the modality's key.

**"Failed to fetch" on upload**
Backend isn't reachable or the request is hitting upload size limits.
Check `MAX_UPLOAD_MB` and your reverse proxy's body-size limit if any.

### Process management

**`Port 8000 already in use`**
Another uvicorn from a previous run. `lsof -i :8000` to find the PID,
`kill <pid>`. Same drill for port 5173.

---

## 23. Production / Deployment Notes <a id="23-deployment"></a>

The codebase is dev-focused; turning it into a deployable service needs
the following changes (none of which are wired in by default):

### Job store

`jobs`, `audio_jobs`, `presence_jobs`, and `visualizations` are
**in-memory Python dicts**. Restarting the backend loses all jobs. For
multi-worker or multi-machine deployments, swap them for Redis hashes:

```python
# Example sketch
import redis
r = redis.Redis(...)
def get_job(jid): return json.loads(r.hget("face:jobs", jid))
def set_job(jid, d): r.hset("face:jobs", jid, json.dumps(d))
```

### Process model

Each route file constructs a `ThreadPoolExecutor` at import time and
serialises with an `asyncio.Semaphore`. That works for a single uvicorn
process. For multiple uvicorn workers (`--workers 4`), the semaphores
are per-worker — total concurrency = workers × semaphore. Cap workers
to 1 if you want the semaphore to be authoritative, or move to a Celery
worker pool with Redis as the broker:

```yaml
celery_worker:
  command: celery -A face.tasks worker --concurrency=8 --pool=prefork
redis:
  image: redis:7-alpine
```

### Reverse proxy

In front of uvicorn, put nginx / caddy / fly's edge to terminate TLS and
set a generous `client_max_body_size` (1500 MB to allow the audio
endpoint's 12-reel uploads).

### Containerisation

Dockerfile sketch:

```dockerfile
FROM python:3.11-slim AS backend
RUN apt-get update && apt-get install -y ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/backend
COPY backend/requirements.txt .
RUN pip install --prefer-binary numpy==1.23.5 \
 && pip install --prefer-binary -r requirements.txt
COPY backend/ .
ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Presence-analyzer can be packaged similarly into a second image and run
as a separate service that the face service POSTs to, eliminating the
subprocess hop in production. Or keep the subprocess if you'd rather
ship one image.

### Secrets and credentials

The audio pipeline can fetch Instagram reels via `instaloader`. If you
need authenticated fetches (private profiles or rate-limit avoidance),
plumb an `INSTALOADER_USERNAME` / `INSTALOADER_SESSION_FILE` env var
into `audio/pipeline/fetcher.py`. There is no built-in secrets manager.

### Observability

The app uses Python's stdlib `logging` (configured in `main.py` from
`LOG_LEVEL`). For prod, ship logs to a structured collector (loki,
cloudwatch, etc). No metrics endpoint exists — add `prometheus-fastapi-instrumentator`
if you need one.

---

## 24. Hand-Off Checklist <a id="24-hand-off-checklist"></a>

Use this when handing the repo to someone new. Walk it together —
takes about 30 minutes if everything is in order.

### Before the meeting

- [ ] Repo cloned on the new dev's machine.
- [ ] Python 3.10 or 3.11 installed (`python3.11 --version` ≥ 3.11.0).
- [ ] Node 18+ installed (`node --version`).
- [ ] ffmpeg installed (`ffmpeg -version`).
- [ ] `./scripts/setup.sh` ran clean. No red errors. (`backend/.venv/`, `presence-analyzer/.venv/`, `frontend/node_modules/` all exist.)
- [ ] `./scripts/dev.sh` starts both servers. http://localhost:5173 loads.

### Verification — Face tab

- [ ] Drag a short MP4 (5–60 s, single face) onto the Face upload zone.
- [ ] Progress bar advances through stages. Result page appears within
      ~15 s for a 30 s clip.
- [ ] `curl http://localhost:8000/api/health | jq .au_backend` → `"mediapipe_geometric"`.

### Verification — Audio tab

- [ ] Drag one MP4 reel or paste an Instagram reel URL.
- [ ] Reaches the "scoring" stage; final `coolness_score` is in [0, 1].

### Verification — Presence tab

- [ ] Drag the same MP4 used for Face.
- [ ] First run downloads MediaPipe models (~30 MB). Subsequent runs are
      fast.
- [ ] `score.score` in [0, 100] returned.

### Walk through the codebase together

Stop at each:

1. `backend/main.py` — show how thin it is (just three `include_router`).
2. `backend/face/routes.py` lines 109–217 — the pipeline body. Explain
   the 5-method parallel scoring at line 152.
3. `backend/face/pipeline/combined_tracker.py` lines 214–350 — the
   geometric AU estimation. Mention OPTIMIZATION_GUIDE.md for context.
4. `frontend/src/App.jsx` — show that it's just a tab switcher.
5. `frontend/src/face/Tab.jsx` — show the upload → poll → display loop
   and `localStorage` resume.
6. `presence-analyzer/README.md` — explain why it lives in its own venv.

### Documents the new owner should read

- `effortlessness_analyzer_guide.md` (this file) — Part I for science,
  Part II for ops.
- `backend/face/OPTIMIZATION_GUIDE.md` — why the optimized path exists
  and how to verify drift if you ever turn py-feat back on.
- `SONIC_REBELLION_SPEC.md` — design rationale for the audio pipeline.
  Read before changing weights in `audio/config.py`.
- `CLAUDE_CODE_BUILD_SPEC.md` — design rationale for the presence
  analyzer. Read before changing anything in `presence-analyzer/`.

### Known sharp edges to flag

- `backend/.env` ships with `TARGET_FPS=12`. The optimized code is
  tuned for `8`. Bump it if you want the 24× speedup (chapter 20).
- `backend/face/pipeline/face_tracker.py` and `au_detector.py` are
  unused. Delete only after you've A/B-tested the optimized path on
  your own footage.
- `py-feat`, `torch`, `torchvision` in `backend/requirements.txt` are
  unused after the optimization. Removing them shrinks the install by
  ~2 GB. Keep them if you might want to A/B against py-feat.
- The job stores are in-memory. Backend restart = jobs forgotten.
  Frontend's localStorage will then loop "Resuming previous run…" —
  tell the dev to clear their browser storage.
- The presence subprocess does a one-time ~30 MB model download on
  first run into `~/.cache/presence/models/`. Don't kill it.

### Cleanup before commit

If you ran the app and want a clean working tree before handing over:

```bash
rm -rf /tmp/effortlessness_uploads /tmp/sonic_rebellion_uploads /tmp/presence_uploads
rm -f backend.dev.log frontend.dev.log
rm -rf frontend/dist
```

These are all gitignored, but it's nice to start clean.

---

*End of guide. Questions about Part I → ping the original ML lead.
Questions about Part II → start with the troubleshooting chapter, then
the relevant route file, then the relevant spec.*
