# ComfyUI-SpeakerSplit

Split an audio clip containing 2–3 (up to 6) speakers into separate per-speaker
tracks, then extract clean **voice-clone reference clips** so a character keeps a
**consistent voice across multiple video/audio generations**.

Built for talking-head / AI-video pipelines where you generate a clip, isolate a
character's voice, and feed that same reference into every following generation.

- **No HuggingFace token required** (unlike pyannote — no gated model, no login).
- **SpeechBrain ECAPA** embeddings by default for accurate separation, even of
  similar-sounding voices. Strongly recommended — install it and leave it on.
- Still runs with **zero extra packages** (MFCC fallback) if you just want to
  wire things up, but ECAPA is what makes the results actually usable.
- Every widget has an inline tooltip explaining what it does and how to tune it.

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [What happens if packages are missing](#what-happens-if-packages-are-missing)
- [Nodes](#nodes)
  - [Speaker Split (Diarization)](#speaker-split-diarization)
  - [Voice Reference Extract](#voice-reference-extract)
  - [Save Speaker Audio](#save-speaker-audio)
- [Example workflow](#example-workflow)
- [Tuning guide](#tuning-guide)
  - [It only detects one speaker](#it-only-detects-one-speaker)
  - [A voice is torn across multiple tracks](#a-voice-is-torn-across-multiple-tracks)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How it works

The diarization node runs a five-stage pipeline:

1. **Downmix** the input to mono 16 kHz for analysis (the original waveform is
   kept at full sample-rate/channels for the output cut).
2. **Voice Activity Detection** (Silero VAD, or a built-in energy VAD fallback)
   finds the speech regions.
3. **Sliding windows** over the speech are turned into **speaker embeddings**
   — SpeechBrain ECAPA-TDNN when available, otherwise an MFCC-statistics
   fallback.
4. **Agglomerative clustering** (cosine) groups the windows into N speakers.
   With `num_speakers = 0` the count is auto-detected (2–4) via silhouette score.
5. **Frame voting + median smoothing** produces clean per-speaker segments;
   low-confidence / overlapping windows are dropped rather than mis-assigned.
   The segments are projected back onto the original waveform.

Speakers are numbered by talk time — `speaker_1` has the most speech.

---

## Installation

### 1. Clone the node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gothdir/ComfyUI-SpeakerSplit
```

### 2. Install the ECAPA dependencies 

The node *can* run on nothing but the torch / torchaudio / numpy that ship with
ComfyUI (MFCC mode), but that mode only tells clearly different voices apart and
falls over on anything subtle. **ECAPA is what makes this node actually work**,
so treat the step below as part of the normal install, not an optional extra.

**Portable ComfyUI — run these from the
`python_embeded` folder** of your install:

```bat
cd /d <your-path>\python_embeded

python.exe -m pip install speechbrain silero-vad --no-deps
python.exe -m pip install hyperpyyaml joblib huggingface_hub sentencepiece scipy tqdm packaging scikit-learn soundfile
```

**venv / Conda:**

```bash
python -m pip install speechbrain silero-vad --no-deps
python -m pip install hyperpyyaml joblib huggingface_hub sentencepiece scipy tqdm packaging scikit-learn soundfile
```

> [!WARNING]
> **`--no-deps` is mandatory** for `speechbrain` and `silero-vad`.
> Both list `torch` and `torchaudio` as dependencies, and without the flag pip
> will overwrite your CUDA build with CPU-only wheels — after which ComfyUI runs
> on the CPU only. The second command installs the *actual* runtime
> dependencies that SpeechBrain needs.

### 3. Verify torch is still on the GPU

```bat
python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This must print `True` and a `+cu12x` (or `+cu13x`) version. If it prints
`False`, a `--no-deps` was missed — reinstall the matching CUDA torch wheel.

### 4. First run

Leave the node's `embedder` widget on its default (`auto`, which uses ECAPA when
present). On the first ECAPA run the **ECAPA-TDNN model (~80 MB)** downloads
automatically to `ComfyUI/models/speaker_embed/spkrec-ecapa-voxceleb`. No
HuggingFace token, no login.

> [!NOTE]
> Skipping step 2 is only meant for a quick "does it wire up" test. For any real
> use — especially separating similar voices for voice-clone references — install
> ECAPA. See the [tuning guide](#it-only-detects-one-speaker) for why MFCC alone
> collapses similar voices into one speaker.

---

## What happens if packages are missing

If you followed the install above you're on the accurate ECAPA path and can skip
this section. If a dependency is missing the node degrades gracefully instead of
crashing — useful to know, but these fallbacks are **not** where you want to stay:

| Missing package | Behaviour |
| --- | --- |
| `silero-vad` | Automatically uses an energy-based VAD (simpler, less robust with background noise). |
| `speechbrain` | With `embedder = auto`, falls back to MFCC embeddings + a warning. With `embedder = ecapa`, aborts with an install hint. MFCC only separates clearly different voices — install SpeechBrain. |
| `scikit-learn` | Uses a built-in spherical k-means (numpy) plus a numpy silhouette score for auto-detection. |

The `report` output always states which VAD and which embedder were actually
used. If it says `mfcc (Fallback)`, SpeechBrain didn't load — go back to the
[install step](#2-install-the-ecapa-dependencies-do-this--dont-skip-it).

---

## Nodes

### Speaker Split (Diarization)

Splits one multi-speaker `AUDIO` into up to three per-speaker `AUDIO` outputs
plus a text `report`.

| Widget | Default | Range / options | What it does |
| --- | --- | --- | --- |
| `num_speakers` | `2` | 0–6 | Number of speakers. `2`/`3` = exact (much more robust — always prefer). `0` = auto-detect 2–4. Too high tears one voice across tracks; too low merges two people. |
| `output_mode` | `concat (für Voice-Referenz)` | concat / mask | **concat**: only this person's speech, gap-free — ideal for voice cloning (video timing is lost). **mask**: original length, other speakers muted — lip-syncable but full of silence. |
| `window_seconds` | `1.5` | 0.5–4.0 | Analysis window per embedding. Longer = more stable signature for similar voices; shorter = faster speaker switches but fuzzier. |
| `hop_seconds` | `0.75` | 0.1–2.0 | Step between windows. Rule of thumb: half of `window_seconds`. Smaller = sharper boundaries but slower. |
| `min_segment_seconds` | `0.6` | 0.1–5.0 | Segments shorter than this are dropped. Higher = cleaner (removes stray words); lower = keeps quick exchanges but pulls in more cross-talk. |
| `overlap_margin` | `0.06` | 0.0–0.5 | Confidence gap between best and 2nd-best cluster. Windows below it are dropped as uncertain. **This is your sensitivity control** — see tuning below. |
| `vad_threshold` | `0.5` | 0.1–0.9 | Silero VAD sensitivity. Lower catches quiet speech (and noise); higher keeps only clear, loud speech. |
| `pad_seconds` | `0.05` | 0.0–0.5 | Padding added before/after each segment so word onsets aren't clipped. |
| `fade_ms` | `20` | 0–200 | Crossfade at every cut to avoid clicks. |
| `smooth_frames` | `5` | 1–21 (odd) | Median smoothing of the labels, in 100 ms frames. Higher suppresses flicker but makes switches sluggish. |
| `embedder` | `auto` | auto / ecapa / mfcc | Voice-fingerprint method. Keep on `auto` (or `ecapa`) with SpeechBrain installed — that's the recommended setup. `ecapa` = accurate neural embeddings. `mfcc` = no extra packages but only separates clearly different voices (test/fallback only). `auto` = ecapa with automatic MFCC fallback. |

**Outputs:** `speaker_1`, `speaker_2`, `speaker_3` (ordered by talk time; unused
outputs are a 1-sample silence), and `report` — a text summary with total
length, speech time, VAD + embedder used, cluster count, and per-speaker talk
time / segment count / timestamps. Wire `report` into a **PreviewText** node to
sanity-check results.

### Voice Reference Extract

Takes a single-speaker track and cuts the **most loudness-consistent** section of
the requested length, normalizes it, and optionally resamples it — ready to use
as a voice-clone reference.

| Widget | Default | Range / options | What it does |
| --- | --- | --- | --- |
| `target_seconds` | `10.0` | 2.0–60.0 | Reference length. 8–15 s is the sweet spot for most voice-clone models. |
| `normalize_db` | `-3.0` | -30.0–0.0 | Target peak level (dBFS). -3 dB is loud with headroom against clipping. |
| `fade_ms` | `15` | 0–200 | In/out fade to avoid clicks at the cut. |
| `force_mono` | `True` | True / False | Downmix to mono — practically all voice-clone/TTS models expect it. |
| `resample_to` | `0` | 0 / 16000 / 22050 / 24000 / 32000 / 44100 / 48000 | Target sample rate. `0` = unchanged. `24000` is common for XTTS/F5/CosyVoice. |

**Outputs:** `reference` (the cut, normalized clip) and `info` (length, source
start time, sample rate, channels, target level).

### Save Speaker Audio

Saves up to three tracks to `ComfyUI/output/` in one node.

| Widget | Default | Range / options | What it does |
| --- | --- | --- | --- |
| `speaker_1` | — | AUDIO | First track — saved as `..._spk1_`. |
| `filename_prefix` | `speakers/voice` | text | Path + name base under `ComfyUI/output/`. A `/` creates a subfolder. An auto-incrementing counter is appended; nothing is overwritten. |
| `format` | `wav` | wav / flac | `wav` = uncompressed, read by everything. `flac` = lossless, ~half the size. |
| `speaker_2` / `speaker_3` | — | AUDIO (optional) | Extra tracks — empty ones are skipped. |

---

## Example workflow

```
LoadAudio
   └─> Speaker Split (Diarization)   [num_speakers=2, embedder=ecapa, output_mode=concat, overlap_margin=0.02]
         ├─ speaker_1 ─> Voice Reference Extract (10 s, 24000 Hz) ─┬─> Save Speaker Audio  → annie_ref.wav
         │                                                          └─> (voice input of your video node)
         ├─ speaker_2 ─> Voice Reference Extract ──────────────────> laura_ref.wav
         └─ report    ─> PreviewText   (check talk-time split & which embedder ran)
```

**For consistent voices across generations:** build each reference **once**, save
it to a file, and load that same file in every following clip. Do **not**
re-extract on every run — that's exactly what breaks the consistency you want.

---

## Tuning guide

Always wire `report` into a **PreviewText** node first and read the `Embedder:`
line — it decides everything below.

### It only detects one speaker

The voices are different but close together. Work through this in order:

1. **Read the `report`.** If it says `mfcc` or `mfcc (Fallback)`, that is the
   cause — MFCC mostly captures timbre and pitch, so similar voices land in one
   cluster no matter what else you change. Install SpeechBrain and set
   `embedder = ecapa`. Everything below assumes ECAPA.
2. **Set `num_speakers = 2`** (not `0`). Auto-detection uses silhouette score,
   which reliably says "one cluster is enough" for similar voices. Forcing 2
   makes it always split.
3. **Lower `overlap_margin` to 0.01–0.02.** This is the real sensitivity knob.
   Similar voices produce small confidence margins everywhere, so the default
   0.06 throws almost everything away and leaves one apparent speaker. A small
   margin keeps the material and lets clustering decide.
4. **Raise `window_seconds` to 2.0–2.5** and `hop_seconds` to 1.0–1.25 — more
   audio per embedding gives ECAPA a more stable fingerprint. This is the single
   biggest quality gain for similar voices.
5. **Lower `min_segment_seconds` to 0.3** so a speaker with only short
   interjections isn't wiped out.
6. **Lower `smooth_frames` to 3** so the dominant speaker doesn't overrule the
   other's short passages during smoothing.

**Preset for similar voices:**

```
num_speakers        2
embedder            ecapa
window_seconds      2.2
hop_seconds         1.1
min_segment_seconds 0.3
overlap_margin      0.015
vad_threshold       0.4
smooth_frames       3
```

Then check the percentage split in `report`. ~90/10 means clustering still found
only one voice; ~60/40 or more balanced means it worked.

### A voice is torn across multiple tracks

The opposite problem — one person shows up on two tracks:

- **`num_speakers` is too high** — set it to the true count.
- **Raise `window_seconds`** so short bursts don't drift between clusters.
- **Raise `smooth_frames`** (7–9) to stabilize the labels.
- **Raise `overlap_margin`** back toward 0.06–0.10 to discard the uncertain
  boundary windows that cause the flip-flopping.

---

## Limitations

- **Overlapping speech cannot be separated.** This is diarization, not source
  separation — simultaneous talk is discarded (via `overlap_margin`) rather than
  split. For true separation on overlap you'd need a source-separation model
  (e.g. Demucs / MossFormer) upstream.
- **Background music or heavy reverb** dominates the embeddings and hurts
  clustering. Put a vocal-isolation / denoise node before this one if needed.
- **Very similar voices** (same gender, same pitch, or two outputs from the same
  TTS voice) are genuinely hard — ECAPA + longer windows helps, but there's a
  physical limit.
- **Minimum useful material** is roughly 6–10 s of clean speech per speaker.

---

## Troubleshooting

**"SpeechBrain fehlt / missing"** — install it with `--no-deps` (see
[Installation](#installation)), or set `embedder = mfcc` to run without it.

**`torch.cuda.is_available()` is `False` after installing** — pip replaced your
CUDA torch with CPU wheels. Reinstall the matching CUDA torch/torchaudio wheel,
and always use `--no-deps` for speechbrain / silero-vad.

**"Input out of range" / "Invalid input" on the node after an update** — a saved
workflow stored widget values in the old order. Right-click the node →
**Fix node (recreate)** to rebuild it with current defaults (links are kept).

**Everything lands on `speaker_1` (90 %+)** — see
[It only detects one speaker](#it-only-detects-one-speaker).

**First run works, second run errors in an unrelated node** (e.g. Impact-Pack
`Switch (Any)` → *maximum recursion depth exceeded*) — that's a known
state-buildup issue in `*`/Any switches, not this node. Restart the ComfyUI
server; if it recurs, replace the Any-switch with a typed switch or a Reroute on
that branch.

---

## License

MIT — see [LICENSE](LICENSE).

Speaker embeddings use **SpeechBrain**'s `spkrec-ecapa-voxceleb`; VAD uses
**Silero VAD**. Both are downloaded/licensed under their own terms.
