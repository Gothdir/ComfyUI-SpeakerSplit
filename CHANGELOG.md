# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-08-14

### Added
- **Speaker Split (Diarization)** node: VAD → speaker embeddings → clustering →
  frame voting, splitting audio into up to 3 per-speaker tracks plus a report.
- **Voice Reference Extract** node: cuts the most loudness-consistent section of
  a single-speaker track, normalizes and optionally resamples it for voice cloning.
- **Save Speaker Audio** node: saves up to three tracks to `output/` at once.
- ECAPA-TDNN embeddings via SpeechBrain (no HuggingFace token required).
- Graceful fallbacks with no extra packages: energy VAD, MFCC embeddings,
  numpy spherical k-means.
- `auto` embedder mode that uses ECAPA when available and falls back to MFCC.
- Inline tooltips on every widget and a diagnostic `report` output.
