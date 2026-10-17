# Model card: beat-level screening models

**Intended use:** research and teaching on low-cost IoMT ECG screening. **Not** for diagnosis,
triage or any clinical decision. Every output is a preliminary screening signal.

## Task

Each detected heartbeat is classified into one of four AAMI EC57 beat classes:

| Class | Meaning | MIT-BIH symbols |
|---|---|---|
| N | normal and bundle-branch-block beats | N, L, R, e, j |
| S | supraventricular ectopic beats | A, a, J, S |
| V | ventricular ectopic beats | V, E |
| F | fusion of ventricular and normal beats | F |

Q (paced and unclassifiable) is **not supported**: MIT-BIH has only 15 such beats outside
the paced records, which is too few to train or test on. These are **beat-level** labels.
They do not describe a rhythm, and they are not a recording-level diagnosis.

A window-level flag ("possible arrhythmia pattern") is raised when at least 3 beats in the
analysis window are classified S or V. This threshold is a research choice. It was not
validated against rhythm annotations.

## Data

- **MIT-BIH Arrhythmia Database** (PhysioNet, ODC-By): 48 half-hour two-lead records from 47
  subjects, 360 Hz. Lead MLII, resampled to 250 Hz to match the hardware target.
- **Inter-patient split** (de Chazal et al., 2004): 22 DS1 records for training and 22 DS2
  records for testing. One record is one patient, so no patient appears in both sets. The
  four paced records (102, 104, 107, 217) are excluded.
- **End-to-end beats:** beats come from this project's own causal detector, not from the
  reference annotations. A detected beat takes the label of the reference beat within
  150 ms. Detections with no reference match are excluded and counted: 57 in DS2. Reference
  beats the detector missed cannot be classified, and they show up in the detection scores.
  Each beat also needs 8 previous beats of history, which skips 198 beats.

| Split | N | S | V | F |
|---|---|---|---|---|
| DS1 (train) | 45,580 | 905 | 3,669 | 410 |
| DS2 (test) | 44,007 | 1,818 | 3,134 | 382 |

## Models

| | Model | Inputs |
|---|---|---|
| A | Heart-rate rules | RR intervals (sustained HR above or below configurable thresholds) |
| B | Random Forest, 300 trees, balanced class weights | 15 causal features |
| C | XGBoost, 400 rounds, depth 6, balanced sample weights | 15 causal features |
| D | 1D-CNN (3 conv blocks, 16-32-32 channels, 7,300 trainable parameters) | 124-sample beat window and 4 RR inputs |
| E | Hybrid: XGBoost on the CNN's 32-unit embedding plus the 15 features | both |

**Features are causal.** They use RR intervals up to the current beat and the signal up to
0.25 s after its R peak, so each beat is classified 0.25 s after it occurs. Training and
live inference call the same extractor (`src/ecg/features.py`) on causally filtered
signals. A test checks that changing later samples never changes an earlier beat's
features.

Seed 42 is used throughout. Rerunning `scripts/train_models.py` reproduced every metric
below exactly.

## Results on DS2 (held-out patients)

| Model | Macro-F1 | Grouped CV macro-F1 (DS1) | Accuracy | Balanced acc. | ROC-AUC (OvR macro) | Latency per beat | Size |
|---|---|---|---|---|---|---|---|
| B Random Forest | 0.483 | 0.468 ± 0.086 | 0.933 | 0.483 | 0.930 | 5.50 ms | 11.0 MB |
| C XGBoost | 0.502 | 0.476 ± 0.093 | 0.898 | 0.527 | 0.900 | 0.27 ms | 1.3 MB |
| D 1D-CNN | 0.596 | not run | 0.774 | 0.713 | 0.917 | 0.10 ms | 0.03 MB |
| E Hybrid | 0.654 | not run | 0.925 | 0.733 | 0.957 | 0.39 ms | 1.1 MB |

Sensitivity / specificity per class:

| Model | N | S | V | F |
|---|---|---|---|---|
| B Random Forest | 0.980 / 0.564 | **0.035** / 0.998 | 0.916 / 0.993 | **0.003** / 0.990 |
| C XGBoost | 0.936 / 0.615 | **0.119** / 0.986 | 0.914 / 0.984 | **0.139** / 0.968 |
| D 1D-CNN | 0.775 / 0.853 | 0.574 / 0.989 | 0.903 / 0.986 | 0.602 / 0.812 |
| E Hybrid | 0.944 / 0.803 | 0.515 / 0.987 | 0.944 / 0.994 | 0.529 / 0.964 |

Positive predictive value for F stays very low for every model (0.002–0.103): most beats
flagged F are not fusion beats.

### Run-to-run variation (CNN models)

Cross-validation was not run for D and E (CPU budget). Instead, both were retrained with
different seeds and scored on DS2 (`scripts/seed_variance.py`):

| Model | Seed 42 (saved model) | Seed 1 | Seed 2 | Seed 3 | Mean ± SD, seeds 1–3 |
|---|---|---|---|---|---|
| D 1D-CNN, macro-F1 | 0.596 | 0.426 | 0.532 | 0.563 | 0.507 ± 0.072 |
| D 1D-CNN, S sensitivity | 0.574 | 0.013 | 0.284 | 0.254 | |
| E Hybrid, macro-F1 | 0.654 | 0.582 | 0.545 | 0.567 | 0.564 ± 0.019 |
| E Hybrid, S sensitivity | 0.515 | 0.279 | 0.240 | 0.260 | |

**The seed-42 models are at the favourable end of this range.** Differences between models
that are smaller than this spread are not evidence that one is better. This prototype makes
**no claim** that the hybrid model is superior.

### What the numbers mean

- **V beats** are recognised reasonably well by every model (sensitivity 0.90–0.94).
- **S beats** are mostly missed by every model. The tree models catch 4–12%. The CNN
  models usually catch about a quarter (24–28% for the hybrid across seeds). The saved
  seed-42 models catch about half, but that is an outlier: the CNN caught as few as 1%
  with another seed.
- **F beats** are rare and poorly recognised by every model.
- **Accuracy is misleading** here: about 89% of test beats are N. Use macro-F1, balanced
  accuracy and per-class figures.
- All of these results are far below what any clinical use would require.

## R-peak detection (Pan–Tompkins, causal live settings, 150 ms tolerance)

| Records | Sensitivity | PPV | F1 |
|---|---|---|---|
| DS1 | 0.9949 | 0.9888 | 0.9919 |
| DS2 | 0.9967 | 0.9988 | 0.9978 |

Records 207 (ventricular flutter) and 208 (frequent ventricular beats) are the hardest.

## Heart-rate rules (Model A)

MIT-BIH has no tachycardia or bradycardia labels. The rule is checked for **agreement**:
the same rule is applied to detected beats and to reference-annotated beats in 660
one-minute DS2 windows. Agreement is **99.4%**. This measures the detection and RR chain.
It is not diagnostic accuracy.

## Explanations

- Global: permutation importance on held-out DS2 beats (drop in macro-F1). For RF and
  XGBoost, the RR prematurity ratio, the preceding RR interval and similarity to recent
  beats rank highest. That matches how ectopic beats are recognised clinically, but it is an
  association and does not show the model reasons like a clinician.
- Local: exact TreeSHAP for XGBoost and the hybrid (built into XGBoost); occlusion for
  Random Forest when `shap` is not installed (approximate, not additive); integrated
  gradients for the CNN (exploratory saliency).
- Reliability warnings accompany every explanation: poor signal, few beats, inputs outside
  the training range, simulated input, and placeholder hardware calibration.

## Known limitations

- Trained on 22 patients from one 1970s–80s Holter dataset with lead MLII. The AD8232 lead
  placement, amplitude calibration and noise are different. Expect a distribution shift.
  The dashboard warns when features fall outside the training range.
- The ESP32 ADC is uncalibrated in this prototype (`AdcCalibration` holds placeholders).
- Nothing in this project has been validated on live AD8232 data.
- Metrics on SIMULATED data are never reported as performance.

## Changes made after first viewing DS2 results

To keep the record honest, these are the changes made after DS2 results had been seen:

1. **Rejected feature experiment.** Three per-patient normalised features were tried. The
   decision used DS1 cross-validation only: RF 0.460 → 0.467 (± 0.049), XGB 0.473 → 0.473.
   With no improvement, the change was reverted.
2. **Detector bug fix.** In causal mode, R peaks were located on a delayed internal
   filter, so every beat was timestamped a few tens of ms late. The fix places R on the
   signal itself. It improved detection (DS1 sensitivity 0.9889 → 0.9949; record 208
   0.878 → 0.992) and changed the features. XGBoost's DS2 macro-F1 moved from 0.553 to
   0.502, while its DS1 cross-validation score was unchanged (0.473 → 0.476). The fix was
   kept because the earlier timing was wrong.
3. **RR rule bug fix.** One borderline premature beat could become a stuck reference and
   flag every later interval. Replacing it with a running median raised rule agreement from
   0.953 to 0.994. Classifier features do not use these flags.
