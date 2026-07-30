# Separation pipeline audit

Audit baseline: commit `112e302` on the `gia-han` branch.

## Executive summary

The repository correctly moved from concatenated systolic intervals to per-segment processing, but the residual is not yet a verified murmur stem. The current implementation can route omitted SSA content, normal-heart detail, reconstruction error, and noise into a signal named `murmur`. Classification results therefore cannot establish separation quality.

The local CirCor copy confirms the boundary risk: `training_data/` contains exactly 3,163 WAV recordings and 3,163 TSV annotations for 942 metadata rows, while recursive scanning of the full `1.0.1/` tree finds 4,379 WAV files.

Strict filename/content validation also found issues hidden by the equal totals: `50782_MV_1.wav` lacks an exact TSV pair, `50782_MV.tsv` is an orphan containing only the invalid row `0 0 28`, and five additional annotations contain overlaps or order discontinuities beyond a 1 ms rounding tolerance (`50150_MV`, `50690_MV_2`, `50690_TV`, `84851_PV`, and `84930_AV`). The full pipeline must exclude or repair these cases rather than reporting that all files are paired.

## Path and data-loading audit

| Area | Files/functions | Current behavior at audit baseline | Risk |
| --- | --- | --- | --- |
| Dataset root | `prepare_labels.py`; feature, spectrogram, separation, and classification scripts | Repeats `Path.home()/physionet.org/.../1.0.1` | Conflicting paths and no single validated boundary |
| Input discovery | `run_separation_per_segment.py` | Uses direct `training_data/*.wav`, which is correct | No count, pairing, readability, or metadata-reference validation before processing |
| Historical/manual tests | `test_data.py`, `test_ssa.py`, `test_zcr.py`, `test_kurtosis.py`, `test_compare.py`, `test_dwt.py` | Uses `/Users/danggiahan/Documents/heart_sounds/wav` | Machine-specific, non-automated, and not suitable as regression tests |
| Generated output | `run_separation_per_segment.py`, feature and spectrogram scripts | Writes under the dataset root (`output_per_seg`, CSVs, spectrograms) | Input/output contamination and accidental recursive ingestion |
| Labels | `prepare_labels.py` and consumers | Creates and reads `labels.csv` under the dataset root | Mutates the source dataset and spreads derived state across input storage |

## Separation-flow audit

The active path is:

`WAV -> TSV systole intervals -> each interval separately -> SSA -> CSSA-ZCR and CSSA-kurtosis -> correlation-only choice -> DWT normal estimate -> original minus estimate`

Per-interval processing in `run_separation_per_segment.get_systolic_segments`, `separate_segment`, and `process_file` is the strongest part of the current design and must be preserved.

### SSA

- `src/ssa.py:ssa_decompose` defaults to exactly 20 reconstructed components.
- The caller subtracts only selected reconstructed content from the original signal.
- Any energy in omitted components is therefore silently included in the residual.
- No input checks, explained-energy criterion, singular values, selected energy, or all-component reconstruction error are exposed.

### CSSA-ZCR

- `src/cssa.py:cssa_zcr` uses a fixed ZCR threshold of `0.05` by default.
- The threshold is repeated in `compare_cssa_methods`, `run_separation_per_segment.py`, and manual scripts.
- Component energy, spectral plausibility, cardiac timing, and noise/artifact assignment are not considered.

### CSSA-kurtosis

- `src/cssa.py:cssa_kurtosis` uses a deterministic GA seed, which is good.
- Its fitness maximizes only excess kurtosis of the combined selected components.
- It can prefer clicks, spikes, motion artifacts, or a tiny high-peaked subset because energy retention and impulse/frequency penalties are absent.

### Selection and DWT

- `src/cssa.py:compare_cssa_methods` chooses the method with the smallest absolute normal/residual correlation.
- Low correlation alone does not imply source quality and can reward a nearly empty residual.
- `src/dwt_refine.py:dwt_refine` is generic wavelet denoising with fixed `db4`, level 4, universal threshold, and soft thresholding.
- `run_separation_per_segment.separate_segment` always applies DWT without an ablation or evidence that it improves separation.

### Naming and outputs

- Variables and artifacts use `murmur` and `seg_*_murmur.npy`, implying a verified source.
- Only a residual and sparse timing metadata are saved.
- There is no explicit noise/artifact candidate, component table, reconstruction audit, run configuration, configuration hash, proxy metrics, or diagnostic plot.

## Feature and classification audit

- `extract_features_rich.py` concatenates already-separated per-cycle residuals only after separation. This does not corrupt SSA itself, but it removes explicit cycle identity before feature aggregation.
- Original and separated feature paths are dataset-owned rather than project-owned.
- Classification scripts run at import time in several files and use duplicated paths.
- Patient grouping is used in the main classifiers, which helps prevent patient leakage.
- Classifier tuning must remain paused until synthetic and real proxy separation results are acceptable.

## Refactor boundaries

The refactor preserves the historical functions where practical while adding:

1. one project-level configuration module;
2. strict dataset validation and a saved JSON report;
3. all-component SSA reconstruction auditing plus configurable explained-energy selection;
4. explicit normal, murmur-candidate, and noise/artifact assignments;
5. configurable ZCR, constrained kurtosis, optional DWT, and deterministic seeds;
6. per-segment artifacts, component features, metrics, plots, and saved configuration;
7. a synthetic ground-truth benchmark and automated regression tests.

Legacy classifier experiments are retained. Their paths are redirected to project outputs; they are not used to decide whether separation is valid.

The repaired real-data audit processes each contiguous S1-systole-S2-diastole cycle without concatenating disjoint intervals. Timing is detected and normalized only within the systolic mask. `s1_leakage_ratio` and `s2_leakage_ratio` measure candidate energy in each heart-sound phase relative to original phase energy, while `outside_murmur_energy_ratio` measures the fraction of total candidate energy outside the annotated systole. This supplies full-cycle leakage context while retaining cycle identity.

Adjacent phase boundaries that differ by no more than the dataset validator's
1 ms annotation tolerance are normalized to one shared sample boundary. Larger
gaps or overlaps remain invalid and are skipped rather than silently repaired.

Phase-aware component selection now augments ZCR or kurtosis assignment with
per-phase energy density. A provisional murmur component must meet configured
systolic-focus and systole-to-S1/S2 thresholds. Rejected components are assigned
whole to the normal-heart estimate rather than truncated at phase boundaries,
preserving exact reconstruction and allowing the leakage metrics to remain an
honest audit. A single best-component fallback keeps short-candidate timing
measurable when no component passes and is exposed as low confidence in the
component table and summary metrics.

Phase thresholds and SSA window duration are tuned only on synthetic
full-cycle mixtures with known normal/murmur/noise stems and absent-murmur
controls. The selection rule first keeps configurations within 0.5 dB of the
best mean murmur SI-SDR, then uses fallback rate, absent false-candidate energy,
normal leakage, outside-systole energy, and retention as ordered tie-breakers.
The frozen real 59-segment audit is evaluation-only and is not used to choose
these settings.

## Frozen 59-segment acceptance decision

The full synthetic grid evaluated 3,264 separations across 48 phase/window
configurations. Its candidate (`20 ms` SSA window, systolic focus `0.06`, and
systole-to-S1/S2 ratio `0.30`) improved synthetic mean murmur SI-SDR from
`2.58 dB` to `4.81 dB` relative to the established defaults. On the one-shot
frozen real audit, however, present-recording fallback increased from `9/36`
to `11/36` and present accepted-candidate retention decreased. The candidate
was therefore rejected and the prior defaults remain in place.

With the retained defaults, label-aware quality gating reports:

- present candidate accepted: `27/36` cycles (75%);
- present candidate missed: `9/36` cycles (25%);
- absent negative control clear: `6/15` cycles (40%);
- absent candidate flagged: `9/15` cycles (60%);
- one annotation-invalid cycle skipped, leaving 59 evaluated segments.

Among present accepted candidates, mean S1 leakage is `0.0427`, mean S2
leakage is `0.0580`, outside-systole energy is `0.6377`, and systolic retention
is `0.1766`. These results do not pass separation acceptance. The reporting
and tuning infrastructure is usable, but the current CSSA component-selection
algorithm must not be treated as a validated murmur source or used to expand
classifier experiments.

## Full-dataset observation export

The audit can now enumerate every exact WAV/TSV pair instead of selecting one
representative location per patient. Present-patient recordings are scored as
`Present` only at locations listed in `Murmur locations`; other locations are
kept as `Unknown` rather than incorrectly used as either positive or negative
controls. On the inspected snapshot this yields 3,162 exact pairs because the
known `50782_MV_1.wav` mismatch is excluded.

Long runs have an atomic per-recording checkpoint and a lightweight `summary`
profile. Resume validates the configuration hash, requested method, recording
scope, and output profile before reusing rows. Invalid cardiac cycles are
skipped with their reason saved separately rather than terminating the batch.

For accepted candidates at location-aware Present recordings, observation
features are calculated inside the detected systolic activity interval:

- onset and offset relative to systole, the cardiac-cycle context, and the
  original recording;
- peak, RMS, mean absolute, envelope, and crest-factor amplitude;
- dominant frequency, centroid, bandwidth, and spectral entropy;
- Welch PSD peak and energy fractions in six frequency bands;
- spectrogram peak time/frequency, time-frequency entropy, and spectral flux.

PSD frequency resolution and spectrogram frame count/window resolution are
exported beside those values. A one-frame spectrogram is retained for audit but
must not be interpreted as evidence of frequency evolution over time.

`adaptive_envelope` and `energy_quantile_fallback` timing results remain
explicitly separated, and fallback separation candidates are excluded from the
observation export. These measurements support descriptive observation and
interpretation; they do not establish a clean murmur ground truth.
