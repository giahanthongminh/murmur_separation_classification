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

Because isolation is performed on a systolic-only segment, complete S1 and S2 intervals are outside that segment. The minimum metrics schema therefore exports `s1_leakage_ratio` and `s2_leakage_ratio` as unavailable rather than falsely reporting zero. A future full-cycle evaluator should estimate those metrics from adjacent annotated states without concatenating cardiac phases before SSA.
