# CirCor murmur isolation

This repository tests whether normal-heart suppression can preserve murmur-related temporal and spectral structure. The residual is called a **murmur candidate** until synthetic ground truth and real-data proxy metrics support a stronger claim.

## Pipeline boundary

- Input is restricted to `~/physionet.org/files/circor-heart-sound/1.0.1/training_data/*.wav`.
- Patient metadata comes from `training_data.csv`.
- Generated arrays, audio, figures, configurations, and reports are written only under this repository's ignored `outputs/` directory.
- Set `CIRCOR_DATASET_ROOT` to override the dataset location.

## Setup and validation

```bash
./setup.sh
python -m src.data_validation
pytest -q
```

Validation checks the 3,163 recordings, 942 patients, filename pairing, readable/non-empty WAVs, sampling rates, TSV states and bounds, duplicate IDs/paths, and metadata recording references. It always writes `outputs/reports/dataset_validation.json`; strict mode exits nonzero when any anomaly is present.

On the audited local snapshot, counts are correct but strict validation intentionally fails: `50782_MV_1.wav` has no exact TSV pair, `50782_MV.tsv` is orphaned and malformed, and five other TSVs contain interval overlap/order errors beyond the allowed 1 ms rounding tolerance. Resolve or explicitly exclude these records before a full experiment. `--skip-dataset-validation` exists only for targeted diagnostic work on already-inspected valid records.

## Real-data audit

```bash
python -m src.separation.audit --limit 20
```

Each selected contiguous S1-systole-S2-diastole cycle is processed separately and receives:

- `original`, `normal_estimate`, `murmur_candidate`, and `noise_candidate` as NPY and WAV;
- `component_features.csv` and `selected_components.json`;
- `metrics.json` and `diagnostic_plot.png`.

The summary is `outputs/reports/separation_summary.csv`. Use `--energy-threshold` to compare 0.95, 0.975, 0.99, and 0.995. DWT is an opt-in ablation via `--use-dwt`, not an assumed improvement.

Real-audit timing is normalized within the annotated systole. S1 and S2
leakage are candidate energy in each heart-sound phase divided by original
energy in that phase. Outside-murmur energy is candidate energy outside the
annotated systole divided by total candidate energy. Use `--run-name` to keep
method outputs and reports isolated during direct comparisons.

When cardiac phase masks are available, provisional murmur components are
filtered by phase-energy density. Components must have sufficient systolic
focus and systole-to-S1/S2 contrast; rejected full-length components return to
the normal-heart estimate, so reconstruction remains exact and leakage is not
hidden by zeroing samples outside systole. If no component passes, the most
systole-focused provisional component is retained and
`phase_selection_used_fallback` is recorded for low-confidence review.
The audit also writes `candidate_quality_status` and separate all-segment,
accepted-only, and quality-stratified reports so fallback candidates cannot be
silently mixed into aggregate separation metrics.
Use `--disable-phase-aware-selection` for a baseline ablation, or adjust
`--minimum-systole-focus` and `--minimum-systole-to-s1-s2-ratio` in explicit
threshold studies. `--disable-phase-selection-fallback` exposes cases where no
provisional component satisfies the phase criteria.

For a restartable scan of every exact WAV/TSV pair, first run a small pilot and
then expand the same configuration:

```bash
python -m src.separation.audit --all-recordings --limit 20 --cycles-per-recording 3 --method auto --use-dwt --output-profile summary --run-name cssa_auto_dwt_all_recordings_pilot --skip-dataset-validation
python -m src.separation.audit --all-recordings --limit 0 --cycles-per-recording 3 --method auto --use-dwt --output-profile summary --run-name cssa_auto_dwt_all_recordings --resume --skip-dataset-validation
```

`--limit 0` means all recordings and `--cycles-per-recording 0` means all valid
cycles. The `summary` profile avoids tens of thousands of per-cycle audio and
plot files. A checkpoint is written after every recording, and `--resume`
skips completed recording/cycle pairs while rejecting a checkpoint created by
a different method, output profile, scope, or separation configuration.

The audit exports `murmur_observations_<run>.csv` only from location-aware
`Present` recordings whose candidates pass the quality gate. It includes
normalized and absolute onset/offset, amplitude envelope and RMS statistics,
dominant frequency and spectral shape, PSD band-energy ratios, and
time-frequency peak/entropy/flux. The grouped means and medians are written to
`murmur_observation_summary_<run>.csv`. These describe an estimated murmur
candidate, not clean-source ground truth. Amplitude is relative to each WAV's
digital full scale and should not be interpreted as calibrated sound pressure
or compared clinically across recording devices.

Phase-aware candidates are tuned by the full-cycle synthetic ground-truth grid
rather than the real audit set. Run
`python -m src.evaluation.synthetic_benchmark --tune-phase-thresholds` to
reproduce `phase_threshold_tuning*.csv` and the selected JSON candidate. A
candidate configuration is adopted only if it also passes the one-shot frozen
real-audit acceptance check; otherwise the established defaults remain locked.

## Synthetic ground truth

```bash
python -m src.evaluation.synthetic_benchmark
# Fast smoke run:
python -m src.evaluation.synthetic_benchmark --quick
```

The benchmark covers eight murmur timing/shapes and sweeps mixture ratios, SNRs, and seeds. It reports SI-SDR, SDR, SNR improvement, correlations, spectral/envelope error, onset/offset error, and cross-source leakage to `outputs/reports/synthetic_benchmark.csv`.
Ground-truth mixtures and per-method estimated stems/metrics are retained under `outputs/synthetic/` for direct inspection.

## Research order

Do not tune the classifiers until separation achieves low reconstruction and S1/S2 leakage, strong murmur-region preservation, stable negative controls, plausible timing, and credible synthetic source metrics. See `docs/separation_pipeline_audit.md` for the baseline audit and affected functions.

The frozen 59-segment quality-gated evaluation currently fails separation
acceptance (27/36 present candidates accepted and 6/15 absent negative controls
clear). Treat the generated candidates as audit artifacts, not validated murmur
sources; the detailed synthetic-to-real decision is recorded in the audit doc.
