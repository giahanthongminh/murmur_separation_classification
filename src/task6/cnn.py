"""Train and evaluate the fixed compact Task 6 1D CNN on a prepared cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import random
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import OUTPUT_ROOT
from src.task6.common import (
    TASK6_VERSION,
    aggregate_cycle_probabilities,
    assert_patient_disjoint,
    classification_metrics,
    git_value,
    make_inner_validation,
    patient_bootstrap_intervals,
    sha256_file,
    sha256_json,
    verify_manifest_artifacts,
)


TOOL_VERSION: Final = f"{TASK6_VERSION}-compact-1d-cnn"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task6_evaluation"


@dataclass(frozen=True)
class CNNConfig:
    seed: int = 20260827
    input_samples: int = 2048
    maximum_training_cycles_per_patient: int = 8
    batch_size: int = 32
    maximum_epochs: int = 60
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    inner_validation_fraction: float = .20
    bootstrap_replicates: int = 2000

    def __post_init__(self) -> None:
        if self.input_samples < 128:
            raise ValueError("input_samples is too small")
        if self.maximum_training_cycles_per_patient < 1:
            raise ValueError("training cycle cap must be positive")
        if self.maximum_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epoch and patience values must be positive")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def frame_signal(signal: np.ndarray, length: int) -> np.ndarray:
    """Deterministically center-crop long systoles and right-pad short ones."""

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(values) > length:
        start = (len(values) - length) // 2
        return values[start : start + length]
    return values


def deterministic_patient_sample(index: pd.DataFrame, *, maximum: int, seed: int) -> pd.DataFrame:
    """Cap cycles per patient without using labels, QA, recording, or location."""

    rows = []
    for patient_id, group in index.groupby("patient_id", sort=True):
        ordered = group.sort_values("candidate_id", kind="mergesort")
        rng = np.random.default_rng(seed + sum(map(ord, str(patient_id))))
        positions = np.sort(rng.choice(len(ordered), size=min(maximum, len(ordered)), replace=False))
        rows.append(ordered.iloc[positions])
    return pd.concat(rows, ignore_index=True)


def fit_signal_normalization(index: pd.DataFrame, cache: Path, input_samples: int) -> tuple[float, float]:
    """Fit scalar waveform normalization using training rows only."""

    total, squared, count = 0.0, 0.0, 0
    for relative in index["signal_path"]:
        values = frame_signal(np.load(cache / relative), input_samples).astype(np.float64)
        total += float(values.sum())
        squared += float(np.square(values).sum())
        count += len(values)
    if count == 0:
        raise ValueError("cannot fit signal normalization on an empty training set")
    mean = total / count
    variance = max(squared / count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))


def verify_cached_signals(index: pd.DataFrame, cache: Path) -> None:
    """Verify every candidate payload against its recorded SHA-256 before use."""

    for row in index.itertuples(index=False):
        path = Path(cache) / str(row.signal_path)
        if not path.exists() or sha256_file(path) != str(row.signal_sha256):
            raise ValueError(f"cached signal hash mismatch: {path}")


class CandidateDataset(Dataset):
    def __init__(self, index: pd.DataFrame, cache: Path, *, input_samples: int, mean: float, std: float):
        self.index = index.reset_index(drop=True)
        self.cache = Path(cache)
        self.input_samples = input_samples
        self.mean, self.std = float(mean), float(std)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.index.iloc[item]
        values = frame_signal(np.load(self.cache / row["signal_path"]), self.input_samples)
        normalized = (values - self.mean) / self.std
        framed = np.zeros(self.input_samples, dtype=np.float32)
        framed[: len(normalized)] = normalized
        target = 1 if row["clinical_outcome"] == "Abnormal" else 0
        return torch.from_numpy(framed[None, :]), torch.tensor(target, dtype=torch.long), item


class CompactWaveformCNN(nn.Module):
    """A fixed small network suitable for the 179-patient cohort."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 9, stride=1, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, stride=1, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.30), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probability = np.empty(len(loader.dataset), dtype=float)
    for values, _, positions in loader:
        output = torch.softmax(model(values.to(device)), dim=1)[:, 1].cpu().numpy()
        probability[positions.numpy()] = output
    return probability


def _patient_validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute early-stopping log loss after equal-weight patient aggregation."""

    probability = _predict(model, loader, device)
    frame = loader.dataset.index[["patient_id", "clinical_outcome"]].copy()
    frame["probability_abnormal"] = probability
    patient = aggregate_cycle_probabilities(frame)
    target = patient["clinical_outcome"].map({"Normal": 0.0, "Abnormal": 1.0}).to_numpy()
    clipped = np.clip(patient["probability_abnormal"].to_numpy(), 1e-7, 1 - 1e-7)
    return float(np.mean(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))))


def _class_weights(train_index: pd.DataFrame, device: torch.device) -> torch.Tensor:
    patient_labels = train_index[["patient_id", "clinical_outcome"]].drop_duplicates("patient_id")
    counts = patient_labels["clinical_outcome"].value_counts()
    weights = [1.0 / counts["Normal"], 1.0 / counts["Abnormal"]]
    scale = 2 / sum(weights)
    return torch.tensor([value * scale for value in weights], dtype=torch.float32, device=device)


def train_fold(
    *, cache: Path, train_index: pd.DataFrame, validation_index: pd.DataFrame,
    test_index: pd.DataFrame, fold_directory: Path, outer_fold: int,
    config: CNNConfig, device: torch.device, resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_determinism(config.seed + outer_fold)
    sampled_train = deterministic_patient_sample(
        train_index, maximum=config.maximum_training_cycles_per_patient,
        seed=config.seed + 7919 * outer_fold,
    )
    mean, std = fit_signal_normalization(sampled_train, cache, config.input_samples)
    datasets = {
        "train": CandidateDataset(sampled_train, cache, input_samples=config.input_samples, mean=mean, std=std),
        "validation": CandidateDataset(validation_index, cache, input_samples=config.input_samples, mean=mean, std=std),
        "test": CandidateDataset(test_index, cache, input_samples=config.input_samples, mean=mean, std=std),
    }
    generator = torch.Generator().manual_seed(config.seed + outer_fold)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = CompactWaveformCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_index, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3)
    fold_directory.mkdir(parents=True, exist_ok=True)
    last_path, best_path = fold_directory / "last_checkpoint.pt", fold_directory / "best_checkpoint.pt"
    start_epoch, best_loss, best_epoch, patience, history = 0, float("inf"), -1, 0, []
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_loss, best_epoch, patience = float(state["best_loss"]), int(state["best_epoch"]), int(state["patience"])
        history = list(state["history"])

    for epoch in range(start_epoch, config.maximum_epochs):
        model.train()
        training_total, training_count = 0.0, 0
        for values, target, _ in train_loader:
            values, target = values.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), target)
            loss.backward()
            optimizer.step()
            training_total += float(loss.item()) * len(target)
            training_count += len(target)
        validation_loss = _patient_validation_loss(model, validation_loader, device)
        scheduler.step(validation_loss)
        history.append({
            "outer_fold": outer_fold, "epoch": epoch,
            "training_loss": training_total / training_count,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = validation_loss, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss}, best_path)
        else:
            patience += 1
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_loss": best_loss, "best_epoch": best_epoch,
            "patience": patience, "history": history,
        }, last_path)
        print(f"Fold {outer_fold + 1}: epoch {epoch + 1}, train={history[-1]['training_loss']:.4f}, val={validation_loss:.4f}, best={best_epoch + 1}")
        if patience >= config.early_stopping_patience:
            break
    if not best_path.exists():
        raise RuntimeError(f"fold {outer_fold} did not create a best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    probability = _predict(model, test_loader, device)
    cycle_predictions = test_index.copy()
    cycle_predictions["outer_fold"] = outer_fold
    cycle_predictions["probability_abnormal"] = probability
    cycle_predictions["predicted_outcome"] = np.where(probability >= .5, "Abnormal", "Normal")
    recording_predictions = cycle_predictions.groupby(
        ["patient_id", "recording_id", "clinical_outcome", "outer_fold"], sort=True, as_index=False
    ).agg(probability_abnormal=("probability_abnormal", "mean"), contributing_cycle_count=("candidate_id", "size"))
    recording_predictions["predicted_outcome"] = np.where(recording_predictions["probability_abnormal"] >= .5, "Abnormal", "Normal")
    audit = {
        "outer_fold": outer_fold, "best_epoch_zero_based": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "normalization_mean_training_only": mean, "normalization_std_training_only": std,
        "training_patient_ids": sorted(train_index["patient_id"].unique()),
        "inner_validation_patient_ids": sorted(validation_index["patient_id"].unique()),
        "test_patient_ids": sorted(test_index["patient_id"].unique()),
        "selected_training_candidate_ids": sorted(sampled_train["candidate_id"]),
        "training_cycle_cap_per_patient": config.maximum_training_cycles_per_patient,
    }
    pd.DataFrame(history).to_csv(fold_directory / "training_history.csv", index=False)
    (fold_directory / "fold_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    cycle_predictions.to_csv(fold_directory / "cycle_predictions.csv", index=False)
    recording_predictions.to_csv(fold_directory / "recording_predictions.csv", index=False)
    (fold_directory / "complete.json").write_text(json.dumps({"status": "complete", "best_epoch": int(best["epoch"])}), encoding="utf-8")
    return cycle_predictions, recording_predictions, audit


def _plot_results(predictions: pd.DataFrame, destination: Path) -> None:
    truth = predictions["clinical_outcome"].map({"Normal": 0, "Abnormal": 1}).to_numpy()
    probability = predictions["probability_abnormal"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_predictions(truth, probability >= .5, display_labels=["Normal", "Abnormal"], cmap="Blues", ax=axes[0], colorbar=False)
    RocCurveDisplay.from_predictions(truth, probability, name="Compact 1D CNN", ax=axes[1])
    axes[0].set_title("Patient-level confusion matrix")
    axes[1].set_title("Patient-level cross-validated ROC")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def run_cnn(
    *, cache_run: Path, baseline_run: Path, run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT, config: CNNConfig = CNNConfig(),
    resume: bool = False, device_name: str = "auto",
) -> Path:
    cache, baseline = Path(cache_run).resolve(), Path(baseline_run).resolve()
    cache_manifest = json.loads((cache / "run_manifest.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads((baseline / "run_manifest.json").read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "complete" or baseline_manifest.get("status") != "complete":
        raise ValueError("cache and baseline runs must both be complete")
    verify_manifest_artifacts(cache, cache_manifest)
    verify_manifest_artifacts(baseline, baseline_manifest)
    index = pd.read_csv(cache / "candidate_index.csv", dtype={"patient_id": str, "recording_id": str})
    verify_cached_signals(index, cache)
    folds = pd.read_csv(baseline / "fold_assignments.csv", dtype={"patient_id": str})
    if set(index["patient_id"]) != set(folds["patient_id"]):
        raise ValueError("cache and baseline eligible patient sets differ")
    observed_labels = index[["patient_id", "clinical_outcome"]].drop_duplicates()
    if len(observed_labels) != len(folds) or not observed_labels.merge(folds, on=["patient_id", "clinical_outcome"], how="outer", indicator=True)["_merge"].eq("both").all():
        raise ValueError("cache labels and frozen fold labels differ")
    destination = Path(output_root).resolve() / run_name
    if destination.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite Task 6 CNN run: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "tool_version": TOOL_VERSION, "config": asdict(config),
        "cache_manifest_sha256": sha256_file(cache / "run_manifest.json"),
        "candidate_index_sha256": sha256_file(cache / "candidate_index.csv"),
        "baseline_manifest_sha256": sha256_file(baseline / "run_manifest.json"),
        "fold_assignments_sha256": sha256_file(baseline / "fold_assignments.csv"),
    }
    preflight_path = destination / "cnn_preflight.json"
    if resume and preflight_path.exists():
        previous = json.loads(preflight_path.read_text(encoding="utf-8"))
        if previous.get("identity_hash") != sha256_json(identity):
            raise ValueError("CNN resume configuration or source hashes differ")
    else:
        preflight_path.write_text(json.dumps({**identity, "identity_hash": sha256_json(identity)}, indent=2), encoding="utf-8")
    device = choose_device(device_name)
    print(f"Task 6 compact 1D CNN device: {device}")
    all_cycle, all_recording, audits = [], [], []
    for outer_fold in sorted(folds["outer_fold"].unique()):
        fold_directory = destination / f"fold_{int(outer_fold)}"
        complete_path = fold_directory / "complete.json"
        if resume and complete_path.exists():
            all_cycle.append(pd.read_csv(fold_directory / "cycle_predictions.csv", dtype={"patient_id": str, "recording_id": str}))
            all_recording.append(pd.read_csv(fold_directory / "recording_predictions.csv", dtype={"patient_id": str, "recording_id": str}))
            audits.append(json.loads((fold_directory / "fold_audit.json").read_text(encoding="utf-8")))
            continue
        test_patients = set(folds.loc[folds["outer_fold"].eq(outer_fold), "patient_id"])
        outer_training = folds.loc[folds["outer_fold"].ne(outer_fold), ["patient_id", "clinical_outcome"]]
        validation_patients = make_inner_validation(
            outer_training, outer_fold=int(outer_fold),
            fraction=config.inner_validation_fraction, seed=config.seed,
        )
        training_patients = set(outer_training["patient_id"]) - validation_patients
        assert_patient_disjoint(training_patients, validation_patients, test_patients)
        train_index = index.loc[index["patient_id"].isin(training_patients)].copy()
        validation_index = index.loc[index["patient_id"].isin(validation_patients)].copy()
        test_index = index.loc[index["patient_id"].isin(test_patients)].copy()
        cycle, recording, audit = train_fold(
            cache=cache, train_index=train_index, validation_index=validation_index,
            test_index=test_index, fold_directory=fold_directory,
            outer_fold=int(outer_fold), config=config, device=device, resume=resume,
        )
        all_cycle.append(cycle)
        all_recording.append(recording)
        audits.append(audit)
    cycle_predictions = pd.concat(all_cycle, ignore_index=True)
    recording_predictions = pd.concat(all_recording, ignore_index=True)
    patient_predictions = aggregate_cycle_probabilities(cycle_predictions)
    patient_predictions["outer_fold"] = patient_predictions["patient_id"].map(folds.set_index("patient_id")["outer_fold"])
    metrics = classification_metrics(patient_predictions)
    intervals = patient_bootstrap_intervals(patient_predictions, replicates=config.bootstrap_replicates, seed=config.seed)
    cycle_predictions.to_csv(destination / "cnn_cycle_predictions.csv", index=False)
    recording_predictions.to_csv(destination / "cnn_recording_predictions.csv", index=False)
    patient_predictions.to_csv(destination / "cnn_patient_predictions.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in metrics.items() if k != "confusion_matrix_normal_abnormal"}, "confusion_matrix_normal_abnormal": json.dumps(metrics["confusion_matrix_normal_abnormal"])}]).to_csv(destination / "cnn_metrics.csv", index=False)
    fold_metric_rows = []
    for outer_fold, fold_predictions in patient_predictions.groupby("outer_fold", sort=True):
        fold_metrics = classification_metrics(fold_predictions)
        fold_metric_rows.append({
            "outer_fold": int(outer_fold),
            **{k: v for k, v in fold_metrics.items() if k != "confusion_matrix_normal_abnormal"},
            "confusion_matrix_normal_abnormal": json.dumps(fold_metrics["confusion_matrix_normal_abnormal"]),
        })
    pd.DataFrame(fold_metric_rows).to_csv(destination / "cnn_fold_metrics.csv", index=False)
    intervals.to_csv(destination / "cnn_bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame([
        {"outer_fold": audit["outer_fold"], "best_epoch_zero_based": audit["best_epoch_zero_based"], "best_validation_loss": audit["best_validation_loss"]}
        for audit in audits
    ]).to_csv(destination / "best_epochs.csv", index=False)
    _plot_results(patient_predictions, destination / "cnn_patient_evaluation.png")
    baseline_metrics = pd.read_csv(baseline / "baseline_metrics.csv")
    fixed = baseline_metrics.loc[
        baseline_metrics["feature_group"].eq("core_plus_exploratory")
        & baseline_metrics["model"].eq("logistic_regression")
    ].copy()
    cnn_row = {"feature_group": "waveform", "model": "compact_1d_cnn", **{k: v for k, v in metrics.items() if k != "confusion_matrix_normal_abnormal"}, "confusion_matrix_normal_abnormal": json.dumps(metrics["confusion_matrix_normal_abnormal"])}
    pd.concat([fixed, pd.DataFrame([cnn_row])], ignore_index=True).to_csv(destination / "fixed_model_comparison.csv", index=False)
    architecture = {
        "primary_input": "raw separated systolic candidate waveform",
        "choice_rationale": "1D avoids image generation and a large pretrained network while preserving waveform information at lower computational cost for 179 patients.",
        "architecture": "Conv1d(16)-Conv1d(32)-Conv1d(64)-global average pool-dropout-linear",
        "parameters": int(sum(parameter.numel() for parameter in CompactWaveformCNN().parameters())),
        "augmentation": "none in the primary analysis",
        "training_balance": f"fixed deterministic maximum of {config.maximum_training_cycles_per_patient} cycles per training patient",
        "primary_inference": "mean cycle probability per patient before metrics",
        "early_stopping": "inner patient-level stratified validation subset; validation loss",
        "limitations": ["small sample", "class imbalance", "no independent test set", "estimated separated candidates are not clean-source ground truth", "no clinical deployment claim"],
    }
    (destination / "cnn_design.json").write_text(json.dumps(architecture, indent=2), encoding="utf-8")
    artifact_names = (
        "cnn_preflight.json", "cnn_cycle_predictions.csv", "cnn_recording_predictions.csv",
        "cnn_patient_predictions.csv", "cnn_metrics.csv",
        "cnn_fold_metrics.csv",
        "cnn_bootstrap_confidence_intervals.csv", "best_epochs.csv",
        "cnn_patient_evaluation.png", "fixed_model_comparison.csv", "cnn_design.json",
    )
    manifest = {
        **identity, "status": "complete", "run_name": run_name,
        "device": str(device), "eligible_patient_count": int(len(patient_predictions)),
        "outcome_counts": {str(k): int(v) for k, v in patient_predictions["clinical_outcome"].value_counts().sort_index().items()},
        "artifact_sha256": {name: sha256_file(destination / name) for name in artifact_names},
        "fold_artifact_sha256": {
            f"fold_{fold}/{name}": sha256_file(destination / f"fold_{fold}" / name)
            for fold in sorted(folds["outer_fold"].unique())
            for name in ("best_checkpoint.pt", "last_checkpoint.pt", "training_history.csv", "fold_audit.json", "cycle_predictions.csv", "recording_predictions.csv", "complete.json")
        },
        "git_branch": git_value("branch", "--show-current"), "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "torch_version": torch.__version__,
        "scope_statement": "Task 6 clinical Outcome classification among Murmur Present patients using separated systolic candidate information.",
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_manifest_artifacts(destination, manifest)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--maximum-training-cycles-per-patient", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    config = CNNConfig(
        maximum_epochs=args.maximum_epochs,
        early_stopping_patience=args.early_stopping_patience,
        maximum_training_cycles_per_patient=args.maximum_training_cycles_per_patient,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    destination = run_cnn(
        cache_run=args.cache_run, baseline_run=args.baseline_run,
        run_name=args.run_name, output_root=args.output_root,
        config=config, resume=args.resume, device_name=args.device,
    )
    print(f"Task 6 CNN evaluation: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
