"""Task 5 Phase 2C multi-level Tier A redundancy analysis.

The runner consumes completed Phase 2A and Phase 2B artifacts.  It calculates
pairwise-complete Spearman correlations with patient-cluster intervals at the
cycle, recording, patient-location, and primary-patient levels; performs
complete-linkage clustering over ``1 - abs(rho)`` at the frozen thresholds;
and exports complete-case VIF/condition-index diagnostics.  It does not use
Outcome, train a classifier, or change the frozen feature pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd
import scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

from config import OUTPUT_ROOT
from src.separation.tier_a_features import (
    FEATURE_SCHEMA_VERSION,
    TIER_A_FEATURE_COLUMNS,
    TIER_A_FEATURE_UNITS,
)


TOOL_VERSION: Final = "task5-phase2c-redundancy-v1.0.0"
DEFAULT_OUTPUT_ROOT: Final = OUTPUT_ROOT / "task5_redundancy"


@dataclass(frozen=True)
class RedundancyConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260827
    confidence_level: float = 0.95
    minimum_overlap: int = 8
    cluster_thresholds: tuple[float, ...] = (.85, .90, .95)

    def __post_init__(self) -> None:
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if self.minimum_overlap < 3:
            raise ValueError("minimum_overlap must be at least three")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if any(not 0 < value < 1 for value in self.cluster_thresholds):
            raise ValueError("cluster thresholds must be between zero and one")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _seed(config: RedundancyConfig, key: Iterable[Any]) -> int:
    offset = int.from_bytes(
        sha256("|".join(map(str, key)).encode()).digest()[:4], "little"
    )
    return (config.bootstrap_seed + offset) % (2**32)


def _read_manifest(run: Path) -> dict[str, Any]:
    path = run / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing source manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"source run is not complete: {run}")
    return manifest


def load_sources(
    phase2a_run: Path, phase2b_run: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase2a_run, phase2b_run = Path(phase2a_run), Path(phase2b_run)
    extraction_manifest = _read_manifest(phase2a_run)
    construct_manifest = _read_manifest(phase2b_run)
    if extraction_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Phase 2A schema differs from the frozen Tier A schema")
    if construct_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Phase 2B schema differs from the frozen Tier A schema")
    feature_path = phase2a_run / "task5_feature_table.csv"
    primary_path = phase2b_run / "primary_patient_feature_long.csv"
    if extraction_manifest.get("artifact_sha256", {}).get(feature_path.name) != _sha256_file(feature_path):
        raise ValueError("Phase 2A feature-table hash mismatch")
    if construct_manifest.get("artifact_sha256", {}).get(primary_path.name) != _sha256_file(primary_path):
        raise ValueError("Phase 2B primary-table hash mismatch")
    if construct_manifest.get("source_feature_table_sha256") != _sha256_file(feature_path):
        raise ValueError("Phase 2B was not generated from the supplied Phase 2A table")
    usecols = [
        "patient_id", "recording_id", "location", "murmur_phase",
        "candidate_quality_status",
        *[item for feature in TIER_A_FEATURE_COLUMNS for item in (feature, f"{feature}_valid")],
    ]
    candidates = pd.read_csv(
        feature_path, usecols=usecols, dtype={"patient_id": str, "recording_id": str}
    )
    for feature in TIER_A_FEATURE_COLUMNS:
        valid = candidates[f"{feature}_valid"].fillna(False).astype(bool)
        candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce").where(valid)
    primary = pd.read_csv(primary_path, dtype={"patient_id": str})
    return candidates, primary, {
        "phase2a_manifest": extraction_manifest,
        "phase2b_manifest": construct_manifest,
        "phase2a_feature_sha256": _sha256_file(feature_path),
        "phase2b_primary_sha256": _sha256_file(primary_path),
    }


def _aggregate_wide(table: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "cycle":
        return table[["patient_id", "recording_id", "location", *TIER_A_FEATURE_COLUMNS]].copy()
    recording = table.groupby(
        ["patient_id", "recording_id", "location"], sort=True, dropna=False
    )[list(TIER_A_FEATURE_COLUMNS)].median().reset_index()
    if level == "recording":
        return recording
    if level == "patient_location":
        return recording.groupby(
            ["patient_id", "location"], sort=True, dropna=False
        )[list(TIER_A_FEATURE_COLUMNS)].median().reset_index()
    raise ValueError(f"unknown aggregation level: {level}")


def build_analysis_contexts(
    candidates: pd.DataFrame, primary: pd.DataFrame
) -> list[tuple[dict[str, str], pd.DataFrame]]:
    contexts: list[tuple[dict[str, str], pd.DataFrame]] = []
    for phase in ("systole", "diastole"):
        phase_table = candidates.loc[candidates["murmur_phase"].eq(phase)]
        for status in ("all", "accepted", "fallback"):
            selected = phase_table if status == "all" else phase_table.loc[
                phase_table["candidate_quality_status"].eq(status)
            ]
            for level in ("cycle", "recording", "patient_location"):
                contexts.append((
                    {"aggregation_level": level, "murmur_phase": phase, "analysis_stratum": status},
                    _aggregate_wide(selected, level),
                ))
    for phase in ("systole", "diastole"):
        for stratum in ("all_valid", "accepted_only", "high_quality"):
            selected = primary.loc[
                primary["murmur_phase"].eq(phase) & primary["stratum"].eq(stratum)
            ]
            wide = selected.pivot(index=["patient_id", "location"], columns="feature", values="value").reset_index()
            for feature in TIER_A_FEATURE_COLUMNS:
                if feature not in wide:
                    wide[feature] = np.nan
            contexts.append((
                {"aggregation_level": "primary_patient", "murmur_phase": phase, "analysis_stratum": stratum},
                wide[["patient_id", "location", *TIER_A_FEATURE_COLUMNS]],
            ))
    return contexts


def _weighted_cluster_interval(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    config: RedundancyConfig,
    context_patients: np.ndarray,
    cluster_draws: np.ndarray,
) -> tuple[float, float, float]:
    complete = frame[["patient_id", x, y]].dropna()
    pair_patients = complete["patient_id"].astype(str).to_numpy()
    present_patients = np.unique(pair_patients)
    if len(complete) < config.minimum_overlap or len(present_patients) < 3:
        return np.nan, np.nan, 1.0
    patient_index = np.searchsorted(context_patients, pair_patients)
    rx = rankdata(complete[x].to_numpy(float), method="average")
    ry = rankdata(complete[y].to_numpy(float), method="average")
    aggregates = np.stack([
        np.bincount(patient_index, weights=values, minlength=len(context_patients))
        for values in (np.ones(len(rx)), rx, ry, rx * rx, ry * ry, rx * ry)
    ])
    n, sx, sy, sx2, sy2, sxy = (
        cluster_draws @ aggregates[index] for index in range(6)
    )
    covariance = sxy - sx * sy / n
    variance_x = sx2 - sx * sx / n
    variance_y = sy2 - sy * sy / n
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0))
    estimates = np.divide(
        covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0
    )
    finite = estimates[np.isfinite(estimates)]
    failure_rate = 1 - len(finite) / config.bootstrap_replicates
    if not len(finite):
        return np.nan, np.nan, float(failure_rate)
    alpha = (1 - config.confidence_level) / 2
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return float(low), float(high), float(failure_rate)


def pairwise_correlations(
    contexts: list[tuple[dict[str, str], pd.DataFrame]], config: RedundancyConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, frame in contexts:
        context_patients = np.unique(frame["patient_id"].astype(str))
        rng = np.random.default_rng(_seed(config, identity.values()))
        cluster_draws = (
            rng.multinomial(
                len(context_patients),
                np.full(len(context_patients), 1 / len(context_patients)),
                size=config.bootstrap_replicates,
            )
            if len(context_patients)
            else np.zeros((config.bootstrap_replicates, 0), dtype=int)
        )
        for left_index, left in enumerate(TIER_A_FEATURE_COLUMNS):
            for right in TIER_A_FEATURE_COLUMNS[left_index + 1:]:
                complete = frame[["patient_id", left, right]].dropna()
                rho = np.nan
                if len(complete) >= config.minimum_overlap and complete[left].nunique() > 1 and complete[right].nunique() > 1:
                    rho = float(spearmanr(complete[left], complete[right]).statistic)
                low, high, failure = _weighted_cluster_interval(
                    frame, left, right, config=config,
                    context_patients=context_patients,
                    cluster_draws=cluster_draws,
                )
                rows.append({
                    **identity,
                    "feature_a": left,
                    "feature_b": right,
                    "overlap_count": int(len(complete)),
                    "independent_patient_count": int(complete["patient_id"].nunique()),
                    "spearman_rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_failure_rate": failure,
                    "inference_status": (
                        "not_estimable" if not np.isfinite(rho)
                        else "descriptive_too_few_patients" if complete["patient_id"].nunique() < 8
                        else "unstable_bootstrap_over_5pct_failures" if failure > .05
                        else "estimable"
                    ),
                })
    return pd.DataFrame(rows)


def correlation_clusters(
    correlations: pd.DataFrame, config: RedundancyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows: list[dict[str, Any]] = []
    keys = ["aggregation_level", "murmur_phase", "analysis_stratum"]
    for identity, group in correlations.groupby(keys, sort=True):
        matrix = np.eye(len(TIER_A_FEATURE_COLUMNS))
        indexes = {feature: index for index, feature in enumerate(TIER_A_FEATURE_COLUMNS)}
        for row in group.itertuples():
            value = float(row.spearman_rho) if np.isfinite(row.spearman_rho) else 0.0
            i, j = indexes[row.feature_a], indexes[row.feature_b]
            matrix[i, j] = matrix[j, i] = value
        tree = linkage(squareform(1 - np.abs(matrix), checks=False), method="complete")
        identity_map = dict(zip(keys, identity, strict=True))
        for threshold in config.cluster_thresholds:
            labels = fcluster(tree, t=1 - threshold, criterion="distance")
            for feature, label in zip(TIER_A_FEATURE_COLUMNS, labels, strict=True):
                cluster_rows.append({
                    **identity_map, "absolute_rho_threshold": threshold,
                    "feature": feature, "cluster_id": int(label),
                })
    clusters = pd.DataFrame(cluster_rows)
    membership = clusters.merge(clusters, on=[*keys, "absolute_rho_threshold"], suffixes=("_a", "_b"))
    membership = membership.loc[membership["feature_a"] < membership["feature_b"]]
    membership["same_cluster"] = membership["cluster_id_a"].eq(membership["cluster_id_b"])
    instability = membership.groupby(
        ["absolute_rho_threshold", "feature_a", "feature_b"], sort=True
    ).agg(
        context_count=("same_cluster", "size"),
        same_cluster_context_count=("same_cluster", "sum"),
        same_cluster_fraction=("same_cluster", "mean"),
    ).reset_index()
    return clusters, instability


def complete_case_diagnostics(
    contexts: list[tuple[dict[str, str], pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    vif_rows: list[dict[str, Any]] = []
    for identity, frame in contexts:
        complete = frame[list(TIER_A_FEATURE_COLUMNS)].dropna()
        if len(complete) <= len(TIER_A_FEATURE_COLUMNS) + 1:
            summary_rows.append({**identity, "complete_case_count": len(complete), "condition_index": np.nan, "status": "insufficient_complete_cases"})
            continue
        standardized = (complete - complete.mean()) / complete.std(ddof=0)
        correlation = np.corrcoef(standardized.to_numpy(), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(correlation)
        positive = eigenvalues[eigenvalues > np.finfo(float).eps]
        condition = float(np.sqrt(eigenvalues.max() / positive.min())) if len(positive) else np.inf
        inverse = np.linalg.pinv(correlation)
        summary_rows.append({**identity, "complete_case_count": len(complete), "condition_index": condition, "status": "secondary_complete_case_diagnostic"})
        for feature, vif in zip(TIER_A_FEATURE_COLUMNS, np.diag(inverse), strict=True):
            vif_rows.append({**identity, "complete_case_count": len(complete), "feature": feature, "vif": float(vif)})
    return pd.DataFrame(summary_rows), pd.DataFrame(vif_rows)


def write_redundancy_figures(
    destination: Path,
    correlations: pd.DataFrame,
    contexts: list[tuple[dict[str, str], pd.DataFrame]],
) -> dict[str, str]:
    """Write clustered heatmaps for every context and primary family pair plots."""

    figure_dir = destination / "figures"
    figure_dir.mkdir()
    hashes: dict[str, str] = {}
    indexes = {feature: index for index, feature in enumerate(TIER_A_FEATURE_COLUMNS)}
    keys = ["aggregation_level", "murmur_phase", "analysis_stratum"]
    for identity, group in correlations.groupby(keys, sort=True):
        matrix = np.eye(len(TIER_A_FEATURE_COLUMNS))
        for row in group.itertuples():
            value = float(row.spearman_rho) if np.isfinite(row.spearman_rho) else 0.0
            i, j = indexes[row.feature_a], indexes[row.feature_b]
            matrix[i, j] = matrix[j, i] = value
        tree = linkage(squareform(1 - np.abs(matrix), checks=False), method="complete")
        order = scipy.cluster.hierarchy.leaves_list(tree)
        ordered = matrix[np.ix_(order, order)]
        labels = [TIER_A_FEATURE_COLUMNS[index].removeprefix("tier_a_") for index in order]
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(ordered, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yticks(range(len(labels)), labels, fontsize=6)
        axis.set_title(" / ".join(map(str, identity)))
        figure.colorbar(image, ax=axis, label="Spearman rho")
        figure.tight_layout()
        filename = "heatmap_" + "_".join(map(str, identity)) + ".png"
        path = figure_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        hashes[str(Path("figures") / filename)] = _sha256_file(path)

    primary = next(
        (
            frame for identity, frame in contexts
            if identity == {
                "aggregation_level": "primary_patient",
                "murmur_phase": "systole",
                "analysis_stratum": "all_valid",
            }
        ),
        None,
    )
    families = {
        "timing": TIER_A_FEATURE_COLUMNS[0:2],
        "envelope": TIER_A_FEATURE_COLUMNS[3:7],
        "spectral": TIER_A_FEATURE_COLUMNS[7:12],
        "ridge": TIER_A_FEATURE_COLUMNS[12:14],
    }
    if primary is not None:
        for family, features in families.items():
            available = [feature for feature in features if feature in primary]
            if len(available) < 2:
                continue
            axes = pd.plotting.scatter_matrix(
                primary[available], figsize=(3 * len(available), 3 * len(available)),
                diagonal="hist", alpha=.55, s=12,
            )
            figure = axes[0, 0].figure
            figure.suptitle(f"Primary systolic all-valid: {family}", y=1.01)
            figure.tight_layout()
            filename = f"pairplot_primary_systole_all_valid_{family}.png"
            path = figure_dir / filename
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            hashes[str(Path("figures") / filename)] = _sha256_file(path)
    return hashes


def run_redundancy_analysis(
    *, phase2a_run: Path, phase2b_run: Path, run_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: RedundancyConfig = RedundancyConfig(),
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    destination = Path(output_root) / run_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {destination}")
    candidates, primary, source = load_sources(phase2a_run, phase2b_run)
    contexts = build_analysis_contexts(candidates, primary)
    correlations = pairwise_correlations(contexts, config)
    clusters, instability = correlation_clusters(correlations, config)
    diagnostics, vifs = complete_case_diagnostics(contexts)
    destination.mkdir(parents=True)
    artifacts = {
        "pairwise_correlations.csv": correlations,
        "redundancy_clusters.csv": clusters,
        "cluster_instability.csv": instability,
        "condition_index_summary.csv": diagnostics,
        "vif_diagnostics.csv": vifs,
    }
    hashes: dict[str, str] = {}
    for filename, frame in artifacts.items():
        path = destination / filename
        frame.to_csv(path, index=False)
        hashes[filename] = _sha256_file(path)
    figure_hashes = write_redundancy_figures(destination, correlations, contexts)
    manifest = {
        "tool_version": TOOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "status": "complete",
        "run_name": run_name,
        "config": asdict(config),
        "phase2a_run": str(Path(phase2a_run).resolve()),
        "phase2b_run": str(Path(phase2b_run).resolve()),
        "phase2a_feature_table_sha256": source["phase2a_feature_sha256"],
        "phase2b_primary_table_sha256": source["phase2b_primary_sha256"],
        "analysis_context_count": len(contexts),
        "correlation_count": len(correlations),
        "bootstrap_method": (
            "patient-cluster multinomial resampling with cluster-weighted "
            "full-sample pairwise ranks"
        ),
        "artifact_sha256": hashes,
        "figure_sha256": figure_hashes,
        "outcome_used": False,
        "prediction_pruning_permitted": False,
        "scope_statement": "Task 5 descriptive redundancy only; any Task 6 pruning must be fitted inside training folds.",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-run", type=Path, required=True)
    parser.add_argument("--phase2b-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = run_redundancy_analysis(
        phase2a_run=args.phase2a_run,
        phase2b_run=args.phase2b_run,
        run_name=args.run_name,
        output_root=args.output_root,
        config=RedundancyConfig(bootstrap_replicates=args.bootstrap_replicates),
    )
    print(f"Task 5 Phase 2C artifacts: {destination}")


if __name__ == "__main__":
    main()
