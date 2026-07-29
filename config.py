"""Shared paths and reproducible separation settings.

The dataset root may be overridden with ``CIRCOR_DATASET_ROOT``. Generated
artifacts are always rooted in this repository and never in the source dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    os.environ.get(
        "CIRCOR_DATASET_ROOT",
        Path.home()
        / "physionet.org"
        / "files"
        / "circor-heart-sound"
        / "1.0.1",
    )
).expanduser().resolve()
AUDIO_DIR = DATASET_ROOT / "training_data"
METADATA_PATH = DATASET_ROOT / "training_data.csv"

EXPECTED_RECORDINGS = 3163
EXPECTED_PATIENTS = 942

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SEPARATION_OUTPUT_DIR = OUTPUT_ROOT / "separation"
FEATURE_OUTPUT_DIR = OUTPUT_ROOT / "features"
REPORT_OUTPUT_DIR = OUTPUT_ROOT / "reports"
SYNTHETIC_OUTPUT_DIR = OUTPUT_ROOT / "synthetic"
LABELS_PATH = FEATURE_OUTPUT_DIR / "labels.csv"


@dataclass(frozen=True)
class SeparationConfig:
    """Every separation hyperparameter that can materially affect a run."""

    sample_rate: int = 4000
    ssa_window_length: int = 100
    explained_energy_threshold: float = 0.99
    maximum_ssa_components: int | None = None
    reconstruction_tolerance: float = 1e-8
    zcr_threshold_strategy: str = "percentile"
    zcr_threshold: float = 0.05
    zcr_percentile: float = 40.0
    kurtosis_population_size: int = 30
    kurtosis_generations: int = 40
    kurtosis_mutation_rate: float = 0.05
    kurtosis_energy_weight: float = 0.25
    kurtosis_impulse_penalty: float = 0.10
    use_dwt: bool = False
    dwt_wavelet: str = "db4"
    dwt_level: int = 4
    dwt_threshold_method: str = "soft"
    onset_threshold_mad: float = 3.0
    minimum_interval_duration_ms: float = 30.0
    gap_merging_duration_ms: float = 20.0
    noise_energy_floor: float = 1e-5
    high_frequency_noise_ratio: float = 0.80
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.explained_energy_threshold <= 1:
            raise ValueError("explained_energy_threshold must be in (0, 1]")
        if self.ssa_window_length < 2:
            raise ValueError("ssa_window_length must be at least 2")
        if self.maximum_ssa_components is not None and self.maximum_ssa_components < 1:
            raise ValueError("maximum_ssa_components must be positive or None")
        if self.zcr_threshold_strategy not in {"fixed", "percentile"}:
            raise ValueError("zcr_threshold_strategy must be 'fixed' or 'percentile'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_SEPARATION_CONFIG = SeparationConfig()


def ensure_output_directories() -> None:
    """Create the project-owned output directories after overlap validation."""

    validate_input_output_isolation(DATASET_ROOT, OUTPUT_ROOT)
    for path in (
        OUTPUT_ROOT,
        SEPARATION_OUTPUT_DIR,
        FEATURE_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
        SYNTHETIC_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_input_output_isolation(dataset_root: Path, output_root: Path) -> None:
    """Reject any configuration that could write into or ingest project output."""

    dataset = dataset_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError(
            f"Dataset and output paths overlap: dataset={dataset}, output={output}"
        )
