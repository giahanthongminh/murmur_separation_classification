from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from src.data_validation import (
    DatasetPaths,
    DatasetValidationError,
    validate_dataset,
)


def _dataset(tmp_path: Path, recordings: int = 1) -> DatasetPaths:
    root = tmp_path / "circor"
    audio = root / "training_data"
    audio.mkdir(parents=True)
    rows = []
    for index in range(recordings):
        patient = str(1000 + index)
        recording = f"{patient}_AV"
        wavfile.write(audio / f"{recording}.wav", 4000, np.zeros(1600, dtype=np.int16))
        (audio / f"{recording}.tsv").write_text(
            "0.00\t0.10\t1\n0.10\t0.20\t2\n0.20\t0.30\t3\n0.30\t0.40\t4\n",
            encoding="utf-8",
        )
        rows.append({"Patient ID": patient, "Locations": "AV"})
    metadata = root / "training_data.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return DatasetPaths(root, audio, metadata)


def test_missing_dataset_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    with pytest.raises(DatasetValidationError):
        validate_dataset(
            DatasetPaths(root, root / "training_data", root / "training_data.csv"),
            expected_recordings=0,
            expected_patients=0,
        )


def test_expected_counts_are_enforced(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    report = validate_dataset(
        paths, expected_recordings=2, expected_patients=2, strict=False
    )
    codes = {error["code"] for error in report["errors"]}
    assert "unexpected_recording_count" in codes
    assert "unexpected_patient_count" in codes


def test_wav_tsv_pairing_is_enforced(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    next(paths.audio_dir.glob("*.tsv")).unlink()
    report = validate_dataset(
        paths, expected_recordings=1, expected_patients=1, strict=False
    )
    assert report["wav_missing_tsv"] == ["1000_AV"]


def test_annotation_bounds_are_checked(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    next(paths.audio_dir.glob("*.tsv")).write_text(
        "0.00\t0.10\t1\n0.10\t0.20\t2\n0.20\t0.30\t3\n0.30\t0.50\t4\n",
        encoding="utf-8",
    )
    report = validate_dataset(
        paths, expected_recordings=1, expected_patients=1, strict=False
    )
    assert report["invalid_annotations"][0]["file"] == "1000_AV.tsv"


def test_valid_fixture_passes(tmp_path: Path) -> None:
    paths = _dataset(tmp_path, recordings=2)
    report = validate_dataset(
        paths, expected_recordings=2, expected_patients=2, strict=True
    )
    assert report["valid"] is True
