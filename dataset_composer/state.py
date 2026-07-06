"""Disk-backed checkpoints and CSV audit logs for the dataset composer.

Two JSON state files let the pipeline resume after interruption:
`conversion_state.json` (keyed by SeriesInstanceUID) and
`classification_state.json` (keyed by NIfTI path).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversionRecord:
    """One DICOM-to-NIfTI conversion entry, keyed by SeriesInstanceUID.

    Attributes:
        series_uid: Primary key — stable across runs and zip-extraction
            layout changes.
        series_dir: Last-known DICOM dir (informational only).
        patient_id: Extracted patient ID, or None if unrecognised.
        timepoint: Extracted timepoint label.
        nifti_path: Path to the converted NIfTI.
        converted_at: ISO timestamp of conversion.
    """

    series_uid:  str
    series_dir:  str
    patient_id:  Optional[str]
    timepoint:   str
    nifti_path:  str
    converted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class ClassificationRecord:
    """One BOA classification result, keyed by NIfTI path.

    Attributes:
        nifti_path: Path to the classified NIfTI.
        patient_id: Extracted patient ID, or None if unrecognised.
        timepoint: Extracted timepoint label.
        body_region: Matched `RegionSpec` label, or `"other"`.
        phase: Classified contrast phase.
        body_confidence: Minimum voxel-count confidence across required
            structures.
        phase_confidence: BOA's probability output for the predicted phase.
        series_description: Original DICOM SeriesDescription tag.
        protocol_name: Original DICOM ProtocolName tag.
        classified_at: ISO timestamp of classification.
    """

    nifti_path:         str
    patient_id:         Optional[str]
    timepoint:          str
    body_region:        str
    phase:              str
    body_confidence:    float
    phase_confidence:   float
    series_description: str = ""
    protocol_name:      str = ""
    classified_at:      str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ComposerState:
    """File-backed checkpoint store for resumable composition.

    Args:
        state_dir: Directory holding the conversion/classification JSON
            state files; created if it doesn't exist.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._conv_path = self.state_dir / "conversion_state.json"
        self._clf_path  = self.state_dir / "classification_state.json"

        self._conv: Dict[str, ConversionRecord]    = {}
        self._clf:  Dict[str, ClassificationRecord] = {}
        self._lock = Lock()

        self._load()

    def _load(self) -> None:
        if self._conv_path.exists():
            with open(self._conv_path) as f:
                data = json.load(f)
            self._conv = {k: ConversionRecord(**v) for k, v in data.items()}
            logger.info("Loaded %d existing conversion records", len(self._conv))

        if self._clf_path.exists():
            with open(self._clf_path) as f:
                data = json.load(f)
            self._clf = {k: ClassificationRecord(**v) for k, v in data.items()}
            logger.info("Loaded %d existing classification records", len(self._clf))

    def _save_conv(self) -> None:
        tmp = self._conv_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({k: asdict(v) for k, v in self._conv.items()}, f, indent=2)
        tmp.replace(self._conv_path)

    def _save_clf(self) -> None:
        tmp = self._clf_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({k: asdict(v) for k, v in self._clf.items()}, f, indent=2)
        tmp.replace(self._clf_path)

    def get_conversion(self, series_uid: str) -> Optional[ConversionRecord]:
        """Return the conversion record for `series_uid`, or None.

        Args:
            series_uid: SeriesInstanceUID to look up.
        """
        if not series_uid:
            return None
        return self._conv.get(series_uid)

    def record_conversion(
        self,
        series_uid: str,
        series_dir: Path,
        patient_id: Optional[str],
        timepoint: str,
        nifti_path: Path,
    ) -> None:
        """Persist one conversion record and flush the JSON state file.

        Args:
            series_uid: SeriesInstanceUID, the primary key.
            series_dir: Last-known DICOM dir.
            patient_id: Extracted patient ID, or None if unrecognised.
            timepoint: Extracted timepoint label.
            nifti_path: Path to the converted NIfTI.
        """
        # Fall back to a synthetic key when DICOM headers are too broken to
        # provide a SeriesInstanceUID. Synthetic keys are unique per series_dir
        # and prefixed so they're easy to spot in the JSON.
        key = series_uid or f"_nouid:{series_dir}"
        with self._lock:
            self._conv[key] = ConversionRecord(
                series_uid=key,
                series_dir=str(series_dir),
                patient_id=patient_id,
                timepoint=timepoint,
                nifti_path=str(nifti_path),
            )
            self._save_conv()

    def all_conversions(self) -> List[ConversionRecord]:
        """Return all conversion records."""
        return list(self._conv.values())

    def get_classification(self, nifti_path: Path) -> Optional[ClassificationRecord]:
        """Return the classification record for `nifti_path`, or None.

        Args:
            nifti_path: NIfTI path to look up.
        """
        return self._clf.get(str(nifti_path))

    def record_classification(self, rec: ClassificationRecord) -> None:
        """Persist one classification record and flush the JSON state file.

        Args:
            rec: Classification record to persist, keyed by `rec.nifti_path`.
        """
        with self._lock:
            self._clf[rec.nifti_path] = rec
            self._save_clf()

    def all_classifications(self) -> List[ClassificationRecord]:
        """Return all classification records."""
        return list(self._clf.values())


def safe_write_row(path: Path, row: List[Any], retries: int = 5, delay: float = 1.0) -> None:
    """Append one CSV row, retrying briefly if the file is locked.

    Args:
        path: CSV file to append to.
        row: Row values to write.
        retries: Number of write attempts before giving up.
        delay: Seconds to wait between retries.
    """
    for attempt in range(retries):
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            return
        except PermissionError:
            if attempt < retries - 1:
                logger.warning("%s locked, retrying in %.1fs", path.name, delay)
                time.sleep(delay)
            else:
                logger.error("Could not write to %s after %d attempts", path.name, retries)


def init_csv(path: Path, header: List[str]) -> None:
    """Create the CSV with a header row if it does not already exist.

    Args:
        path: CSV file to create.
        header: Column names to write as the header row.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def merge_shard_states(output_root: Path, n_shards: int, force: bool = False) -> None:
    """Merge per-shard state files (from `--shard K/N` runs) into the canonical state.

    Reads `_state_shard{k}of{n_shards}/*.json` for `k` in `range(n_shards)`
    and unions them into `_state/*.json`.

    Args:
        output_root: Composer output root containing the `_state_shard*/` dirs.
        n_shards: Number of shards to merge (must match the array size used).
        force: Ignore any pre-existing canonical state instead of starting
            the merge from it.
    """
    canon_dir = output_root / "_state"
    canon_dir.mkdir(parents=True, exist_ok=True)

    for stem in ("classification_state.json", "conversion_state.json"):
        merged: dict = {}
        if not force and (canon_dir / stem).exists():
            merged.update(json.loads((canon_dir / stem).read_text()))

        n_from_shards = 0
        for k in range(n_shards):
            shard_path = output_root / f"_state_shard{k}of{n_shards}" / stem
            if not shard_path.exists():
                logger.info("  shard %d: no %s (skipping)", k, stem)
                continue
            data = json.loads(shard_path.read_text())
            n_from_shards += len(data)
            merged.update(data)

        out = canon_dir / stem
        out.write_text(json.dumps(merged, indent=2))
        logger.info("%s: %d from shards → %d merged → %s",
                    stem, n_from_shards, len(merged), out)


SKIPPED_HEADER = [
    "patient_id", "timepoint", "series_dir", "folder_name", "reason",
]
CONVERSION_HEADER = [
    "timestamp", "patient_id", "timepoint", "n_slices", "thickness_mm",
    "series_description", "status", "series_dir", "nifti_path", "details",
]
CLASSIFICATION_HEADER = [
    "timestamp", "patient_id", "timepoint", "body_region", "phase",
    "body_confidence", "phase_confidence", "kept",
    "series_description", "protocol_name", "nifti_path",
]
