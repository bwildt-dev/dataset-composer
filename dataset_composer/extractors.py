"""Patient ID, timepoint, and DICOM tag extraction.

All logic is deterministic and header-based. `IdExtractor` and
`TimepointExtractor` are configured via the YAML `id` and
`timepoint` sections and work identically for DICOM and NIfTI paths.

`IdExtractor` ships no default patterns — naming conventions are
cohort-specific, so `id.patterns` must be set in the config.
`TimepointExtractor` defaults to `PRERAND` → `t0`, `EVAL` → `t1`; override
via config for other cohorts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pydicom

logger = logging.getLogger(__name__)


@dataclass
class IdExtractor:
    """Extract and normalise a patient/subject ID from a file path.

    No default patterns are shipped — naming conventions are cohort-specific,
    so `patterns` must be set for this to return anything other than `None`.

    Attributes:
        patterns: Tried in order; the first capture group of the first
            match is used (or full match if no groups).
        format: Applied to the captured text — `"SUBJ-{:03d}"` zero-pads
            numeric IDs, `"{}"` keeps non-numeric IDs verbatim.
    """

    patterns: List[str] = field(default_factory=list)
    format: str = "{}"

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def __call__(self, path: Path) -> Optional[str]:
        """Return the normalised ID extracted from any component of `path`, or None.

        Args:
            path: File path to search for a matching ID pattern.
        """
        for part in path.parts:
            for pat in self._compiled:
                m = pat.search(str(part))
                if m:
                    captured = m.group(1) if m.lastindex else m.group(0)
                    try:
                        return self.format.format(int(captured))
                    except (ValueError, TypeError):
                        return self.format.format(captured)
        return None


@dataclass
class TimepointExtractor:
    """Extract a timepoint label from a file path.

    When `enabled=False`, always returns `""` and files are placed under
    `<phase>/<patient_id>/` with no timepoint subdirectory.

    Attributes:
        enabled: Whether timepoint extraction is active.
        labels: Label names mapped to pattern lists; the first match across
            all path components wins.
        fallback: Returned when nothing matches.
    """

    enabled: bool = True
    labels:  Dict[str, List[str]] = field(default_factory=lambda: {
        "t0": ["PRERAND"],
        "t1": ["EVAL"],
    })
    fallback: str = "unknown"

    def __post_init__(self) -> None:
        self._compiled: Dict[str, List[re.Pattern]] = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in self.labels.items()
        }

    def __call__(self, path: Path) -> str:
        if not self.enabled:
            return ""
        for part in path.parts:
            for label, patterns in self._compiled.items():
                for pat in patterns:
                    if pat.search(str(part)):
                        return label
        return self.fallback


_DEFAULT_ID_EXTRACTOR = IdExtractor()
_DEFAULT_TP_EXTRACTOR = TimepointExtractor()


_INTERESTING_TAGS = (
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
    "ImageType",
    "ImageOrientationPatient",
    "SliceThickness",
    "Manufacturer",
    "ManufacturerModelName",
    "SeriesNumber",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "AcquisitionTime",
    "SeriesTime",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "ContrastBolusStartTime",
    "AcquisitionNumber",
    "Modality",
)


def read_series_tags(series_dir: Path) -> Dict[str, str]:
    """Read DICOM tags from a representative slice of the series.

    Uses the middle slice rather than the first — some scanners only write
    full metadata on body slices, not the initial localiser slices.

    Args:
        series_dir: DICOM series directory to read from.

    Returns:
        A dict of tag values (plus `_n_slices`), or an empty dict if
        nothing can be read.
    """
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(p for p in series_dir.iterdir() if p.is_file())
    if not dcm_files:
        return {}

    sample = dcm_files[len(dcm_files) // 2]
    try:
        ds = pydicom.dcmread(sample, stop_before_pixels=True, force=True)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", sample, exc)
        return {}

    out: Dict[str, str] = {"_n_slices": str(len(dcm_files))}
    for tag in _INTERESTING_TAGS:
        val = getattr(ds, tag, "")
        out[tag] = str(val) if val is not None else ""
    return out
