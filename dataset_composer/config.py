"""YAML configuration loader for the dataset composer.

All fields have defaults; only override what you need. Environment variables
in path strings are expanded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .classifier import RegionSpec


@dataclass
class IdConfig:
    """Patient ID extraction config.

    Naming conventions are cohort-specific, so `patterns` must be set in
    the config for `IdExtractor` to return anything other than `None`.

    Attributes:
        patterns: Tried in order; the first capture group of the first hit
            is used.
        format: Applied to the captured text — `"SUBJ-{:03d}"` zero-pads
            numeric IDs; `"{}"` keeps the match verbatim.
    """
    patterns: List[str] = field(default_factory=list)
    format: str = "{}"


@dataclass
class TimepointConfig:
    """Timepoint extraction config.

    Attributes:
        enabled: When False, the timepoint subdirectory is omitted and
            files land under `<output>/<phase>/<patient_id>/`.
        labels: Ordered mapping of `label → [patterns]`; the first match
            wins.
        fallback: Label used when no pattern matches.
    """
    enabled:  bool                  = True
    labels:   Dict[str, List[str]]  = field(default_factory=lambda: {
        "t0": ["PRERAND"],
        "t1": ["EVAL"],
    })
    fallback: str                   = "unknown"


@dataclass
class PathsConfig:
    """Input/output path config for the composer.

    Attributes:
        output_root: Root directory for the final composed dataset.
        source_dir: Full-pipeline: DICOM source tree.
        scratch_dir: Full-pipeline: intermediate NIfTIs.
        nifti_dir: Classify-only: existing NIfTI tree.
        dcm2niix: Explicit dcm2niix binary path; None searches PATH.
    """

    output_root: Path
    source_dir:   Optional[Path] = None
    scratch_dir:  Optional[Path] = None
    nifti_dir:    Optional[Path] = None
    dcm2niix:     Optional[Path] = None


DEFAULT_REFORMAT_PATTERNS: List[str] = [
    "MPR", "MIP",
    "coronal", "sagittal", "coronaal", "sagitaal",
    r"_cor$", r"_sag$",
    r"_cor[_,\s]", r"_sag[_,\s]",
    r"^cor[\s_]", r"^sag[\s_]",
    r"IMR[\s_]?(long|cor|sag|mip)",
    r"^longsetting", "rawdun", "raw_dun",
]

DEFAULT_TOPOGRAM_PATTERNS: List[str] = [
    "topogram", "topo", "scout", "locator", "tracker", "tracking",
    r"exam[\s_]summary", r"dose[\s_]info",
    r"^SV$", r"^1_1$",
    r"C\+\d+\s?min", r"MonoE\d+keV", r"Spectral[\s_]\(\d+\)",
]


@dataclass
class ScannerConfig:
    """Folder-name regex filters applied before any DICOM is opened.

    Attributes:
        reformat_patterns: Fragments matching reformatted series.
        topogram_patterns: Fragments matching scout/topogram/admin series.
    """
    reformat_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_REFORMAT_PATTERNS)
    )
    topogram_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_TOPOGRAM_PATTERNS)
    )


@dataclass
class ClassifierConfig:
    """Classifier backend config (BOA or dummy).

    Attributes:
        backend: `"boa"` or `"dummy"`.
        device: Device string passed to the BOA backend.
        regions: Body-region acceptance rules, tried in order.
        keep_phases: Contrast phases to keep in the final output.
    """

    backend:     str            = "boa"
    device:      str            = "cuda"
    regions:     List[RegionSpec] = field(
        default_factory=lambda: [RegionSpec("abdomen", {"liver": 5000})]
    )
    keep_phases: List[str]      = field(default_factory=lambda: ["arterial", "venous"])


@dataclass
class WandBConfig:
    """Weights & Biases reporting config.

    Attributes:
        enabled: Whether to report to wandb.
        project: wandb project name.
        run_name: Optional explicit run name.
        tags: Tags applied to the run.
    """
    enabled:  bool          = False
    project:  str           = "pdac-dataset-composer"
    run_name: Optional[str] = None
    tags:     List[str]     = field(default_factory=list)


@dataclass
class ComposerConfig:
    """Top-level config for the dataset composer pipeline.

    Attributes:
        paths: Input/output path config.
        classifier: Classifier backend config.
        id: Patient ID extraction config.
        timepoint: Timepoint extraction config.
        scanner: Folder-name regex filter config.
        wandb: Weights & Biases reporting config.
        keep_all: Copy every classified scan regardless of `keep_phases`.
        unzip_archives: Stage 0: recursively unzip `*.zip` under source_dir.
        delete_zips_after: Delete archives after successful extraction.
        unzip_workers: Extraction worker count; None means
            `min(cpu_count(), 8)`.
    """

    paths:      PathsConfig
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    id:         IdConfig         = field(default_factory=IdConfig)
    timepoint:  TimepointConfig  = field(default_factory=TimepointConfig)
    scanner:    ScannerConfig    = field(default_factory=ScannerConfig)
    wandb:      WandBConfig      = field(default_factory=WandBConfig)
    keep_all:   bool             = False
    unzip_archives:    bool          = True
    delete_zips_after: bool          = False
    unzip_workers:     Optional[int] = None

    @property
    def keep_phases(self) -> Tuple[str, ...]:
        """Return the configured phases to keep as a tuple."""
        return tuple(self.classifier.keep_phases)

    @property
    def is_classify_only(self) -> bool:
        """Return True when running classification on pre-converted NIfTIs."""
        return self.paths.nifti_dir is not None

    def build_extractors(self):
        """Return `(IdExtractor, TimepointExtractor)` from this config."""
        from .extractors import IdExtractor, TimepointExtractor
        return (
            IdExtractor(patterns=self.id.patterns, format=self.id.format),
            TimepointExtractor(
                enabled  = self.timepoint.enabled,
                labels   = self.timepoint.labels,
                fallback = self.timepoint.fallback,
            ),
        )


def _expand(value: object) -> object:
    """Recursively expand environment variables in string values.

    Args:
        value: A string, dict, list, or other value to expand.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: Path) -> ComposerConfig:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML config file.

    Raises:
        ValueError: If `paths.output_root` is missing, or if neither
            classify-only (`paths.nifti_dir`) nor full-pipeline
            (`paths.source_dir` + `paths.scratch_dir`) paths are given.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _expand(raw) or {}

    paths_raw = raw.get("paths", {})
    if "output_root" not in paths_raw:
        raise ValueError("Config is missing required field: paths.output_root")

    def _p(key: str) -> Optional[Path]:
        v = paths_raw.get(key)
        return Path(v) if v else None

    paths = PathsConfig(
        output_root = Path(paths_raw["output_root"]),
        source_dir  = _p("source_dir"),
        scratch_dir = _p("scratch_dir"),
        nifti_dir   = _p("nifti_dir"),
        dcm2niix    = _p("dcm2niix"),
    )

    if paths.nifti_dir is None and paths.source_dir is None:
        raise ValueError(
            "Config must specify either paths.nifti_dir (classify-only) "
            "or paths.source_dir + paths.scratch_dir (full pipeline)."
        )
    if paths.nifti_dir is None and paths.scratch_dir is None:
        raise ValueError(
            "Full-pipeline mode requires both paths.source_dir and paths.scratch_dir."
        )

    clf_raw = raw.get("classifier", {})

    regions = [
        RegionSpec(
            label      = str(r["label"]),
            structures = {str(k): int(v) for k, v in r.get("structures", {}).items()},
        )
        for r in clf_raw.get("regions", [{"label": "abdomen", "structures": {"liver": 5000}}])
    ]

    classifier = ClassifierConfig(
        backend     = str(clf_raw.get("backend", "boa")),
        device      = str(clf_raw.get("device", "cuda")),
        regions     = regions,
        keep_phases = [str(p) for p in clf_raw.get("keep_phases", ["arterial", "venous"])],
    )

    id_raw = raw.get("id", {})
    _default_id = IdConfig()
    id_cfg = IdConfig(
        patterns = [str(p) for p in id_raw.get("patterns", _default_id.patterns)],
        format   = str(id_raw.get("format", _default_id.format)),
    )

    tp_raw = raw.get("timepoint", {})
    _default_tp = TimepointConfig()

    labels = (
        {str(k): [str(p) for p in v] for k, v in tp_raw["labels"].items()}
        if "labels" in tp_raw
        else _default_tp.labels
    )

    tp_cfg = TimepointConfig(
        enabled  = bool(tp_raw.get("enabled", True)),
        labels   = labels,
        fallback = str(tp_raw.get("fallback", _default_tp.fallback)),
    )

    sc_raw = raw.get("scanner", {}) or {}
    sc_cfg = ScannerConfig(
        reformat_patterns = [str(p) for p in sc_raw.get(
            "reformat_patterns", DEFAULT_REFORMAT_PATTERNS)],
        topogram_patterns = [str(p) for p in sc_raw.get(
            "topogram_patterns", DEFAULT_TOPOGRAM_PATTERNS)],
    )

    wb_raw = raw.get("wandb", {}) or {}
    wb_cfg = WandBConfig(
        enabled  = bool(wb_raw.get("enabled", False)),
        project  = str(wb_raw.get("project", "pdac-dataset-composer")),
        run_name = wb_raw.get("run_name"),
        tags     = [str(t) for t in wb_raw.get("tags", [])],
    )

    return ComposerConfig(
        paths             = paths,
        classifier        = classifier,
        id                = id_cfg,
        timepoint         = tp_cfg,
        scanner           = sc_cfg,
        keep_all          = bool(raw.get("keep_all", False)),
        wandb             = wb_cfg,
        unzip_archives    = bool(raw.get("unzip_archives", True)),
        delete_zips_after = bool(raw.get("delete_zips_after", False)),
        unzip_workers     = raw.get("unzip_workers"),
    )


def config_to_dict(cfg: ComposerConfig) -> dict:
    """Serialise a config back to a plain dict (for logging / saving).

    Args:
        cfg: Config to serialise.
    """
    return {
        "paths": {
            "output_root": str(cfg.paths.output_root),
            "source_dir":  str(cfg.paths.source_dir)  if cfg.paths.source_dir  else None,
            "scratch_dir": str(cfg.paths.scratch_dir) if cfg.paths.scratch_dir else None,
            "nifti_dir":   str(cfg.paths.nifti_dir)   if cfg.paths.nifti_dir   else None,
            "dcm2niix":    str(cfg.paths.dcm2niix)    if cfg.paths.dcm2niix    else None,
        },
        "id": {
            "patterns": cfg.id.patterns,
            "format":   cfg.id.format,
        },
        "timepoint": {
            "enabled":  cfg.timepoint.enabled,
            "labels":   cfg.timepoint.labels,
            "fallback": cfg.timepoint.fallback,
        },
        "classifier": {
            "backend":     cfg.classifier.backend,
            "device":      cfg.classifier.device,
            "regions":     [{"label": r.label, "structures": r.structures}
                            for r in cfg.classifier.regions],
            "keep_phases": cfg.classifier.keep_phases,
        },
        "scanner": {
            "reformat_patterns": cfg.scanner.reformat_patterns,
            "topogram_patterns": cfg.scanner.topogram_patterns,
        },
        "keep_all": cfg.keep_all,
    }
