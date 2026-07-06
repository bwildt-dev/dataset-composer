"""DICOM → NIfTI dataset assembly."""

from .classifier import (
    BOAClassifier,
    Classification,
    DummyClassifier,
    RegionSpec,
    SeriesClassifier,
    build_classifier,
)
from .config import ClassifierConfig, ComposerConfig, PathsConfig, WandBConfig, load_config
from .reporting import WandBReporter, init_wandb
from .pipeline import compose, compose_from_niftis, populate_state_from_nifti_dir
from .scanner import SeriesCandidate, SkippedSeries
from .state import ComposerState, merge_shard_states
from .unzipper import ExtractionResult, extract_all_zips

__all__ = [
    # Classifier
    "BOAClassifier",
    "Classification",
    "DummyClassifier",
    "SeriesClassifier",
    "build_classifier",
    "RegionSpec",
    # Config
    "ClassifierConfig",
    "ComposerConfig",
    "PathsConfig",
    "WandBConfig",
    "load_config",
    # Reporting
    "WandBReporter",
    "init_wandb",
    # Pipeline
    "compose",
    "compose_from_niftis",
    "populate_state_from_nifti_dir",
    # Scanner
    "SeriesCandidate",
    "SkippedSeries",
    # State
    "ComposerState",
    "merge_shard_states",
    # Unzipper
    "ExtractionResult",
    "extract_all_zips",
]
