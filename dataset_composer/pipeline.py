"""End-to-end dataset composition orchestrator.

Three resumable phases: discover (DICOM header filters), convert (dcm2niix → scratch),
classify (AI region + phase → final output tree).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .classifier import Classification, SeriesClassifier
from .converter import ConversionResult, convert_series, find_dcm2niix
from .extractors import (
    IdExtractor,
    TimepointExtractor,
    _DEFAULT_ID_EXTRACTOR,
    _DEFAULT_TP_EXTRACTOR,
    read_series_tags,
)
from .scanner import (
    SeriesCandidate,
    filter_series,
    find_series_dirs,
)
from .state import (
    CLASSIFICATION_HEADER,
    CONVERSION_HEADER,
    SKIPPED_HEADER,
    ClassificationRecord,
    ComposerState,
    init_csv,
    safe_write_row,
)
from .unzipper import extract_all_zips
from .reporting import _NullReporter

logger = logging.getLogger(__name__)


def _shard_of(path: object, n: int) -> int:
    """Return which of `n` shards owns `path`, via a stable path hash.

    Args:
        path: NIfTI path (or its string form) to assign to a shard.
        n: Total number of shards.

    Returns:
        Shard index `k` in `[0, n)`.
    """
    return int(hashlib.md5(str(path).encode()).hexdigest(), 16) % n


def discover(
    source_dir: Path,
    state: ComposerState,
    skipped_log: Path,
    id_extractor: IdExtractor = _DEFAULT_ID_EXTRACTOR,
    tp_extractor: TimepointExtractor = _DEFAULT_TP_EXTRACTOR,
    reformat_re: Optional[object] = None,
    topogram_re: Optional[object] = None,
) -> List[SeriesCandidate]:
    """Walk `source_dir` and return the header-filtered candidate list.

    Args:
        source_dir: DICOM source tree to walk.
        state: Composer state, used to skip already-converted series.
        skipped_log: CSV path to append skipped-series rows to.
        id_extractor: Extracts the patient ID from each series path.
        tp_extractor: Extracts the timepoint label from each series path.
        reformat_re: Compiled regex from
            `scanner.compile_filter_patterns(cfg.scanner.reformat_patterns)`;
            None falls back to the module-level default in scanner.py.
        topogram_re: Same as `reformat_re`, for topogram/admin series.

    Returns:
        The surviving `SeriesCandidate` list.
    """
    init_csv(skipped_log, SKIPPED_HEADER)

    logger.info("Discovery: walking %s", source_dir)
    series_dirs = find_series_dirs(source_dir)
    logger.info("Discovery: found %d series directories", len(series_dirs))

    convs = state.all_conversions()
    already_done_uids  = {r.series_uid for r in convs if r.series_uid}
    already_done_paths = {r.series_dir for r in convs if r.series_dir}
    candidates, skipped = filter_series(
        series_dirs,
        skip_existing_uids=already_done_uids,
        skip_existing_paths=already_done_paths,
        id_extractor=id_extractor,
        tp_extractor=tp_extractor,
        reformat_re=reformat_re,
        topogram_re=topogram_re,
    )

    for s in skipped:
        safe_write_row(skipped_log, [
            s.patient_id or "UNKNOWN", s.timepoint,
            str(s.series_dir), s.series_dir.name, s.reason,
        ])

    logger.info("Discovery: %d candidates, %d skipped, %d already converted",
                len(candidates), len(skipped), len(already_done_uids))
    return candidates


def convert(
    candidates: List[SeriesCandidate],
    raw_nifti_dir: Path,
    state: ComposerState,
    conversion_log: Path,
    dcm2niix_path: Optional[Path] = None,
    timeout_s: int = 600,
) -> List[Path]:
    """Convert each candidate to NIfTI; return list of produced paths.

    Args:
        candidates: Series to convert.
        raw_nifti_dir: Root directory for converted NIfTIs.
        state: Composer state; conversions are recorded here for resume.
        conversion_log: CSV path to append per-series conversion rows to.
        dcm2niix_path: Explicit dcm2niix binary path; None searches PATH.
        timeout_s: Per-series subprocess timeout, in seconds.

    Returns:
        Paths to the produced NIfTIs, one per successfully converted
        candidate.
    """
    init_csv(conversion_log, CONVERSION_HEADER)
    raw_nifti_dir.mkdir(parents=True, exist_ok=True)
    dcm2niix = find_dcm2niix(dcm2niix_path)

    produced: List[Path] = []
    n_total = len(candidates)
    for i, cand in enumerate(candidates, 1):
        existing = state.get_conversion(cand.series_uid)
        if existing is not None and Path(existing.nifti_path).exists():
            produced.append(Path(existing.nifti_path))
            logger.debug("[%d/%d] %s: cache hit", i, n_total, cand.series_dir.name)
            continue

        # Suffix with the last 8 chars of series UID to avoid collisions when
        # two series share identical folder names.
        pid = cand.patient_id or "UNKNOWN"
        uid_short = (cand.series_uid or "noUID")[-8:]
        stem = f"{pid}__{cand.timepoint}__{uid_short}"

        out_subdir = raw_nifti_dir / pid / cand.timepoint

        # On-disk safety net: if a NIfTI matching this stem already exists,
        # adopt it instead of re-running dcm2niix.
        import re as _re_local
        sanitized = _re_local.sub(r"_+", "_", stem)
        on_disk: List[Path] = []
        if out_subdir.exists():
            on_disk = sorted(
                set(out_subdir.glob(f"{stem}*.nii.gz"))
                | set(out_subdir.glob(f"{sanitized}*.nii.gz"))
            )
        if on_disk:
            primary = max(on_disk, key=lambda p: p.stat().st_size)
            state.record_conversion(
                series_uid=cand.series_uid,
                series_dir=cand.series_dir,
                patient_id=cand.patient_id,
                timepoint=cand.timepoint,
                nifti_path=primary,
            )
            produced.append(primary)
            logger.info("[%d/%d] %s: skip — already on disk (%s)",
                        i, n_total, cand.series_dir.name, primary.name)
            continue

        t_start = time.time()
        result: ConversionResult = convert_series(
            series_dir=cand.series_dir,
            output_dir=out_subdir,
            filename_stem=stem,
            dcm2niix=dcm2niix,
            timeout_s=timeout_s,
        )
        elapsed = time.time() - t_start

        ts = datetime.now().isoformat(timespec="seconds")
        if result:
            state.record_conversion(
                series_uid=cand.series_uid,
                series_dir=cand.series_dir,
                patient_id=cand.patient_id,
                timepoint=cand.timepoint,
                nifti_path=result.nifti,
            )
            safe_write_row(conversion_log, [
                ts, pid, cand.timepoint, cand.n_slices,
                cand.slice_thickness_mm if cand.slice_thickness_mm is not None else "",
                cand.series_description, "converted",
                str(cand.series_dir), str(result.nifti),
                f"{elapsed:.1f}s",
            ])
            produced.append(result.nifti)
            logger.info("[%d/%d] %s: converted in %.1fs", i, n_total, cand.series_dir.name, elapsed)
        else:
            safe_write_row(conversion_log, [
                ts, pid, cand.timepoint, cand.n_slices,
                cand.slice_thickness_mm if cand.slice_thickness_mm is not None else "",
                cand.series_description, "failed",
                str(cand.series_dir), "", result.stderr,
            ])
            logger.error("[%d/%d] %s: conversion failed: %s",
                         i, n_total, cand.series_dir.name, result.stderr)

    logger.info("Conversion: %d NIfTIs in %s", len(produced), raw_nifti_dir)
    return produced


_DEFAULT_KEEP_PHASES: Tuple[str, ...] = ("arterial", "venous")

import os as _os  # noqa: E402
MAX_NIFTI_BYTES_FOR_BOA = int(
    _os.environ.get("COMPOSER_MAX_NIFTI_MB", "700")
) * 1_000_000


def classify_and_finalise(
    classifier: SeriesClassifier,
    state: ComposerState,
    output_root: Path,
    classification_log: Path,
    keep_only_relevant: bool = True,
    keep_phases: Tuple[str, ...] = _DEFAULT_KEEP_PHASES,
    reporter: Optional[object] = None,
) -> None:
    """Classify every converted NIfTI and copy keepers into the final layout.

    A scan is kept when `body_region != "other"` and `phase in keep_phases`.
    Already-classified NIfTIs are read from state and re-placed without
    re-running inference.

    Args:
        classifier: Backend used to classify each NIfTI.
        state: Composer state; source of conversions, sink for
            classifications.
        output_root: Root directory for the final composed dataset.
        classification_log: CSV path to append per-scan classification
            rows to.
        keep_only_relevant: If False, copy every classified scan
            regardless of `keep_phases`.
        keep_phases: Contrast phases to keep in the final output.
        reporter: Optional progress reporter; defaults to a no-op.
    """
    init_csv(classification_log, CLASSIFICATION_HEADER)
    rep = reporter if reporter is not None else _NullReporter()

    def _is_relevant(body_region: str, phase: str) -> bool:
        return body_region != "other" and phase in keep_phases

    records = state.all_conversions()
    logger.info("Classification: %d NIfTIs to consider", len(records))
    logger.info("  keep_phases   : %s", ", ".join(keep_phases))

    n_kept, n_dropped, n_failed = 0, 0, 0
    recent_scan_times: List[float] = []
    last_scan_seconds = 0.0  # most recent BOA call duration

    for i, rec in enumerate(records, 1):
        # Periodic progress to wandb
        if i == 1 or i % 20 == 0 or i == len(records):
            avg_scan = (
                sum(recent_scan_times) / len(recent_scan_times)
                if recent_scan_times else 0.0
            )
            rep.log({
                "classify/processed":        i,
                "classify/kept":             n_kept,
                "classify/dropped":          n_dropped,
                "classify/failed":           n_failed,
                "classify/progress_pct":     100.0 * i / max(1, len(records)),
                "classify/last_scan_seconds": round(last_scan_seconds, 2),
                "classify/avg_scan_seconds":  round(avg_scan, 2),
            })

        nifti = Path(rec.nifti_path)
        if not nifti.exists():
            logger.warning("[%d/%d] %s: NIfTI missing on disk, skipping",
                           i, len(records), nifti)
            continue

        prev = state.get_classification(nifti)
        if prev is not None:
            if not keep_only_relevant or _is_relevant(prev.body_region, prev.phase):
                _place_in_final(nifti, prev, output_root)
                n_kept += 1
            else:
                n_dropped += 1
            continue

        try:
            size_bytes = nifti.stat().st_size
        except OSError:
            size_bytes = 0
        if size_bytes > MAX_NIFTI_BYTES_FOR_BOA:
            size_mb = size_bytes / 1e6
            logger.warning(
                "[%d/%d] %s: too large (%.0f MB > %.0f MB) — skipping BOA",
                i, len(records), nifti.name, size_mb,
                MAX_NIFTI_BYTES_FOR_BOA / 1e6,
            )
            n_failed += 1
            state.record_classification(ClassificationRecord(
                nifti_path=str(nifti),
                patient_id=rec.patient_id,
                timepoint=rec.timepoint,
                body_region="other",
                phase="classify_error",
                body_confidence=0.0,
                phase_confidence=0.0,
                series_description=f"ERROR: NIfTI too large ({size_mb:.0f} MB)",
                protocol_name="",
            ))
            continue

        try:
            t_scan = time.time()
            cls: Classification = classifier.classify(nifti)
            last_scan_seconds = time.time() - t_scan
            recent_scan_times.append(last_scan_seconds)
            # Keep rolling window bounded so the average reflects current speed.
            if len(recent_scan_times) > 50:
                recent_scan_times.pop(0)
        except Exception as exc:
            logger.error("[%d/%d] %s: classification failed: %s",
                         i, len(records), nifti.name, exc, exc_info=True)
            n_failed += 1
            # Record the failure in state so resumes don't retry forever.
            state.record_classification(ClassificationRecord(
                nifti_path=str(nifti),
                patient_id=rec.patient_id,
                timepoint=rec.timepoint,
                body_region="other",
                phase="classify_error",
                body_confidence=0.0,
                phase_confidence=0.0,
                series_description=f"ERROR: {type(exc).__name__}: {str(exc)[:200]}",
                protocol_name="",
            ))
            continue

        # In classify-only mode series_dir is a NIfTI path, so the DICOM source
        # may not exist, read tags only when the path is a real directory.
        series_desc, protocol = "", ""
        src = Path(rec.series_dir)
        try:
            if src.is_dir():
                tags = read_series_tags(src)
                series_desc = tags.get("SeriesDescription", "")
                protocol    = tags.get("ProtocolName", "")
        except (PermissionError, OSError) as exc:
            logger.debug("Could not read DICOM tags from %s: %s", src, exc)

        crec = ClassificationRecord(
            nifti_path=str(nifti),
            patient_id=rec.patient_id,
            timepoint=rec.timepoint,
            body_region=cls.body_region,
            phase=cls.phase,
            body_confidence=cls.body_confidence,
            phase_confidence=cls.phase_confidence,
            series_description=series_desc,
            protocol_name=protocol,
        )
        state.record_classification(crec)

        relevant = _is_relevant(cls.body_region, cls.phase)
        if (not keep_only_relevant) or relevant:
            _place_in_final(nifti, crec, output_root)
            n_kept += 1
        else:
            n_dropped += 1

        safe_write_row(classification_log, [
            datetime.now().isoformat(timespec="seconds"),
            rec.patient_id or "UNKNOWN", rec.timepoint,
            cls.body_region, cls.phase,
            f"{cls.body_confidence:.3f}", f"{cls.phase_confidence:.3f}",
            "yes" if relevant else "no",
            series_desc, protocol, str(nifti),
        ])
        logger.info("[%d/%d] %s → %s/%s (%s)",
                    i, len(records), nifti.name, cls.body_region, cls.phase,
                    "kept" if relevant else "dropped")

    logger.info("Classification: kept=%d  dropped=%d  failed=%d",
                n_kept, n_dropped, n_failed)


def _place_in_final(nifti: Path, rec: ClassificationRecord, output_root: Path) -> None:
    """Copy a kept NIfTI to `<output>/<phase>/<patient_id>/[<timepoint>/]`.

    Args:
        nifti: Source NIfTI to copy.
        rec: Classification record providing patient_id/phase/timepoint.
        output_root: Root directory for the final composed dataset.
    """
    if not rec.patient_id:
        logger.warning("Skipping placement: %s has no patient_id", nifti.name)
        return
    parts = [output_root, rec.phase, rec.patient_id]
    if rec.timepoint:
        parts.append(rec.timepoint)
    dest_dir = Path(*parts)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / nifti.name
    if dest.exists() and dest.stat().st_size == nifti.stat().st_size:
        return
    shutil.copy2(nifti, dest)


def populate_state_from_nifti_dir(
    nifti_dir: Path,
    state: ComposerState,
    id_extractor: IdExtractor = _DEFAULT_ID_EXTRACTOR,
    tp_extractor: TimepointExtractor = _DEFAULT_TP_EXTRACTOR,
    shard: Optional[Tuple[int, int]] = None,
) -> int:
    """Scan `nifti_dir` for `*.nii.gz` and inject conversion records.

    Filenames in the composer's canonical form (`<PID>__<TP>__<uid>.nii.gz`)
    are parsed directly; otherwise `id_extractor` and `tp_extractor` are
    applied to the path. When `tp_extractor.enabled` is False, timepoint is
    stored as `""`.

    Args:
        nifti_dir: Directory tree to scan recursively for `*.nii.gz`.
        state: Composer state to inject conversion records into.
        id_extractor: Extracts the patient ID from each NIfTI path.
        tp_extractor: Extracts the timepoint label from each NIfTI path.
        shard: `(k, n)` registers only NIfTIs assigned to shard `k` by
            `_shard_of`, a stable path hash.

    Returns:
        The number of new records injected.
    """
    niftis = sorted(nifti_dir.rglob("*.nii.gz"))
    if shard is not None:
        k, n = shard
        total = len(niftis)
        niftis = [p for p in niftis if _shard_of(p, n) == k]
        logger.info(
            "Shard %d/%d (path-hash): selected %d of %d NIfTIs",
            k + 1, n, len(niftis), total,
        )
    already_represented = {
        Path(r.nifti_path) for r in state.all_conversions() if r.nifti_path
    }
    n_new = 0
    for nifti in niftis:
        if nifti in already_represented:
            continue
        # Split on one-or-more underscores to handle both.
        import re as _re
        parts = _re.split(r"_+", nifti.stem.replace(".nii", ""))
        if (
            tp_extractor.enabled
            and len(parts) >= 2
            and (parts[1] in tp_extractor.labels
                 or _re.fullmatch(r"t\d+", parts[1]) is not None)
        ):
            pid, tp = parts[0], parts[1]
        else:
            pid = id_extractor(nifti) or "UNKNOWN"
            tp  = tp_extractor(nifti)
        # No DICOM source → no SeriesInstanceUID. Use a synthetic key so the
        # record is uniquely identifiable but obviously not a real UID.
        state.record_conversion(
            series_uid=f"_nifti:{nifti}",
            series_dir=nifti,
            patient_id=pid,
            timepoint=tp,
            nifti_path=nifti,
        )
        n_new += 1

    logger.info("populate_state_from_nifti_dir: injected %d records from %s",
                n_new, nifti_dir)
    return n_new


def compose(
    source_dir:         Path,
    output_root:        Path,
    raw_nifti_dir:      Path,
    classifier:         SeriesClassifier,
    dcm2niix_path:      Optional[Path] = None,
    keep_only_relevant: bool = True,
    keep_phases:        Tuple[str, ...] = _DEFAULT_KEEP_PHASES,
    id_extractor:       IdExtractor = _DEFAULT_ID_EXTRACTOR,
    tp_extractor:       TimepointExtractor = _DEFAULT_TP_EXTRACTOR,
    unzip_archives:     bool = True,
    delete_zips_after:  bool = False,
    unzip_workers:      Optional[int] = None,
    reporter:           Optional[object] = None,
    convert_only:       bool = False,
    reformat_patterns:  Optional[List[str]] = None,
    topogram_patterns:  Optional[List[str]] = None,
) -> None:
    """Run all four phases (unzip → discover → convert → classify).

    Args:
        source_dir: DICOM source tree to walk.
        output_root: Root directory for the final composed dataset.
        raw_nifti_dir: Root directory for converted (pre-classification)
            NIfTIs.
        classifier: Backend used to classify each converted NIfTI.
        dcm2niix_path: Explicit dcm2niix binary path; None searches PATH.
        keep_only_relevant: If False, copy every classified scan
            regardless of `keep_phases`.
        keep_phases: Contrast phases to keep in the final output.
        id_extractor: Extracts the patient ID from each series path.
        tp_extractor: Extracts the timepoint label from each series path.
        unzip_archives: Whether to recursively unzip `*.zip` under
            `source_dir` before discovery.
        delete_zips_after: Delete archives after successful extraction.
        unzip_workers: Extraction worker count; None means
            `min(cpu_count(), 8)`.
        reporter: Pass a `WandBReporter` (from `reporting.init_wandb`) for
            progress reporting; defaults to a no-op reporter.
        convert_only: Stop after conversion (CPU-only run); resume
            classification later with `compose_from_niftis`.
        reformat_patterns: Config-driven reformat-filter regex fragments;
            None falls back to the scanner module defaults.
        topogram_patterns: Config-driven topogram-filter regex fragments;
            None falls back to the scanner module defaults.
    """
    rep = reporter if reporter is not None else _NullReporter()
    state_dir = output_root / "_state"
    log_dir   = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    state           = ComposerState(state_dir)
    skipped_log     = log_dir / f"skipped_{ts}.csv"
    conversion_log  = log_dir / f"conversion_{ts}.csv"
    classification_log = log_dir / f"classification_{ts}.csv"

    logger.info("Dataset composer")
    logger.info("  source        : %s", source_dir)
    logger.info("  output        : %s", output_root)
    logger.info("  raw nifti dir : %s", raw_nifti_dir)
    logger.info("  state dir     : %s", state_dir)
    logger.info("  classifier    : %s", classifier.__class__.__name__)

    if unzip_archives:
        with rep.stage("unzip"):
            extracted = extract_all_zips(source_dir,
                                         delete_after=delete_zips_after,
                                         n_workers=unzip_workers)
        rep.summary({
            "unzip/archives_extracted": len(extracted),
            "unzip/total_members": sum(e.n_members for e in extracted),
        })

    from .scanner import compile_filter_patterns as _compile_pats
    reformat_re = _compile_pats(reformat_patterns) if reformat_patterns is not None else None
    topogram_re = _compile_pats(topogram_patterns) if topogram_patterns is not None else None

    with rep.stage("discover"):
        candidates = discover(source_dir, state, skipped_log,
                              id_extractor=id_extractor, tp_extractor=tp_extractor,
                              reformat_re=reformat_re, topogram_re=topogram_re)
    rep.summary({
        "discover/candidates": len(candidates),
        "discover/unique_patients": len({c.patient_id for c in candidates if c.patient_id}),
    })

    with rep.stage("convert"):
        produced = convert(candidates, raw_nifti_dir, state, conversion_log,
                           dcm2niix_path=dcm2niix_path)
    rep.summary({
        "convert/produced": len(produced),
        "convert/failed":   len(candidates) - len(produced),
    })

    if convert_only:
        logger.info("convert_only=True — stopping after convert. "
                    "Run compose_from_niftis later on %s to classify.", raw_nifti_dir)
        rep.summary({
            "total/candidates": len(candidates),
            "total/converted":  len(produced),
            "mode":             "prep-only (no classification)",
        })
        rep.finish()
        return

    with rep.stage("classify"):
        classify_and_finalise(
            classifier, state, output_root, classification_log,
            keep_only_relevant=keep_only_relevant,
            keep_phases=keep_phases,
            reporter=rep,
        )

    # Final summary
    classifications = state.all_classifications()
    from collections import Counter
    phase_counts  = Counter(c.phase for c in classifications)
    region_counts = Counter(c.body_region for c in classifications)
    rep.summary({
        "total/candidates":     len(candidates),
        "total/converted":      len(produced),
        "total/classified":     len(classifications),
        **{f"phase/{p}":  n for p, n in phase_counts.items()},
        **{f"region/{r}": n for r, n in region_counts.items()},
    })
    rep.finish()

    logger.info("Composer done. Logs in %s", log_dir)


def compose_from_niftis(
    nifti_dir:          Path,
    output_root:        Path,
    classifier:         SeriesClassifier,
    keep_only_relevant: bool = True,
    keep_phases:        Tuple[str, ...] = _DEFAULT_KEEP_PHASES,
    id_extractor:       IdExtractor = _DEFAULT_ID_EXTRACTOR,
    tp_extractor:       TimepointExtractor = _DEFAULT_TP_EXTRACTOR,
    reporter:           Optional[object] = None,
    shard:              Optional[Tuple[int, int]] = None,
) -> None:
    """Classification-only pipeline for an existing NIfTI tree.

    Skips DICOM discovery and dcm2niix conversion. `nifti_dir` is walked
    recursively for `*.nii.gz` files; each is registered as a synthetic
    conversion record so `classify_and_finalise` can run normally.

    Args:
        nifti_dir: Directory tree to scan recursively for `*.nii.gz`.
        output_root: Root directory for the final composed dataset.
        classifier: Backend used to classify each NIfTI.
        keep_only_relevant: If False, copy every classified scan
            regardless of `keep_phases`.
        keep_phases: Contrast phases to keep in the final output.
        id_extractor: Extracts the patient ID from each NIfTI path.
        tp_extractor: Extracts the timepoint label from each NIfTI path.
        reporter: Optional progress reporter; defaults to a no-op.
        shard: Speed up classify-only mode by splitting a large NIfTI
            backlog across N parallel tasks.
    """
    suffix = ""
    if shard is not None:
        k, n = shard
        suffix = f"_shard{k}of{n}"
        logger.info("Sharding enabled: this is shard %d of %d", k + 1, n)

    state_dir = output_root / f"_state{suffix}"
    log_dir   = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    canonical_state_dir = output_root / "_state"
    if shard is not None and canonical_state_dir.exists():
        state_dir.mkdir(parents=True, exist_ok=True)
        k, n = shard

        # Filter conversion records by nifti_path → shard.
        conv_src = canonical_state_dir / "conversion_state.json"
        conv_dst = state_dir / "conversion_state.json"
        if conv_src.exists() and not conv_dst.exists():
            import json as _json
            full = _json.loads(conv_src.read_text())
            subset = {
                key: rec for key, rec in full.items()
                if _shard_of(rec.get("nifti_path", ""), n) == k
            }
            conv_dst.write_text(_json.dumps(subset, indent=2))
            logger.info(
                "Seeded shard state: %d of %d conversion records belong to shard %d/%d",
                len(subset), len(full), k + 1, n,
            )

        # Same filter for classification records.
        clf_src = canonical_state_dir / "classification_state.json"
        clf_dst = state_dir / "classification_state.json"
        if clf_src.exists() and not clf_dst.exists():
            import json as _json
            full = _json.loads(clf_src.read_text())
            subset = {
                path: rec for path, rec in full.items()
                if _shard_of(path, n) == k
            }
            clf_dst.write_text(_json.dumps(subset, indent=2))
            logger.info(
                "Seeded shard state: %d of %d classification records belong to shard %d/%d",
                len(subset), len(full), k + 1, n,
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    state              = ComposerState(state_dir)
    classification_log = log_dir / f"classification{suffix}_{ts}.csv"
    rep = reporter if reporter is not None else _NullReporter()

    logger.info("Dataset composer (classify-only mode%s)",
                f", shard {shard[0] + 1}/{shard[1]}" if shard else "")
    logger.info("  nifti_dir  : %s", nifti_dir)
    logger.info("  output     : %s", output_root)
    logger.info("  state dir  : %s", state_dir)
    logger.info("  classifier : %s", classifier.__class__.__name__)

    with rep.stage("populate_state"):
        n = populate_state_from_nifti_dir(nifti_dir, state,
                                           id_extractor=id_extractor,
                                           tp_extractor=tp_extractor,
                                           shard=shard)
    logger.info("Registered %d new NIfTIs for classification", n)
    rep.summary({"classify/total_niftis": len(state.all_conversions())})

    with rep.stage("classify"):
        classify_and_finalise(
            classifier, state, output_root, classification_log,
            keep_only_relevant=keep_only_relevant,
            keep_phases=keep_phases,
            reporter=rep,
        )
    rep.finish()

    logger.info("Composer (classify-only) done. Logs in %s", log_dir)
