from .schema import Datatype
from .utils import is_epi, collect_tokens
from .rules import (
    LOCALIZER_KEYWORDS,
    DERIVED_MAP_KEYWORDS,
    SURGICAL_NAV_KEYWORDS,
    DERIVED_TIMESTAMP_PATTERN,
    DIFFUSION_KEYWORDS,
    PERFUSION_KEYWORDS,
    FUNC_KEYWORDS,
    FMAP_KEYWORDS,
    ANAT_KEYWORDS,
)
from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset

from pathlib import Path
from typing import Optional

from rich.progress import track
import pandas as pd

def score_exclude(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score for exclusion from BIDS raw dataset.
    High scores indicate series that should be excluded:
    localizers, derived maps (ADC/FA/TRACE, CBF/CBV, etc.), MIPs,
    surgical navigation exports, ORIG duplicates, heavily reformatted/projection images, etc.
    """
    score = 0

    imagetype = series.image_type
    textmeta = series.text
    temporal = series.temporal
    spatial = series.spatial

    # --- Strong positive evidence for exclusion ---

    # Localizers (text tokens)
    if tokens & LOCALIZER_KEYWORDS:
        score += 6

    # Derived maps (FA, Trace, ADC, CBF, CBV, etc.) from text
    if tokens & DERIVED_MAP_KEYWORDS:
        score += 6

    # Surgical navigation exports
    if tokens & SURGICAL_NAV_KEYWORDS:
        score += 6

    # Timestamp-stamped derived maps in raw text (e.g., "ADC 2021-05-03 10:23")
    if textmeta and textmeta.series_description and textmeta.series_description.text:
        desc = textmeta.series_description.text
        if DERIVED_TIMESTAMP_PATTERN.search(desc):
            score += 5

        # "ORIG:" prefix often indicates duplicates/backups
        if desc.lower().startswith("orig:"):
            score += 4

    # ImageType-based exclusion cues
    if imagetype:
        # Explicit localizer flag
        if imagetype.is_localizer and imagetype.is_localizer.value:
            score += 6

        # MIP/projection/reformatted images
        if imagetype.is_mip and imagetype.is_mip.value:
            score += 4
        if imagetype.is_projection and imagetype.is_projection.value:
            score += 4
        if imagetype.is_reformatted and imagetype.is_reformatted.value:
            score += 4

        # Derived diffusion/perfusion maps
        if imagetype.is_adc and imagetype.is_adc.value:
            score += 5
        if imagetype.is_fa and imagetype.is_fa.value:
            score += 5
        if imagetype.is_trace and imagetype.is_trace.value:
            score += 5
        if imagetype.is_cbf and imagetype.is_cbf.value:
            score += 5
        if imagetype.is_cbv and imagetype.is_cbv.value:
            score += 5

    # --- Weaker structural cues for exclusion ---

    # Extremely thick slices + very few slices → often scouts/localizers
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    if thickness is not None and thickness >= 8.0:
        # Only if there is no strong evidence for being a main anat series
        score += 2

    # Very few slices and few timepoints: likely scout/localizer (already mostly caught above)
    n_tp = temporal.num_timepoints if temporal else None
    num_slices = series.geometry.num_slices if series.geometry else None
    if num_slices is not None and num_slices <= 5 and (n_tp is None or n_tp <= 3):
        score += 2

    return score

def score_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for DWI datatype."""
    score = 0

    diffusion = series.diffusion
    temporal = series.temporal
    imagetype = series.image_type
    perfusion = series.perfusion
    encoding = series.encoding

    # --- Convenience aliases ---
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    bvals = diffusion.b_values if (diffusion and diffusion.b_values) else None
    num_dirs = diffusion.num_diffusion_directions if diffusion else None
    num_vols = diffusion.num_diffusion_volumes if diffusion else None
    num_b0 = diffusion.num_b0 if diffusion else None
    has_diffusion_flag = bool(diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value)

    has_diffusion_imagetype = bool(
        imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value
    )
    has_adc = bool(imagetype and imagetype.is_adc and imagetype.is_adc.value)
    has_fa = bool(imagetype and imagetype.is_fa and imagetype.is_fa.value)
    has_trace = bool(imagetype and imagetype.is_trace and imagetype.is_trace.value)

    has_perfusion_flag = bool(
        (imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value)
        or (perfusion and (
            perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent
        ))
    )

    # --- Positive evidence for DWI ---

    # Non-zero b-values -> strong diffusion evidence
    has_nonzero_bvals = bool(
        bvals and any(b > 50 for b in bvals)
    )
    if has_nonzero_bvals:
        score += 4

    # High b-values (e.g., >= 800–1000) -> even stronger DWI
    if bvals and any(b >= 800 for b in bvals):
        score += 2

    # Many diffusion directions -> strong evidence
    has_gradients = bool(num_dirs is not None and num_dirs >= 6)
    if has_gradients:
        score += 4

    # Many diffusion volumes (multi-shell / multi-direction)
    if num_vols is not None and num_vols >= 20:
        score += 2

    # Presence of multiple b0 volumes also supports DWI
    if num_b0 is not None and num_b0 >= 2:
        score += 1

    # ImageType signals: "DIFFUSION"
    if has_diffusion_imagetype or has_diffusion_flag:
        score += 3

    # Text signals - "dti" or "dwi" or "diffusion"
    if tokens & DIFFUSION_KEYWORDS:
        score += 3

    # Typical DWI is EPI with moderate timepoints; n_tp ~ number of volumes
    if epi and n_tp is not None:
        # Many volumes but not crazy-high like fMRI multi-run
        if n_tp >= 10:
            score += 1
        if n_tp >= 20:
            score += 1

    # --- Support for derived diffusion maps as DWI datatype vs fmap/perf/func ---

    # ADC / FA / TRACE maps: we still want them under DWI datatype, not fmap/func
    if has_adc or has_fa or has_trace:
        score += 2

    # --- Negative / mutual-exclusion evidence ---

    # Perfusion-like evidence: penalize DWI if strong perfusion cues dominate
    if has_perfusion_flag:
        score -= 4

    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40:
        score -= 2

    # Strong functional evidence: EPI + many timepoints + FUNC_KEYWORDS
    if epi and n_tp is not None and n_tp > 50 and (tokens & FUNC_KEYWORDS):
        score -= 4

    # Strong anatomical naming with no diffusion physics
    if (tokens & ANAT_KEYWORDS) and not (has_nonzero_bvals or has_gradients or has_diffusion_imagetype):
        score -= 4

    # If ImageType explicitly says perfusion, penalize
    if imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value:
        score -= 3

    return score


def score_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perfusion datatype."""
    score = 0

    perf = series.perfusion
    temporal = series.temporal
    imagetype = series.image_type
    diffusion = series.diffusion
    encoding = series.encoding

    # --- Convenience aliases ---
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    has_perfusion_imagetype = bool(
        imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value
    )

    # Strong diffusion evidence (for penalties)
    has_diffusion_flag = bool(
        diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value
    )

    # --- Positive evidence for perfusion ---

    if perf:
        # ASL / perfusion-specific tags
        if perf.perfusion_labeling_type and perf.perfusion_labeling_type.value:
            # Strong ASL evidence
            score += 5

        if perf.perfusion_series_type and perf.perfusion_series_type.value:
            # e.g., "PERFUSION", "CBF", "CBV", etc.
            score += 4

        if perf.contrast_agent and perf.contrast_agent.value:
            # Contrast bolus metadata present
            score += 2

        # Many timepoints with regular temporal spacing is typical for DSC/ASL
        if perf.num_timepoints and perf.num_timepoints > 40:
            score += 2

        if perf.temporal_spacing and perf.temporal_spacing.value:
            # Presence of a well-defined temporal spacing supports perfusion time-series
            score += 1

    # ImageType perfusion indicator
    if has_perfusion_imagetype:
        score += 4

    # Text tokens - strong perfusion cues (ASL, CBF, DSC, perfusion, etc.)
    if tokens & PERFUSION_KEYWORDS:
        score += 4

    # EPI perfusion (ASL or DSC) pattern
    if epi and (tokens & PERFUSION_KEYWORDS):
        score += 2

    # Generic time-series evidence: many timepoints, not classic fMRI task naming
    if n_tp is not None and n_tp > 40:
        score += 1

    # --- Negative / mutual-exclusion evidence ---

    # Strong diffusion physics: penalize perf if clearly DWI
    if has_diffusion_flag:
        score -= 4

    if diffusion and diffusion.num_diffusion_directions and diffusion.num_diffusion_directions >= 6:
        score -= 2

    # Strong functional cues: EPI + many timepoints + functional tokens → less likely pure perfusion
    if epi and n_tp is not None and n_tp > 50 and (tokens & FUNC_KEYWORDS):
        score -= 3

    # Strong anatomical naming without perfusion tags
    if (tokens & ANAT_KEYWORDS) and not (
        tokens & PERFUSION_KEYWORDS or has_perfusion_imagetype or (perf and perf.perfusion_labeling_type)
    ):
        score -= 2

    # If ImageType explicitly says DIFFUSION, penalize as perf
    if imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value:
        score -= 2

    return score

def score_fmap(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for fieldmap datatype."""
    score = 0

    imagetype = series.image_type
    multiecho = series.multi_echo
    temporal = series.temporal
    encoding = series.encoding
    diffusion = series.diffusion
    perfusion = series.perfusion

    # --- Convenience aliases ---
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    # --- Positive evidence for fieldmap ---

    # Text tokens (fieldmap, topup, phasecorr, etc.)
    if tokens & FMAP_KEYWORDS:
        score += 5

    # Classic phase/magnitude with few timepoints
    if imagetype:
        is_mag = bool(imagetype.is_magnitude and imagetype.is_magnitude.value)
        is_phase = bool(imagetype.is_phase and imagetype.is_phase.value)
        if (is_mag or is_phase) and n_tp is not None and n_tp <= 2:
            score += 4

        # Real/imaginary pairs (alternative phase encoding representation)
        is_real = bool(imagetype.is_real and imagetype.is_real.value)
        is_imag = bool(imagetype.is_imaginary and imagetype.is_imaginary.value)
        if (is_real or is_imag) and n_tp is not None and n_tp <= 2:
            score += 3

    # Multi-echo GRE (common for fieldmaps), especially if not anatomical
    if multiecho and multiecho.num_echoes and multiecho.num_echoes > 1:
        if not (tokens & ANAT_KEYWORDS):
            score += 3
        else:
            score += 1  # Still possible, but less confident

    # EPI fieldmaps: short time-series (2-4 volumes) + fmap keywords
    if epi and n_tp is not None and 2 <= n_tp <= 4 and (tokens & FMAP_KEYWORDS):
        score += 3

    # --- Negative / mutual-exclusion evidence ---

    # Many timepoints → unlikely to be a classic fmap
    if n_tp is not None and n_tp > 10:
        score -= 3

    # Strong diffusion physics
    if diffusion and (
        (diffusion.has_diffusion and diffusion.has_diffusion.value) or
        (diffusion.num_diffusion_directions and diffusion.num_diffusion_directions >= 6)
    ):
        score -= 4

    # Strong perfusion cues
    if perfusion and (
        (perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value) or
        (perfusion.num_timepoints and perfusion.num_timepoints > 40)
    ):
        score -= 3

    # Strong functional evidence: EPI + many timepoints + func tokens
    if epi and n_tp is not None and n_tp > 20 and (tokens & FUNC_KEYWORDS):
        score -= 3

    # Strong anatomical naming + multi-echo + high-resolution → likely ME-MPRAGE
    if (tokens & ANAT_KEYWORDS) and multiecho and multiecho.num_echoes and multiecho.num_echoes > 1:
        score -= 2

    return score

def score_func(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for functional datatype."""
    score = 0

    temporal = series.temporal
    encoding = series.encoding
    imagetype = series.image_type
    diffusion = series.diffusion
    perfusion = series.perfusion

    # --- Convenience aliases ---
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)
    tr_bucket = temporal.tr_bucket if temporal else None

    # --- Positive evidence for functional ---

    # Text tokens - "fmri", task names, "bold", "rest", etc.
    if tokens & FUNC_KEYWORDS:
        score += 5

    # Classic BOLD/fMRI: EPI + many timepoints + short/medium TR
    if epi:
        if n_tp is not None:
            if n_tp >= 50:
                score += 4
            elif n_tp >= 20:
                score += 3
            elif n_tp >= 10:
                score += 1

        if tr_bucket in ("short", "medium"):
            score += 2

    # Multiband / parallel imaging (common for high-res fMRI)
    if encoding:
        if (encoding.multiband_factor and encoding.multiband_factor.value and encoding.multiband_factor.value > 1):
            score += 1
        if (encoding.parallel_reduction_factor_in_plane and encoding.parallel_reduction_factor_in_plane.value):
            score += 1

    # Task/resting-state text without conflicting modality keywords
    if tokens & FUNC_KEYWORDS and not (tokens & (ANAT_KEYWORDS | DIFFUSION_KEYWORDS | PERFUSION_KEYWORDS)):
        score += 1

    # --- Negative / mutual-exclusion evidence ---

    # Very few timepoints and EPI (likely fmap or dwi)
    if epi and n_tp is not None and n_tp < 10:
        score -= 3

    # Strong fieldmap cues
    if tokens & FMAP_KEYWORDS:
        score -= 2

    # Strong diffusion physics
    if diffusion and (
        (diffusion.has_diffusion and diffusion.has_diffusion.value) or
        (diffusion.num_diffusion_directions and diffusion.num_diffusion_directions >= 6)
    ):
        score -= 4

    # Strong perfusion cues
    if perfusion and (
        (perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value) or
        (perfusion.num_timepoints and perfusion.num_timepoints > 40)
    ):
        score -= 3

    # Strong anatomical naming + few timepoints + non-EPI
    if (tokens & ANAT_KEYWORDS) and n_tp is not None and n_tp <= 3 and not epi:
        score -= 3

    return score

def score_anat(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for anatomical datatype."""
    score = 0

    temporal = series.temporal
    spatial = series.spatial
    sequence = series.sequence
    multiecho = series.multi_echo
    diffusion = series.diffusion
    perfusion = series.perfusion
    imagetype = series.image_type
    contrast = series.contrast
    encoding = series.encoding

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)
    is3d = bool(temporal and temporal.is_3D and temporal.is_3D)
    is_iso = bool(temporal and temporal.is_isotropic and temporal.is_isotropic)
    tr_bucket = temporal.tr_bucket if temporal else None

    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None

    has_mpr = bool(imagetype and imagetype.is_mpr and imagetype.is_mpr.value)
    has_diffusion_flag = bool(
        (imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value)
        or (diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value)
    )
    has_perfusion_flag = bool(
        (imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value)
        or (perfusion and (
            perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent
        ))
    )
    has_contrast = bool(contrast and contrast.contrast_agent and contrast.contrast_agent.value)

    num_echoes = multiecho.num_echoes if multiecho else None
    is_multi_echo = bool(num_echoes and num_echoes > 1)

    # --- Strong positive evidence ---

    # Text tokens - T1, T2, FLAIR, T2*, MAGiC, etc.
    if tokens & ANAT_KEYWORDS:
        score += 5

    # ImageType indicates MPR / MPRAGE-style anatomical
    if has_mpr:
        score += 3

    # Single/few-volume non-EPI is likely anatomical
    if n_tp is not None and n_tp <= 3 and not epi:
        score += 3

    # 3D acquisition (from temporal or sequence)
    if is3d or (
        sequence and sequence.mr_acquisition_type 
        and sequence.mr_acquisition_type.value 
        and str(sequence.mr_acquisition_type.value).lower() == "3d"
        ):
        score += 2

    # High-resolution geometry (thin slices, isotropic)
    if thickness is not None:
        if thickness < 1.5:
            score += 2
        elif thickness < 2.5:
            score += 1

    if is_iso:
        score += 1

    # Long TR + non-EPI + few timepoints (structural-like)
    if tr_bucket == "long" and not epi and n_tp is not None and n_tp <= 3:
        score += 1

    # Post-contrast anatomical: contrast agent + non-EPI + few TPs
    if has_contrast and not epi and n_tp is not None and n_tp <= 5:
        score += 1

    # --- Negative / mutual-exclusion evidence ---

    # Strong DWI evidence: diffusion flags, many directions, high b-values
    if has_diffusion_flag:
        score -= 4

    # Strong perfusion evidence: ASL/DSC tags or many perfusion timepoints
    if has_perfusion_flag:
        score -= 4

    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40:
        score -= 2

    # Functional-like EPI with many timepoints and functional tokens
    if epi and n_tp is not None and n_tp > 10 and (tokens & FUNC_KEYWORDS):
        score -= 4

    # Generic penalty: many timepoints for non-multi-echo series
    if n_tp is not None and n_tp > 10 and not is_multi_echo:
        score -= 3

    # Very thick slices without strong anat cues → likely scout/localizer
    if thickness is not None and thickness > 5.0 and not (tokens & ANAT_KEYWORDS) and not has_mpr:
        score -= 2

    # If ImageType explicitly says localizer, penalize heavily (normally excluded earlier)
    if imagetype and imagetype.is_localizer and imagetype.is_localizer.value:
        score -= 5

    return score

def score_datatype(
    series: SeriesFeatures,
    tokens: set[str],
    exclude_threshold: int = 5
    ) -> tuple[dict[str, int], str, float]:
    """
    Compute scores for each BIDS datatype and return the best guess with confidence thresholds.
    
    Returns:
        scores: dict mapping each datatype to its score
        best_type: the datatype with the highest score (or "exclude"/"unknown" if
                     thresholds not met)
        confidence: a float between 0 and 1 representing confidence in the best_type assignment
        exclude_threshold: minimum score for the "exclude" label to override other datatypes (tune based on dev-set performance)
    The confidence calculation can be based on the best score, margin over runner-up, and absolute thresholds.
        
    The scoring thresholds and confidence calculation can be tuned based on dev-set performance.
    """
    
    # Compute all scores
    scores = {
        Datatype.ANAT.value: score_anat(series, tokens),
        Datatype.FUNC.value: score_func(series, tokens),
        Datatype.DWI.value: score_dwi(series, tokens),
        Datatype.PERF.value: score_perf(series, tokens),
        Datatype.FMAP.value: score_fmap(series, tokens),
        Datatype.EXCLUDE.value: score_exclude(series, tokens),
    }
    
    # Exclusion takes priority: if exclude score is high, return exclude regardless of others
    exclude_score = scores[Datatype.EXCLUDE.value]
    if exclude_score >= exclude_threshold:  # Tune this threshold based on dev-set precision
        confidence: float = exclude_score / 10  # Example confidence scaling (0.5 for score of 5, 0.8 for score of 8, etc.)
        return scores, Datatype.EXCLUDE.value, confidence
    
    # Filter out exclude and pick best among remaining datatypes
    candidate_scores = {k: v for k, v in scores.items()
                       if k != Datatype.EXCLUDE.value}
    
    best_type, best_score = max(candidate_scores.items(), key=lambda kv: kv[1])
    
    # Require minimum evidence threshold
    if best_score < 2:
        return scores, Datatype.UNKNOWN.value, 0.0
    
    # Require margin over runner-up
    sorted_scores = sorted(candidate_scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0
    margin = best_score - second_best
    
    if margin < 1:
        return scores, Datatype.UNKNOWN.value, 0.0
    
    confidence: float = min(1.0, (best_score * margin) / 20.0)
    return scores, best_type, confidence

def all_datatype_scores(datasets: list[Dataset]) -> pd.DataFrame:
    """
    Compute datatype scores for all series in all datasets and return a DataFrame.
    This can be used for analysis, threshold tuning, and error analysis.
    """
    records = []
    for dataset in track(datasets, description="Computing datatype scores..."):
        dataset_series_features: dict[str, dict[str, dict[str, SeriesFeatures]]] = dataset.generate_features()
        for subject_id, sessions in dataset_series_features.items():
            for session_id, series_dict in sessions.items():
                for series_id, series_features in series_dict.items():
                    tokens = collect_tokens(series_features)
                    scores, best_type, confidence = score_datatype(series_features, tokens)
                    record = {
                        "dataset": dataset.dir_root,
                        "subject": subject_id,
                        "session": session_id,
                        "series": series_id,
                        **scores,
                        "best_type": best_type,
                        "confidence": confidence,
                    }
                    records.append(record)
    df = pd.DataFrame.from_records(records)
    return df