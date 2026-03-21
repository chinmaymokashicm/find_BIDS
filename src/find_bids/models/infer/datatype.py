from .acquisition import (
    DIFFUSION_INTENT_THRESHOLD,
    EPI_FAMILY_THRESHOLD,
    FIELDMAP_INTENT_THRESHOLD,
    GRE_FAMILY_THRESHOLD,
    get_acquisition_family_scores,
    get_acquisition_intent_scores,
    is_epi,
    LOCALIZER_INTENT_THRESHOLD,
    PERFUSION_INTENT_THRESHOLD,
    SE_TSE_FAMILY_THRESHOLD,
)
from .schema import Datatype
from .utils import (
    collect_tokens,
    softmax,
    ScoringRule,
    apply_rules,
    flag_true,
    token_matches
    )
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


# ---------------------------------------------------------------------------
# Shared exclusion signals (used by score_exclude and score_derived)
# ---------------------------------------------------------------------------

def _score_exclusion_signals(series: SeriesFeatures, tokens: set[str]) -> float:
    imagetype = series.image_type
    textmeta = series.text
    temporal = series.temporal
    spatial = series.spatial

    desc = textmeta.series_description.text if (textmeta and textmeta.series_description) else None
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    n_tp = temporal.num_timepoints if temporal else None
    num_slices = series.geometry.num_slices if series.geometry else None

    return apply_rules(
        (token_matches(tokens, LOCALIZER_KEYWORDS), 6),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 6),
        (token_matches(tokens, SURGICAL_NAV_KEYWORDS), 6),
        (desc and bool(DERIVED_TIMESTAMP_PATTERN.search(desc)), 5),
        (desc and desc.lower().startswith("orig:"), 4),
        (imagetype and flag_true(imagetype, "is_localizer"), 6),
        (imagetype and flag_true(imagetype, "is_mip"), 4),
        (imagetype and flag_true(imagetype, "is_projection"), 4),
        (imagetype and flag_true(imagetype, "is_reformatted"), 4),
        (imagetype and flag_true(imagetype, "is_adc"), 5),
        (imagetype and flag_true(imagetype, "is_fa"), 5),
        (imagetype and flag_true(imagetype, "is_trace"), 5),
        (imagetype and flag_true(imagetype, "is_cbf"), 5),
        (imagetype and flag_true(imagetype, "is_cbv"), 5),
        (thickness is not None and thickness >= 8.0, 2),
        (num_slices is not None and num_slices <= 5 and (n_tp is None or n_tp <= 3), 2),
    )


# ! Not using it currently
def score_exclude(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for exclusion from BIDS raw dataset. High = should be excluded."""
    return _score_exclusion_signals(series, tokens)


# ! Not using it currently
def score_derived(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for being a derived map that should be excluded from raw BIDS."""
    return _score_exclusion_signals(series, tokens)

def score_localizer(series: SeriesFeatures, tokens: set[str]) -> float:
    imagetype = series.image_type
    textmeta = series.text

    desc = textmeta.series_description.text if (textmeta and textmeta.series_description) else None

    return apply_rules(
        (token_matches(tokens, LOCALIZER_KEYWORDS), 6),
        (desc and desc.lower().startswith("loc"), 4),
        (imagetype and flag_true(imagetype, "is_localizer"), 6),
    )

def score_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    temporal = series.temporal
    spatial = series.spatial
    sequence = series.sequence
    multiecho = series.multi_echo
    diffusion = series.diffusion
    perfusion = series.perfusion
    imagetype = series.image_type
    contrast = series.contrast

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series, tokens)
    is3d = bool(temporal and temporal.is_3D)
    is_iso = bool(temporal and temporal.is_isotropic)
    tr_bucket = temporal.tr_bucket if temporal else None
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    num_echoes = multiecho.num_echoes if multiecho else None
    is_multi_echo = bool(num_echoes and num_echoes > 1)

    has_mpr = bool(imagetype and flag_true(imagetype, "is_mpr"))
    has_diffusion_flag = bool(
        (imagetype and flag_true(imagetype, "has_diffusion"))
        or (diffusion and flag_true(diffusion, "has_diffusion"))
    )
    has_perfusion_flag = bool(
        (imagetype and flag_true(imagetype, "has_perfusion"))
        or (perfusion and (perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent))
    )
    has_contrast = bool(
        contrast and contrast.contrast_agent and contrast.contrast_agent.value
        and len(contrast.contrast_agent.value) > 0
    )
    is_3d_sequence = bool(
        sequence and sequence.mr_acquisition_type
        and sequence.mr_acquisition_type.value
        and str(sequence.mr_acquisition_type.value).lower() == "3d"
    )
    # TE ranges from 10 to 60 - weak evidence
    in_anat_te_range = bool(
        multiecho and multiecho.echo_times and multiecho.echo_times[0] >= 10 and multiecho.echo_times[-1] <= 60 # Echo times are expected to be in ascending order, so check first and last for range
    )

    return apply_rules(
        # Positive signals for anatomical scans
        (token_matches(tokens, ANAT_KEYWORDS), 5),
        (has_mpr, 3),
        (n_tp is not None and n_tp <= 3 and not epi, 3),
        (is3d or is_3d_sequence, 2),
        (thickness is not None and thickness < 1.5, 2),
        (thickness is not None and 1.5 <= thickness < 2.5, 1),
        (is_iso, 1),
        (tr_bucket == "long" and not epi and n_tp is not None and n_tp <= 3, 1),
        (has_contrast and not epi and n_tp is not None and n_tp <= 5, 1),
        (n_tp in (None, 1) and is3d and not epi and not has_diffusion_flag and not has_perfusion_flag, 4),
        (in_anat_te_range, 1),
        
        # Negative signals for anatomical scans
        (token_matches(tokens, LOCALIZER_KEYWORDS), -5),
        (has_contrast and n_tp is not None and n_tp > 10, -20),
        (token_matches(tokens, PERFUSION_KEYWORDS), -10),
        (has_diffusion_flag, -4),
        (has_perfusion_flag, -4),
        (perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40, -2),
        (epi and n_tp is not None and n_tp > 10 and token_matches(tokens, FUNC_KEYWORDS), -4),
        (n_tp is not None and n_tp > 10 and not is_multi_echo and not has_contrast, -3),
        (thickness is not None and thickness > 5.0 and not token_matches(tokens, ANAT_KEYWORDS) and not has_mpr, -2),
        (imagetype and flag_true(imagetype, "is_localizer"), -5),
    )

def score_dwi(series: SeriesFeatures, tokens: set[str]) -> float:
    diffusion = series.diffusion
    temporal = series.temporal
    imagetype = series.image_type
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series, tokens)
    bvals = diffusion.b_values if (diffusion and diffusion.b_values) else None
    num_dirs = diffusion.num_diffusion_directions if diffusion else None
    num_vols = diffusion.num_diffusion_volumes if diffusion else None
    num_b0 = diffusion.num_b0 if diffusion else None

    has_diffusion_flag = bool(
        (imagetype and flag_true(imagetype, "has_diffusion"))
        or (diffusion and flag_true(diffusion, "has_diffusion"))
    )
    has_adc = imagetype and flag_true(imagetype, "is_adc")
    has_fa = imagetype and flag_true(imagetype, "is_fa")
    has_trace = imagetype and flag_true(imagetype, "is_trace")
    has_perfusion_flag = bool(
        (imagetype and flag_true(imagetype, "has_perfusion"))
        or (perfusion and (perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent))
    )
    has_nonzero_bvals = bool(bvals and any(b > 50 for b in bvals))

    return apply_rules(
        # Positive signals for diffusion scans
        (has_nonzero_bvals, 4),
        (bvals and any(b >= 800 for b in bvals), 2),
        (num_dirs is not None and num_dirs >= 6, 4),
        (num_vols is not None and num_vols >= 20, 2),
        (num_b0 is not None and num_b0 >= 2, 1),
        (epi and n_tp is not None and n_tp >= 10, 1),
        (epi and n_tp is not None and n_tp >= 20, 1),
        (has_diffusion_flag, 3),
        (token_matches(tokens, DIFFUSION_KEYWORDS), 5),
        (epi and token_matches(tokens, DIFFUSION_KEYWORDS), 2),
        (has_adc or has_fa or has_trace, 2),
        ((has_adc or has_fa or has_trace) and n_tp is not None and n_tp <= 2, 1),
        
        # Negative signals for diffusion scans
        (token_matches(tokens, LOCALIZER_KEYWORDS), -5),
        (has_perfusion_flag, -4),
        (perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40, -2),
        (epi and n_tp is not None and n_tp > 50 and token_matches(tokens, FUNC_KEYWORDS), -4),
        (token_matches(tokens, ANAT_KEYWORDS) and not (has_nonzero_bvals or has_diffusion_flag or token_matches(tokens, DIFFUSION_KEYWORDS)), -3),
        (imagetype and flag_true(imagetype, "has_perfusion"), -3),
    )


def score_perf(series: SeriesFeatures, tokens: set[str]) -> float:
    perf = series.perfusion
    temporal = series.temporal
    imagetype = series.image_type
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series, tokens)

    has_contrast = bool(
        series.contrast and series.contrast.contrast_agent
        and series.contrast.contrast_agent.value
        and len(series.contrast.contrast_agent.value) > 0
    )

    return apply_rules(
        # Positive signals for perfusion scans
        (perf and perf.perfusion_labeling_type and perf.perfusion_labeling_type.value, 10),
        (perf and n_tp is not None and 60 <= n_tp <= 120, 4),
        (perf and perf.contrast_agent and perf.contrast_agent.value, 3),
        (imagetype and flag_true(imagetype, "has_perfusion"), 5),
        (has_contrast and epi and n_tp is not None and n_tp > 10, 10),
        (epi and token_matches(tokens, PERFUSION_KEYWORDS), 6),
        
        # Negative signals for perfusion scans
        (token_matches(tokens, LOCALIZER_KEYWORDS), -5),
        (token_matches(tokens, ANAT_KEYWORDS) and not (token_matches(tokens, PERFUSION_KEYWORDS) or perf or has_contrast), -5),
    )


def score_func(series: SeriesFeatures, tokens: set[str]) -> float:
    temporal = series.temporal
    diffusion = series.diffusion
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series, tokens)
    tr_bucket = temporal.tr_bucket if temporal else None
    is3d = bool(temporal and temporal.is_3D)

    has_contrast = bool(
        series.contrast and series.contrast.contrast_agent
        and series.contrast.contrast_agent.value
        and len(series.contrast.contrast_agent.value) > 0
    )

    return apply_rules(
        # Positive signals for functional scans
        (token_matches(tokens, FUNC_KEYWORDS), 5),
        (epi and n_tp is not None and n_tp >= 50, 4),
        (epi and n_tp is not None and 20 <= n_tp < 50, 3),
        (epi and tr_bucket in ("short", "medium"), 2),
        
        # Negative signals for functional scans
        (token_matches(tokens, LOCALIZER_KEYWORDS), -5),
        (is3d, -8),
        (diffusion and flag_true(diffusion, "has_diffusion"), -10),
        (not epi, -10),
        (n_tp is not None and n_tp <= 1, -15),
        (token_matches(tokens, PERFUSION_KEYWORDS), -15),
        (has_contrast, -20),
        (token_matches(tokens, ANAT_KEYWORDS) and (n_tp is None or n_tp < 10 or not epi), -15),
        (perfusion and perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value, -20),
    )
    
def score_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    imagetype = series.image_type
    multiecho = series.multi_echo
    temporal = series.temporal
    spatial = series.spatial
    encoding = series.encoding

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series, tokens)
    is3d = bool(temporal and temporal.is_3D)
    num_echoes = multiecho.num_echoes if multiecho else None
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    is_mag = imagetype and flag_true(imagetype, "is_magnitude")
    is_phase = imagetype and flag_true(imagetype, "is_phase")

    return apply_rules(
        # Positive signals for fieldmaps
        (epi and n_tp is not None and 2 <= n_tp <= 6, 3),
        (n_tp is not None and n_tp <= 2 and (is_mag or is_phase), 5),
        (token_matches(tokens, FMAP_KEYWORDS), 6),
        (num_echoes is not None and num_echoes > 2, 4),
        (epi and series.num_volumes is not None and series.num_volumes <= 10, 3), # EPI fieldmaps often have multiple volumes but still a low number
        (encoding and encoding.phase_encoding_direction and encoding.phase_encoding_direction.value is not None, 2), # Fieldmaps typically populate phase encoding direction
        
        # Negative signals for fieldmaps
        (token_matches(tokens, LOCALIZER_KEYWORDS), -5),
        (token_matches(tokens, ANAT_KEYWORDS) and not token_matches(tokens, FMAP_KEYWORDS), -10),
        (is3d, -20),
        (thickness is not None and thickness < 1.5, -15),
        (token_matches(tokens, ANAT_KEYWORDS) and num_echoes is not None and num_echoes > 1, -15),
    )


def _apply_acquisition_family_guidance(
    raw_scores: dict[str, float],
    series: SeriesFeatures,
    tokens: set[str],
) -> None:
    family_scores = get_acquisition_family_scores(series, tokens)
    intent_scores = get_acquisition_intent_scores(series, tokens)
    num_echoes = series.multi_echo.num_echoes if series.multi_echo else None

    raw_scores[Datatype.ANAT.value] += apply_rules(
        (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 2),
        (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -5),
        (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, -4),
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -4),
    )
    raw_scores[Datatype.FUNC.value] += apply_rules(
        (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 2),
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -5),
        (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, -3),
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -3),
        (intent_scores["localizer"] >= LOCALIZER_INTENT_THRESHOLD, -4),
    )
    raw_scores[Datatype.DWI.value] += apply_rules(
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, 5),
        (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 1),
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -3),
        (intent_scores["localizer"] >= LOCALIZER_INTENT_THRESHOLD, -4),
    )
    raw_scores[Datatype.PERF.value] += apply_rules(
        (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 4),
        (family_scores["epi"] >= EPI_FAMILY_THRESHOLD and intent_scores["fieldmap"] < FIELDMAP_INTENT_THRESHOLD, 1),
        (family_scores["gre"] >= GRE_FAMILY_THRESHOLD and intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 1),
        (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD and intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 1),
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -4),
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -3),
    )
    raw_scores[Datatype.FMAP.value] += apply_rules(
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 5),
        (family_scores["epi"] >= EPI_FAMILY_THRESHOLD and intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 2),
        (family_scores["gre"] >= GRE_FAMILY_THRESHOLD and num_echoes is not None and num_echoes > 1, 2),
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -4),
        (intent_scores["localizer"] >= LOCALIZER_INTENT_THRESHOLD, -4),
    )
    raw_scores[Datatype.LOCALIZER.value] += apply_rules(
        (intent_scores["localizer"] >= LOCALIZER_INTENT_THRESHOLD, 6),
        (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 1),
        (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 1),
        (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -4),
        (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -2),
    )

def score_datatype(
    series: SeriesFeatures,
    tokens: set[str],
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for each BIDS datatype.

    Returns
    -------
    probs : dict[str, float]
        Probability for each datatype (softmax output)

    best_type : str
        Predicted datatype or "unknown"

    confidence : float
        Margin between top two candidate probabilities
    """
    raw_scores = {
        Datatype.ANAT.value: score_anat(series, tokens),
        Datatype.FUNC.value: score_func(series, tokens),
        Datatype.DWI.value: score_dwi(series, tokens),
        Datatype.PERF.value: score_perf(series, tokens),
        Datatype.FMAP.value: score_fmap(series, tokens),
        Datatype.LOCALIZER.value: score_localizer(series, tokens),
        # Exclude and unknown are not scored here - they are fallbacks based on margins and thresholds
    }

    _apply_acquisition_family_guidance(raw_scores, series, tokens)

    probs = softmax(raw_scores)
    if not probs:
        return {}, "unknown", 0.0

    best_datatype = max(probs, key=lambda k: probs[k])

    sorted_probs = sorted(probs.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    if margin < 0.2:
        best_datatype = "unknown"
        confidence = 0.0
    else:
        confidence = margin

    return probs, best_datatype, confidence