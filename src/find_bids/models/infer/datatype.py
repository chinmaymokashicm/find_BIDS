from .schema import Datatype
from .utils import is_epi, collect_tokens, softmax
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

# ! Not using it currently
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

# ! Not using it currently
def score_derived(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score for being a derived map (FA/ADC/Trace, CBF/CBV, etc.) that should be excluded from raw BIDS.   
    High scores indicate series that are likely derived maps, which should be labeled with appropriate suffixes but not included as raw data.
    This is a separate score from "exclude" because some derived maps may still be useful to include in the BIDS dataset with proper labeling, while others (e.g., surgical nav exports) should be excluded entirely.
    Note: This is a heuristic score and may not perfectly separate all cases, but it can help flag likely derived maps for further review or special handling.
    A high derived score combined with a high exclude score would strongly suggest a series is a derived map that should be excluded from raw BIDS, while a high derived score with a low exclude score might indicate a series that should be included but labeled as a derivative.
    
    Args:
        series: SeriesFeatures object containing extracted features of the series
        tokens: Set of text tokens extracted from the series metadata (e.g., SeriesDescription, SequenceName, etc.)
        
    Returns:
        score: Integer score indicating likelihood of being a derived map (higher means more likely)
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

def score_fmap(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0

    imagetype = series.image_type
    multiecho = series.multi_echo
    temporal = series.temporal
    spatial = series.spatial
    
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)
    is3d = bool(temporal and temporal.is_3D)
    num_echoes = multiecho.num_echoes if multiecho else None
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None

    is_mag = bool(imagetype and imagetype.is_magnitude and imagetype.is_magnitude.value)
    is_phase = bool(imagetype and imagetype.is_phase and imagetype.is_phase.value)

    # ---------------------------------------------------------
    # 1. THE RESOLUTION GATE
    # ---------------------------------------------------------
    # Fieldmaps are 2D; 3D volumes (like ME-MPRAGE) are anatomical.
    if is3d:
        score -= 20

    # High-res isotropic scans are structural, not BIDS fieldmaps.
    if thickness is not None and thickness < 1.5:
        score -= 15

    # ---------------------------------------------------------
    # 2. POSITIVE EVIDENCE
    # ---------------------------------------------------------
    if tokens & FMAP_KEYWORDS:
        score += 6

    # Classic Magnitude/Phase pattern for fieldmaps
    if n_tp is not None and n_tp <= 2:
        if is_mag or is_phase:
            score += 5

    # Short EPI PE-polar fieldmaps (TOPUP)
    if epi and n_tp is not None and 2 <= n_tp <= 6:
        score += 3

    # ---------------------------------------------------------
    # 3. ANATOMICAL EXCLUSION
    # ---------------------------------------------------------
    # Multi-echo anatomical mapping (e.g. T2*)
    if (tokens & ANAT_KEYWORDS) and num_echoes and num_echoes > 1:
        score -= 15

    if (tokens & ANAT_KEYWORDS) and not (tokens & FMAP_KEYWORDS):
        score -= 10

    return score

def score_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0
    diffusion = series.diffusion
    temporal = series.temporal
    imagetype = series.image_type
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    bvals = diffusion.b_values if (diffusion and diffusion.b_values) else None
    num_dirs = diffusion.num_diffusion_directions if diffusion else None
    num_vols = diffusion.num_diffusion_volumes if diffusion else None
    num_b0 = diffusion.num_b0 if diffusion else None

    has_diffusion_flag = bool(diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value)
    has_diffusion_imagetype = bool(imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value)
    
    has_adc = bool(imagetype and imagetype.is_adc and imagetype.is_adc.value)
    has_fa = bool(imagetype and imagetype.is_fa and imagetype.is_fa.value)
    has_trace = bool(imagetype and imagetype.is_trace and imagetype.is_trace.value)

    has_perfusion_flag = bool(
        (imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value)
        or (perfusion and (perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent))
    )

    # ---------------------------------------------------------
    # 1. PHYSICS
    # ---------------------------------------------------------
    has_nonzero_bvals = bool(bvals and any(b > 50 for b in bvals))
    if has_nonzero_bvals: score += 4
    if bool(bvals and any(b >= 800 for b in bvals)): score += 2
    if bool(num_dirs is not None and num_dirs >= 6): score += 4
    if num_vols is not None and num_vols >= 20: score += 2
    if num_b0 is not None and num_b0 >= 2: score += 1

    if epi and n_tp is not None and n_tp > 1:
        if n_tp >= 10: score += 1
        if n_tp >= 20: score += 1

    if has_diffusion_flag or has_diffusion_imagetype:
        score += 3

    # ---------------------------------------------------------
    # 2. THE TEXTUAL FALLBACK (Fix for missing DWI)
    # ---------------------------------------------------------
    if tokens & DIFFUSION_KEYWORDS:
        score += 5  # Boosted significantly to catch missing b-value headers
        if epi:
            score += 5  # EPI + Diffusion name = Almost certainly DWI

    if has_adc or has_fa or has_trace:
        score += 2
        if n_tp is not None and n_tp <= 2:
            score += 1

    # ---------------------------------------------------------
    # 3. EXCLUSIONS
    # ---------------------------------------------------------
    if has_perfusion_flag: score -= 4
    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40: score -= 2
    if epi and n_tp is not None and n_tp > 50 and (tokens & FUNC_KEYWORDS): score -= 4
    if (tokens & ANAT_KEYWORDS) and not (has_nonzero_bvals or has_diffusion_imagetype or (tokens & DIFFUSION_KEYWORDS)): score -= 3
    if imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value: score -= 3

    return score


def score_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0
    perf = series.perfusion
    temporal = series.temporal
    imagetype = series.image_type
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    if perf:
        if perf.perfusion_labeling_type and perf.perfusion_labeling_type.value: score += 10
        if n_tp is not None and 60 <= n_tp <= 120: score += 4
        if perf.contrast_agent and perf.contrast_agent.value: score += 3

    if imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value:
        score += 5

    has_contrast = bool(series.contrast and series.contrast.contrast_agent and 
                        series.contrast.contrast_agent.value and 
                        len(series.contrast.contrast_agent.value) > 0)

    # ---------------------------------------------------------
    # 1. THE DCE FIX (Allow non-EPI contrast series to win)
    # ---------------------------------------------------------
    # DCE is usually NOT EPI. It's a 3D T1 sequence run many times.
    if has_contrast and n_tp is not None and n_tp > 10:
        score += 10

    if epi and (tokens & PERFUSION_KEYWORDS):
        score += 6

    if (tokens & ANAT_KEYWORDS) and not (tokens & PERFUSION_KEYWORDS or perf or has_contrast):
        score -= 5

    return score


def score_func(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0
    temporal = series.temporal
    encoding = series.encoding
    diffusion = series.diffusion
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)
    tr_bucket = temporal.tr_bucket if temporal else None
    is3d = bool(temporal and temporal.is_3D)

    if not epi: score -= 15
    if n_tp is not None and n_tp <= 1: score -= 15
    if is3d: score -= 8

    if tokens & FUNC_KEYWORDS: score += 5

    if epi and n_tp is not None:
        if n_tp >= 50: score += 4
        elif n_tp >= 20: score += 3

    if epi and tr_bucket in ("short", "medium"): score += 2

    has_contrast = bool(series.contrast and series.contrast.contrast_agent and 
                        series.contrast.contrast_agent.value and 
                        len(series.contrast.contrast_agent.value) > 0)
    
    if has_contrast: score -= 20
    if (tokens & ANAT_KEYWORDS) and (n_tp is None or n_tp < 10 or not epi): score -= 15
    if diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value: score -= 10

    # ---------------------------------------------------------
    # 1. THE ASL FIX (Prevent ASL from dominating func)
    # ---------------------------------------------------------
    if perfusion and perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value:
        score -= 20
    if tokens & PERFUSION_KEYWORDS:
        score -= 15

    return score


def score_anat(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0
    temporal = series.temporal
    spatial = series.spatial
    sequence = series.sequence
    multiecho = series.multi_echo
    diffusion = series.diffusion
    perfusion = series.perfusion
    imagetype = series.image_type
    contrast = series.contrast

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)
    is3d = bool(temporal and temporal.is_3D)
    is_iso = bool(temporal and temporal.is_isotropic)
    tr_bucket = temporal.tr_bucket if temporal else None
    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    num_echoes = multiecho.num_echoes if multiecho else None
    is_multi_echo = bool(num_echoes and num_echoes > 1)

    has_mpr = bool(imagetype and imagetype.is_mpr and imagetype.is_mpr.value)
    has_diffusion_flag = bool((imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value) or (diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value))
    has_perfusion_flag = bool((imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value) or (perfusion and (perfusion.perfusion_labeling_type or perfusion.perfusion_series_type or perfusion.contrast_agent)))
    has_contrast = bool(contrast and contrast.contrast_agent and contrast.contrast_agent.value and len(contrast.contrast_agent.value) > 0)

    if tokens & ANAT_KEYWORDS: score += 5
    if has_mpr: score += 3
    if n_tp is not None and n_tp <= 3 and not epi: score += 3

    if is3d or (sequence and sequence.mr_acquisition_type and sequence.mr_acquisition_type.value and str(sequence.mr_acquisition_type.value).lower() == "3d"): score += 2
    if thickness is not None:
        if thickness < 1.5: score += 2
        elif thickness < 2.5: score += 1
    if is_iso: score += 1

    if tr_bucket == "long" and not epi and n_tp is not None and n_tp <= 3: score += 1
    if has_contrast and not epi and n_tp is not None and n_tp <= 5: score += 1
        
    if n_tp in (None, 1) and is3d and not epi and not has_diffusion_flag and not has_perfusion_flag: score += 4

    # ---------------------------------------------------------
    # 1. THE DCE FIX (Penalize contrast time-series)
    # ---------------------------------------------------------
    # A standard post-contrast T1 has 1 volume. If it has 10+, it's a DCE perfusion scan.
    if has_contrast and n_tp is not None and n_tp > 10:
        score -= 20

    if tokens & PERFUSION_KEYWORDS:
        score -= 10

    if has_diffusion_flag: score -= 4
    if has_perfusion_flag: score -= 4
    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40: score -= 2
    if epi and n_tp is not None and n_tp > 10 and (tokens & FUNC_KEYWORDS): score -= 4
    if n_tp is not None and n_tp > 10 and not is_multi_echo and not has_contrast: score -= 3

    if thickness is not None and thickness > 5.0 and not (tokens & ANAT_KEYWORDS) and not has_mpr: score -= 2
    if imagetype and imagetype.is_localizer and imagetype.is_localizer.value: score -= 5

    return score

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

    # ---------------------------------------------------------
    # RAW SCORE COMPUTATION
    # ---------------------------------------------------------

    raw_scores = {
        Datatype.ANAT.value: float(score_anat(series, tokens)),
        Datatype.FUNC.value: float(score_func(series, tokens)),
        Datatype.DWI.value: float(score_dwi(series, tokens)),
        Datatype.PERF.value: float(score_perf(series, tokens)),
        Datatype.FMAP.value: float(score_fmap(series, tokens)),
        # Datatype.EXCLUDE.value: float(score_exclude(series, tokens)),
    }

    # ---------------------------------------------------------
    # PROBABILITY CALIBRATION
    # ---------------------------------------------------------

    probs = softmax(raw_scores)

    # Remove excluded datatype if present
    candidate_probs = {
        k: v for k, v in probs.items()
        if k != Datatype.EXCLUDE.value
    }

    # Renormalize candidate probabilities
    total = sum(candidate_probs.values())
    if total == 0:
        return probs, Datatype.UNKNOWN.value, 0.0

    candidate_probs = {
        k: v / total
        for k, v in candidate_probs.items()
    }

    # ---------------------------------------------------------
    # SELECT BEST DATATYPE
    # ---------------------------------------------------------

    best_type = max(candidate_probs, key=lambda k: candidate_probs[k])
    best_prob = candidate_probs[best_type]

    # ---------------------------------------------------------
    # CONFIDENCE ESTIMATION
    # ---------------------------------------------------------

    sorted_probs = sorted(candidate_probs.values(), reverse=True)

    if len(sorted_probs) > 1:
        margin = sorted_probs[0] - sorted_probs[1]
    else:
        margin = sorted_probs[0]

    # Low evidence safeguard
    if best_prob < 0.40:
        return probs, Datatype.UNKNOWN.value, 0.0

    confidence = margin

    return probs, best_type, confidence