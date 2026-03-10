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

def score_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score likelihood that a series belongs to the DWI datatype.

    Steps:
        1. Extract relevant features and tokens from the series
        2. Apply physics-informed heuristics to assign points for diffusion-consistent features (e.g., high b-values, many diffusion directions, EPI pattern)
        3. Apply penalties for features that are inconsistent with DWI (e.g., strong perfusion evidence, functional EPI pattern, anatomical naming without diffusion physics)
        4. Aggregate the scores to produce a final likelihood estimate
        5. Optionally convert the raw score to a probability using a softmax function across all datatypes (not implemented here, but could be done in the final classification step)
        6. Return the final score (higher means more likely to be DWI)
        
    Note: This is a heuristic scoring system and may not perfectly classify all series, but it can help flag likely DWI series for further review or automatic labeling.
    
    Args:
        series: SeriesFeatures object containing extracted features of the series
        tokens: Set of text tokens extracted from the series metadata (e.g., SeriesDescription, SequenceName, etc.)
    """

    score = 0

    # ============================================================
    # 1. Extract commonly used signals
    # ============================================================

    diffusion = series.diffusion
    temporal = series.temporal
    imagetype = series.image_type
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    # Diffusion physics
    bvals = diffusion.b_values if (diffusion and diffusion.b_values) else None
    num_dirs = diffusion.num_diffusion_directions if diffusion else None
    num_vols = diffusion.num_diffusion_volumes if diffusion else None
    num_b0 = diffusion.num_b0 if diffusion else None

    has_diffusion_flag = bool(
        diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value
    )

    has_diffusion_imagetype = bool(
        imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value
    )

    # Derived diffusion maps
    has_adc = bool(imagetype and imagetype.is_adc and imagetype.is_adc.value)
    has_fa = bool(imagetype and imagetype.is_fa and imagetype.is_fa.value)
    has_trace = bool(imagetype and imagetype.is_trace and imagetype.is_trace.value)

    # Perfusion evidence (used for penalties)
    has_perfusion_flag = bool(
        (imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value)
        or (perfusion and (
            perfusion.perfusion_labeling_type
            or perfusion.perfusion_series_type
            or perfusion.contrast_agent
        ))
    )

    # ============================================================
    # 2. Strong diffusion physics evidence
    # ============================================================

    has_nonzero_bvals = bool(bvals and any(b > 50 for b in bvals))
    has_high_bvals = bool(bvals and any(b >= 800 for b in bvals))
    has_gradients = bool(num_dirs is not None and num_dirs >= 6)

    if has_nonzero_bvals:
        score += 4

    if has_high_bvals:
        score += 2

    if has_gradients:
        score += 4

    if num_vols is not None and num_vols >= 20:
        score += 2

    if num_b0 is not None and num_b0 >= 2:
        score += 1

    # ============================================================
    # 3. Typical diffusion acquisition pattern
    # ============================================================

    # DWI is typically EPI with multiple volumes
    if epi and n_tp is not None and n_tp > 1:

        if n_tp >= 10:
            score += 1

        if n_tp >= 20:
            score += 1

    # ============================================================
    # 4. Metadata / naming evidence
    # ============================================================

    # ImageType diffusion indicators
    if has_diffusion_flag or has_diffusion_imagetype:
        score += 3

    # Text-based naming hints
    if tokens & DIFFUSION_KEYWORDS:
        score += 2

    # ============================================================
    # 5. Derived diffusion maps (ADC / FA / TRACE)
    # ============================================================

    if has_adc or has_fa or has_trace:

        # Weak evidence for diffusion datatype
        score += 1

        # Derived maps often single-volume
        if n_tp is not None and n_tp <= 2:
            score += 1

    # ============================================================
    # 6. Mutual exclusion penalties
    # ============================================================

    # Perfusion evidence
    if has_perfusion_flag:
        score -= 4

    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40:
        score -= 2

    # Functional pattern (EPI + long time-series + functional tokens)
    if epi and n_tp is not None and n_tp > 50 and (tokens & FUNC_KEYWORDS):
        score -= 4

    # Anatomical naming without diffusion physics
    if (tokens & ANAT_KEYWORDS) and not (
        has_nonzero_bvals or has_gradients or has_diffusion_imagetype
    ):
        score -= 3

    # Explicit perfusion ImageType
    if imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value:
        score -= 3

    return score

def score_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score likelihood that a series belongs to the PERF (perfusion) datatype.

    The scoring logic considers multiple lines of evidence, including:
    1. Explicit naming cues (e.g., "ASL", "CBF", "perfusion" in text tokens)
    2. ImageType indicators (e.g., has_perfusion flag)
    3. Typical acquisition patterns (e.g., long time-series, EPI pattern for DSC)
    4. Mutual exclusion with other datatypes (e.g., strong diffusion evidence, functional EPI pattern, anatomical naming without perfusion cues)
    5. Penalties for inconsistent features (e.g., strong diffusion physics, functional EPI pattern, anatomical naming without perfusion indicators)
    
    Args:
        series: SeriesFeatures object containing extracted features of the series
        tokens: Set of text tokens extracted from the series metadata (e.g., SeriesDescription, SequenceName, etc.)
    """

    score = 0

    # ============================================================
    # 1. Extract commonly used signals
    # ============================================================

    perf = series.perfusion
    temporal = series.temporal
    imagetype = series.image_type
    diffusion = series.diffusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    # Perfusion indicators
    has_perfusion_imagetype = bool(
        imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value
    )

    # Diffusion indicators (used for penalties)
    has_diffusion_flag = bool(
        diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value
    )

    num_dirs = diffusion.num_diffusion_directions if diffusion else None

    # ============================================================
    # 2. Strong perfusion metadata evidence
    # ============================================================

    if perf:
        # ASL / labeling type (very strong perfusion signal)
        if perf.perfusion_labeling_type and perf.perfusion_labeling_type.value:
            score += 5

        # Explicit perfusion series classification
        if perf.perfusion_series_type and perf.perfusion_series_type.value:
            score += 4

        # Contrast bolus (typical for DSC / DCE)
        if perf.contrast_agent and perf.contrast_agent.value:
            score += 2

        # Long perfusion time-series
        if perf.num_timepoints and perf.num_timepoints > 40:
            score += 2

        # Regular temporal spacing (common in dynamic perfusion)
        if perf.temporal_spacing and perf.temporal_spacing.value:
            score += 1

    # ImageType indicator
    if has_perfusion_imagetype:
        score += 4

    # ============================================================
    # 3. Typical perfusion acquisition pattern
    # ============================================================

    # Perfusion often appears as an EPI time-series
    if epi and n_tp is not None:

        if n_tp >= 30:
            score += 1

        if n_tp >= 50:
            score += 1

    # ============================================================
    # 4. Naming / token evidence
    # ============================================================

    if tokens & PERFUSION_KEYWORDS:
        score += 4

        # EPI perfusion naming (e.g., ASL, DSC)
        if epi:
            score += 2

    # ============================================================
    # 5. Generic temporal evidence
    # ============================================================

    # Long time-series without strong functional tokens
    if n_tp is not None and n_tp > 40 and not (tokens & FUNC_KEYWORDS):
        score += 1

    # ============================================================
    # 6. Mutual exclusion penalties
    # ============================================================

    # Strong diffusion physics
    if has_diffusion_flag:
        score -= 4

    if num_dirs and num_dirs >= 6:
        score -= 2

    # Functional BOLD pattern
    if epi and n_tp is not None and n_tp > 50 and (tokens & FUNC_KEYWORDS):
        score -= 3

    # Anatomical naming without perfusion indicators
    if (tokens & ANAT_KEYWORDS) and not (
        tokens & PERFUSION_KEYWORDS
        or has_perfusion_imagetype
        or (perf and perf.perfusion_labeling_type)
    ):
        score -= 2

    # Explicit diffusion ImageType
    if imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value:
        score -= 2

    return score

def score_fmap(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score likelihood that a series belongs to the FMAP (fieldmap) datatype.

    The scoring logic considers multiple lines of evidence, including:
    1. Explicit naming cues (e.g., "fieldmap", "fmap", "phase", "magnitude" in text tokens)
    2. ImageType indicators (e.g., is_phase, is_magnitude)
    3. Typical acquisition patterns (e.g., few timepoints, EPI-like for PE-polar fieldmaps)
    4. Mutual exclusion with other datatypes (e.g., strong diffusion evidence, strong functional evidence, anatomical naming without fieldmap cues)
    5. Multi-echo GRE patterns (common for fieldmaps but can be ME-MPRAGE, so consider anatomical tokens for context)
    6. Penalties for inconsistent features (e.g., many timepoints, strong diffusion/perfusion evidence, functional EPI pattern)
    
    Args:
        series: SeriesFeatures object containing extracted features of the series
        tokens: Set of text tokens extracted from the series metadata (e.g., SeriesDescription, SequenceName, etc.)
    """

    score = 0

    # ============================================================
    # 1. Extract commonly used signals
    # ============================================================

    imagetype = series.image_type
    multiecho = series.multi_echo
    temporal = series.temporal
    diffusion = series.diffusion
    perfusion = series.perfusion

    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    num_echoes = multiecho.num_echoes if multiecho else None

    # ImageType indicators
    is_mag = bool(imagetype and imagetype.is_magnitude and imagetype.is_magnitude.value)
    is_phase = bool(imagetype and imagetype.is_phase and imagetype.is_phase.value)
    is_real = bool(imagetype and imagetype.is_real and imagetype.is_real.value)
    is_imag = bool(imagetype and imagetype.is_imaginary and imagetype.is_imaginary.value)

    # ============================================================
    # 2. Explicit naming evidence
    # ============================================================

    if tokens & FMAP_KEYWORDS:
        score += 5

    # ============================================================
    # 3. Classic GRE phase/magnitude fieldmap pattern
    # ============================================================

    # Fieldmaps are commonly exported as:
    #   magnitude1 / magnitude2
    #   phase
    # with very few volumes.

    if n_tp is not None and n_tp <= 2:

        if is_mag or is_phase:
            score += 4

        if is_real or is_imag:
            score += 3

    # ============================================================
    # 4. Multi-echo GRE fieldmaps
    # ============================================================

    if num_echoes and num_echoes > 1:

        # Multi-echo GRE is commonly used for fieldmap estimation
        if not (tokens & ANAT_KEYWORDS):
            score += 3
        else:
            # Could be ME-MPRAGE or other anatomical
            score += 1

    # ============================================================
    # 5. EPI PE-polar fieldmaps (TOPUP / SEFM)
    # ============================================================

    # Often look like short EPI runs with 2–4 volumes
    if epi and n_tp is not None and 2 <= n_tp <= 4:

        if tokens & FMAP_KEYWORDS:
            score += 3
        else:
            # Weak hint for PE-polar fieldmap
            score += 1

    # ============================================================
    # 6. Mutual exclusion penalties
    # ============================================================

    # Many timepoints → unlikely to be fieldmap
    if n_tp is not None and n_tp > 10:
        score -= 3

    # Strong diffusion evidence
    if diffusion and (
        (diffusion.has_diffusion and diffusion.has_diffusion.value)
        or (diffusion.num_diffusion_directions and diffusion.num_diffusion_directions >= 6)
    ):
        score -= 4

    # Strong perfusion evidence
    if perfusion and (
        (perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value)
        or (perfusion.num_timepoints and perfusion.num_timepoints > 40)
    ):
        score -= 3

    # Functional pattern (EPI + long time-series + functional tokens)
    if epi and n_tp is not None and n_tp > 20 and (tokens & FUNC_KEYWORDS):
        score -= 3

    # Multi-echo anatomical sequences (e.g., ME-MPRAGE)
    if (tokens & ANAT_KEYWORDS) and num_echoes and num_echoes > 1:
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

    # Strong diffusion evidence
    has_diffusion_flag = bool(
        diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value
    )

    # Strong perfusion evidence
    has_perfusion_flag = bool(
        perfusion and (
            (perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value)
            or (perfusion.num_timepoints and perfusion.num_timepoints > 40)
        )
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE EVIDENCE
    # ---------------------------------------------------------

    # Explicit functional keywords: bold, fmri, rest, task
    if tokens & FUNC_KEYWORDS:
        score += 5

    # Canonical BOLD pattern: EPI + many timepoints
    if epi and n_tp is not None:
        if n_tp >= 50:
            score += 4
        elif n_tp >= 20:
            score += 3
        elif n_tp >= 10:
            score += 1

    # ---------------------------------------------------------
    # MODERATE POSITIVE EVIDENCE
    # ---------------------------------------------------------

    # Short/medium TR typical for BOLD acquisitions
    if epi and tr_bucket in ("short", "medium"):
        score += 2

    # ---------------------------------------------------------
    # SUPPORTING ACQUISITION FEATURES
    # ---------------------------------------------------------

    if encoding:
        # Multiband acceleration (very common in fMRI)
        if (
            encoding.multiband_factor
            and encoding.multiband_factor.value
            and encoding.multiband_factor.value > 1
        ):
            score += 1

        # Parallel imaging
        if (
            encoding.parallel_reduction_factor_in_plane
            and encoding.parallel_reduction_factor_in_plane.value
        ):
            score += 1

    # Functional tokens without competing modality cues
    if tokens & FUNC_KEYWORDS and not (
        tokens & (ANAT_KEYWORDS | DIFFUSION_KEYWORDS | PERFUSION_KEYWORDS)
    ):
        score += 1

    # ---------------------------------------------------------
    # NEGATIVE / MODALITY EXCLUSION EVIDENCE
    # ---------------------------------------------------------

    # Very short EPI series → likely fieldmap or scout
    if epi and n_tp is not None and n_tp < 10:
        score -= 3

    # Fieldmap naming cues
    if tokens & FMAP_KEYWORDS:
        score -= 2

    # Strong diffusion physics
    if has_diffusion_flag:
        score -= 4

    if diffusion and diffusion.num_diffusion_directions and diffusion.num_diffusion_directions >= 6:
        score -= 2

    # Strong perfusion indicators
    if has_perfusion_flag:
        score -= 3

    # ---------------------------------------------------------
    # ANATOMICAL CONFLICTS
    # ---------------------------------------------------------

    # Structural naming + few volumes + non-EPI
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

    # --- Convenience aliases ---
    n_tp = temporal.num_timepoints if temporal else None
    epi = is_epi(series)

    is3d = bool(temporal and temporal.is_3D)
    is_iso = bool(temporal and temporal.is_isotropic)

    tr_bucket = temporal.tr_bucket if temporal else None

    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None

    num_echoes = multiecho.num_echoes if multiecho else None
    is_multi_echo = bool(num_echoes and num_echoes > 1)

    has_mpr = bool(imagetype and imagetype.is_mpr and imagetype.is_mpr.value)

    has_diffusion_flag = bool(
        (imagetype and imagetype.has_diffusion and imagetype.has_diffusion.value)
        or (diffusion and diffusion.has_diffusion and diffusion.has_diffusion.value)
    )

    has_perfusion_flag = bool(
        (imagetype and imagetype.has_perfusion and imagetype.has_perfusion.value)
        or (perfusion and (
            perfusion.perfusion_labeling_type
            or perfusion.perfusion_series_type
            or perfusion.contrast_agent
        ))
    )

    has_contrast = bool(
        contrast and contrast.contrast_agent and contrast.contrast_agent.value
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE EVIDENCE
    # ---------------------------------------------------------

    # Classic anatomical keywords
    if tokens & ANAT_KEYWORDS:
        score += 5

    # MPR/MPRAGE indicator
    if has_mpr:
        score += 3

    # Few volumes and non-EPI → typical structural
    if n_tp is not None and n_tp <= 3 and not epi:
        score += 3

    # ---------------------------------------------------------
    # ACQUISITION STRUCTURE
    # ---------------------------------------------------------

    # 3D acquisition
    if is3d or (
        sequence
        and sequence.mr_acquisition_type
        and sequence.mr_acquisition_type.value
        and str(sequence.mr_acquisition_type.value).lower() == "3d"
    ):
        score += 2

    # Thin slices typical of high-resolution structural scans
    if thickness is not None:
        if thickness < 1.5:
            score += 2
        elif thickness < 2.5:
            score += 1

    # Isotropic voxels
    if is_iso:
        score += 1

    # ---------------------------------------------------------
    # STRUCTURAL SCAN PATTERNS
    # ---------------------------------------------------------

    # Long TR + few volumes + non-EPI
    if tr_bucket == "long" and not epi and n_tp is not None and n_tp <= 3:
        score += 1

    # Post-contrast structural scan
    if has_contrast and not epi and n_tp is not None and n_tp <= 5:
        score += 1

    # ---------------------------------------------------------
    # NEGATIVE / MODALITY EXCLUSION
    # ---------------------------------------------------------

    # Strong diffusion evidence
    if has_diffusion_flag:
        score -= 4

    # Strong perfusion evidence
    if has_perfusion_flag:
        score -= 4

    if perfusion and perfusion.num_timepoints and perfusion.num_timepoints > 40:
        score -= 2

    # Functional-like EPI time-series
    if epi and n_tp is not None and n_tp > 10 and (tokens & FUNC_KEYWORDS):
        score -= 4

    # ---------------------------------------------------------
    # GENERIC TIME-SERIES PENALTIES
    # ---------------------------------------------------------

    # Many timepoints but not multi-echo
    if n_tp is not None and n_tp > 10 and not is_multi_echo:
        score -= 3

    # ---------------------------------------------------------
    # LOW-QUALITY / LOCALIZER SIGNALS
    # ---------------------------------------------------------

    # Very thick slices without structural naming
    if (
        thickness is not None
        and thickness > 5.0
        and not (tokens & ANAT_KEYWORDS)
        and not has_mpr
    ):
        score -= 2

    # Explicit localizer flag
    if imagetype and imagetype.is_localizer and imagetype.is_localizer.value:
        score -= 5

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