from ..extract.series import SeriesFeatures
from .utils import collect_tokens, softmax, is_epi, is_fieldmap_epi
from .rules import (
    T1_KEYWORDS,
    T2_KEYWORDS,
    FLAIR_KEYWORDS,
    PD_KEYWORDS,
    T2STAR_KEYWORDS,
    T1MAP_KEYWORDS,
    T2MAP_KEYWORDS,
    GRE_KEYWORDS,
    SWI_KEYWORDS,
    FUNC_KEYWORDS,
    DIFFUSION_KEYWORDS,
    FMAP_KEYWORDS,
    BOLD_KEYWORDS,
    SBREF_KEYWORDS
)
from .schema import Datatype, BIDS_SCHEMA


# =====ANATOMIC SUFFIX SCORING FUNCTIONS (T1w, T2w, FLAIR, PD, T2starw, SWI)====

def score_t1w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T1w suffix."""
    score = 0.0

    temporal = series.temporal
    imagetype = series.image_type
    contrast = series.contrast

    # --- TE bucket ---
    te_bucket = temporal.te_bucket if temporal else None

    # --- Geometry ---
    is_iso = bool(temporal and temporal.is_isotropic)
    is_3d = bool(temporal and temporal.is_3D)

    # --- Inversion recovery ---
    has_inversion = bool(
        temporal
        and temporal.inversion_time
        and temporal.inversion_time.valid_fraction > 0.5
        and temporal.inversion_time.stable
        and temporal.inversion_time.value is not None
    )

    # --- MPR reconstruction ---
    has_mpr = bool(
        imagetype
        and imagetype.is_mpr
        and imagetype.is_mpr.value
    )

    # --- Contrast ---
    has_contrast = bool(
        contrast
        and contrast.contrast_agent
        and contrast.contrast_agent.value
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE EVIDENCE
    # ---------------------------------------------------------

    if tokens & T1_KEYWORDS:
        score += 5

    if has_mpr:
        score += 4

    # ---------------------------------------------------------
    # ACQUISITION PHYSICS
    # ---------------------------------------------------------

    if has_inversion:
        score += 4

    # ---------------------------------------------------------
    # STRUCTURAL GEOMETRY
    # ---------------------------------------------------------

    if is_iso:
        score += 2

    if is_3d:
        score += 1

    if has_contrast:
        score += 3

    # ---------------------------------------------------------
    # NEGATIVE EVIDENCE
    # ---------------------------------------------------------

    if te_bucket == "long":
        score -= 3

    if tokens & FLAIR_KEYWORDS:
        score -= 4

    return max(-10, min(score, 10))

def score_t2w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T2w suffix."""
    score = 0.0

    temporal = series.temporal
    sequence = series.sequence

    te_bucket = temporal.te_bucket if temporal else None

    inversion_time = (
        temporal.inversion_time.value
        if temporal
        and temporal.inversion_time
        and temporal.inversion_time.valid_fraction > 0.5
        and temporal.inversion_time.stable
        else None
    )

    is_3d = bool(
        sequence
        and sequence.mr_acquisition_type
        and sequence.mr_acquisition_type.consistency
        and sequence.mr_acquisition_type.consistency > 0.8
        and str(sequence.mr_acquisition_type.value).lower() == "3d"
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE
    # ---------------------------------------------------------

    if tokens & T2_KEYWORDS:
        score += 5

    # ---------------------------------------------------------
    # ACQUISITION PHYSICS
    # ---------------------------------------------------------

    if te_bucket == "long":
        score += 4
    elif te_bucket == "medium":
        score += 1

    if not is_3d:
        score += 2

    # ---------------------------------------------------------
    # NEGATIVE EVIDENCE
    # ---------------------------------------------------------

    if inversion_time and inversion_time > 1500:
        score -= 2

    if te_bucket == "short":
        score -= 4

    if tokens & T1_KEYWORDS:
        score -= 4

    return max(-10, min(score, 10)) # 

def score_flair_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/FLAIR suffix."""
    score = 0.0

    temporal = series.temporal

    te_bucket = temporal.te_bucket if temporal else None

    inversion_time = (
        temporal.inversion_time.value
        if temporal
        and temporal.inversion_time
        and temporal.inversion_time.valid_fraction > 0.5
        and temporal.inversion_time.stable
        else None
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE
    # ---------------------------------------------------------

    if tokens & FLAIR_KEYWORDS:
        score += 6

    if inversion_time and inversion_time > 1800:
        score += 4

    # ---------------------------------------------------------
    # SUPPORTING
    # ---------------------------------------------------------

    if te_bucket == "long":
        score += 2

    # ---------------------------------------------------------
    # NEGATIVE
    # ---------------------------------------------------------

    if te_bucket == "short":
        score -= 4

    if tokens & T1_KEYWORDS:
        score -= 3

    return max(-10, min(score, 10))

def score_pd_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/PD suffix."""
    score = 0.0

    temporal = series.temporal

    te_bucket = temporal.te_bucket if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    has_inversion = bool(
        temporal
        and temporal.inversion_time
        and temporal.inversion_time.valid_fraction > 0.5
        and temporal.inversion_time.stable
        and temporal.inversion_time.value
    )

    # ---------------------------------------------------------
    # STRONG POSITIVE
    # ---------------------------------------------------------

    if tokens & PD_KEYWORDS:
        score += 5

    # ---------------------------------------------------------
    # PHYSICS
    # ---------------------------------------------------------

    if te_bucket == "short" and tr_bucket == "long":
        score += 3

    # ---------------------------------------------------------
    # NEGATIVE
    # ---------------------------------------------------------

    if tokens & T2_KEYWORDS:
        score -= 1

    if has_inversion:
        score -= 3

    return max(-10, min(score, 10))

def score_t2starw_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/T2starw."""
    score = 0.0

    seq = series.sequence
    img_type = series.image_type

    # --------------------------------------------------
    # STRONG TEXTUAL
    # --------------------------------------------------

    if tokens & T2STAR_KEYWORDS:
        score += 6

    # --------------------------------------------------
    # GRE PHYSICS
    # --------------------------------------------------

    if tokens & GRE_KEYWORDS:
        score += 2

    if (
        seq
        and seq.scanning_sequence
        and seq.scanning_sequence.value == "GR"
        and seq.scanning_sequence.consistency
        and seq.scanning_sequence.consistency > 0.7
    ):
        score += 2
        
    # --------------------------------------------------
    # T2* USUALLY HAS MULTI-ECHO (but not always, so only a weak signal)
    # --------------------------------------------------
    
    if series.multi_echo and series.multi_echo.num_echoes and series.multi_echo.num_echoes > 1:
        score += 1

    # --------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------

    if tokens & SWI_KEYWORDS:
        score -= 4

    if tokens & FUNC_KEYWORDS:
        score -= 5

    if tokens & DIFFUSION_KEYWORDS:
        score -= 6

    if "epi" in tokens:
        score -= 3

    # --------------------------------------------------
    # WEAK ANATOMICAL BIAS
    # --------------------------------------------------

    if img_type and img_type.is_magnitude and img_type.is_magnitude.value:
        score += 1

    return max(-10, min(score, 10))

def score_swi_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/swi."""
    score = 0.0

    seq = series.sequence
    img_type = series.image_type

    # --------------------------------------------------
    # STRONG TEXTUAL
    # --------------------------------------------------

    if tokens & SWI_KEYWORDS:
        score += 6

    # --------------------------------------------------
    # GRE PHYSICS
    # --------------------------------------------------

    if tokens & GRE_KEYWORDS:
        score += 2

    if (
        seq
        and seq.scanning_sequence
        and seq.scanning_sequence.value == "GR"
        and seq.scanning_sequence.consistency
        and seq.scanning_sequence.consistency > 0.7
    ):
        score += 2

    # --------------------------------------------------
    # SWI SPECIFIC SIGNAL
    # --------------------------------------------------

    if "phase" in tokens or "pha" in tokens:
        score += 1

    if "mag" in tokens or "magnitude" in tokens:
        score += 1

    # --------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------

    if tokens & T2STAR_KEYWORDS:
        score -= 3

    if tokens & FUNC_KEYWORDS:
        score -= 5

    if tokens & DIFFUSION_KEYWORDS:
        score -= 6

    if "epi" in tokens:
        score -= 3

    # --------------------------------------------------
    # WEAK ANATOMICAL BIAS
    # --------------------------------------------------

    if img_type and img_type.is_magnitude and img_type.is_magnitude.value:
        score += 1

    return max(-10, min(score, 10))

# ----Derived maps (T1map, T2map)

def score_t1map_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T1map (Quantitative T1 mapping)."""
    score = 0.0
    it = series.image_type

    # ---------------------------------------------------------
    # 1. STRONG TEXTUAL EVIDENCE
    # ---------------------------------------------------------
    if tokens & T1MAP_KEYWORDS:
        score += 7

    # ---------------------------------------------------------
    # 2. DERIVED STATUS (The 'r' axis)
    # ---------------------------------------------------------
    if it:
        # Quantitative maps are almost never 'Original' or 'Primary'
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 3
            
        # Reconstructed or secondary status is a positive signal
        if it.is_reformatted and it.is_reformatted.value:
            score += 2

    # ---------------------------------------------------------
    # 3. VOLUME COUNT
    # ---------------------------------------------------------
    # While the source acquisition has many volumes (VFA or IR), 
    # the resulting map is usually a single calculated volume.
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 5:
        score -= 6 # Likely the raw source data, not the map

    # ---------------------------------------------------------
    # 4. NEGATIVE EVIDENCE
    # ---------------------------------------------------------
    if tokens & {"t2map", "adc", "cbv"}:
        score -= 5

    return max(-10, min(score, 10))


def score_t2map_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T2map (Quantitative T2 mapping)."""
    score = 0.0
    it = series.image_type

    # ---------------------------------------------------------
    # 1. STRONG TEXTUAL EVIDENCE
    # ---------------------------------------------------------
    if tokens & T2MAP_KEYWORDS:
        score += 7

    # ---------------------------------------------------------
    # 2. DERIVED STATUS
    # ---------------------------------------------------------
    if it:
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 3

    # ---------------------------------------------------------
    # 3. VOLUME COUNT
    # ---------------------------------------------------------
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 5:
        score -= 6

    # ---------------------------------------------------------
    # 4. SEQUENCE CONTEXT
    # ---------------------------------------------------------
    # T2 maps often come from multi-echo spin echo sequences.
    if series.multi_echo and series.multi_echo.num_echoes and series.multi_echo.num_echoes > 1:
        # If it's 1 volume but the metadata says multi-echo, it's highly likely a map
        score += 2

    # ---------------------------------------------------------
    # 5. NEGATIVE EVIDENCE
    # ---------------------------------------------------------
    if tokens & {"t1map", "flair", "bold"}:
        score -= 5

    return max(-10, min(score, 10))

def score_anat_suffix(
    series: SeriesFeatures,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for anat suffixes.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype.ANAT].keys())
    
    # Compute raw suffix scores
    raw_scores = {}
    for suffix in valid_suffixes:
        score_fn_name = f"score_{suffix}_anat"
        if score_fn_name in globals():
            raw_scores[suffix] = globals()[score_fn_name](series, tokens)
        else:
            raw_scores[suffix] = 0  # Default for unhandled suffixes
    
    # Convert to calibrated probabilities via softmax
    probs = softmax(raw_scores)
    
    # Pick best
    best_suffix = max(probs, key=lambda k: probs[k])
    best_prob = probs[best_suffix]
    
    # Confidence thresholds (same logic as datatype)
    if best_prob < 0.4:  # Low evidence
        best_suffix = "unknown"
        confidence = 0.0
    else:
        # Margin over runner-up
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    
    return probs, best_suffix, confidence

# =====FUNCTIONAL SUFFIX SCORING (BOLD, CBV, CBF)====

def score_bold_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/bold."""
    score = 0.0

    temporal = series.temporal
    encoding = series.encoding

    n_tp = temporal.num_timepoints if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    # --------------------------------------------------
    # STRONG TEXTUAL
    # --------------------------------------------------

    if tokens & BOLD_KEYWORDS:
        score += 6

    # --------------------------------------------------
    # EPI PHYSICS
    # --------------------------------------------------

    if is_epi(series):
        score += 2

    if tr_bucket == "short":
        score += 2

    # --------------------------------------------------
    # TEMPORAL STRUCTURE
    # --------------------------------------------------

    if n_tp is not None and n_tp > 80:
        score += 4
    elif n_tp is not None and n_tp > 20:
        score += 2

    # --------------------------------------------------
    # ACCELERATION (COMMON IN BOLD)
    # --------------------------------------------------

    if encoding:

        if (
            encoding.multiband_factor
            and encoding.multiband_factor.value
            and encoding.multiband_factor.value > 1
        ):
            score += 1

        if (
            encoding.parallel_reduction_factor_in_plane
            and encoding.parallel_reduction_factor_in_plane.value
            and encoding.parallel_reduction_factor_in_plane.value > 1
        ):
            score += 1

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    if tokens & SBREF_KEYWORDS:
        score -= 4

    if n_tp is not None and n_tp < 10:
        score -= 4

    if tokens & FMAP_KEYWORDS:
        score -= 4

    if is_fieldmap_epi(series, tokens):
        score -= 4

    return max(-10, min(score, 10))

def score_sbref_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/sbref."""
    score = 0.0

    temporal = series.temporal

    n_tp = temporal.num_timepoints if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    # --------------------------------------------------
    # STRONG TEXTUAL
    # --------------------------------------------------

    if tokens & SBREF_KEYWORDS:
        score += 6

    # --------------------------------------------------
    # EPI PHYSICS
    # --------------------------------------------------

    if is_epi(series):
        score += 2

    if tr_bucket in ("short", "medium"):
        score += 1

    # --------------------------------------------------
    # TEMPORAL STRUCTURE
    # --------------------------------------------------
    if n_tp is not None:
        if n_tp == 1:
            score += 4
        elif n_tp <= 3:
            score += 2

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    if n_tp is not None and n_tp >= 50:
        score -= 5

    if tokens & BOLD_KEYWORDS:
        score -= 4

    if is_fieldmap_epi(series, tokens):
        score -= 4
        
    if tokens & FMAP_KEYWORDS:
        score -= 3

    return max(-10, min(score, 10))

def score_func_suffix(
    series: SeriesFeatures,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for func suffixes.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype.FUNC].keys())
    
    # Compute raw suffix scores
    raw_scores = {}
    for suffix in valid_suffixes:
        score_fn_name = f"score_{suffix}_func"
        if score_fn_name in globals():
            raw_scores[suffix] = globals()[score_fn_name](series, tokens)
        else:
            raw_scores[suffix] = 0  # Default for unhandled suffixes
    
    # Convert to calibrated probabilities via softmax
    probs = softmax(raw_scores)
    
    # Pick best
    best_suffix = max(probs, key=lambda k: probs[k])
    best_prob = probs[best_suffix]
    
    # Confidence thresholds (same logic as datatype)
    if best_prob < 0.4:  # Low evidence
        best_suffix = "unknown"
        confidence = 0.0
    else:
        # Margin over runner-up
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    
    return probs, best_suffix, confidence

# =====DIFFUSION SUFFIX SCORING (dwi, adc, fa)====

def score_dwi_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score confidence that a series is a DWI acquisition (dwi/dwi).

    Uses a combination of:
    - textual cues
    - diffusion metadata
    - acquisition properties
    - negative evidence
    """

    score = 0

    diffusion = series.diffusion
    seq = series.sequence
    temporal = series.temporal

    # --------------------------------------------------
    # 1. Strong textual evidence
    # --------------------------------------------------

    DWI_KEYWORDS = {"dwi", "diffusion", "dti", "dsi", "hardi"}

    if tokens & DWI_KEYWORDS:
        score += 5

    # --------------------------------------------------
    # 2. Diffusion metadata (strongest signal)
    # --------------------------------------------------

    if diffusion:

        # presence of b-values
        if diffusion.b_values and len(set(diffusion.b_values)) > 1:
            score += 4

            max_b = max(diffusion.b_values)

            if max_b > 700:
                score += 2

        # diffusion directions
        if diffusion.num_diffusion_directions:

            n_dir = diffusion.num_diffusion_directions

            if n_dir >= 30:
                score += 3

            elif n_dir >= 6:
                score += 2

            elif n_dir > 1:
                score += 1

    # --------------------------------------------------
    # 3. Acquisition characteristics
    # --------------------------------------------------

    # most DWI acquisitions are EPI
    if is_epi(series):
        score += 1

    # diffusion series typically have multiple volumes
    if temporal and temporal.num_timepoints:

        n_tp = temporal.num_timepoints

        if n_tp >= 10:
            score += 1

    # --------------------------------------------------
    # 4. Weak textual hints
    # --------------------------------------------------

    if tokens & {"b0", "trace", "adc"}:
        score += 1

    # --------------------------------------------------
    # 5. Negative evidence
    # --------------------------------------------------

    if tokens & {"bold", "fmri", "task", "rest"}:
        score -= 5

    if tokens & {"t1", "t2", "flair", "mprage"}:
        score -= 4

    if tokens & {"fieldmap", "topup"}:
        score -= 3

    return max(-10, min(score, 10))

def score_adc_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score confidence that a series is an ADC (Apparent Diffusion Coefficient) map.
    """
    score = 0
    it = series.image_type

    # --------------------------------------------------
    # 1. Explicit Indicators (ImageType & Keywords)
    # --------------------------------------------------
    
    # Highest confidence: DICOM explicitly flags it as ADC
    if it and it.is_adc and it.is_adc.value:
        score += 7
        
    if tokens & {"adc", "apparent diffusion coefficient"}:
        score += 5

    # --------------------------------------------------
    # 2. Derived Status Check
    # --------------------------------------------------
    
    if it:
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 4
            
        # Often flagged as derived or secondary
        if (it.is_reformatted and it.is_reformatted.value) or \
           (not (it.is_original and it.is_original.value)):
            score += 2

    # --------------------------------------------------
    # 3. Volume and Diffusion Characteristics
    # --------------------------------------------------
    
    # ADC maps are almost always a single calculated 3D volume
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 5:
        # A true ADC map shouldn't have many volumes; if it does, it's likely raw DWI
        score -= 6 

    # --------------------------------------------------
    # 4. Negative Evidence
    # --------------------------------------------------
    
    if tokens & {"fa", "fractional anisotropy", "trace", "tensor"}:
        score -= 4

    return max(-10, min(score, 10))



def score_fa_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score confidence that a series is an FA (Fractional Anisotropy) map.
    """
    score = 0
    it = series.image_type

    # --------------------------------------------------
    # 1. Explicit Indicators (ImageType & Keywords)
    # --------------------------------------------------
    
    # Highest confidence: DICOM explicitly flags it as FA
    if it and it.is_fa and it.is_fa.value:
        score += 7
        
    if tokens & {"fa", "fractional anisotropy", "colfa", "color fa"}:
        score += 6

    # --------------------------------------------------
    # 2. Derived Status Check
    # --------------------------------------------------
    
    if it:
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 4

    # --------------------------------------------------
    # 3. Volume Characteristics
    # --------------------------------------------------
    
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 5:
        score -= 6

    # --------------------------------------------------
    # 4. Negative Evidence
    # --------------------------------------------------
    
    # Prevent collision with flip angle maps (often just labeled "fa" in anat/fmap)
    # Though datatype gating usually prevents this, it's safe to penalize anat keywords
    if tokens & {"adc", "trace", "t1", "b1"}:
        score -= 4

    return max(-10, min(score, 10))

def score_dwi_suffix(
    series: SeriesFeatures,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for dwi suffixes.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype.DWI].keys())
    
    # Since we only have one DWI suffix (dwi), this is effectively a binary classification (dwi vs unknown) but with probability based on confidence in the DWI features.
    raw_score = score_dwi_dwi(series, tokens)
    if raw_score >= 3:
        probs = {valid_suffixes[0]: 0.95}  # High confidence in DWI
        confidence = 0.9
        best_suffix = valid_suffixes[0]
    elif raw_score >= 0:
        probs = {valid_suffixes[0]: 0.6}  # Moderate confidence in DWI
        confidence = 0.2
        best_suffix = valid_suffixes[0]
    else:
        probs = {"unknown": 0.9}  # Likely not DWI
        confidence = 0.8
        best_suffix = "unknown"
    
    return probs, best_suffix, confidence

# =====PERFUSION SUFFIX SCORING (ASL, M0SCAN, DSC, DCE, CBF, CBV, MTT)====

def score_asl_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    score = 0

    temporal = series.temporal

    # --------------------------------------------------
    # Definitive ASL tag
    # --------------------------------------------------

    if (
        series.perfusion
        and series.perfusion.perfusion_labeling_type
        and series.perfusion.perfusion_labeling_type.value
    ):
        score += 6

    # --------------------------------------------------
    # ASL keywords
    # --------------------------------------------------

    if tokens & {"asl", "pcasl", "pasl", "arterial"}:
        score += 4

    # --------------------------------------------------
    # Temporal structure
    # --------------------------------------------------

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None and n_tp >= 20:
        score += 3

    # --------------------------------------------------
    # Acquisition
    # --------------------------------------------------

    if is_epi(series):
        score += 1

    # --------------------------------------------------
    # Negative evidence
    # --------------------------------------------------

    if series.contrast and series.contrast.contrast_agent:
        score -= 5

    if tokens & {"dsc", "dce"}:
        score -= 3

    return max(-10, min(score, 10))


def score_m0scan_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/m0scan suffix (M0 reference scan for ASL)."""
    score = 0

    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast

    n_tp = temporal.num_timepoints if temporal else None

    # --------------------------------------------------
    # M0 reference scan indicators
    # --------------------------------------------------

    if tokens & {"m0", "m0scan", "m0ref"}:
        score += 6

    # --------------------------------------------------
    # ASL-compatible acquisition
    # --------------------------------------------------

    is_asl_related = False
    if (
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    ):
        score += 3
        is_asl_related = True
    
    if tokens & {"asl", "pcasl", "pasl"}:
        is_asl_related = True

    # --------------------------------------------------
    # Temporal structure (few volumes for reference)
    # --------------------------------------------------

    if (n_tp is not None and n_tp <= 5) and is_asl_related:
        # If we have strong ASL indicators, then having very few timepoints is a strong signal for M0 reference scan (often just 1 volume)
        score += 4

    # --------------------------------------------------
    # Negative evidence
    # --------------------------------------------------

    if n_tp is not None and n_tp > 10:
        score -= 4

    if contrast and contrast.contrast_agent:
        score -= 3
        
    if series.image_type and series.image_type.is_derived:
        # M0 scans are typically not marked as "derived" images
        score -= 5

    return max(-10, min(score, 10))


def score_dsc_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/dsc suffix (Dynamic Susceptibility Contrast)."""
    score = 0

    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast

    n_tp = temporal.num_timepoints if temporal else None

    # --------------------------------------------------
    # DSC-specific indicators
    # --------------------------------------------------

    if tokens & {"dsc", "t2* perf", "dsc-mri", "perfusion"}:
        score += 4

    # --------------------------------------------------
    # EPI + contrast agent + many timepoints
    # --------------------------------------------------

    if n_tp is not None and n_tp > 40:
        score += 3
        if is_epi(series):
            score += 2 # Additional boost for being EPI, bringing total to 5

    if (
        contrast
        and contrast.contrast_agent
        and contrast.contrast_agent.value
    ):
        score += 3

    # --------------------------------------------------
    # Negative evidence
    # --------------------------------------------------

    if (
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    ):
        score -= 1

    if not is_epi(series):
        score -= 3

    return max(-10, min(score, 10))


def score_dce_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/dce suffix (Dynamic Contrast Enhanced)."""
    score = 0

    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast
    sequence = series.sequence

    te_bucket = temporal.te_bucket if temporal else None
    n_tp = temporal.num_timepoints if temporal else None

    # --------------------------------------------------
    # DCE-specific keywords
    # --------------------------------------------------

    if tokens & {"dce", "dce-mri", "permeability", "ktrans"}:
        score += 4

    # --------------------------------------------------
    # T1-weighted acquisition + contrast + many timepoints
    # --------------------------------------------------

    if te_bucket == "short":
        score += 2

    if (
        contrast
        and contrast.contrast_agent
        and contrast.contrast_agent.value
    ):
        score += 3

    if n_tp is not None and n_tp > 30:
        score += 3

    # --------------------------------------------------
    # GRE/SPGR typical for DCE
    # --------------------------------------------------

    if (
        sequence
        and sequence.scanning_sequence
        and any(s in str(sequence.scanning_sequence.value).lower() for s in ["gre", "spgr", "gradient", "tfl"])
    ):
        score += 2

    # --------------------------------------------------
    # Negative evidence
    # --------------------------------------------------

    if is_epi(series):
        score -= 3

    if (
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    ):
        score -= 1

    return max(-10, min(score, 10))

# ---Derived perf suffixes---

def score_cbf_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/cbf suffix using structured ImageType features."""
    score = 0
    it = series.image_type
    
    # --------------------------------------------------
    # 1. Explicit Indicators (ImageType & Keywords)
    # --------------------------------------------------
    # Use the structured boolean flag from DICOM parsing
    if it and it.is_cbf and it.is_cbf.value:
        score += 7

    if tokens & {"cbf", "flow", "cerebral blood flow"}:
        score += 5

    # --------------------------------------------------
    # 2. Derived Status vs. Raw Acquisition
    # --------------------------------------------------
    # Derived maps are usually NOT Original/Primary
    if it:
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 3
        
        # If it's flagged as reformatted or projection, it's likely a map
        if (it.is_reformatted and it.is_reformatted.value) or \
           (it.is_projection and it.is_projection.value):
            score += 2

    # --------------------------------------------------
    # 3. Volume Count (CBF maps are static/single-volume)
    # --------------------------------------------------
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 10:
        score -= 6  # Strong penalty: actual ASL/DSC scans have many volumes

    return max(-10, min(score, 10))


def score_cbv_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/cbv suffix using structured ImageType features."""
    score = 0
    it = series.image_type

    # --------------------------------------------------
    # 1. Explicit Indicators
    # --------------------------------------------------
    if it and it.is_cbv and it.is_cbv.value:
        score += 7

    if tokens & {"cbv", "blood volume", "cerebral blood volume"}:
        score += 5

    # --------------------------------------------------
    # 2. Derived Status
    # --------------------------------------------------
    if it:
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 3

    # --------------------------------------------------
    # 3. Volume Count
    # --------------------------------------------------
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 10:
        score -= 6

    return max(-10, min(score, 10))


def score_mtt_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/mtt suffix (Mean Transit Time)."""
    score = 0
    it = series.image_type

    # --------------------------------------------------
    # 1. Explicit Indicators 
    # (Note: MTT isn't in your ImageTypeFeature booleans, so we use tokens)
    # --------------------------------------------------
    if tokens & {"mtt", "transit time", "mean transit time"}:
        score += 8

    # --------------------------------------------------
    # 2. Derived Status
    # --------------------------------------------------
    if it:
        # If it's not original/primary, it's more likely to be a derived map like MTT
        if it.is_original and it.is_original.value:
            score -= 5
        if it.is_primary and it.is_primary.value:
            score -= 3
            
        # Perfusion indicator in ImageType (often present in vendor MTT maps)
        if it.has_perfusion and it.has_perfusion.value:
            score += 2

    # --------------------------------------------------
    # 3. Volume Count
    # --------------------------------------------------
    if series.num_volumes == 1:
        score += 3
    elif series.num_volumes > 10:
        score -= 6

    return max(-10, min(score, 10))

def score_perf_suffix(
    series: SeriesFeatures,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for perf suffixes.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype.PERF].keys())
    
    # Compute raw suffix scores
    raw_scores = {}
    for suffix in valid_suffixes:
        score_fn_name = f"score_{suffix}_perf"
        if score_fn_name in globals():
            raw_scores[suffix] = globals()[score_fn_name](series, tokens)
        else:
            raw_scores[suffix] = 0  # Default for unhandled suffixes
    
    # Convert to calibrated probabilities via softmax
    probs = softmax(raw_scores)
    
    # Pick best
    best_suffix = max(probs, key=lambda k: probs[k])
    best_prob = probs[best_suffix]
    
    # Confidence thresholds (same logic as datatype)
    if best_prob < 0.4:  # Low evidence
        best_suffix = "unknown"
        confidence = 0.0
    else:
        # Margin over runner-up
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    
    return probs, best_suffix, confidence

# =====FMAP SUFFIX SCORING (PHASEDIFF, MAGNITUDE1, MAGNITUDE2, EPI, FIELDMAP)====

def score_phasediff_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/phasediff suffix."""
    score = 0.0

    temporal = series.temporal
    image_type = series.image_type

    # --------------------------------------------------
    # STRONG PHYSICAL EVIDENCE
    # --------------------------------------------------

    is_phase = (
        image_type
        and image_type.is_phase
        and image_type.is_phase.value
    )

    if is_phase:
        score += 6

    # --------------------------------------------------
    # TYPICAL FIELDMAP STRUCTURE
    # --------------------------------------------------

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None and n_tp <= 2:
        score += 3

    # --------------------------------------------------
    # TEXTUAL CUES
    # --------------------------------------------------

    if tokens & {"phase", "phasediff", "phasediffmap", "phase_diff"}:
        score += 4

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    if tokens & {"magnitude", "mag"}:
        score -= 4

    return max(-10, min(score, 10))


def score_magnitude1_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/magnitude1 suffix."""
    score = 0.0

    temporal = series.temporal
    image_type = series.image_type

    # --------------------------------------------------
    # STRONG IMAGE EVIDENCE
    # --------------------------------------------------

    is_magnitude = (
        image_type
        and image_type.is_magnitude
        and image_type.is_magnitude.value
    )

    if is_magnitude:
        score += 5

    # --------------------------------------------------
    # TYPICAL FMAP ACQUISITION
    # --------------------------------------------------

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None and n_tp <= 2:
        score += 3

    # --------------------------------------------------
    # TEXTUAL CUES
    # --------------------------------------------------

    if tokens & {"magnitude", "mag", "mag1"}:
        score += 2

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    is_phase = (
        image_type
        and image_type.is_phase
        and image_type.is_phase.value
    )

    if is_phase:
        score -= 4

    return max(-10, min(score, 10))


def score_magnitude2_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/magnitude2 suffix."""
    score = 0.0

    temporal = series.temporal
    image_type = series.image_type

    # --------------------------------------------------
    # STRONG EVIDENCE
    # --------------------------------------------------

    is_magnitude = (
        image_type
        and image_type.is_magnitude
        and image_type.is_magnitude.value
    )

    if is_magnitude:
        score += 5

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None and n_tp <= 2:
        score += 3

    # --------------------------------------------------
    # TEXTUAL CUES
    # --------------------------------------------------

    if tokens & {"mag2"}:
        score += 3

    if tokens & {"magnitude", "mag"}:
        score += 1

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    is_phase = (
        image_type
        and image_type.is_phase
        and image_type.is_phase.value
    )

    if is_phase:
        score -= 4

    if tokens & {"mag1"}:
        score -= 2

    return max(-10, min(score, 10))


def score_epi_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/epi suffix (EPI fieldmap)."""
    score = 0.0

    temporal = series.temporal
    encoding = series.encoding
    image_type = series.image_type

    # --------------------------------------------------
    # STRONG STRUCTURAL EVIDENCE
    # --------------------------------------------------

    is_epi_seq = is_epi(series)

    if is_epi_seq:
        score += 4

    n_tp = temporal.num_timepoints if temporal else None

    if is_epi_seq and n_tp is not None and 2 <= n_tp <= 6:
        score += 4

    # --------------------------------------------------
    # ACQUISITION METADATA
    # --------------------------------------------------

    has_pe_dir = (
        encoding
        and encoding.phase_encoding_direction
        and encoding.phase_encoding_direction.value
    )

    if has_pe_dir:
        score += 2

    # --------------------------------------------------
    # TEXTUAL CUES
    # --------------------------------------------------

    if tokens & {"fieldmap", "fmap", "topup", "phasecorr"}:
        score += 3

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    if n_tp is not None and n_tp > 10:
        score -= 4

    is_phase = (
        image_type
        and image_type.is_phase
        and image_type.is_phase.value
    )

    is_magnitude = (
        image_type
        and image_type.is_magnitude
        and image_type.is_magnitude.value
    )

    if is_phase or is_magnitude:
        score -= 3
    
    if series.multi_echo and series.multi_echo.num_echoes and series.multi_echo.num_echoes > 1:
        score -= 2

    return max(-10, min(score, 10))


def score_fieldmap_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/fieldmap suffix."""
    score = 0.0

    temporal = series.temporal
    image_type = series.image_type

    # --------------------------------------------------
    # TEXTUAL EVIDENCE
    # --------------------------------------------------

    if tokens & {"fieldmap", "fmap", "b0", "shimming"}:
        score += 4

    # --------------------------------------------------
    # TYPICAL ACQUISITION
    # --------------------------------------------------

    num_echoes = (
        series.multi_echo.num_echoes
        if series.multi_echo and series.multi_echo.num_echoes
        else None
    )

    if num_echoes and num_echoes > 1:
        score += 3

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None and n_tp <= 4:
        score += 2

    # --------------------------------------------------
    # NEGATIVE EVIDENCE
    # --------------------------------------------------

    is_phase = (
        image_type
        and image_type.is_phase
        and image_type.is_phase.value
    )

    is_magnitude = (
        image_type
        and image_type.is_magnitude
        and image_type.is_magnitude.value
    )

    if is_phase or is_magnitude:
        score -= 3

    is_epi_seq = is_epi(series)

    if is_epi_seq:
        score -= 2

    return max(-10, min(score, 10))

def score_fmap_suffix(
    series: SeriesFeatures,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated probabilities for fmap suffixes.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype.FMAP].keys())
    
    # Compute raw suffix scores
    raw_scores = {}
    for suffix in valid_suffixes:
        score_fn_name = f"score_{suffix}_fmap"
        if score_fn_name in globals():
            raw_scores[suffix] = globals()[score_fn_name](series, tokens)
        else:
            raw_scores[suffix] = 0  # Default for unhandled suffixes
    
    # Convert to calibrated probabilities via softmax
    probs = softmax(raw_scores)
    
    # Pick best
    best_suffix = max(probs, key=lambda k: probs[k])
    best_prob = probs[best_suffix]
    
    # Confidence thresholds (same logic as datatype)
    if best_prob < 0.4:  # Low evidence
        best_suffix = "unknown"
        confidence = 0.0
    else:
        # Margin over runner-up
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    
    return probs, best_suffix, confidence

# =====GENERATING ALL SUFFIXES FOR A GIVEN DATATYPE====

def score_suffix(
    series: SeriesFeatures,
    datatype: str,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Generic suffix scoring for any datatype.
    
    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    if datatype not in BIDS_SCHEMA:
        return {}, "unknown", 0.0
    
    valid_suffixes = list(BIDS_SCHEMA[Datatype(datatype)].keys())
    
    raw_scores = {}
    for suffix in valid_suffixes:
        score_fn_name = f"score_{suffix.lower()}_{datatype.lower()}"
        if score_fn_name in globals():
            raw_scores[suffix] = globals()[score_fn_name](series, tokens)
        else:
            # Fallback to generic suffix scoring
            # raw_scores[suffix] = score_generic_suffix(suffix, series, tokens)
            pass
    
    probs = softmax(raw_scores)
    if not probs:
        return {}, "unknown", 0.0
    best_suffix = max(probs, key=lambda k: probs[k])
    best_prob = probs[best_suffix]
    
    if best_prob < 0.4:
        best_suffix = "unknown"
        confidence = 0.0
    else:
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    
    return probs, best_suffix, confidence
