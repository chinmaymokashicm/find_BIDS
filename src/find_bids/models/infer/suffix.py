from typing import Callable, cast

from ..extract.series import SeriesFeatures
from .acquisition import (
    DIFFUSION_INTENT_THRESHOLD,
    EPI_FAMILY_THRESHOLD,
    FIELDMAP_INTENT_THRESHOLD,
    GRE_FAMILY_THRESHOLD,
    get_acquisition_family_scores,
    get_acquisition_intent_scores,
    is_epi,
    is_fieldmap_epi,
    PERFUSION_INTENT_THRESHOLD,
    SE_TSE_FAMILY_THRESHOLD,
)
from .utils import (
    apply_rules,
    softmax,
    is_gre_gr,
    has_multi_echo,
    feature_gt,
    apply_rules_clipped,
    stable_value,
    flag_true,
    token_matches
    )
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
    SBREF_KEYWORDS,
    DERIVED_MAP_KEYWORDS,
    REFORMAT_KEYWORDS,
    SEGMENTATION_KEYWORDS,
    FUNC_PREPROC_KEYWORDS,
    FUNC_STATMAP_KEYWORDS,
    FUNC_SUMMARY_KEYWORDS,
    LOCALIZER_KEYWORDS,
    PERF_TIMING_KEYWORDS,
    FMAP_PHASELIKE_KEYWORDS,
)
from .schema import Datatype, BIDS_SCHEMA


def _is_derived_like(series: SeriesFeatures, tokens: set[str]) -> bool:
    """Detect whether a series is likely derived."""
    it = series.image_type

    strong_image_flags = (
        flag_true(it, "is_reformatted")
        or flag_true(it, "is_projection")
        or flag_true(it, "is_mip")
        or flag_true(it, "is_adc")
        or flag_true(it, "is_fa")
        or flag_true(it, "is_trace")
        or flag_true(it, "is_cbf")
        or flag_true(it, "is_cbv")
    )

    strong_tokens = {
        "adc", "fa", "trace", "cbf", "cbv",
        "parametric", "quantitative", "qmap",
        "statmap", "statisticmap"
    }

    return bool(
        strong_image_flags
        or token_matches(tokens, strong_tokens)
    )


def _is_derived_bucket_label(suffix: str, datatype: str) -> bool:
    """Coarse derived buckets are prefixed with their datatype (e.g., anatParamMap)."""
    suffix_l = suffix.lower()
    datatype_l = datatype.lower()
    return suffix_l.startswith(datatype_l) and suffix_l != datatype_l

# =====ANATOMIC SUFFIX SCORING FUNCTIONS (T1w, T2w, FLAIR, PD, T2starw, SWI)====

def score_t1w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T1w suffix."""

    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None
    inversion_time = (
        stable_value(temporal.inversion_time)
        if temporal and temporal.inversion_time
        else None
    )
    is_iso = bool(temporal and temporal.is_isotropic)
    is_3d = bool(temporal and temporal.is_3D)

    return apply_rules_clipped(
        (token_matches(tokens, T1_KEYWORDS), 4),
        (flag_true(series.image_type, "is_mpr"), 4),

        # MRI physics
        (te_bucket == "short" and tr_bucket == "short", 3),

        # inversion-prepared sequences (MPRAGE)
        (inversion_time is not None, 3),

        (is_iso, 2),
        (is_3d, 1),

        (flag_true(series.contrast, "contrast_agent"), 2),

        # negatives
        (te_bucket == "long", -3),
        (token_matches(tokens, FLAIR_KEYWORDS), -4),
        (token_matches(tokens, T2_KEYWORDS), -2),
    )

def score_t2w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T2w suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None
    inversion_time = (
        stable_value(temporal.inversion_time)
        if temporal and temporal.inversion_time
        else None
    )
    is_3d = bool(
        series.sequence
        and series.sequence.mr_acquisition_type
        and series.sequence.mr_acquisition_type.value
        and str(series.sequence.mr_acquisition_type.value).lower() == "3d"
    )

    return apply_rules_clipped(
        (token_matches(tokens, T2_KEYWORDS), 4),

        # MRI physics
        (te_bucket == "long" and tr_bucket == "long", 3),
        (te_bucket == "long", 3),

        (not is_3d, 2),

        # penalties
        (inversion_time is not None and inversion_time > 1500, -2),
        (te_bucket == "short", -4),
        (token_matches(tokens, T1_KEYWORDS), -4),
        (token_matches(tokens, FLAIR_KEYWORDS), -3),
    )

def score_flair_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/FLAIR suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None

    inversion_time = stable_value(temporal.inversion_time) if temporal and temporal.inversion_time else None

    return apply_rules_clipped(
        (token_matches(tokens, FLAIR_KEYWORDS), 6),
        (inversion_time is not None and inversion_time > 1800, 4),
        (te_bucket == "long", 2),
        (te_bucket == "short", -4),
        (token_matches(tokens, T1_KEYWORDS), -3),
        (token_matches(tokens, T2_KEYWORDS), -2),
    )

def score_pd_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/PD suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    has_inversion = bool(
        temporal
        and temporal.inversion_time
        and stable_value(temporal.inversion_time)
    )

    return apply_rules_clipped(
        (token_matches(tokens, PD_KEYWORDS), 5),
        (te_bucket == "short" and tr_bucket == "long", 3),
        (token_matches(tokens, T2_KEYWORDS), -1),
        (has_inversion, -3),
    )

def score_t2starw_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/T2starw."""
    return apply_rules_clipped(
        (token_matches(tokens, T2STAR_KEYWORDS), 6),
        (token_matches(tokens, GRE_KEYWORDS), 2),
        (is_gre_gr(series.sequence), 2),
        (has_multi_echo(series), 1),
        # negatives
        (token_matches(tokens, SWI_KEYWORDS), -4),
        (token_matches(tokens, FUNC_KEYWORDS), -5),
        (token_matches(tokens, DIFFUSION_KEYWORDS), -6),
        (is_epi(series, tokens), -4),

        (flag_true(series.image_type, "is_magnitude"), 1),
    )

def score_swi_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/swi."""
    return apply_rules_clipped(
        (token_matches(tokens, SWI_KEYWORDS), 6),
        (token_matches(tokens, GRE_KEYWORDS), 2),
        (is_gre_gr(series.sequence), 2),
        ("phase" in tokens or "pha" in tokens, 1),
        ("mag" in tokens or "magnitude" in tokens, 1),
        (token_matches(tokens, T2STAR_KEYWORDS), -3),
        (token_matches(tokens, FUNC_KEYWORDS), -5),
        (token_matches(tokens, DIFFUSION_KEYWORDS), -6),
        (is_epi(series, tokens), -3),
        (flag_true(series.image_type, "is_magnitude"), 1),
    )

# ---- Coarse derived buckets for anat (Ideas v3)

def score_anatparammap_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket anatParamMap."""
    it = series.image_type
    map_keywords = (
        T1MAP_KEYWORDS
        | T2MAP_KEYWORDS
        | {"parametric", "qmap", "quantitative", "quant"}
    )

    return apply_rules_clipped(
        (token_matches(tokens, map_keywords), 6),
        # vendor map flags
        (
            flag_true(it, "is_adc")
            or flag_true(it, "is_fa")
            or flag_true(it, "is_trace"),
            3,
        ),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        # weaker penalty (maps can be 4D)
        (series.num_volumes > 50, -4),

        # negatives
        (flag_true(it, "is_original"), -5),
        (flag_true(it, "is_primary"), -3),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_anatreformat_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket anatReformat."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, REFORMAT_KEYWORDS), 6),
        (flag_true(it, "is_reformatted"), 5),
        (flag_true(it, "is_projection") or flag_true(it, "is_mip") or flag_true(it, "is_mpr"), 3),
        (series.num_volumes == 1, 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, T1MAP_KEYWORDS | T2MAP_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_anatsegmentation_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket anatSegmentation."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, SEGMENTATION_KEYWORDS), 7),
        (series.num_volumes == 1, 3),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
        (token_matches(tokens, T1MAP_KEYWORDS | T2MAP_KEYWORDS), -2),
    )


def score_anatotherderived_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Catch-all score for coarse bucket anatOtherDerived."""
    it = series.image_type

    return apply_rules_clipped(
        (_is_derived_like(series, tokens), 5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 3),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, T1MAP_KEYWORDS | T2MAP_KEYWORDS), -2),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -2),
    )

# =====FUNCTIONAL SUFFIX SCORING (raw + coarse derived buckets)====

def score_bold_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/bold."""
    temporal = series.temporal
    num_volumes = series.num_volumes or 1
    is_long_timeseries = num_volumes > 30
    is_short_timeseries = 5 < num_volumes <= 30

    return apply_rules_clipped(
        (token_matches(tokens, BOLD_KEYWORDS), 5),

        # strong physics signals
        (is_epi(series, tokens), 4),
        (is_long_timeseries, 5),
        (is_short_timeseries, 2),
        # SBREF should not be bold
        (token_matches(tokens, SBREF_KEYWORDS), -6),
        # diffusion penalty
        (token_matches(tokens, DIFFUSION_KEYWORDS), -6),
        # fieldmaps often EPI but short
        (token_matches(tokens, FMAP_KEYWORDS), -4),
        # derived
        (_is_derived_like(series, tokens), -4),
        # image type
        (flag_true(series.image_type, "is_original"), 1),
    )

def score_sbref_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/sbref."""
    num_volumes = series.num_volumes or 1
    return apply_rules_clipped(
        (token_matches(tokens, SBREF_KEYWORDS), 6),
        # physics
        (is_epi(series, tokens), 3),
        (num_volumes == 1, 4),
        # penalties
        (num_volumes > 5, -6),
        (token_matches(tokens, BOLD_KEYWORDS), -2),
        (token_matches(tokens, DIFFUSION_KEYWORDS), -5),
        (_is_derived_like(series, tokens), -4),
    )


# ---- Coarse derived buckets for func (Ideas v3)

def score_funcpreprocbold_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket funcPreprocBold."""
    temporal = series.temporal
    it = series.image_type
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (token_matches(tokens, FUNC_PREPROC_KEYWORDS), 7),
        (token_matches(tokens, BOLD_KEYWORDS), 2),
        (is_epi(series, tokens), 2),
        (n_tp is not None and n_tp > 30, 2),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, FUNC_STATMAP_KEYWORDS), -3),
        (token_matches(tokens, FUNC_SUMMARY_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_funcstatmap_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket funcStatMap."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, FUNC_STATMAP_KEYWORDS), 8),
        (token_matches(tokens, BOLD_KEYWORDS), 1),
        (series.num_volumes == 1, 3),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_functimeseriessummary_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket funcTimeseriesSummary."""
    temporal = series.temporal
    it = series.image_type
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (token_matches(tokens, FUNC_SUMMARY_KEYWORDS), 8),
        (token_matches(tokens, BOLD_KEYWORDS), 2),
        (n_tp is not None and n_tp > 10, 2),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, FUNC_STATMAP_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_funcsegmentationormask_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket funcSegmentationOrMask."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, SEGMENTATION_KEYWORDS), 7),
        (token_matches(tokens, BOLD_KEYWORDS | FUNC_KEYWORDS), 2),
        (series.num_volumes == 1, 3),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, FUNC_STATMAP_KEYWORDS), -2),
    )


def score_funcotherderived_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Catch-all score for coarse bucket funcOtherDerived."""
    it = series.image_type

    return apply_rules_clipped(
        (_is_derived_like(series, tokens), 5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 3),
        (token_matches(tokens, BOLD_KEYWORDS | FUNC_KEYWORDS), 1),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, FUNC_PREPROC_KEYWORDS), -2),
        (token_matches(tokens, FUNC_STATMAP_KEYWORDS), -2),
        (token_matches(tokens, FUNC_SUMMARY_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -2),
    )

# =====DIFFUSION SUFFIX SCORING (raw + coarse derived buckets)====

def score_dwi_dwi(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for dwi/dwi."""

    diffusion = series.diffusion

    has_bvals = bool(diffusion and diffusion.b_values and len(diffusion.b_values) > 0)
    has_directions = bool(diffusion and diffusion.num_diffusion_directions and diffusion.num_diffusion_directions > 0)

    num_volumes = series.num_volumes or 1

    return apply_rules_clipped(
        (token_matches(tokens, DIFFUSION_KEYWORDS), 5),
        # strongest signals
        (has_bvals, 6),
        (has_directions, 5),
        # diffusion often multi-volume
        (num_volumes > 5, 3),
        (is_epi(series, tokens), 2),
        # penalties
        (_is_derived_like(series, tokens), -5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), -6),
        (token_matches(tokens, FMAP_KEYWORDS), -4),
    )

def score_dwiparammap_dwi(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket dwiParamMap."""
    it = series.image_type

    return apply_rules_clipped(
        (flag_true(it, "is_adc"), 6),
        (flag_true(it, "is_fa"), 6),
        (flag_true(it, "is_trace"), 5),
        (token_matches(tokens, {"adc", "fa", "trace", "md", "rd", "ad", "tensor", "colfa"}), 6),
        (token_matches(tokens, {"fractional", "anisotropy", "apparent", "diffusion"}), 2),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -4),
        (flag_true(it, "is_original"), -5),
        (flag_true(it, "is_primary"), -3),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_dwisegmentationormask_dwi(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket dwiSegmentationOrMask."""
    it = series.image_type

    score = apply_rules_clipped(
        (token_matches(tokens, SEGMENTATION_KEYWORDS), 7),
        (token_matches(tokens, DIFFUSION_KEYWORDS), 2),
        (flag_true(it, "has_diffusion"), 2),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, {"adc", "fa", "trace", "tensor"}), -2),
    )
    return float(score)


def score_dwiotherderived_dwi(series: SeriesFeatures, tokens: set[str]) -> float:
    """Catch-all score for coarse bucket dwiOtherDerived."""
    it = series.image_type

    score = apply_rules_clipped(
        (_is_derived_like(series, tokens), 5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 3),
        (token_matches(tokens, DIFFUSION_KEYWORDS), 2),
        (flag_true(it, "has_diffusion"), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, {"adc", "fa", "trace", "tensor"}), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -2),
    )
    return float(score)

# =====PERFUSION SUFFIX SCORING (raw + coarse derived buckets)====

def score_asl_perf(series: SeriesFeatures, tokens: set[str]) -> float:
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (
            bool(
                series.perfusion
                and series.perfusion.perfusion_labeling_type
                and series.perfusion.perfusion_labeling_type.value
            ),
            6,
        ),
        (token_matches(tokens, {"asl", "pcasl", "pasl", "arterial"}), 4),
        (n_tp is not None and n_tp >= 20, 3),
        (is_epi(series, tokens), 1),
        (bool(series.contrast and series.contrast.contrast_agent), -5),
        (token_matches(tokens, {"dsc", "dce"}), -3),
    )


def score_m0scan_perf(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for perf/m0scan suffix (M0 reference scan for ASL)."""
    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast
    n_tp = temporal.num_timepoints if temporal else None

    has_asl_label = bool(
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    )
    is_asl_related = has_asl_label or token_matches(tokens, {"asl", "pcasl", "pasl"})

    return apply_rules_clipped(
        (token_matches(tokens, {"m0", "m0scan", "m0ref"}), 6),
        (has_asl_label, 3),
        ((n_tp is not None and n_tp <= 5) and is_asl_related, 4),
        (n_tp is not None and n_tp > 10, -4),
        (bool(contrast and contrast.contrast_agent), -3),
        (bool(series.image_type and series.image_type.is_derived), -5),
    )


def score_dsc_perf(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for perf/dsc suffix (Dynamic Susceptibility Contrast)."""
    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast
    n_tp = temporal.num_timepoints if temporal else None

    is_epi_seq = is_epi(series, tokens)
    has_perfusion_label = bool(
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    )
    return apply_rules_clipped(
        (token_matches(tokens, {"dsc", "t2* perf", "dsc-mri", "perfusion"}), 4),
        (n_tp is not None and n_tp > 40, 3),
        (n_tp is not None and n_tp > 40 and is_epi_seq, 2),
        (flag_true(contrast, "contrast_agent"), 3),
        (has_perfusion_label, -1),
        (not is_epi_seq, -3),
    )


def score_dce_perf(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for perf/dce suffix (Dynamic Contrast Enhanced)."""
    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast
    sequence = series.sequence
    te_bucket = temporal.te_bucket if temporal else None
    n_tp = temporal.num_timepoints if temporal else None

    has_dce_sequence_hint = bool(
        sequence
        and sequence.scanning_sequence
        and any(
            s in str(sequence.scanning_sequence.value).lower()
            for s in ["gre", "spgr", "gradient", "tfl"]
        )
    )
    has_perfusion_label = bool(
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    )
    score = apply_rules_clipped(
        (token_matches(tokens, {"dce", "dce-mri", "permeability", "ktrans"}), 4),
        (te_bucket == "short", 2),
        (flag_true(contrast, "contrast_agent"), 3),
        (n_tp is not None and n_tp > 30, 3),
        (has_dce_sequence_hint, 2),
        (is_epi(series, tokens), -3),
        (has_perfusion_label, -1),
    )
    return int(score)

# --- Coarse derived buckets for perf (Ideas v3)

def score_perfcbflikemap_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for coarse bucket perfCbfLikeMap."""
    it = series.image_type

    score = apply_rules_clipped(
        (flag_true(it, "is_cbf"), 7),
        (token_matches(tokens, {"cbf", "flow", "cerebral", "blood", "perfusion"}), 5),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -5),
        (flag_true(it, "is_original"), -5),
        (flag_true(it, "is_primary"), -3),
        (token_matches(tokens, {"cbv"} | PERF_TIMING_KEYWORDS), -3),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )
    return int(score)


def score_perfcbvlikemap_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for coarse bucket perfCbvLikeMap."""
    it = series.image_type

    score = apply_rules_clipped(
        (flag_true(it, "is_cbv"), 7),
        (token_matches(tokens, {"cbv", "blood", "volume", "cerebral"}), 5),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -5),
        (flag_true(it, "is_original"), -5),
        (flag_true(it, "is_primary"), -3),
        (token_matches(tokens, {"cbf"} | PERF_TIMING_KEYWORDS), -3),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )
    return int(score)


def score_perftimingmap_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for coarse bucket perfTimingMap."""
    it = series.image_type

    score = apply_rules_clipped(
        (token_matches(tokens, PERF_TIMING_KEYWORDS), 8),
        (token_matches(tokens, {"transit", "delay", "arrival"}), 3),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "has_perfusion"), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -5),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, {"cbf", "cbv"}), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )
    return int(score)


def score_perfsegmentationorroi_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for coarse bucket perfSegmentationOrRoi."""
    it = series.image_type

    score = apply_rules_clipped(
        (token_matches(tokens, SEGMENTATION_KEYWORDS | {"roi"}), 8),
        (token_matches(tokens, {"perfusion", "cbf", "cbv"}), 2),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, PERF_TIMING_KEYWORDS), -2),
    )
    return int(score)


def score_perfotherderived_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Catch-all score for coarse bucket perfOtherDerived."""
    it = series.image_type

    score = apply_rules_clipped(
        (_is_derived_like(series, tokens), 5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 3),
        (token_matches(tokens, {"perfusion", "cbf", "cbv"} | PERF_TIMING_KEYWORDS), 1),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, {"cbf", "cbv"}), -2),
        (token_matches(tokens, PERF_TIMING_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -2),
    )
    return int(score)

# =====FMAP SUFFIX SCORING (PHASEDIFF, MAGNITUDE1, MAGNITUDE2, EPI, FIELDMAP)====

def score_phasediff_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/phasediff suffix."""
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (token_matches(tokens, {"phase", "phasediff", "phasediffmap", "phase_diff"}), 4),
        # physics
        (has_multi_echo(series), 4),
        (series.num_volumes == 1, 2),

        # GRE often used
        (is_gre_gr(series.sequence), 2),

        # penalties
        (token_matches(tokens, BOLD_KEYWORDS), -5),
        (token_matches(tokens, DIFFUSION_KEYWORDS), -5),

        (_is_derived_like(series, tokens), -4),
    )


def score_magnitude1_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/magnitude1 suffix."""
    temporal = series.temporal
    echo_numbers = (
        series.multi_echo.echo_numbers
        if series.multi_echo and series.multi_echo.echo_numbers
        else None
    )
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (flag_true(series.image_type, "is_magnitude"), 5),
        (echo_numbers and echo_numbers[0] == 1, 2),
        (n_tp is not None and n_tp <= 2, 3),
        (token_matches(tokens, {"magnitude", "mag", "mag1"}), 2),
        (flag_true(series.image_type, "is_phase"), -4),
        (token_matches(tokens, {"mag2"}), -2),
        (echo_numbers and echo_numbers[0] == 2, -2),
    )


def score_magnitude2_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/magnitude2 suffix."""
    temporal = series.temporal
    echo_numbers = (
        series.multi_echo.echo_numbers
        if series.multi_echo and series.multi_echo.echo_numbers
        else None
    )
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (flag_true(series.image_type, "is_magnitude"), 5),
        (echo_numbers and echo_numbers[0] == 2, 2),
        (n_tp is not None and n_tp <= 2, 3),
        (token_matches(tokens, {"mag2"}), 3),
        (token_matches(tokens, {"magnitude", "mag"}), 1),
        (flag_true(series.image_type, "is_phase"), -4),
        (token_matches(tokens, {"mag1"}), -2),
        (echo_numbers and echo_numbers[0] == 1, -2),
    )


def score_epi_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/epi suffix (EPI fieldmap)."""
    temporal = series.temporal
    encoding = series.encoding
    is_epi_seq = is_epi(series, tokens)
    is_epi_fieldmap = is_fieldmap_epi(series, tokens)
    n_tp = temporal.num_timepoints if temporal else None

    has_pe_dir = bool(
        encoding
        and encoding.phase_encoding_direction
        and encoding.phase_encoding_direction.value
    )

    return apply_rules_clipped(
        (is_epi_fieldmap, 5),
        (is_epi_seq and n_tp is not None and 2 <= n_tp <= 6, 4),
        (has_pe_dir, 2),
        (token_matches(tokens, {"fieldmap", "fmap", "topup", "phasecorr"}), 3),
        (not is_epi_seq, -4),
        (n_tp is not None and n_tp >= 10, -4),
        (flag_true(series.image_type, "is_phase") or flag_true(series.image_type, "is_magnitude"), -3),
        (has_multi_echo(series), -2),
    )


def score_fieldmap_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/fieldmap suffix."""
    temporal = series.temporal
    num_echoes = series.multi_echo.num_echoes if series.multi_echo and series.multi_echo.num_echoes else None
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (token_matches(tokens, {"fieldmap", "fmap", "b0", "shimming"}), 4),
        (num_echoes and num_echoes > 1, 3),
        (n_tp is not None and n_tp <= 4, 2),
        (flag_true(series.image_type, "is_phase") or flag_true(series.image_type, "is_magnitude"), -3),
        (is_fieldmap_epi(series, tokens), -4),
        (is_epi(series, tokens), -2),
    )


# ---- Coarse derived buckets for fmap (Ideas v3)

def score_fmapsusceptibilityorphasemap_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket fmapSusceptibilityOrPhaseMap."""
    it = series.image_type
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None

    return apply_rules_clipped(
        (token_matches(tokens, FMAP_PHASELIKE_KEYWORDS), 6),
        (flag_true(it, "is_phase"), 5),
        (flag_true(it, "is_magnitude"), 1),
        (_is_derived_like(series, tokens), 2),
        (n_tp is not None and n_tp <= 4, 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
    )


def score_fmapreformat_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket fmapReformat."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, REFORMAT_KEYWORDS), 6),
        (flag_true(it, "is_reformatted"), 5),
        (flag_true(it, "is_projection") or flag_true(it, "is_mip"), 3),
        (token_matches(tokens, FMAP_KEYWORDS), 2),
        (_is_derived_like(series, tokens), 2),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -3),
        (token_matches(tokens, FMAP_PHASELIKE_KEYWORDS), -2),
    )


def score_fmapsegmentationormask_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for coarse bucket fmapSegmentationOrMask."""
    it = series.image_type

    return apply_rules_clipped(
        (token_matches(tokens, SEGMENTATION_KEYWORDS), 8),
        (token_matches(tokens, FMAP_KEYWORDS | FMAP_PHASELIKE_KEYWORDS), 2),
        (_is_derived_like(series, tokens), 2),
        (series.num_volumes == 1, 3),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
    )


def score_fmapotherderived_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Catch-all score for coarse bucket fmapOtherDerived."""
    it = series.image_type

    return apply_rules_clipped(
        (_is_derived_like(series, tokens), 5),
        (token_matches(tokens, DERIVED_MAP_KEYWORDS), 3),
        (token_matches(tokens, FMAP_KEYWORDS | FMAP_PHASELIKE_KEYWORDS), 1),
        (flag_true(it, "is_original"), -5),
        (token_matches(tokens, REFORMAT_KEYWORDS), -2),
        (token_matches(tokens, SEGMENTATION_KEYWORDS), -2),
    )


def _apply_acquisition_family_guidance(
    raw_scores: dict[str, float],
    datatype: str,
    series: SeriesFeatures,
    tokens: set[str],
) -> None:
    family_scores = get_acquisition_family_scores(series, tokens)
    intent_scores = get_acquisition_intent_scores(series, tokens)

    if datatype == Datatype.ANAT.value:
        if "T1w" in raw_scores:
            raw_scores["T1w"] += apply_rules(
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, -1),
            )
        if "T2w" in raw_scores:
            raw_scores["T2w"] += apply_rules(
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 2),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, -2),
            )
        if "FLAIR" in raw_scores:
            raw_scores["FLAIR"] += apply_rules(
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 2),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, -1),
            )
        if "PD" in raw_scores:
            raw_scores["PD"] += apply_rules(
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 1),
            )
        if "T2starw" in raw_scores:
            raw_scores["T2starw"] += apply_rules(
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, -2),
            )
        if "SWI" in raw_scores:
            raw_scores["SWI"] += apply_rules(
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, -2),
            )
        return

    if datatype == Datatype.FUNC.value:
        if "bold" in raw_scores:
            raw_scores["bold"] += apply_rules(
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 2),
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -3),
                (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -4),
                (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, -2),
            )
        if "sbref" in raw_scores:
            raw_scores["sbref"] += apply_rules(
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 2),
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -1),
                (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -3),
            )
        return

    if datatype == Datatype.DWI.value:
        if "dwi" in raw_scores:
            raw_scores["dwi"] += apply_rules(
                (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, 4),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 1),
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, -3),
            )
        return

    if datatype == Datatype.PERF.value:
        if "asl" in raw_scores:
            raw_scores["asl"] += apply_rules(
                (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 3),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 1),
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, 1),
            )
        if "m0scan" in raw_scores:
            raw_scores["m0scan"] += apply_rules(
                (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 2),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 1),
            )
        if "dsc" in raw_scores:
            raw_scores["dsc"] += apply_rules(
                (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 3),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 2),
            )
        if "dce" in raw_scores:
            raw_scores["dce"] += apply_rules(
                (intent_scores["perfusion"] >= PERFUSION_INTENT_THRESHOLD, 3),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, -1),
                (family_scores["se_tse"] >= SE_TSE_FAMILY_THRESHOLD, -2),
            )
        return

    if datatype == Datatype.FMAP.value:
        if "phasediff" in raw_scores:
            raw_scores["phasediff"] += apply_rules(
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 3),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, -2),
            )
        if "magnitude1" in raw_scores:
            raw_scores["magnitude1"] += apply_rules(
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 2),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 1),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, -2),
            )
        if "magnitude2" in raw_scores:
            raw_scores["magnitude2"] += apply_rules(
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 2),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 1),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, -2),
            )
        if "epi" in raw_scores:
            raw_scores["epi"] += apply_rules(
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 3),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, 2),
                (intent_scores["diffusion"] >= DIFFUSION_INTENT_THRESHOLD, -3),
            )
        if "fieldmap" in raw_scores:
            raw_scores["fieldmap"] += apply_rules(
                (intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD, 3),
                (family_scores["gre"] >= GRE_FAMILY_THRESHOLD, 2),
                (family_scores["epi"] >= EPI_FAMILY_THRESHOLD, -2),
            )

# =====GENERATING ALL SUFFIXES FOR A GIVEN DATATYPE====

def score_suffix(
    series: SeriesFeatures,
    datatype: str,
    tokens: set[str],
    is_derived: bool | None = None,
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated suffix probabilities for a given datatype.

    Looks up score_{suffix}_{datatype} functions for each valid suffix in
    BIDS_SCHEMA, applies softmax, and returns the best suffix with a
    confidence margin. When `is_derived` is provided, suffix candidates are
    filtered to raw suffixes (`False`) or coarse derived buckets (`True`).
    Returns "unknown" when the margin between the top two candidates is
    below 0.2.

    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    if datatype not in BIDS_SCHEMA:
        return {}, "unknown", 0.0

    valid_suffixes = list(BIDS_SCHEMA[Datatype(datatype)].keys())
    if is_derived is True:
        derived_suffixes = [s for s in valid_suffixes if _is_derived_bucket_label(s, datatype)]
        if derived_suffixes:
            valid_suffixes = derived_suffixes
    elif is_derived is False:
        raw_suffixes = [s for s in valid_suffixes if not _is_derived_bucket_label(s, datatype)]
        if raw_suffixes:
            valid_suffixes = raw_suffixes

    raw_scores: dict[str, float] = {}
    for suffix in valid_suffixes:
        score_fn_obj = globals().get(f"score_{suffix.lower()}_{datatype.lower()}")
        if callable(score_fn_obj):
            score_fn = cast(Callable[[SeriesFeatures, set[str]], float], score_fn_obj)
            raw_scores[suffix] = float(score_fn(series, tokens))
        else:
            raw_scores[suffix] = 0.0

    _apply_acquisition_family_guidance(raw_scores, datatype, series, tokens)

    probs = softmax(raw_scores)
    if not probs:
        return {}, "unknown", 0.0

    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    best_suffix, best_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_prob - second_prob

    if margin < 0.2:
        return probs, "unknown", 0.0
    return probs, best_suffix, margin
