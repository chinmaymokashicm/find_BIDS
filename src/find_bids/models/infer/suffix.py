from typing import Callable, cast
from pydantic import BaseModel, ConfigDict

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
    SBREF_KEYWORDS,
    DERIVED_MAP_KEYWORDS
)
from .schema import Datatype, BIDS_SCHEMA

CLIPPING_SCORE_MIN = -15
CLIPPING_SCORE_MAX = 15


class ScoringRule(BaseModel):
    """Validated scoring rule used by `_apply_rules`."""
    condition: bool
    delta: float

    model_config = ConfigDict(extra="forbid", frozen=True)


def _clip_score(score: float) -> float:
    return max(CLIPPING_SCORE_MIN, min(score, CLIPPING_SCORE_MAX))


def _stable_value(field, min_fraction: float = 0.5):
    """Return a stable scalar value from a feature field, else None."""
    if not field or field.valid_fraction <= min_fraction or not field.stable:
        return None
    return field.value


def _flag_true(obj, attr: str) -> bool:
    """Safely read boolean-like flags exposed as nested objects with `.value`."""
    value = getattr(obj, attr, None) if obj else None
    return bool(value and getattr(value, "value", False))


def _feature_gt(obj, attr: str, threshold: float) -> bool:
    feature = getattr(obj, attr, None) if obj else None
    return bool(feature and feature.value and feature.value > threshold)


def _has(tokens: set[str], keywords: set[str]) -> bool:
    """Return whether any keyword is present in a token set."""
    return bool(tokens & keywords)


def _as_scoring_rule(rule: ScoringRule | tuple[object, float]) -> ScoringRule:
    if isinstance(rule, ScoringRule):
        return rule
    condition, delta = rule
    return ScoringRule(condition=bool(condition), delta=float(delta))


def _apply_rules(*rules: ScoringRule | tuple[object, float], base: float = 0.0) -> float:
    """Accumulate validated rules and clip the final score."""
    score = base
    for raw_rule in rules:
        rule = _as_scoring_rule(raw_rule)
        if rule.condition:
            score += rule.delta
    return _clip_score(score)


def _is_gre_gr(sequence) -> bool:
    return bool(
        sequence
        and sequence.scanning_sequence
        and sequence.scanning_sequence.value == "GR"
        and sequence.scanning_sequence.consistency
        and sequence.scanning_sequence.consistency > 0.7
    )


def _has_multi_echo(series: SeriesFeatures, min_echoes: int = 1) -> bool:
    return bool(
        series.multi_echo
        and series.multi_echo.num_echoes
        and series.multi_echo.num_echoes > min_echoes
    )

# =====ANATOMIC SUFFIX SCORING FUNCTIONS (T1w, T2w, FLAIR, PD, T2starw, SWI)====

def score_t1w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T1w suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None

    is_iso = bool(temporal and temporal.is_isotropic)
    is_3d = bool(temporal and temporal.is_3D)
    has_inversion = bool(
        temporal
        and temporal.inversion_time
        and _stable_value(temporal.inversion_time) is not None
    )

    return _apply_rules(
        (_has(tokens, T1_KEYWORDS), 5),
        (_flag_true(series.image_type, "is_mpr"), 4),
        (has_inversion, 4),
        (is_iso, 2),
        (is_3d, 1),
        (_flag_true(series.contrast, "contrast_agent"), 3),
        (te_bucket == "long", -3),
        (_has(tokens, FLAIR_KEYWORDS), -4),
    )

def score_t2w_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T2w suffix."""
    temporal = series.temporal
    sequence = series.sequence
    te_bucket = temporal.te_bucket if temporal else None

    inversion_time = _stable_value(temporal.inversion_time) if temporal and temporal.inversion_time else None

    is_3d = bool(
        sequence
        and sequence.mr_acquisition_type
        and sequence.mr_acquisition_type.consistency
        and sequence.mr_acquisition_type.consistency > 0.8
        and str(sequence.mr_acquisition_type.value).lower() == "3d"
    )

    return _apply_rules(
        (_has(tokens, T2_KEYWORDS), 5),
        (te_bucket == "long", 4),
        (te_bucket == "medium", 1),
        (not is_3d, 2),
        (inversion_time is not None and inversion_time > 1500, -2),
        (te_bucket == "short", -4),
        (_has(tokens, T1_KEYWORDS), -4),
        (_has(tokens, FLAIR_KEYWORDS), -3),
    )

def score_flair_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/FLAIR suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None

    inversion_time = _stable_value(temporal.inversion_time) if temporal and temporal.inversion_time else None

    return _apply_rules(
        (_has(tokens, FLAIR_KEYWORDS), 6),
        (inversion_time is not None and inversion_time > 1800, 4),
        (te_bucket == "long", 2),
        (te_bucket == "short", -4),
        (_has(tokens, T1_KEYWORDS), -3),
        (_has(tokens, T2_KEYWORDS), -2),
    )

def score_pd_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/PD suffix."""
    temporal = series.temporal
    te_bucket = temporal.te_bucket if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    has_inversion = bool(
        temporal
        and temporal.inversion_time
        and _stable_value(temporal.inversion_time)
    )

    return _apply_rules(
        (_has(tokens, PD_KEYWORDS), 5),
        (te_bucket == "short" and tr_bucket == "long", 3),
        (_has(tokens, T2_KEYWORDS), -1),
        (has_inversion, -3),
    )

def score_t2starw_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/T2starw."""
    return _apply_rules(
        (_has(tokens, T2STAR_KEYWORDS), 6),
        (_has(tokens, GRE_KEYWORDS), 2),
        (_is_gre_gr(series.sequence), 2),
        (_has_multi_echo(series), 1),
        (_has(tokens, SWI_KEYWORDS), -4),
        (_has(tokens, FUNC_KEYWORDS), -5),
        (_has(tokens, DIFFUSION_KEYWORDS), -6),
        ("epi" in tokens, -3),
        (_flag_true(series.image_type, "is_magnitude"), 1),
    )

def score_swi_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for anat/swi."""
    return _apply_rules(
        (_has(tokens, SWI_KEYWORDS), 6),
        (_has(tokens, GRE_KEYWORDS), 2),
        (_is_gre_gr(series.sequence), 2),
        ("phase" in tokens or "pha" in tokens, 1),
        ("mag" in tokens or "magnitude" in tokens, 1),
        (_has(tokens, T2STAR_KEYWORDS), -3),
        (_has(tokens, FUNC_KEYWORDS), -5),
        (_has(tokens, DIFFUSION_KEYWORDS), -6),
        ("epi" in tokens, -3),
        (_flag_true(series.image_type, "is_magnitude"), 1),
    )

# ----Derived maps (T1map, T2map)

def score_t1map_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T1map (Quantitative T1 mapping)."""
    it = series.image_type

    return _apply_rules(
        (_has(tokens, T1MAP_KEYWORDS), 7),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -3),
        (_flag_true(it, "is_reformatted"), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 5, -6),
        (_has(tokens, {"t2map", "adc", "cbv"}), -5),
    )


def score_t2map_anat(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for anat/T2map (Quantitative T2 mapping)."""
    it = series.image_type

    return _apply_rules(
        (_has(tokens, T2MAP_KEYWORDS), 7),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -3),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 5, -6),
        (_has_multi_echo(series), 2),
        (_has(tokens, {"t1map", "flair", "bold"}), -5),
    )

# =====FUNCTIONAL SUFFIX SCORING (BOLD, CBV, CBF)====

def score_bold_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/bold."""
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    return _apply_rules(
        (_has(tokens, BOLD_KEYWORDS), 6),
        (is_epi(series), 2),
        (tr_bucket == "short", 2),
        (n_tp is not None and n_tp > 80, 4),
        (n_tp is not None and 20 < n_tp <= 80, 2),
        (_feature_gt(series.encoding, "multiband_factor", 1), 1),
        (_feature_gt(series.encoding, "parallel_reduction_factor_in_plane", 1), 1),
        (_has(tokens, SBREF_KEYWORDS), -4),
        (n_tp is not None and n_tp < 10, -4),
        (_has(tokens, FMAP_KEYWORDS), -4),
        (is_fieldmap_epi(series, tokens), -4),
    )

def score_sbref_func(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score likelihood for func/sbref."""
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None
    tr_bucket = temporal.tr_bucket if temporal else None

    return _apply_rules(
        (_has(tokens, SBREF_KEYWORDS), 6),
        (is_epi(series), 2),
        (tr_bucket in ("short", "medium"), 1),
        (n_tp is not None and n_tp == 1, 4),
        (n_tp is not None and 1 < n_tp <= 3, 2),
        (n_tp is not None and n_tp >= 50, -5),
        (_has(tokens, BOLD_KEYWORDS), -4),
        (is_fieldmap_epi(series, tokens), -4),
        (_has(tokens, FMAP_KEYWORDS), -3),
    )

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

    score = 5 if tokens & {"dwi", "diffusion", "dti", "dsi", "hardi"} else 0

    diffusion = series.diffusion
    n_tp = series.temporal.num_timepoints if series.temporal else None

    if diffusion:
        b_values = diffusion.b_values or []
        if b_values and len(set(b_values)) > 1:
            score += 4
            if max(b_values) > 700:
                score += 2

        n_dir = diffusion.num_diffusion_directions or 0
        if n_dir >= 30:
            score += 3
        elif n_dir >= 6:
            score += 2
        elif n_dir > 1:
            score += 1

    score = _apply_rules(
        (is_epi(series), 1),
        (n_tp is not None and n_tp >= 10, 1),
        (_has(tokens, {"b0", "trace", "adc"}), 1),
        (_has(tokens, {"bold", "fmri", "task", "rest"}), -5),
        (_has(tokens, {"t1", "t2", "flair", "mprage"}), -4),
        (_has(tokens, FMAP_KEYWORDS), -3),
        base=float(score),
    )
    return int(score)

def score_adc_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score confidence that a series is an ADC (Apparent Diffusion Coefficient) map.
    """
    it = series.image_type

    derived_like = bool(it and (_flag_true(it, "is_reformatted") or not _flag_true(it, "is_original")))
    score = _apply_rules(
        (_flag_true(it, "is_adc"), 7),
        (_has(tokens, {"adc", "apparent diffusion coefficient"}), 5),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -4),
        (derived_like, 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 5, -6),
        (_has(tokens, {"fa", "fractional anisotropy", "trace", "tensor"}), -4),
    )
    return int(score)

def score_fa_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Score confidence that a series is an FA (Fractional Anisotropy) map.
    """
    it = series.image_type

    score = _apply_rules(
        (_flag_true(it, "is_fa"), 7),
        (_has(tokens, {"fa", "fractional anisotropy", "colfa", "color fa"}), 6),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -4),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 5, -6),
        (_has(tokens, {"adc", "trace", "t1", "b1"}), -4),
    )
    return int(score)

# =====PERFUSION SUFFIX SCORING (ASL, M0SCAN, DSC, DCE, CBF, CBV, MTT)====

def score_asl_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None

    score = _apply_rules(
        (
            bool(
                series.perfusion
                and series.perfusion.perfusion_labeling_type
                and series.perfusion.perfusion_labeling_type.value
            ),
            6,
        ),
        (_has(tokens, {"asl", "pcasl", "pasl", "arterial"}), 4),
        (n_tp is not None and n_tp >= 20, 3),
        (is_epi(series), 1),
        (bool(series.contrast and series.contrast.contrast_agent), -5),
        (_has(tokens, {"dsc", "dce"}), -3),
    )
    return int(score)


def score_m0scan_perf(series: SeriesFeatures, tokens: set[str]) -> int:
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
    is_asl_related = has_asl_label or _has(tokens, {"asl", "pcasl", "pasl"})

    score = _apply_rules(
        (_has(tokens, {"m0", "m0scan", "m0ref"}), 6),
        (has_asl_label, 3),
        ((n_tp is not None and n_tp <= 5) and is_asl_related, 4),
        (n_tp is not None and n_tp > 10, -4),
        (bool(contrast and contrast.contrast_agent), -3),
        (bool(series.image_type and series.image_type.is_derived), -5),
    )
    return int(score)


def score_dsc_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/dsc suffix (Dynamic Susceptibility Contrast)."""
    temporal = series.temporal
    perfusion = series.perfusion
    contrast = series.contrast
    n_tp = temporal.num_timepoints if temporal else None

    is_epi_seq = is_epi(series)
    has_perfusion_label = bool(
        perfusion
        and perfusion.perfusion_labeling_type
        and perfusion.perfusion_labeling_type.value
    )
    score = _apply_rules(
        (_has(tokens, {"dsc", "t2* perf", "dsc-mri", "perfusion"}), 4),
        (n_tp is not None and n_tp > 40, 3),
        (n_tp is not None and n_tp > 40 and is_epi_seq, 2),
        (_flag_true(contrast, "contrast_agent"), 3),
        (has_perfusion_label, -1),
        (not is_epi_seq, -3),
    )
    return int(score)


def score_dce_perf(series: SeriesFeatures, tokens: set[str]) -> int:
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
    score = _apply_rules(
        (_has(tokens, {"dce", "dce-mri", "permeability", "ktrans"}), 4),
        (te_bucket == "short", 2),
        (_flag_true(contrast, "contrast_agent"), 3),
        (n_tp is not None and n_tp > 30, 3),
        (has_dce_sequence_hint, 2),
        (is_epi(series), -3),
        (has_perfusion_label, -1),
    )
    return int(score)

# ---Derived perf suffixes---

def score_cbf_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/cbf suffix using structured ImageType features."""
    it = series.image_type

    score = _apply_rules(
        (_flag_true(it, "is_cbf"), 7),
        (_has(tokens, {"cbf", "flow", "cerebral blood flow"}), 5),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -3),
        (_flag_true(it, "is_reformatted") or _flag_true(it, "is_projection"), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -6),
    )
    return int(score)

def score_cbv_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/cbv suffix using structured ImageType features."""
    it = series.image_type

    score = _apply_rules(
        (_flag_true(it, "is_cbv"), 7),
        (_has(tokens, {"cbv", "blood volume", "cerebral blood volume"}), 5),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -3),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -6),
    )
    return int(score)


def score_mtt_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perf/mtt suffix (Mean Transit Time)."""
    it = series.image_type

    score = _apply_rules(
        (_has(tokens, {"mtt", "transit time", "mean transit time"}), 8),
        (_flag_true(it, "is_original"), -5),
        (_flag_true(it, "is_primary"), -3),
        (_flag_true(it, "has_perfusion"), 2),
        (series.num_volumes == 1, 3),
        (series.num_volumes > 10, -6),
    )
    return int(score)

# =====FMAP SUFFIX SCORING (PHASEDIFF, MAGNITUDE1, MAGNITUDE2, EPI, FIELDMAP)====

def score_phasediff_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/phasediff suffix."""
    temporal = series.temporal
    n_tp = temporal.num_timepoints if temporal else None

    return _apply_rules(
        (_flag_true(series.image_type, "is_phase"), 6),
        (n_tp is not None and n_tp <= 2, 3),
        (_has(tokens, {"phase", "phasediff", "phasediffmap", "phase_diff"}), 4),
        (_has(tokens, {"magnitude", "mag"}), -4),
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

    return _apply_rules(
        (_flag_true(series.image_type, "is_magnitude"), 5),
        (echo_numbers and echo_numbers[0] == 1, 2),
        (n_tp is not None and n_tp <= 2, 3),
        (_has(tokens, {"magnitude", "mag", "mag1"}), 2),
        (_flag_true(series.image_type, "is_phase"), -4),
        (_has(tokens, {"mag2"}), -2),
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

    return _apply_rules(
        (_flag_true(series.image_type, "is_magnitude"), 5),
        (echo_numbers and echo_numbers[0] == 2, 2),
        (n_tp is not None and n_tp <= 2, 3),
        (_has(tokens, {"mag2"}), 3),
        (_has(tokens, {"magnitude", "mag"}), 1),
        (_flag_true(series.image_type, "is_phase"), -4),
        (_has(tokens, {"mag1"}), -2),
        (echo_numbers and echo_numbers[0] == 1, -2),
    )


def score_epi_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/epi suffix (EPI fieldmap)."""
    temporal = series.temporal
    encoding = series.encoding
    is_epi_seq = is_epi(series)
    n_tp = temporal.num_timepoints if temporal else None

    has_pe_dir = bool(
        encoding
        and encoding.phase_encoding_direction
        and encoding.phase_encoding_direction.value
    )

    return _apply_rules(
        (is_epi_seq, 4),
        (is_epi_seq and n_tp is not None and 2 <= n_tp <= 6, 4),
        (has_pe_dir, 2),
        (_has(tokens, {"fieldmap", "fmap", "topup", "phasecorr"}), 3),
        (n_tp is not None and n_tp >= 10, -4),
        (_flag_true(series.image_type, "is_phase") or _flag_true(series.image_type, "is_magnitude"), -3),
        (_has_multi_echo(series), -2),
    )


def score_fieldmap_fmap(series: SeriesFeatures, tokens: set[str]) -> float:
    """Score for fmap/fieldmap suffix."""
    temporal = series.temporal
    num_echoes = series.multi_echo.num_echoes if series.multi_echo and series.multi_echo.num_echoes else None
    n_tp = temporal.num_timepoints if temporal else None

    return _apply_rules(
        (_has(tokens, {"fieldmap", "fmap", "b0", "shimming"}), 4),
        (num_echoes and num_echoes > 1, 3),
        (n_tp is not None and n_tp <= 4, 2),
        (_flag_true(series.image_type, "is_phase") or _flag_true(series.image_type, "is_magnitude"), -3),
        (is_epi(series), -2),
    )

# =====GENERATING ALL SUFFIXES FOR A GIVEN DATATYPE====

def score_suffix(
    series: SeriesFeatures,
    datatype: str,
    tokens: set[str]
) -> tuple[dict[str, float], str, float]:
    """
    Compute calibrated suffix probabilities for a given datatype.

    Looks up score_{suffix}_{datatype} functions for each valid suffix in
    BIDS_SCHEMA, applies softmax, and returns the best suffix with a
    confidence margin. Returns "unknown" when the margin between the top
    two candidates is below 0.2.

    Returns:
        probs: dict mapping suffixes to probabilities (sums to 1)
        best_suffix: argmax suffix (or "unknown")
        confidence: max(prob) - second_max(prob) [0-1]
    """
    if datatype not in BIDS_SCHEMA:
        return {}, "unknown", 0.0

    valid_suffixes = BIDS_SCHEMA[Datatype(datatype)].keys()
    raw_scores: dict[str, float] = {}
    for suffix in valid_suffixes:
        score_fn_obj = globals().get(f"score_{suffix.lower()}_{datatype.lower()}")
        if callable(score_fn_obj):
            score_fn = cast(Callable[[SeriesFeatures, set[str]], float], score_fn_obj)
            raw_scores[suffix] = float(score_fn(series, tokens))
        else:
            raw_scores[suffix] = 0.0

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
