"""Score doc-aligned acquisition families and orthogonal intent flags."""

from ..extract.series import SeriesFeatures
from .rules import (
    DIFFUSION_KEYWORDS,
    FMAP_KEYWORDS,
    GRE_KEYWORDS,
    LOCALIZER_KEYWORDS,
    PERFUSION_KEYWORDS,
)
from .utils import (
    apply_rules,
    collect_tokens,
    flag_true,
    get_num_timepoints,
    stable_value,
    token_matches,
)

EPI_FAMILY_THRESHOLD = 5
GRE_FAMILY_THRESHOLD = 5
SE_TSE_FAMILY_THRESHOLD = 5
OTHER_FAMILY_THRESHOLD = 4

DIFFUSION_INTENT_THRESHOLD = 6
PERFUSION_INTENT_THRESHOLD = 6
FIELDMAP_INTENT_THRESHOLD = 6
LOCALIZER_INTENT_THRESHOLD = 5

OTHER_SPECIALIZED_KEYWORDS = {
    "radial", "spiral", "mra", "angio", "swi",
    "spectroscopy", "mrs", "mrsi", "svs", "csi",
}

SE_TSE_KEYWORDS = {"se", "tse", "fse", "fastspin", "fastspinecho", "turbo"}


def _sequence_value(series: SeriesFeatures, attr: str) -> str:
    sequence = series.sequence
    feature = getattr(sequence, attr, None) if sequence else None
    if feature and feature.value:
        return str(feature.value).upper()
    return ""


def _stable_numeric(obj, attr: str) -> float | None:
    field = getattr(obj, attr, None) if obj else None
    return stable_value(field) if field else None


def _echo_time_ms(series: SeriesFeatures) -> float | None:
    return _stable_numeric(series.temporal, "echo_time")


def _echo_spacing_ms(series: SeriesFeatures) -> float | None:
    return _stable_numeric(series.encoding, "echo_spacing")


def _echo_train_length(series: SeriesFeatures) -> float | None:
    return _stable_numeric(series.temporal, "echo_train_length")


def _mr_acquisition_type(series: SeriesFeatures) -> str | None:
    sequence = series.sequence
    if sequence and sequence.mr_acquisition_type and sequence.mr_acquisition_type.value:
        return str(sequence.mr_acquisition_type.value).upper()
    return None


def _has_diffusion_cues(series: SeriesFeatures, tokens: set[str]) -> bool:
    diffusion = series.diffusion
    return bool(
        (diffusion and diffusion.b_values)
        or (diffusion and diffusion.num_diffusion_directions)
        or flag_true(series.image_type, "has_diffusion")
        or token_matches(tokens, DIFFUSION_KEYWORDS)
    )


def _has_perfusion_cues(series: SeriesFeatures, tokens: set[str]) -> bool:
    perfusion = series.perfusion
    return bool(
        flag_true(series.image_type, "has_perfusion")
        or (perfusion and perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value)
        or (perfusion and perfusion.perfusion_series_type and perfusion.perfusion_series_type.value)
        or token_matches(tokens, PERFUSION_KEYWORDS)
    )


def score_epi_family(series: SeriesFeatures, tokens: set[str]) -> float:
    sequence_value = _sequence_value(series, "scanning_sequence")
    echo_spacing = _echo_spacing_ms(series)
    echo_train_length = _echo_train_length(series)
    echo_time = _echo_time_ms(series)
    n_tp = get_num_timepoints(series)

    return apply_rules(
        ("EP" in sequence_value, 5),
        (echo_spacing is not None and echo_spacing < 1.0, 3),
        (echo_train_length is not None and echo_train_length > 10, 2),
        (n_tp is not None and n_tp > 10, 1),
        (echo_time is not None and echo_time < 5.0, -2),
        (_mr_acquisition_type(series) == "3D", -2),
    )


def score_gre_family(series: SeriesFeatures, tokens: set[str]) -> float:
    sequence_value = _sequence_value(series, "scanning_sequence")
    sequence_variant = _sequence_value(series, "sequence_variant")
    echo_time = _echo_time_ms(series)
    echo_train_length = _echo_train_length(series)

    return apply_rules(
        ("GR" in sequence_value, 5),
        ("SP" in sequence_variant or "SS" in sequence_variant, 2),
        (echo_time is not None and echo_time < 10.0, 3),
        (echo_train_length is None or echo_train_length <= 1, 2),
        (_mr_acquisition_type(series) == "3D", 2),
        (token_matches(tokens, GRE_KEYWORDS), 1),
        (_has_diffusion_cues(series, tokens) or _has_perfusion_cues(series, tokens), -3),
    )


def score_se_tse_family(series: SeriesFeatures, tokens: set[str]) -> float:
    sequence_value = _sequence_value(series, "scanning_sequence")
    echo_spacing = _echo_spacing_ms(series)
    echo_train_length = _echo_train_length(series)
    echo_time = _echo_time_ms(series)

    return apply_rules(
        (
            any(tag in sequence_value for tag in ("SE", "TSE", "FSE"))
            or token_matches(tokens, SE_TSE_KEYWORDS),
            5,
        ),
        (echo_train_length is not None and echo_train_length >= 2, 3),
        (echo_time is not None and echo_time > 50.0, 2),
        (_mr_acquisition_type(series) == "2D", 1),
        (echo_spacing is not None and echo_spacing < 1.0, -2),
    )


def score_other_family(series: SeriesFeatures, tokens: set[str], named_scores: dict[str, float] | None = None) -> float:
    if named_scores is None:
        named_scores = {
            "epi": score_epi_family(series, tokens),
            "gre": score_gre_family(series, tokens),
            "se_tse": score_se_tse_family(series, tokens),
        }

    strongest_named = max(named_scores.values()) if named_scores else 0.0
    return apply_rules(
        (strongest_named < min(EPI_FAMILY_THRESHOLD, GRE_FAMILY_THRESHOLD, SE_TSE_FAMILY_THRESHOLD), 4),
        (token_matches(tokens, OTHER_SPECIALIZED_KEYWORDS), 3),
        (flag_true(series.image_type, "has_angio"), 3),
    )


def score_diffusion_intent(series: SeriesFeatures, tokens: set[str]) -> float:
    diffusion = series.diffusion

    return apply_rules(
        (diffusion and diffusion.b_values, 6),
        (diffusion and diffusion.num_diffusion_directions, 6),
        (diffusion and diffusion.num_diffusion_volumes and diffusion.num_diffusion_volumes >= 5, 2),
        (flag_true(series.image_type, "has_diffusion"), 4),
        (token_matches(tokens, DIFFUSION_KEYWORDS), 3),
    )


def score_perfusion_intent(series: SeriesFeatures, tokens: set[str]) -> float:
    perfusion = series.perfusion
    n_tp = perfusion.num_timepoints if perfusion else get_num_timepoints(series)
    has_contrast = bool(
        (perfusion and perfusion.contrast_agent and perfusion.contrast_agent.value)
        or (series.contrast and series.contrast.contrast_agent and series.contrast.contrast_agent.value)
    )

    return apply_rules(
        (perfusion and perfusion.perfusion_labeling_type and perfusion.perfusion_labeling_type.value, 6),
        (perfusion and perfusion.perfusion_series_type and perfusion.perfusion_series_type.value, 4),
        (flag_true(series.image_type, "has_perfusion"), 4),
        (token_matches(tokens, PERFUSION_KEYWORDS), 3),
        (has_contrast and n_tp is not None and n_tp > 10, 2),
    )


def score_fieldmap_intent(series: SeriesFeatures, tokens: set[str]) -> float:
    image_type = series.image_type
    num_echoes = series.multi_echo.num_echoes if series.multi_echo else None
    n_tp = get_num_timepoints(series)
    has_mag_or_phase = flag_true(image_type, "is_magnitude") or flag_true(image_type, "is_phase")
    has_pe_dir = bool(
        series.encoding
        and series.encoding.phase_encoding_direction
        and series.encoding.phase_encoding_direction.value
    )

    return apply_rules(
        (token_matches(tokens, FMAP_KEYWORDS), 4),
        (num_echoes is not None and num_echoes > 1, 3),
        (has_mag_or_phase and n_tp is not None and n_tp <= 2, 2),
        (n_tp is not None and 2 <= n_tp <= 6, 2),
        (has_pe_dir, 1),
    )


def score_localizer_intent(series: SeriesFeatures, tokens: set[str]) -> float:
    image_type = series.image_type
    num_slices = series.geometry.num_slices if series.geometry else None
    n_tp = get_num_timepoints(series)
    thickness = (
        series.spatial.slice_thickness.value
        if series.spatial and series.spatial.slice_thickness
        else None
    )

    return apply_rules(
        (token_matches(tokens, LOCALIZER_KEYWORDS), 5),
        (flag_true(image_type, "is_localizer"), 6),
        (thickness is not None and thickness >= 8.0, 2),
        (num_slices is not None and num_slices <= 5, 2),
        (n_tp is not None and n_tp <= 3, 1),
    )


def get_acquisition_family_scores(series: SeriesFeatures, tokens: set[str]) -> dict[str, float]:
    family_scores = {
        "epi": score_epi_family(series, tokens),
        "gre": score_gre_family(series, tokens),
        "se_tse": score_se_tse_family(series, tokens),
    }
    family_scores["other"] = score_other_family(series, tokens, family_scores)
    return family_scores


def get_acquisition_intent_scores(series: SeriesFeatures, tokens: set[str]) -> dict[str, float]:
    return {
        "diffusion": score_diffusion_intent(series, tokens),
        "perfusion": score_perfusion_intent(series, tokens),
        "fieldmap": score_fieldmap_intent(series, tokens),
        "localizer": score_localizer_intent(series, tokens),
    }


def is_epi(series: SeriesFeatures, tokens: set[str] | None = None) -> bool:
    tokens = collect_tokens(series) if tokens is None else tokens
    return score_epi_family(series, tokens) >= EPI_FAMILY_THRESHOLD


def is_fieldmap_epi(series: SeriesFeatures, tokens: set[str] | None = None) -> bool:
    tokens = collect_tokens(series) if tokens is None else tokens
    family_scores = get_acquisition_family_scores(series, tokens)
    intent_scores = get_acquisition_intent_scores(series, tokens)
    return (
        family_scores["epi"] >= EPI_FAMILY_THRESHOLD
        and intent_scores["fieldmap"] >= FIELDMAP_INTENT_THRESHOLD
        and intent_scores["diffusion"] < DIFFUSION_INTENT_THRESHOLD
    )