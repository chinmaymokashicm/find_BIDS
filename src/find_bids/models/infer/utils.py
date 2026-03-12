from ..extract.series import SeriesFeatures

import math

def collect_tokens(series: SeriesFeatures) -> set[str]:
    """Union of all text tokens from SeriesDescription / ProtocolName / SequenceName."""
    tokens: set[str] = set()
    if series.text:
        for feat in (
            series.text.series_description,
            series.text.protocol_name,
            series.text.sequence_name,
        ):
            if feat and feat.tokens:
                # tokens are already lowercased by extract_dicom_tokens
                tokens |= set(feat.tokens.keys())
    return tokens

def get_num_timepoints(series: SeriesFeatures) -> int | None:
    return series.temporal.num_timepoints if series.temporal else None

# def is_epi(series: SeriesFeatures) -> bool:
#     """
#     Vendor-robust EPI detection using standard DICOM tags
#     and physics-informed heuristics.
#     """

#     score = 0

#     seq = series.sequence
#     temp = series.temporal
#     enc = series.encoding

#     # ---------------------------
#     # Strongest signal: ScanningSequence
#     # ---------------------------
#     if seq and seq.scanning_sequence and seq.scanning_sequence.value:
#         val = str(seq.scanning_sequence.value).upper()
#         if "EP" in val:          # DICOM standard for Echo Planar
#             score += 5

#     # ---------------------------
#     # ScanOptions may contain "EPI"
#     # ---------------------------
#     if seq and seq.scan_options and seq.scan_options.value:
#         val = str(seq.scan_options.value).upper()
#         if "EPI" in val:
#             score += 4

#     # ---------------------------
#     # Echo Train Length (EPI hallmark)
#     # ---------------------------
#     if temp and temp.echo_train_length and temp.echo_train_length.value:
#         if temp.echo_train_length.value >= 10:
#             score += 3

#     # ---------------------------
#     # Acquisition type (2D typical for EPI)
#     # ---------------------------
#     if seq and seq.mr_acquisition_type and seq.mr_acquisition_type.value:
#         if str(seq.mr_acquisition_type.value).upper() == "2D":
#             score += 1

#     # ---------------------------
#     # Temporal behavior
#     # ---------------------------
#     n_tp = get_num_timepoints(series)
#     if n_tp and n_tp > 10:
#         score += 2  # time-series EPI (BOLD, DSC)

#     # ---------------------------
#     # Phase encoding direction (common in EPI)
#     # ---------------------------
#     if enc and enc.phase_encoding_direction and enc.phase_encoding_direction.value:
#         score += 1

#     # ---------------------------
#     # Geometry sanity check
#     # ---------------------------
#     if series.spatial and series.spatial.slice_thickness and series.spatial.slice_thickness.value:
#         thickness = series.spatial.slice_thickness.value
#         if 1.0 <= thickness <= 5.0:  # EPI typically 2-4mm, but allow wider range
#             score += 1

#     # ---------------------------
#     # Guard against obvious non-EPI
#     # ---------------------------
#     if seq and seq.scanning_sequence and seq.scanning_sequence.value:
#         val = str(seq.scanning_sequence.value).upper()
#         if "SE" in val and "EP" not in val:
#             score -= 3  # classic spin echo
#         if "GR" in val and "EP" not in val:
#             score -= 2  # gradient echo non-EPI

#     return score >= 5

def is_epi(series: SeriesFeatures) -> bool:
    """
    Vendor-robust EPI detection using DICOM metadata
    and acquisition physics heuristics.
    """

    score = 0

    seq = series.sequence
    temp = series.temporal
    enc = series.encoding

    # ---------------------------
    # ScanningSequence (strongest)
    # ---------------------------
    if seq and seq.scanning_sequence and seq.scanning_sequence.value:
        val = str(seq.scanning_sequence.value).upper()

        if "EP" in val:
            score += 5
        elif val == "SE":
            score -= 3
        elif val == "GR":
            score -= 2
            
    if temp and temp.is_3D:
        score -= 4  # EPI is almost always 2D

    # ---------------------------
    # ScanOptions (weak signal)
    # ---------------------------
    if seq and seq.scan_options and seq.scan_options.value:
        val = str(seq.scan_options.value).upper()
        if "EPI" in val:
            score += 2

    # ---------------------------
    # Echo Train Length
    # ---------------------------
    if temp and temp.echo_train_length and temp.echo_train_length.value:
        if temp.echo_train_length.value >= 10:
            score += 3

    # ---------------------------
    # Acquisition type
    # ---------------------------
    if seq and seq.mr_acquisition_type and seq.mr_acquisition_type.value:
        if str(seq.mr_acquisition_type.value).upper() == "2D":
            score += 1

    # ---------------------------
    # Temporal structure
    # ---------------------------
    n_tp = get_num_timepoints(series)
    if n_tp and n_tp > 10:
        score += 2

    # ---------------------------
    # Phase encoding
    # ---------------------------
    if enc and enc.phase_encoding_direction and enc.phase_encoding_direction.value:
        score += 1

    # ---------------------------
    # Slice thickness sanity
    # ---------------------------
    if series.spatial and series.spatial.slice_thickness and series.spatial.slice_thickness.value:
        thickness = series.spatial.slice_thickness.value
        if 1.0 <= thickness <= 5.0:
            score += 1

    return score >= 5

def is_fieldmap_epi(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Detect EPI fieldmap acquisitions (e.g. topup / SE-EPI).
    """

    score = 0

    seq = series.sequence
    temporal = series.temporal

    # --------------------------------------------------
    # Must be EPI
    # --------------------------------------------------

    if is_epi(series):
        score += 4
    else:
        return False

    # --------------------------------------------------
    # Strong textual signals
    # --------------------------------------------------

    FMAP_EPI_KEYWORDS = {
        "fieldmap",
        "topup",
        "pepolar",
        "sefm",
        "spin_echo_fieldmap",
        "field_map"
    }

    if tokens & FMAP_EPI_KEYWORDS:
        score += 5

    # --------------------------------------------------
    # Temporal structure
    # --------------------------------------------------

    n_tp = temporal.num_timepoints if temporal else None

    if n_tp is not None:

        if n_tp <= 3:
            score += 3

        elif n_tp <= 10:
            score += 1

        elif n_tp > 50:
            score -= 4

    # --------------------------------------------------
    # Spin echo EPI (common for fieldmaps)
    # --------------------------------------------------

    if seq and seq.scanning_sequence and seq.scanning_sequence.value:

        val = str(seq.scanning_sequence.value).upper()

        if "SE" in val and "EP" in val:
            score += 2

    # --------------------------------------------------
    # Negative evidence
    # --------------------------------------------------

    if tokens & {"sbref"}:
        score -= 4

    if tokens & {"bold", "fmri", "task", "rest"}:
        score -= 5

    return score >= 5

def softmax(scores: dict[str, float], temperature: float = 1.5) -> dict[str, float]:
    """
    Convert raw scores to probabilities using temperature-scaled softmax.
    """

    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {
        k: math.exp((v - max_score) / temperature)
        for k, v in scores.items()
    }
    total = sum(exp_scores.values())
    if total == 0:
        # fallback uniform distribution
        n = len(scores)
        return {k: 1.0 / n for k in scores}

    return {k: v / total for k, v in exp_scores.items()}

def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))