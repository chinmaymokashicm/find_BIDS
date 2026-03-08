from ..extract.series import SeriesFeatures

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

def is_epi(series: SeriesFeatures) -> bool:
    """
    Vendor-robust EPI detection using standard DICOM tags
    and physics-informed heuristics.
    """

    score = 0

    seq = series.sequence
    temp = series.temporal
    enc = series.encoding

    # ---------------------------
    # Strongest signal: ScanningSequence
    # ---------------------------
    if seq and seq.scanning_sequence and seq.scanning_sequence.value:
        val = str(seq.scanning_sequence.value).upper()
        if "EP" in val:          # DICOM standard for Echo Planar
            score += 5

    # ---------------------------
    # ScanOptions may contain "EPI"
    # ---------------------------
    if seq and seq.scan_options and seq.scan_options.value:
        val = str(seq.scan_options.value).upper()
        if "EPI" in val:
            score += 4

    # ---------------------------
    # Echo Train Length (EPI hallmark)
    # ---------------------------
    if temp and temp.echo_train_length and temp.echo_train_length.value:
        if temp.echo_train_length.value >= 10:
            score += 3

    # ---------------------------
    # Acquisition type (2D typical for EPI)
    # ---------------------------
    if seq and seq.mr_acquisition_type and seq.mr_acquisition_type.value:
        if str(seq.mr_acquisition_type.value).upper() == "2D":
            score += 1

    # ---------------------------
    # Temporal behavior
    # ---------------------------
    n_tp = get_num_timepoints(series)
    if n_tp and n_tp > 10:
        score += 2  # time-series EPI (BOLD, DSC)

    # ---------------------------
    # Phase encoding direction (common in EPI)
    # ---------------------------
    if enc and enc.phase_encoding_direction and enc.phase_encoding_direction.value:
        score += 1

    # ---------------------------
    # Geometry sanity check
    # ---------------------------
    if series.spatial and series.spatial.slice_thickness and series.spatial.slice_thickness.value:
        thickness = series.spatial.slice_thickness.value
        if 1.0 <= thickness <= 5.0:  # EPI typically 2-4mm, but allow wider range
            score += 1

    # ---------------------------
    # Guard against obvious non-EPI
    # ---------------------------
    if seq and seq.scanning_sequence and seq.scanning_sequence.value:
        val = str(seq.scanning_sequence.value).upper()
        if "SE" in val and "EP" not in val:
            score -= 3  # classic spin echo
        if "GR" in val and "EP" not in val:
            score -= 2  # gradient echo non-EPI

    return score >= 5