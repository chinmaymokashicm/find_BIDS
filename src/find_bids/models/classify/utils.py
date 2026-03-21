import numpy as np
import pandas as pd

TIER1_FEATURES = [
    "num_instances",
    "num_unique_slices",
    "num_volumes",
    "manufacturer",
    "model",
    "field_strength",

    # geometry / spatial
    "rows",
    "columns",
    "num_slices",
    "voxel_size",
    "matrix_size",
    # "geometry_hash",  # intentionally excluded (acts like a series ID)

    # temporal
    "repetition_time",
    "echo_time",
    "inversion_time",
    "flip_angle",
    # "num_timepoints", # intentionally excluded - used in perfusion
    "temporal_variation",
    "tr_bucket",
    "te_bucket",
    "is_3D",
    "is_isotropic",
    "echo_train_length",

    # diffusion
    "has_diffusion",
    "b_values",
    "num_b0",
    "num_diffusion_volumes",
    "num_diffusion_directions",

    # perfusion
    "perfusion_labeling_type",
    "perfusion_series_type",
    "bolus_arrival_time",
    "num_timepoints",
    "temporal_spacing",

    # sequence
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",

    # multi-echo
    "num_echoes",
    "echo_times",
    "echo_numbers",

    # spatial (more)
    "slice_thickness",
    "spacing_between_slices",
    "pixel_spacing",
    "image_orientation",

    # encoding
    "phase_encoding_direction",
    "phase_encoding_axis",
    "phase_encoding_polarity",
    "echo_spacing",
    "parallel_reduction_factor_in_plane",
    "parallel_reduction_factor_out_of_plane",
    "multiband_factor",

    # contrast
    "contrast_agent",
    "injection_time",
    # "signal_intensity_shift",  # currently None in implementation

    # image type flags (critical for is_derived and derived buckets)
    "is_original",
    "is_primary",
    "is_localizer",
    "is_mpr",
    "is_mip",
    "is_projection",
    "is_reformatted",
    "has_angio",
    "has_diffusion_token",
    "has_perfusion_token",
    "is_adc",
    "is_fa",
    "is_trace",
    "is_cbf",
    "is_cbv",
    "is_magnitude",
    "is_phase",
    "is_real",
    "is_imaginary",

    # text fields (raw text; you’ll vectorize separately)
    "series_description",
    "protocol_name",
    "sequence_name",

    # acquisition / provenance
    # "acquisition_time",
    # "series_time",
    # "acquisition_order",
    # "source_image_sequences",
]

NUMERIC_FEATURES = [
    "num_instances",
    "num_unique_slices",
    "num_volumes",
    "rows",
    "columns",
    "repetition_time",
    "echo_time",
    "inversion_time",
    "flip_angle",
    "echo_train_length",
    "slice_thickness",
    "spacing_between_slices",
    "echo_spacing",
    "num_echoes",
    "num_diffusion_volumes",
    "num_diffusion_directions",
    "num_b0",
    "temporal_variation",
    "temporal_spacing",
    "num_timepoints",
    "multiband_factor",
]

BOOLEAN_FEATURES = [
    "is_3D",
    "is_isotropic",
    "has_diffusion",
    "is_original",
    "is_primary",
    "is_localizer",
    "is_mpr",
    "is_mip",
    "is_projection",
    "is_reformatted",
    "has_angio",
    "has_diffusion_token",
    "has_perfusion_token",
    "is_adc",
    "is_fa",
    "is_trace",
    "is_cbf",
    "is_cbv",
    "is_magnitude",
    "is_phase",
    "is_real",
    "is_imaginary",
]

CATEGORICAL_FEATURES = [
    "manufacturer",
    "model",
    "field_strength",
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",
    "phase_encoding_direction",
    "phase_encoding_axis",
    "phase_encoding_polarity",
    "perfusion_labeling_type",
    "perfusion_series_type",
    "contrast_agent",
    "tr_bucket",
    "te_bucket",
]

TEXT_FEATURES = [
    "series_description",
    "protocol_name",
    "sequence_name",
]

def safe_divide(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Perform element-wise division of a by b, returning NaN where b is zero or where either a or b is NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = np.nan
    return result