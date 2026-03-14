from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset
from ..infer.core import DatasetsInference

from upath import UPath
from typing import Optional

import pandas as pd

TIER1_FEATURES = [
    # # core / identifiers (keep for joins, but you can drop later before modeling)
    # "series_uid",
    # "study_uid",
    # "modality",
    # "series_number",
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
    "acquisition_time",
    "series_time",
    "acquisition_order",
    "source_image_sequences",
]


def get_features_from_series(features_root: UPath, save: bool = True, sample_subjects: Optional[int] = 5) -> pd.DataFrame:
    rows = []
    for dataset_name in features_root.iterdir():
        if not dataset_name.is_dir():
            continue
        dataset_json_path: UPath = dataset_name / "dataset.json"
        if not dataset_json_path.exists():
            print(f"Warning: No dataset.json found for {dataset_name}, skipping.")
            continue
        dataset = Dataset.from_json(dataset_json_path)
        all_features: dict[str, dict[str, dict[str, SeriesFeatures]]] = dataset.generate_features(
            skip_unavailable=True,
            sample_subjects=sample_subjects
            )
        print(all_features)
        for subject_id, sessions in all_features.items():
            for session_id, series_dict in sessions.items():
                for series_id, features in series_dict.items():
                    flattened_features: dict = features.flatten()
                    row = {
                        "dataset": dataset_name.name,
                        "subject": subject_id,
                        "session": session_id,
                        "series": series_id,
                        **{k: flattened_features.get(k) for k in TIER1_FEATURES}
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        with (features_root / "all_series_features.csv").open("w") as f:
            df.to_csv(f, index=False)
    return df