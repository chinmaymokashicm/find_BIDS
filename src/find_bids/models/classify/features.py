from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset
from ..infer.core import DatasetsInference

from upath import UPath
from typing import Optional

import pandas as pd
from rich.progress import track

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

def get_features_from_series(
    features_root: UPath,
    load_existing: bool = True,
    save: bool = True,
    sample_subjects: Optional[int] = None,
    rebuild_old_root: Optional[str | UPath] = None,
    rebuild_new_root: Optional[str | UPath] = None,
    datasets: Dataset | list[Dataset] | None = None
    ) -> pd.DataFrame:
    index_cols = ["dataset", "subject", "session", "series"]
    if load_existing:
        existing_csv_path = features_root / "all_series_features.csv"
        if existing_csv_path.exists():
            print(f"Loading existing features from {existing_csv_path}")
            with existing_csv_path.open("r") as f:
                df = pd.read_csv(f, index_col=index_cols)
            return df
        else:
            print(f"No existing features found at {existing_csv_path}, regenerating...")
    rows = []
    
    if isinstance(datasets, Dataset):
        datasets = [datasets]
    
    if not all(isinstance(ds, Dataset) for ds in datasets or []):
        raise ValueError("All items in datasets must be instances of Dataset.")
    
    for dataset in datasets or []:
        if rebuild_old_root and rebuild_new_root:
            dataset.rebuild_paths_with_new_root(old_root=rebuild_old_root, new_root=rebuild_new_root)
        dataset_features: dict[str, dict[str, dict[str, SeriesFeatures]]] = dataset.generate_features(
            skip_unavailable=True,
            sample_subjects=sample_subjects
        )
        for subject_id, sessions in track(dataset_features.items(), description=f"Processing dataset {dataset.dir_root.name}", total=len(dataset_features)):
            for session_id, series_dict in sessions.items():
                for series_id, series_features in series_dict.items():
                    flattened_features: dict = series_features.flatten()
                    row = {
                        "dataset": dataset.dir_root.name,
                        "subject": subject_id,
                        "session": session_id,
                        "series": series_id,
                        **{k: flattened_features.get(k) for k in TIER1_FEATURES}
                    }
                    rows.append(row)

    df = pd.DataFrame(rows).set_index(index_cols)
    if save:
        with (features_root / "all_series_features.csv").open("w") as f:
            df.to_csv(f, index=True)
        # Also save the table to a local directory for easy access
        local_save_path = UPath("all_series_features.csv")
        if not local_save_path.parent.exists():
            local_save_path.parent.mkdir(parents=True, exist_ok=True)
        with local_save_path.open("w") as f:
            df.to_csv(f, index=True)
    return df