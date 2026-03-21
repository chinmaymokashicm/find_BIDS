from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset
from ..infer.core import DatasetsInference
from .utils import TIER1_FEATURES

from upath import UPath
from typing import Optional
import sqlite3

import pandas as pd
from rich.progress import track

def get_features_from_series(
    features_root: UPath,
    load_existing: bool = True,
    save: bool = True,
    sample_subjects: Optional[int] = None,
    rebuild_old_root: Optional[str | UPath] = None,
    rebuild_new_root: Optional[str | UPath] = None,
    datasets: Dataset | list[Dataset] | None = None,
    conn: Optional[sqlite3.Connection] = None
    ) -> pd.DataFrame:
    index_cols = ["dataset", "subject", "session", "series"]
    csv_path = features_root / "all_series_features.csv"
    if load_existing:
        if csv_path.exists():
            print(f"Loading existing features from {csv_path}")
            with csv_path.open("r") as f:
                df = pd.read_csv(f, index_col=index_cols)
            return df
        else:
            print(f"No existing features found at {csv_path}, regenerating...")
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
            sample_subjects=sample_subjects,
            conn=conn
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
        with csv_path.open("w") as f:
            df.to_csv(f, index=True)
        # Also save the table to a local directory for easy access
        local_save_path = UPath("all_series_features.csv")
        if not local_save_path.parent.exists():
            local_save_path.parent.mkdir(parents=True, exist_ok=True)
        with local_save_path.open("w") as f:
            df.to_csv(f, index=True)
    return df