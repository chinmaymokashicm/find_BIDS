from src.find_bids.models.extract.series import initialize_features_db, SeriesFeatures
from src.find_bids.models.extract.dataset import Dataset
from src.find_bids.models.annotate.core import initialize_annotations_metrics_db, AllSessionsAnnotation

# from pathlib import Path
from upath import UPath
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rich.progress import track

DB_PATH = UPath("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/features.db")

SAMPLE_SIZE: Optional[int] = None  # Set to an integer to limit the number of subjects processed for testing
if sys.argv and len(sys.argv) > 1 and sys.argv[1] == "test":
    print("Running in test mode: limiting to 2.")
    SAMPLE_SIZE = 2

def process_dataset(dataset_name: str, paths: dict) -> str:
    if paths["subject_level"]:
        dataset = Dataset.from_dir_with_subject_level(
            dir_root=paths["dicom_root"],
            features_root=paths["features_root"],
            dtype="DICOM",
            session_subdir_path=paths["session_subdir_path"],
            series_subdir_path=paths["series_subdir_path"],
            n_subjects=SAMPLE_SIZE
        )
    else:
        dataset = Dataset.from_dir_without_subject_level(
            dir_root=paths["dicom_root"],
            features_root=paths["features_root"],
            dtype="DICOM",
            session_subdir_path=paths["session_subdir_path"],
            series_subdir_path=paths["series_subdir_path"],
            n_sessions=SAMPLE_SIZE # Use n_sessions to limit total number of series
        )

    dataset.generate_bids_ids(replace_existing=True)
    dataset.to_json()

    # Use a thread-local SQLite connection to avoid cross-thread connection usage.
    conn = initialize_features_db(DB_PATH)
    try:
        dataset.generate_features(skip_unavailable=True, conn=conn)
    finally:
        conn.close()
    # dataset.generate_features(skip_unavailable=True)

    # dataset.generate_bids_ids(replace_existing=True)
    # dataset.to_json()
    return dataset_name

# Generate features for multiple datasets and merge into a single table
dataset_info: dict[str, dict] = {
    "PROACTIVE": {
        "dicom_root": UPath("/rsrch5/home/csi/Quarles_Lab/PROACTIVE_De-identidied"),
        "features_root": UPath("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/PROACTIVE"),
        "subject_level": True,
        "session_subdir_path": "",
        "series_subdir_path": "DICOM/"
    },
    "QIAC": {
        "dicom_root": UPath("/rsrch5/home/csi/Quarles_Lab/Bajaj_Projects/Melanoma_Data_QIAC/Raw_MRI"),
        "features_root": UPath("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/QIAC"),
        "subject_level": False,
        "session_subdir_path": "SCANS/",
        "series_subdir_path": "DICOM/"
    },
    # Add more datasets as needed
}

# Initialize DB schema once before parallel workers start.
# conn = initialize_features_db(db_path)
# conn.close()

futures = {}
with ThreadPoolExecutor() as executor:
    for dataset_name, paths in dataset_info.items():
        # future = executor.submit(process_dataset, dataset_name, paths, db_path)
        future = executor.submit(process_dataset, dataset_name, paths)
        futures[future] = dataset_name

for future in as_completed(futures):
    dataset_name = futures[future]
    future.result()