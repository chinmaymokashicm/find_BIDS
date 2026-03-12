from src.find_bids.models.extract.series import initialize_features_db, SeriesFeatures
from src.find_bids.models.extract.dataset import Dataset
from src.find_bids.models.annotate.core import initialize_annotations_metrics_db, AllSessionsAnnotation

# from pathlib import Path
from upath import UPath
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rich.progress import track

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
db_path = UPath("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/features.db")
conn = initialize_features_db(db_path)
futures = {}
with ThreadPoolExecutor() as executor:
    for dataset_name, paths in dataset_info.items():
        if paths["subject_level"]:
            future = executor.submit(
                Dataset.from_dir_with_subject_level,
                dir_root=paths["dicom_root"],
                features_root=paths["features_root"],
                dtype="DICOM",
                session_subdir_path=paths["session_subdir_path"],
                series_subdir_path=paths["series_subdir_path"]
            )
        else:
            future = executor.submit(
                Dataset.from_dir_without_subject_level,
                dir_root=paths["dicom_root"],
                features_root=paths["features_root"],
                dtype="DICOM",
                session_subdir_path=paths["session_subdir_path"],
                series_subdir_path=paths["series_subdir_path"]
            )
        futures[future] = dataset_name

for future in as_completed(futures):
    dataset_name = futures[future]
    dataset: Dataset = future.result()
    dataset.generate_bids_ids(replace_existing=True)
    dataset.to_json()
    dataset.generate_features(skip_unavailable=True, conn=conn)
    dataset.generate_bids_ids(replace_existing=True)
    dataset.to_json()