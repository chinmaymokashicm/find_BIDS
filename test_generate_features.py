from src.find_bids.models.extract.series import initialize_db, SeriesFeatures
from src.find_bids.models.extract.dataset import Dataset

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rich.progress import track

# Generate features for multiple datasets and merge into a single table
dataset_info: dict[str, dict] = {
    "PROACTIVE": {
        "dicom_root": Path("/rsrch5/home/csi/Quarles_Lab/PROACTIVE_De-identidied"),
        "features_root": Path("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/PROACTIVE"),
        "subject_level": True,
        "session_subdir_path": "",
        "series_subdir_path": "DICOM/"
    },
    "QIAC": {
        "dicom_root": Path("/rsrch5/home/csi/Quarles_Lab/Bajaj_Projects/Melanoma_Data_QIAC/Raw_MRI"),
        "features_root": Path("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/QIAC"),
        "subject_level": False,
        "session_subdir_path": "SCANS/",
        "series_subdir_path": "DICOM/"
    },
    # Add more datasets as needed
}
db_path = Path("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/features.db")
conn = initialize_db(db_path)

datasets = []
for dataset_name, paths in dataset_info.items():
    with ThreadPoolExecutor() as executor:
        dir_root: Path = paths["dicom_root"]
        features_root: Path = paths["features_root"]
        subject_level: bool = paths["subject_level"]
        if subject_level:
            future = executor.submit(
                Dataset.from_dir_with_subject_level,
                dir_root=dir_root,
                features_root=features_root,
                dtype="DICOM",
                session_subdir_path=paths["session_subdir_path"],
                series_subdir_path=paths["series_subdir_path"]
            )
            print(f"Submitted task for {dataset_name} dataset with subject-level structure: {dir_root}")
        else:
            future = executor.submit(
                Dataset.from_dir_without_subject_level,
                dir_root=dir_root,
                features_root=features_root,
                dtype="DICOM",
                session_subdir_path=paths["session_subdir_path"],
                series_subdir_path=paths["series_subdir_path"]
            )
            print(f"Submitted task for {dataset_name} dataset with session-level structure: {dir_root}")
        dataset = future.result()
    dataset.generate_bids_ids(replace_existing=True)
    dataset.to_json()
    dataset.generate_features(conn)
    # dataset.export_all_features_to_table()
    # datasets.append(dataset)
    
# Merge features tables from all datasets into a single table
# merged_table_save_path = Path("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/all_features.csv")
# datasets[0].merge_features_tables(datasets[1:], save_path=merged_table_save_path)

conn.close()