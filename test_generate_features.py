from src.find_bids.models.extract.series import SeriesFeatures
from src.find_bids.models.extract.dataset import Dataset

from pathlib import Path

dicom_root: Path = Path("/rsrch5/home/csi/Quarles_Lab/PROACTIVE_De-identidied")

features_root: Path = Path("/rsrch5/home/csi/Quarles_Lab/find_BIDS/features/PROACTIVE")
dataset: Dataset = Dataset.from_dir_with_session_level(dicom_root, features_root=features_root, dtype="DICOM", series_subdir_path="DICOM/")

dataset.generate_bids_ids(replace_existing=True)
dataset.export()

dataset.generate_features()