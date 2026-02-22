"""
Extract and normalize key features from a DICOM or Nifti dataset that are relevant for BIDS conversion.

The types of raw datasets we expect to encounter include:
- Subject > Session > Series
- Session > Series

Series can either be:
- DICOM series: a folder containing multiple DICOM files, each representing one slice in the series. The series-level features are extracted by aggregating information across all DICOM files in the series.
- Nifti series: a single Nifti file containing the entire 3D or 4D volume. The series-level features are extracted from the Nifti header and filename.
"""
from .series import SeriesFeatures

import os, re, json
from pathlib import Path, Path
from typing import Optional, Iterable, Iterator, Self, Any, Literal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sqlite3

from pydantic import BaseModel, field_validator, ConfigDict
import pydicom as dicom
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, track
import numpy as np
import pandas as pd

class Series(BaseModel):
    series_id: str
    path: Path

class Session(BaseModel):
    session_id: str
    path: Path
    series: Optional[list[Series]] = None
    bids_session_id: Optional[str] = None

class Subject(BaseModel):
    subject_id: str
    sessions: Optional[list[Session]] = None
    bids_participant_id: Optional[str] = None

class Dataset(BaseModel):
    dir_root: Path
    features_root: Path
    dtype: Literal["DICOM", "Nifti"]
    subjects: Optional[list[Subject]] = None
    
    @property
    def sorted_subjects(self) -> list[Subject]:
        if self.subjects is None:
            return []
        return sorted(self.subjects, key=lambda s: s.subject_id)
    
    @property
    def csv_export_path(self) -> Path:
        return self.features_root / "features.csv"
    
    @staticmethod
    def validate_dicom_dir(dir_path: Path) -> bool:
        """
        Validate that the given directory contains DICOM files by checking for the presence of files with a .dcm extension or by attempting to read a sample file with pydicom.
        """
        dicom_files = list(dir_path.glob("*.dcm"))
        if dicom_files:
            return True
        # If no .dcm files are found, try to read a sample file to check if it's a DICOM directory
        for file in dir_path.iterdir():
            if file.is_file():
                try:
                    dicom.dcmread(file, stop_before_pixels=True)
                    return True
                except Exception:
                    continue
        return False
    
    @staticmethod
    def validate_nifti_dir(dir_path: Path) -> bool:
        """
        Validate that the given directory contains Nifti files by checking for the presence of files with .nii or .nii.gz extensions.
        """
        nifti_files = list(dir_path.glob("*.nii")) + list(dir_path.glob("*.nii.gz"))
        return len(nifti_files) > 0
    
    @classmethod
    def from_dir_with_subject_level(
        cls,
        dir_root: str | Path,
        features_root: str | Path,
        dtype: Literal["DICOM", "Nifti"],
        session_subdir_path: str = "",
        series_subdir_path: str = "",
        n_subjects: Optional[int] = None,
    ) -> Self:
        """
        Recursively traverse the directory structure starting from dir_root, and construct the nested Subject > Session > Series hierarchy based on the provided subdirectory paths.

        The directory structure is assumed to be:
        dir_root -|
            subject_id_1 -|
                session_id_1 -|
                    series_id_1 -| (DICOM or Nifti files)
                    series_id_2 -| (DICOM or Nifti files)
                session_id_2 -|
                    series_id_3 -| (DICOM or Nifti files)
            subject_id_2 -|
                session_id_3 -|
                    series_id_4 -| (DICOM or Nifti files)
        
        Args:
            dir_root: The root directory of the dataset.
            dtype: The type of dataset ("DICOM" or "Nifti").
            session_subdir_path: The relative path from the subject directory to the session directory. If empty, assumes session directories are directly under subject directories.
            series_subdir_path: The relative path from the session directory to the series directory. If empty, assumes series directories are directly under session directories.
            n_subjects: If provided, limits the number of subjects to process for testing purposes.
            
        Returns:
            An instance of Dataset with the extracted hierarchy of subjects, sessions, and series.
        """
        dir_root = Path(dir_root)
        features_root = Path(features_root)
        if not features_root.exists():
            features_root.mkdir(parents=True, exist_ok=True)
        
        series_count: int = 0
        
        subject_dirs = [d for d in dir_root.iterdir() if d.is_dir()]
        if n_subjects is not None:
            subject_dirs = subject_dirs[:n_subjects]
        subjects: list[Subject] = []
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn(f"Processed {{task.fields[series_count]}} series")
        ) as progress:
            task_id = progress.add_task("Processing subjects", total=len(subject_dirs), series_count=0)

            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                session_dirs = [(d, (d / session_subdir_path)) for d in subject_dir.iterdir() if d.is_dir()]
                sessions: list[Session] = []
                for session_dir_without_path_pattern, session_dir in session_dirs:
                    session_id = session_dir_without_path_pattern.name
                    series_dirs = [(d, (d / series_subdir_path)) for d in session_dir.iterdir() if d.is_dir()]
                    series_list: list[Series] = []
                    for series_dir_without_path_pattern, series_dir in series_dirs:
                        series_id = series_dir_without_path_pattern.name
                        if dtype == "DICOM":
                            if not series_dir.is_dir() or not cls.validate_dicom_dir(series_dir):
                                continue
                        elif dtype == "Nifti":
                            if not cls.validate_nifti_dir(series_dir):
                                continue
                        series_list.append(Series(series_id=series_id, path=series_dir))
                        series_count += 1
                        progress.update(task_id, series_count=series_count)
                    sessions.append(Session(session_id=session_id, path=session_dir, series=series_list))
                subjects.append(Subject(subject_id=subject_id, sessions=sessions))
                progress.advance(task_id)
        
        return cls(dir_root=dir_root, features_root=features_root, dtype=dtype, subjects=subjects)
    
    @classmethod
    def from_dir_without_subject_level(
        cls,
        dir_root: str | Path,
        features_root: str | Path,
        dtype: Literal["DICOM", "Nifti"],
        session_subdir_path: str = "",
        series_subdir_path: str = "",
        n_sessions: Optional[int] = None,
    ) -> Self:
        """
        Similar to from_dir_with_session_level, but assumes there is no subject level in the directory structure. The hierarchy is Session > Series.
        Session ID is assumed to be XXXX-XXXX-XXXX-XXXX, and subject ID is derived from the session ID by taking the first 2 segments (e.g. XXXX-XXXX).

        The directory structure is assumed to be:
        dir_root -|
            session_id_1 -|
                series_id_1 -| (DICOM or Nifti files)
                series_id_2 -| (DICOM or Nifti files)
            session_id_2 -|
                series_id_3 -| (DICOM or Nifti files)
        
        Args:
            dir_root: The root directory of the dataset.
            dtype: The type of dataset ("DICOM" or "Nifti").
            session_subdir_path: The relative path from the root directory to the session directory. If empty, assumes session directories are directly under the root directory.
            series_subdir_path: The relative path from the session directory to the series directory. If empty, assumes series directories are directly under session directories.
            n_sessions: If provided, limits the number of sessions to process for testing purposes.
            
        Returns:
            An instance of Dataset with the extracted hierarchy of sessions and series.
        """
        dir_root = Path(dir_root)
        features_root = Path(features_root)
        if not features_root.exists():
            features_root.mkdir(parents=True, exist_ok=True)
        
        series_count: int = 0
        
        session_dirs = [(d, (d / session_subdir_path)) for d in dir_root.iterdir() if d.is_dir()]
        if n_sessions is not None:
            session_dirs = session_dirs[:n_sessions]
        subjects: list[Subject] = []
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn(f"Processed {{task.fields[series_count]}} series")
        ) as progress:
            task_id = progress.add_task("Processing sessions", total=len(session_dirs), series_count=0)

            for session_dir_without_path_pattern, session_dir in session_dirs:
                session_id = session_dir_without_path_pattern.name
                subject_id = "-".join(session_id.split("-")[:2])
                series_dirs = [(d, (d / series_subdir_path)) for d in session_dir.iterdir() if d.is_dir()]
                series_list: list[Series] = []
                for series_dir_without_path_pattern, series_dir in series_dirs:
                    series_id = series_dir_without_path_pattern.name
                    if dtype == "DICOM":
                        if not series_dir.is_dir() or not cls.validate_dicom_dir(series_dir):
                            continue
                    elif dtype == "Nifti":
                        if not cls.validate_nifti_dir(series_dir):
                            continue
                    series_list.append(Series(series_id=series_id, path=series_dir))
                    series_count += 1
                    progress.update(task_id, series_count=series_count)
                subjects.append(Subject(subject_id=subject_id, sessions=[Session(session_id=session_id, path=session_dir, series=series_list)]))
                progress.advance(task_id)
                
        return cls(dir_root=dir_root, features_root=features_root, dtype=dtype, subjects=subjects)
    
    def generate_bids_ids(self, replace_existing: bool = True) -> None:
        """
        Generate BIDS participant_id for each subject and session_id for each session in the dataset, based on their order in the sorted list of subjects and sessions. 
        If replace_existing is False, will only generate IDs for subjects/sessions that do not already have a bids_participant_id/bids_session_id.
        
        Args:
            replace_existing: If True, will replace existing BIDS IDs. If False, will only
                generate BIDS IDs for subjects/sessions that do not already have them.
        """
        if self.subjects is None:
            return
        subject_counter = 1
        for subject in self.sorted_subjects:
            if not replace_existing and subject.bids_participant_id is not None:
                continue
            subject.bids_participant_id = f"{subject_counter:04d}"
            subject_counter += 1
            session_counter = 1
            for session in subject.sessions or []:
                if not replace_existing and session.bids_session_id is not None:
                    continue
                session.bids_session_id = f"{session_counter:04d}"
                session_counter += 1
    
    def generate_features(self, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Generate features for all series in the dataset and save them to the features_root directory, maintaining the same hierarchy of subject/session/series.
        """
        if self.dtype == "DICOM":
            self._generate_dicom_features(conn)
        elif self.dtype == "Nifti":
            self._generate_nifti_features()
            
    def _generate_dicom_features(self, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Generate features for DICOM series by reading the DICOM files in each series directory and extracting relevant metadata and image statistics. Save the features to the features_root directory.
        """
        if self.subjects is None:
            return
        for subject in track(self.subjects):
            for session in subject.sessions or []:
                for series in session.series or []:
                    features_save_path = self.features_root / subject.subject_id / session.session_id / f"{series.series_id}.json"
                    if features_save_path.exists():
                        continue  # Skip if features already exist for this series
                    features = SeriesFeatures.from_dicom_series(series.path)
                    features_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(features_save_path, "w") as f:
                        json.dump(features.model_dump(), f, indent=4)
                    if conn is not None:
                        features.to_sqlite(conn, subject_id=subject.subject_id, session_id=session.session_id)
                        
    def _generate_nifti_features(self) -> None:
        """
        Generate features for Nifti series by reading the Nifti files in each series directory and extracting relevant metadata and image statistics. Save the features to the features_root directory.
        """
        if self.subjects is None:
            return
        # for subject in self.subjects or []:
        #     for session in subject.sessions or []:
        #         for series in session.series or []:
        #             features = SeriesFeatures.from_nifti_series(series.path)
        #             features_save_path = self.features_root / subject.subject_id / session.session_id / f"{series.series_id}.json"
        
        
    def to_json(self) -> dict:
        if self.subjects is None:
            return {}
        export_path: Path = self.features_root / "dataset.json"
        with open(export_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=4)
        return self.model_dump()
    
    def export_all_features_to_table(self) -> None:
        """
        Export all features for all series in the dataset to a single CSV file for easier analysis and visualization. The CSV file will have one row per series, with columns for subject_id, session_id, series_id, and all extracted features.
        """
        if self.subjects is None:
            return
        records: list[dict[str, Any]] = []
        for subject in self.subjects:
            for session in subject.sessions or []:
                for series in session.series or []:
                    features_path = self.features_root / subject.subject_id / session.session_id / f"{series.series_id}.json"
                    if not features_path.exists():
                        if self.dtype == "DICOM":
                            series_features = SeriesFeatures.from_dicom_series(series.path)
                        elif self.dtype == "Nifti":
                            series_features = SeriesFeatures.from_nifti_series(series.path)
                        else:
                            continue
                    else:
                        series_features = SeriesFeatures.from_json(features_path)
                    record: dict = series_features.flatten()
                    record = {
                        "subject_id": subject.subject_id,
                        "session_id": session.session_id,
                        "series_id": series.series_id,
                        **record
                    }
                    records.append(record)
        df = pd.DataFrame(records)
        df.to_csv(self.csv_export_path, index=False)
    
    def merge_features_tables(self, other_datasets: list[Self], save_path: str | Path) -> None:
        """
        Merge the features tables from multiple datasets into a single CSV file. This is useful for comparing features across different datasets or for combining datasets for larger analyses.
        
        Args:
            other_datasets: A list of other Dataset instances to merge with this dataset.
            save_path: The path to save the merged CSV file.
        """
        if any(not isinstance(ds, Dataset) for ds in other_datasets):
            raise ValueError("All items in other_datasets must be instances of Dataset")
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        # Add column for dataset identifier to each dataset's features table
        df = pd.read_csv(self.csv_export_path, low_memory=False)
        df["root_data"] = str(self.dir_root)
        df["features_data"] = str(self.features_root)
        for other_ds in other_datasets:
            other_df = pd.read_csv(other_ds.csv_export_path, low_memory=False)
            other_df["root_data"] = str(other_ds.dir_root)
            other_df["features_data"] = str(other_ds.features_root)
            df = pd.concat([df, other_df], ignore_index=True)
        df.to_csv(save_path, index=False)