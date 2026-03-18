"""
Inference pipeline for BIDS dataset structure inference. 
This module contains the core logic for inferring the BIDS structure of a dataset based on extracted features and metadata. 
It defines the main classes and functions for performing inference, including loading features, applying inference rules, and generating the inferred BIDS structure.
"""
from .schema import BIDSEntities, Datatype
from .utils import collect_tokens, softmax, logistic
from .rules import (
    LOCALIZER_KEYWORDS,
    DERIVED_MAP_KEYWORDS,
    SURGICAL_NAV_KEYWORDS,
    DERIVED_TIMESTAMP_PATTERN,
    DIFFUSION_KEYWORDS,
    PERFUSION_KEYWORDS,
    FUNC_KEYWORDS,
    FMAP_KEYWORDS,
    ANAT_KEYWORDS,
)
from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset
from .datatype import score_datatype
from .suffix import score_suffix

import os, json, re, math
from enum import Enum
from typing import Optional, Self
# from pathlib import Path
from upath import UPath
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
from pydantic import BaseModel, model_validator
from rich.progress import track

def score_derived(series: SeriesFeatures, tokens: set[str]) -> int:
    """
    Node-level heuristic score for whether a DICOM series is derived
    (quantitative maps, reformats, projections, etc.).
    Positive score → derived
    Negative score → raw acquisition
    """

    score = 0

    imagetype = series.image_type
    spatial = series.spatial
    temporal = series.temporal
    textmeta = series.text

    # ----------------------------
    # STRONG DERIVED SIGNALS
    # ----------------------------

    if tokens & DERIVED_MAP_KEYWORDS:
        score += 6

    if tokens & LOCALIZER_KEYWORDS:
        score += 6

    if tokens & SURGICAL_NAV_KEYWORDS:
        score += 6

    if imagetype:
        if imagetype.is_adc and imagetype.is_adc.value:
            score += 6
        if imagetype.is_fa and imagetype.is_fa.value:
            score += 6
        if imagetype.is_trace and imagetype.is_trace.value:
            score += 6
        if imagetype.is_cbf and imagetype.is_cbf.value:
            score += 6
        if imagetype.is_cbv and imagetype.is_cbv.value:
            score += 6

    # ----------------------------
    # MODERATE DERIVED SIGNALS
    # ----------------------------

    if imagetype:
        if imagetype.is_mip and imagetype.is_mip.value:
            score += 4
        if imagetype.is_projection and imagetype.is_projection.value:
            score += 4
        if imagetype.is_reformatted and imagetype.is_reformatted.value:
            score += 4

    if textmeta and textmeta.series_description and textmeta.series_description.text:
        desc = textmeta.series_description.text

        if DERIVED_TIMESTAMP_PATTERN.search(desc):
            score += 4

        if desc.lower().startswith("orig:"):
            score += 3

    # ----------------------------
    # WEAK STRUCTURAL SIGNALS
    # ----------------------------

    if temporal and temporal.num_timepoints == 1:
        score += 1

    if series.num_volumes == 1:
        score += 1

    thickness = spatial.slice_thickness.value if (spatial and spatial.slice_thickness) else None
    if thickness and thickness >= 8:
        score += 2

    num_slices = series.geometry.num_slices if series.geometry else None
    if num_slices and num_slices <= 5:
        score += 2

    # ----------------------------
    # RAW ACQUISITION SIGNALS
    # ----------------------------

    if imagetype:
        if imagetype.is_original and imagetype.is_original.value:
            score -= 5

        if imagetype.is_primary and imagetype.is_primary.value:
            score -= 3

        if imagetype.is_mpr and imagetype.is_mpr.value:
            score -= 2

    if temporal and temporal.num_timepoints and temporal.num_timepoints > 10:
        score -= 4

    if spatial and spatial.slice_thickness and spatial.slice_thickness.value:
        if spatial.slice_thickness.value < 2:
            score -= 2

    if series.geometry and series.geometry.num_slices:
        if series.geometry.num_slices >= 50:
            score -= 1

    if temporal and temporal.is_3D:
        score -= 1

    return score

def score_is_derived(
    series: SeriesFeatures,
    tokens: set[str],
    threshold: float = 0.7,
    uncertainty_band: float = 0.2,
) -> tuple[dict[str, float], bool | None, float]:
    """
    Convert derived score into probability and label.

    Returns:
        probs: {'derived': p1, 'raw': p0}
        label: True (derived), False (raw), None (uncertain)
        confidence: |p1 - p0|
    """

    derived_score = score_derived(series, tokens)

    p_derived = logistic(derived_score / 3)  # temperature scaling
    p_raw = 1 - p_derived

    confidence = abs(p_derived - p_raw)

    if p_derived > threshold:
        label = True
    elif p_derived < (1 - threshold):
        label = False
    else:
        label = None

    return {"derived": p_derived, "raw": p_raw}, label, confidence

class SeriesInference(BaseModel):
    """Data class representing the inferred BIDS labels for a single series, along with confidence scores."""
    subject_id: str
    session_id: str
    series_id: str
    series_description: Optional[str]
    inferred_datatype: Optional[str]
    datatype_confidence: Optional[float]
    inferred_suffix: Optional[str]
    suffix_confidence: Optional[float]
    is_derived: Optional[bool]
    derived_confidence: Optional[float]
    
    @property
    def label(self) -> str:
        datatype = self.inferred_datatype if self.inferred_datatype else "unknown"
        suffix = self.inferred_suffix if self.inferred_suffix else "unknown"
        # derived = "derived" if self.is_derived else ("raw" if self.is_derived == False else "unknown")
        return f"{datatype}_{suffix}"

    @property
    def min_confidence(self) -> Optional[float]:
        confidences = [c for c in [self.datatype_confidence, self.suffix_confidence, self.derived_confidence] if c is not None]
        if not confidences:
            return None
        return min(confidences)
    
class DatasetInference(BaseModel):
    """Data class representing the inferred BIDS labels for all series within a dataset."""
    dataset: str
    series_inferences: list[SeriesInference]
    
    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for si in self.series_inferences:
            record = {
                "dataset": self.dataset,
                "subject_id": si.subject_id,
                "session_id": si.session_id,
                "series_id": si.series_id,
                "series_description": si.series_description,
                "inferred_datatype": si.inferred_datatype,
                "datatype_confidence": si.datatype_confidence,
                "inferred_suffix": si.inferred_suffix,
                "suffix_confidence": si.suffix_confidence,
                "is_derived": si.is_derived,
                "derived_confidence": si.derived_confidence,
                "label": si.label,
                "min_confidence": si.min_confidence
            }
            records.append(record)
        return pd.DataFrame(records)
    
class DatasetsInference(BaseModel):
    """Data class representing the inferred BIDS labels for multiple datasets."""
    datasets: list[DatasetInference]
    
    @classmethod
    def from_csv(cls, csv_path: str) -> Self:
        df = pd.read_csv(csv_path)
        df["is_derived"] = df["is_derived"].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})
        datasets = []
        for dataset_name, group in df.groupby("dataset"):
            series_inferences = []
            for _, row in group.iterrows():
                si = SeriesInference(
                    subject_id=row["subject_id"],
                    session_id=row["session_id"],
                    series_id=row["series_id"],
                    series_description=row.get("series_description"),
                    inferred_datatype=row.get("inferred_datatype"),
                    datatype_confidence=row.get("datatype_confidence"),
                    inferred_suffix=row.get("inferred_suffix"),
                    suffix_confidence=row.get("suffix_confidence"),
                    is_derived=row.get("is_derived"),
                    derived_confidence=row.get("derived_confidence")
                )
                series_inferences.append(si)
            di = DatasetInference(dataset=str(dataset_name), series_inferences=series_inferences)
            datasets.append(di)
        return cls(datasets=datasets)
    
    @classmethod
    def from_datasets(cls, datasets: list[Dataset], sample_subjects_per_subjects: Optional[int] = None, conn: Optional[sqlite3.Connection] = None) -> Self:
        dataset_inferences = []
        for dataset in datasets:
            dataset_features: dict[str, dict[str, dict[str, SeriesFeatures]]] = dataset.generate_features(
                conn=conn,
                skip_unavailable=True,
                sample_subjects=sample_subjects_per_subjects
                )
            series_inferences: list[SeriesInference] = []
            for subject_id, sessions in dataset_features.items():
                for session_id, series_dict in sessions.items():
                    for series_id, series_features in series_dict.items():
                        tokens = collect_tokens(series_features)
                        _, inferred_datatype, datatype_confidence = score_datatype(series=series_features, tokens=tokens)
                        _, is_derived, derived_confidence = score_is_derived(series=series_features, tokens=tokens)
                        _, inferred_suffix, suffix_confidence = score_suffix(
                            series=series_features,
                            tokens=tokens,
                            datatype=inferred_datatype,
                            is_derived=is_derived,
                        )
                        min_confidence = min(datatype_confidence, suffix_confidence, derived_confidence)
                        # series = dataset.search_series_by_id(subject_id, session_id, series_id)
                        series_description = series_features.text.series_description.text if series_features.text and series_features.text.series_description else None
                        record = {
                            "dataset": dataset.dir_root.name,
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "series_id": series_id,
                            "series_description": series_description,
                            "inferred_datatype": inferred_datatype,
                            "datatype_confidence": datatype_confidence,
                            "inferred_suffix": inferred_suffix,
                            "suffix_confidence": suffix_confidence,
                            "is_derived": is_derived,
                            "derived_confidence": derived_confidence,
                            "min_confidence": min_confidence
                        }
                        series_inferences.append(SeriesInference(**record))
            dataset_inference = DatasetInference(dataset=dataset.dir_root.name, series_inferences=series_inferences)
            dataset_inferences.append(dataset_inference)
        return cls(datasets=dataset_inferences)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([di.to_dataframe() for di in self.datasets], ignore_index=True)
    
    def to_csv(self, csv_path: UPath | str) -> None:
        if isinstance(csv_path, str):
            csv_path = UPath(csv_path)
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False) # type: ignore