"""
Extract and normalize key features from DICOM series to facilitate BIDS inference and dataset characterization.

This module defines a comprehensive set of features that can be extracted from a series of DICOM files, including-
    - temporal
    - diffusion
    - perfusion
    - spatial
    - encoding
    - contrast-related
    - and textual metadata features.

Each feature is designed to capture relevant information that can aid in the classification and organization of neuroimaging data according to the BIDS standard. 
The features are structured to provide robust summaries across all instances in a series, allowing for effective inference even in cases where some metadata may be missing or inconsistent.

Extraction phase is purely data-driven, relying on the presence of relevant DICOM fields and their values across the series.
Stages:
    1. Read DICOM files and tags
    2. Normalize and aggregate numeric, categorical, boolean, and textual features across the series
    3. Handle missing values and variability to provide robust feature summaries
    4. Compute derived features such as number of timepoints, diffusion directions, and consistency measures
    5. Output a structured SeriesFeatures object that encapsulates all extracted information for downstream inference and dataset organization.

The next step is to implement an inference pipeline that takes these extracted features and applies rule-based logic or machine learning classifiers to predict BIDS entities and organize the data accordingly.
"""

import os, re, json
from datetime import datetime
from pathlib import Path, Path
from typing import Optional, Iterable, Iterator, Self, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from pydantic import BaseModel, field_validator, ConfigDict
import pydicom as dicom
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn
import numpy as np
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat
# from tinydb import TinyDB, Query
import sqlite3

CAMEL_TO_SNAKE_CASE_PATTERN: re.Pattern = re.compile(r'(?<!^)(?=[A-Z])')
SNAKE_TO_CAMEL_CASE_PATTERN: re.Pattern = re.compile(r'(_)([a-z])')
PASCAL_TO_SNAKE_CASE_PATTERN: re.Pattern = re.compile(r'(?<!^)(?=[A-Z])')
SNAKE_TO_PASCAL_CASE_PATTERN: re.Pattern = re.compile(r'(_)([a-z])')

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

def indent_block(text: str, indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(pad + line for line in text.splitlines())

def humanize_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, SeriesNumericFeature):
        return f"{value.value} (iqr={value.iqr}, valid_fraction={value.valid_fraction}, stable={value.stable})"
    if isinstance(value, SeriesCategoricalFeature):
        return f"{value.value} (consistency={value.consistency})"
    if isinstance(value, SeriesBooleanFeature):
        return f"{value.value} (true_fraction={value.true_fraction})"
    if isinstance(value, SeriesTextFeature):
        return f"{len(value.tokens)} tokens"
    return str(value)

def format_section(title: str, items: list[tuple[str, Any]]) -> str:
    lines = [title, "-" * len(title)]
    for label, value in items:
        lines.append(f"  {label}: {humanize_value(value)}")
    return "\n".join(lines)

def read_dicom_header(path: Path) -> Optional[dicom.Dataset]:
    try:
        return dicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None

def parse_dicom_time(time_str: Optional[str]) -> Optional[float]:
    if time_str is None:
        return None
    try:
        if "." in time_str:
            time_str, fractional = time_str.split(".")
            fractional_seconds = float("0." + fractional)
        else:
            fractional_seconds = 0.0
        hours = int(time_str[0:2]) if len(time_str) >= 2 else 0
        minutes = int(time_str[2:4]) if len(time_str) >= 4 else 0
        seconds = int(time_str[4:6]) if len(time_str) >= 6 else 0
        total_seconds = hours * 3600 + minutes * 60 + seconds + fractional_seconds
        return total_seconds
    except Exception:
        return None

def parse_dicom_datetime(datetime_str: Optional[str]) -> Optional[Any]:
    """Parse DICOM datetime string (format: YYYYMMDDHHMMSS.FFFFFF) into a datetime object"""
    if datetime_str is None:
        return None
    try:
        if isinstance(datetime_str, str):
            # Remove fractional seconds if present
            if "." in datetime_str:
                datetime_str = datetime_str.split(".")[0]
            return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        return None
    except Exception:
        return None

def parse_dicom_date_time(date_str: Optional[str], time_str: Optional[str]) -> Optional[Any]:
    """Parse DICOM date (YYYYMMDD) and time (HHMMSS.FFFFFF) strings into a datetime object"""
    if date_str is None or time_str is None:
        return None
    try:
        from datetime import datetime
        # Remove fractional seconds if present
        if "." in time_str:
            time_str = time_str.split(".")[0]
        datetime_combined = f"{date_str}{time_str}"
        return datetime.strptime(datetime_combined, "%Y%m%d%H%M%S")
    except Exception:
        return None

def get_tag_value(ds: dicom.Dataset, tag: str, default=None) -> Optional[Any]:
    if hasattr(ds, tag):
        return getattr(ds, tag)
    return default

def normalize_category(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return "\\".join(map(str, v))
    return str(v).strip().lower()

# def multivalue_to_string(value: Optional[MultiValue | str]) -> Optional[str]:
#     if value is None:
#         return None
#     if not isinstance(value, (MultiValue, str)):
#         raise ValueError(f"Expected a MultiValue or str, got {type(value)}")
#     if isinstance(value, str):
#         return value
#     return " ".join(str(v) for v in value)

def multivalue_to_string(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, MultiValue):
        return " ".join(str(v) for v in value)
    return str(value)

def dsfloat_to_float(value: Optional[DSfloat]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, DSfloat):
        return float(value)

def extract_dicom_tokens(text: Optional[str]) -> list[str]:
    if isinstance(text, MultiValue):
        text = multivalue_to_string(text)
    if not isinstance(text, str):
        raise ValueError(f"Expected a string for text analysis, got {type(text)}")
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)
    return _TOKEN_PATTERN.findall(text)

def initialize_features_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS series_features (
                    subject_id TEXT NOT NULL,
                    session_id TEXT,
                    series_id TEXT NOT NULL,
                    series_description TEXT,
                    data JSON NOT NULL,
                    PRIMARY KEY (subject_id, session_id, series_id, series_description)
                )
            """)
            conn.commit()
    return sqlite3.connect(db_path)

class SeriesNumericFeature(BaseModel):
    value: Optional[float] # Robust central tendency measure (e.g. median)
    iqr: Optional[float] # Interquartile range (IQR) as a measure of variability
    valid_fraction: float # Fraction of instances with valid values for this feature (0 to 1)
    stable: Optional[bool] = None # Indicates if the feature is stable across instances (e.g. low IQR relative to median)
    
    def __eq__(self, other: Self | object) -> bool:
        if not isinstance(other, SeriesNumericFeature):
            return self.value == other
        return (
            self.value == other.value and
            self.iqr == other.iqr and
            self.valid_fraction == other.valid_fraction and
            self.stable == other.stable
        )
    
    @classmethod
    def from_values(cls, values: Iterable[Optional[float]]) -> Self:
        """
        Creates a SeriesNumericFeature instance from an iterable of numeric values, which may include None for missing values. 
        This method computes the median and interquartile range (IQR) while ignoring None values, and calculates the fraction of valid (non-None) values to provide a robust summary of the feature across a series of DICOM instances.
        """
        values_list = list(values)
        valid_values = [v for v in values_list if v is not None]
        if not valid_values:
            return cls(value=None, iqr=None, valid_fraction=0.0)
        value = np.median(valid_values)
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        valid_fraction = len(valid_values) / len(values_list)
        stable = (
            True if len(valid_values) >= 2 and iqr == 0
            else False if len(valid_values) >= 2 and value != 0 and iqr / abs(value) > 0.1
            else None  # Not enough data to determine stability or value is zero (can't compute relative IQR)
            )
        return cls(value=value, iqr=iqr, valid_fraction=valid_fraction, stable=stable)

class SeriesCategoricalFeature(BaseModel):
    value: Optional[str] # Mode or most common category
    category_counts: dict[str, int] # Counts of each category across instances
    consistency: Optional[float]  # Fraction of instances that match the mode category (0 to 1), indicating consistency across the series
    
    def __eq__(self, other: Self | object) -> bool:
        if not isinstance(other, SeriesCategoricalFeature):
            return self.value == other
        return (
            self.value == other.value and
            self.category_counts == other.category_counts and
            self.consistency == other.consistency
        )
    
    @classmethod
    def from_values(cls, values: Iterable[Optional[str]]) -> Self:
        """
        Creates a SeriesCategoricalFeature instance from an iterable of categorical values, which may include None for missing values. 
        This method computes the mode (most common category) and counts of each category while ignoring None values, and calculates the consistency of the mode category across the series to provide a summary of the categorical feature across a series of DICOM instances.
        """
        values_list = list(values)
        valid_values = [v for v in values_list if v is not None]
        if not valid_values:
            return cls(value=None, category_counts={}, consistency=None)
        category_counts = {}
        for v in valid_values:
            # Convert unhashable types (like MultiValue) to hashable string representation
            # key = str(v) if not isinstance(v, (str, int, float, bool)) else v
            key = multivalue_to_string(v) if isinstance(v, MultiValue) else str(v)
            key = normalize_category(key)
            category_counts[key] = category_counts.get(key, 0) + 1
        mode_category = max(category_counts, key=lambda x: category_counts[x])
        consistency = category_counts[mode_category] / len(valid_values)
        return cls(value=mode_category, category_counts=category_counts, consistency=consistency)

class SeriesBooleanFeature(BaseModel):
    value: Optional[bool]
    true_fraction: float # Fraction of instances with value True (0 to 1), indicating prevalence of the feature across the series
    
    def __eq__(self, other: Self | object) -> bool:
        if not isinstance(other, SeriesBooleanFeature):
            return self.value == other
        return (
            self.value == other.value and
            self.true_fraction == other.true_fraction
        )
    
    @classmethod
    def from_values(cls, values: Iterable[Optional[bool]]) -> Self:
        """
        Creates a SeriesBooleanFeature instance from an iterable of boolean values, which may include None for missing values. 
        This method computes the fraction of True values while ignoring None values to provide a summary of the boolean feature across a series of DICOM instances.
        """
        values_list = list(values)
        valid_values = [v for v in values_list if v is not None]
        if not valid_values:
            return cls(value=None, true_fraction=0.0)
        true_count = sum(1 for v in valid_values if v)
        true_fraction = true_count / len(valid_values)
        mode_value = True if true_fraction >= 0.5 else False
        return cls(value=mode_value, true_fraction=true_fraction)

class SeriesTextFeature(BaseModel):
    text: str # Mode or most common text value (could be empty if no valid text)
    tokens: dict[str, int]
    valid_fraction: float # Fraction of instances with valid text values (0 to 1), indicating how often this textual feature is present across the series

    def __eq__(self, other: Self | object) -> bool:
        if not isinstance(other, SeriesTextFeature):
            return self.text == other
        return (
            self.text == other.text and
            self.tokens == other.tokens and
            self.valid_fraction == other.valid_fraction
        )

    @classmethod
    def from_values(cls, values: Iterable[Optional[str]]) -> Self:
        values_list = list(values)
        valid_texts = [v for v in values_list if isinstance(v, str) and v.strip()]
        if not valid_texts:
            return cls(text="", tokens={}, valid_fraction=0.0)
        text_counts = {}
        token_counts = {}
        for text in valid_texts:
            text_counts[text] = text_counts.get(text, 0) + 1
            tokens = extract_dicom_tokens(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        mode_text = max(text_counts, key=lambda x: text_counts[x])
        valid_fraction = len(valid_texts) / len(values_list)
        return cls(text=mode_text, tokens=token_counts, valid_fraction=valid_fraction)

class GeometryFeatures(BaseModel):
    rows: Optional[SeriesNumericFeature] = None
    columns: Optional[SeriesNumericFeature] = None
    num_slices: Optional[int] = None
    voxel_size: Optional[tuple[float, float, float]] = None
    matrix_size: Optional[tuple[int, int, int]] = None
    geometry_hash: Optional[str] = None

    def __str__(self) -> str:
        return format_section(
            "Geometry Features",
            [
                ("Rows", self.rows),
                ("Columns", self.columns),
                ("Number of slices", self.num_slices),
                ("Voxel size (mm)", self.voxel_size),
                ("Matrix size (voxels)", self.matrix_size),
                ("Geometry hash", self.geometry_hash),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["rows", "columns", "num_slices", "voxel_size", "matrix_size", "geometry_hash"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)

        rows = SeriesNumericFeature.from_values(
            get_tag_value(ds, "Rows", None) for ds in datasets
        )
        columns = SeriesNumericFeature.from_values(
            get_tag_value(ds, "Columns", None) for ds in datasets
        )

        slice_positions = [
            get_tag_value(ds, "ImagePositionPatient", None)
            for ds in datasets
        ]
        num_slices = len(set(str(p) for p in slice_positions if p is not None)) or None

        dx = dy = dz = None
        if rows.value and columns.value:
            # px = SeriesNumericFeature.from_values(
            #     get_tag_value(ds, "PixelSpacing", [None, None])[0]
            #     if hasattr(ds, "PixelSpacing") else None
            #     for ds in datasets
            # )
            # py = SeriesNumericFeature.from_values(
            #     get_tag_value(ds, "PixelSpacing", [None, None])[1]
            #     if hasattr(ds, "PixelSpacing") else None
            #     for ds in datasets
            # )
            # dz = SeriesNumericFeature.from_values(
            #     get_tag_value(ds, "SliceThickness", None)
            #     for ds in datasets
            # )

            # if px.value and py.value and dz.value:
            #     dx = px.value
            #     dy = py.value
            #     dz = dz.value
            for ds in datasets:
                if hasattr(ds, "PixelSpacing") and hasattr(ds, "SliceThickness"):
                    ps = get_tag_value(ds, "PixelSpacing", [None, None])
                    st = get_tag_value(ds, "SliceThickness", None)
                    if ps and isinstance(ps, (list, tuple)) and len(ps) == 2 and st:
                        try:
                            dx_candidate = float(ps[0])
                            dy_candidate = float(ps[1])
                            dz_candidate = float(st)
                            if dx_candidate > 0 and dy_candidate > 0 and dz_candidate > 0:
                                dx = dx_candidate
                                dy = dy_candidate
                                dz = dz_candidate
                                break
                        except (ValueError, TypeError):
                            continue

        voxel_size = (dx, dy, dz) if dx and dy and dz else None
        matrix_size = (
            int(rows.value),
            int(columns.value),
            num_slices
        ) if rows.value and columns.value and num_slices else None

        geometry_hash = None
        if matrix_size and voxel_size:
            geometry_hash = hash((matrix_size, voxel_size))

        return cls(
            rows=rows,
            columns=columns,
            num_slices=num_slices,
            voxel_size=voxel_size,
            matrix_size=matrix_size,
            geometry_hash=str(geometry_hash) if geometry_hash else None
        )
        
    def flatten(self) -> dict[str, Optional[int | float | tuple[int, int, int] | tuple[float, float, float] | str]]:
        return {
            "rows": self.rows.value if self.rows else None,
            "columns": self.columns.value if self.columns else None,
            "num_slices": self.num_slices,
            "voxel_size": self.voxel_size,
            "matrix_size": self.matrix_size,
            "geometry_hash": self.geometry_hash,
        }

class TemporalFeatures(BaseModel):
    repetition_time: Optional[SeriesNumericFeature] = None
    echo_time: Optional[SeriesNumericFeature] = None
    inversion_time: Optional[SeriesNumericFeature] = None
    flip_angle: Optional[SeriesNumericFeature] = None

    num_timepoints: Optional[int] = None
    temporal_variation: Optional[SeriesNumericFeature] = None  # seconds
    tr_bucket: Optional[str] = None  # e.g. "short", "medium", "long" based on typical TR ranges for fMRI, structural, etc.
    te_bucket: Optional[str] = None  # e.g. "short", "medium", "long" based on typical TE ranges for different sequence types
    is_3D: Optional[bool] = None  # Inferred from geometry features or specific DICOM tags
    is_isotropic: Optional[bool] = None  # Inferred from geometry features (e.g. equal voxel dimensions)
    echo_train_length: Optional[SeriesNumericFeature] = None  # For sequences with multiple echoes, the number of echoes acquired

    def __str__(self) -> str:
        return format_section(
            "Temporal Features",
            [
                ("Repetition time", self.repetition_time),
                ("Echo time", self.echo_time),
                ("Inversion time", self.inversion_time),
                ("Flip angle", self.flip_angle),
                ("Number of timepoints", self.num_timepoints),
                ("Temporal variation (s)", self.temporal_variation),
                ("TR bucket", self.tr_bucket),
                ("TE bucket", self.te_bucket),
                ("Is 3D", self.is_3D),
                ("Is isotropic", self.is_isotropic),
                ("Echo train length", self.echo_train_length),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["repetition_time", "echo_time", "inversion_time", "flip_angle", "num_timepoints", "temporal_variation", "tr_bucket", "te_bucket", "is_3D", "is_isotropic", "echo_train_length"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset], geometry: Optional[GeometryFeatures] = None) -> Self:
        datasets = list(datasets)

        repetition_time = SeriesNumericFeature.from_values(
            get_tag_value(ds, "RepetitionTime", None) for ds in datasets
        )
        echo_time = SeriesNumericFeature.from_values(
            get_tag_value(ds, "EchoTime", None) for ds in datasets
        )
        inversion_time = SeriesNumericFeature.from_values(
            get_tag_value(ds, "InversionTime", None) for ds in datasets
        )
        flip_angle = SeriesNumericFeature.from_values(
            get_tag_value(ds, "FlipAngle", None) for ds in datasets
        )

        # ---- Timepoint inference ----
        tpi = [
            get_tag_value(ds, "TemporalPositionIdentifier", None)
            for ds in datasets
        ]
        tpi = [v for v in tpi if v is not None]

        frt = [
            get_tag_value(ds, "FrameReferenceTime", None)
            for ds in datasets
        ]
        frt = [float(v) for v in frt if v is not None]

        times = [
            parse_dicom_time(get_tag_value(ds, "AcquisitionTime", None))
            for ds in datasets
        ]
        times = [t for t in times if t is not None]

        if tpi:
            num_timepoints = len(set(tpi))
        elif frt:
            num_timepoints = len(set(frt))
        elif times and repetition_time.value:
            tr_sec = repetition_time.value / 1000.0
            bins = {round(t / tr_sec) for t in times}
            num_timepoints = len(bins)
        else:
            num_timepoints = None

        # ---- Temporal variation (inter-volume deltas) ----
        temporal_variation = None
        if times:
            times_sorted = sorted(times)
            deltas = [
                t2 - t1 for t1, t2 in zip(times_sorted, times_sorted[1:])
                if t2 > t1
            ]
            if deltas:
                temporal_variation = SeriesNumericFeature.from_values(deltas)
                
        # ----------------------------
        # TR bucket (ms)
        # ----------------------------
        tr_bucket = None
        if repetition_time.value is not None:
            tr = repetition_time.value
            if tr < 1000:
                tr_bucket = "short"      # fast EPI
            elif tr < 3000:
                tr_bucket = "medium"     # standard fMRI
            else:
                tr_bucket = "long"       # structural or slow

        # ----------------------------
        # TE bucket (ms)
        # ----------------------------
        te_bucket = None
        if echo_time.value is not None:
            te = echo_time.value * 1000 if echo_time.value < 10 else echo_time.value
            # handle seconds vs ms ambiguity

            if te < 30:
                te_bucket = "short"
            elif te < 80:
                te_bucket = "medium"
            else:
                te_bucket = "long"
        
        echo_train_length = SeriesNumericFeature.from_values(
            get_tag_value(ds, "EchoTrainLength", None)
            for ds in datasets
        )

        # ----------------------------
        # Geometry-derived flags
        # ----------------------------
        is_3D = None
        is_isotropic = None

        if geometry:
            if geometry.num_slices and num_timepoints:
                is_3D = False
            elif geometry.num_slices and not num_timepoints:
                is_3D = True

            if geometry.voxel_size:
                dx, dy, dz = geometry.voxel_size
                if dx and dy and dz:
                    tol = 0.1
                    is_isotropic = (
                        abs(dx - dy) < tol and
                        abs(dx - dz) < tol
                    )

        return cls(
            repetition_time=repetition_time,
            echo_time=echo_time,
            inversion_time=inversion_time,
            flip_angle=flip_angle,
            num_timepoints=num_timepoints,
            temporal_variation=temporal_variation,
            tr_bucket=tr_bucket,
            te_bucket=te_bucket,
            is_3D=is_3D,
            is_isotropic=is_isotropic,
            echo_train_length=echo_train_length,
        )
        
    def flatten(self) -> dict[str, Optional[float | int | str | bool]]:
        return {
            "repetition_time": self.repetition_time.value if self.repetition_time else None,
            "echo_time": self.echo_time.value if self.echo_time else None,
            "inversion_time": self.inversion_time.value if self.inversion_time else None,
            "flip_angle": self.flip_angle.value if self.flip_angle else None,
            "num_timepoints": self.num_timepoints,
            "temporal_variation": self.temporal_variation.value if self.temporal_variation else None,
            "tr_bucket": self.tr_bucket,
            "te_bucket": self.te_bucket,
            "is_3D": self.is_3D,
            "is_isotropic": self.is_isotropic,
        }

class DiffusionFeatures(BaseModel):
    b_values: Optional[list[float]] = None  # unique shells
    num_b0: Optional[int] = None
    num_diffusion_volumes: Optional[int] = None
    num_diffusion_directions: Optional[int] = None
    has_diffusion: Optional[SeriesBooleanFeature] = None

    def __str__(self) -> str:
        return format_section(
            "Diffusion Features",
            [
                ("B-values", self.b_values),
                ("Number of b0", self.num_b0),
                ("Diffusion volumes", self.num_diffusion_volumes),
                ("Diffusion directions", self.num_diffusion_directions),
                ("Has diffusion", self.has_diffusion),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["b_values", "num_b0", "num_diffusion_volumes", "num_diffusion_directions", "has_diffusion"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)

        volume_map = {}  # (bval, gradient) -> count
        has_diffusion_flags = []

        for ds in datasets:
            b = get_tag_value(ds, "DiffusionBValue", None)
            g = get_tag_value(ds, "DiffusionGradientDirection", None)

            if b is None:
                has_diffusion_flags.append(False)
                continue

            has_diffusion_flags.append(True)

            b = float(b)
            gradient = (
                tuple(round(float(x), 5) for x in g)
                if g is not None and len(g) == 3
                else None
            )

            vol_key = (b, gradient)

            volume_map.setdefault(vol_key, 0)
            volume_map[vol_key] += 1

        has_diffusion = SeriesBooleanFeature.from_values(has_diffusion_flags)

        if not volume_map:
            return cls(
                b_values=None,
                num_b0=None,
                num_diffusion_volumes=None,
                num_diffusion_directions=None,
                has_diffusion=has_diffusion,
            )

        bvals = [b for (b, _) in volume_map.keys()]
        gradients = {g for (b, g) in volume_map.keys() if b > 0 and g is not None}

        return cls(
            b_values=sorted(set(bvals)),
            num_b0=sum(1 for b in bvals if b == 0),
            num_diffusion_volumes=len(volume_map),
            num_diffusion_directions=len(gradients) if gradients else None,
            has_diffusion=has_diffusion,
        )
        
    def flatten(self) -> dict[str, list[float] | int | bool | None]:
        return {
            "b_values": self.b_values if self.b_values else None,
            "num_b0": self.num_b0,
            "num_diffusion_volumes": self.num_diffusion_volumes,
            "num_diffusion_directions": self.num_diffusion_directions,
            "has_diffusion": self.has_diffusion.value if self.has_diffusion else None,
        }

class PerfusionFeatures(BaseModel):
    perfusion_labeling_type: Optional[SeriesCategoricalFeature] = None
    bolus_arrival_time: Optional[SeriesNumericFeature] = None
    contrast_agent: Optional[SeriesCategoricalFeature] = None
    perfusion_series_type: Optional[SeriesCategoricalFeature] = None

    num_timepoints: Optional[int] = None
    temporal_spacing: Optional[SeriesNumericFeature] = None

    def flatten(self) -> dict[str, str | float | int | None]:
        return {
            "perfusion_labeling_type": self.perfusion_labeling_type.value if self.perfusion_labeling_type else None,
            "bolus_arrival_time": self.bolus_arrival_time.value if self.bolus_arrival_time else None,
            "contrast_agent": self.contrast_agent.value if self.contrast_agent else None,
            "perfusion_series_type": self.perfusion_series_type.value if self.perfusion_series_type else None,
            "num_timepoints": self.num_timepoints,
            "temporal_spacing": self.temporal_spacing.value if self.temporal_spacing else None,
        }

    def __str__(self) -> str:
        return format_section(
            "Perfusion Features",
            [
                ("Labeling type", self.perfusion_labeling_type),
                ("Bolus arrival time", self.bolus_arrival_time),
                ("Contrast agent", self.contrast_agent),
                ("Series type", self.perfusion_series_type),
                ("Number of timepoints", self.num_timepoints),
                ("Temporal spacing", self.temporal_spacing),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["perfusion_labeling_type", "bolus_arrival_time", "contrast_agent", "perfusion_series_type", "num_timepoints", "temporal_spacing"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)

        # Extract raw DICOM fields
        perfusion_labeling_type = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ASLLabelingType", None) for ds in datasets
        )
        bolus_arrival_time = SeriesNumericFeature.from_values(
            get_tag_value(ds, "BolusArrivalTime", None) for ds in datasets
        )
        contrast_agent = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ContrastBolusAgent", None) for ds in datasets
        )
        perfusion_series_type = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "PerfusionSeriesType", None) for ds in datasets
        )

        # ---- Timepoint inference ----
        tpi = [
            get_tag_value(ds, "TemporalPositionIdentifier", None)
            for ds in datasets
        ]
        tpi = [v for v in tpi if v is not None]

        frt = [
            get_tag_value(ds, "FrameReferenceTime", None)
            for ds in datasets
        ]
        frt = [float(v) for v in frt if v is not None]

        times = [
            parse_dicom_time(get_tag_value(ds, "AcquisitionTime", None))
            for ds in datasets
        ]
        times = [t for t in times if t is not None]

        if tpi:
            num_timepoints = len(set(tpi))
        elif frt:
            num_timepoints = len(set(frt))
        elif times:
            num_timepoints = len(set(times))
        else:
            num_timepoints = None

        temporal_spacing = None
        if len(times) > 1:
            deltas = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
            if deltas:
                temporal_spacing = SeriesNumericFeature.from_values(deltas)

        return cls(
            perfusion_labeling_type=perfusion_labeling_type,
            bolus_arrival_time=bolus_arrival_time,
            contrast_agent=contrast_agent,
            perfusion_series_type=perfusion_series_type,
            num_timepoints=num_timepoints,
            temporal_spacing=temporal_spacing,
        )

class SequenceFeatures(BaseModel):
    """DICOM sequence type tags (vendor-standardized)"""
    scanning_sequence: Optional[SeriesCategoricalFeature] = None
    sequence_variant: Optional[SeriesCategoricalFeature] = None
    scan_options: Optional[SeriesCategoricalFeature] = None
    mr_acquisition_type: Optional[SeriesCategoricalFeature] = None  # 2D vs 3D
    
    def __str__(self) -> str:
        return format_section(
            "Sequence Features",
            [
                ("Scanning sequence", self.scanning_sequence),
                ("Sequence variant", self.sequence_variant),
                ("Scan options", self.scan_options),
                ("MR acquisition type", self.mr_acquisition_type),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["scanning_sequence", "sequence_variant", "scan_options", "mr_acquisition_type"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        return cls(
            scanning_sequence=SeriesCategoricalFeature.from_values(
                normalize_category(get_tag_value(ds, "ScanningSequence", None)) 
                for ds in datasets
            ),
            sequence_variant=SeriesCategoricalFeature.from_values(
                normalize_category(get_tag_value(ds, "SequenceVariant", None)) 
                for ds in datasets
            ),
            scan_options=SeriesCategoricalFeature.from_values(
                normalize_category(get_tag_value(ds, "ScanOptions", None)) 
                for ds in datasets
            ),
            mr_acquisition_type=SeriesCategoricalFeature.from_values(
                normalize_category(get_tag_value(ds, "MRAcquisitionType", None)) 
                for ds in datasets
            ),
        )
        
    def flatten(self) -> dict[str, Optional[str]]:
        return {
            "scanning_sequence": self.scanning_sequence.value if self.scanning_sequence else None,
            "sequence_variant": self.sequence_variant.value if self.sequence_variant else None,
            "scan_options": self.scan_options.value if self.scan_options else None,
            "mr_acquisition_type": self.mr_acquisition_type.value if self.mr_acquisition_type else None,
        }
        
class MultiEchoFeatures(BaseModel):
    """Detection of multi-echo sequences"""
    num_echoes: Optional[int] = None
    echo_times: Optional[list[float]] = None  # Sorted unique echo times
    echo_numbers: Optional[list[int]] = None  # If EchoNumbers tag present
    
    def __str__(self) -> str:
        return format_section(
            "Multi-echo Features",
            [
                ("Number of echoes", self.num_echoes),
                ("Echo times", self.echo_times),
                ("Echo numbers", self.echo_numbers),
            ],
        )
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["num_echoes", "echo_times", "echo_numbers"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        echo_times_raw = [get_tag_value(ds, "EchoTime", None) for ds in datasets]
        echo_times_valid = [float(et) for et in echo_times_raw if et is not None]
        
        unique_echo_times = sorted(set(echo_times_valid)) if echo_times_valid else None
        
        echo_numbers_raw = [get_tag_value(ds, "EchoNumbers", None) for ds in datasets]
        echo_numbers_valid = []
        for en in echo_numbers_raw:
            if en is not None:
                try:
                    en_str = multivalue_to_string(en) if isinstance(en, MultiValue) else str(en)
                    if en_str is not None:
                        echo_numbers_valid.append(int(en_str))
                except (ValueError, TypeError):
                    pass
        unique_echo_numbers = sorted(set(echo_numbers_valid)) if echo_numbers_valid else None
        
        num_echoes = (
            len(unique_echo_numbers) if unique_echo_numbers
            else len(unique_echo_times) if unique_echo_times and len(unique_echo_times) > 1
            else None
        )
        
        is_multi_echo = num_echoes is not None and num_echoes > 1
        
        return cls(
            num_echoes=num_echoes,
            echo_times=unique_echo_times,
            echo_numbers=unique_echo_numbers,
        )
        
    def flatten(self) -> dict[str, Optional[int | list[float] | list[int]]]:
        return {
            "num_echoes": self.num_echoes,
            "echo_times": self.echo_times if self.echo_times else None,
            "echo_numbers": self.echo_numbers if self.echo_numbers else None,
        }
    
class SpatialFeatures(BaseModel):
    slice_thickness: Optional[SeriesNumericFeature] = None
    spacing_between_slices: Optional[SeriesNumericFeature] = None
    pixel_spacing: Optional[tuple[SeriesNumericFeature, SeriesNumericFeature]] = None
    image_orientation: Optional[SeriesCategoricalFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Spatial Features",
            [
                ("Slice thickness", self.slice_thickness),
                ("Spacing between slices", self.spacing_between_slices),
                ("Pixel spacing", self.pixel_spacing),
                ("Image orientation", self.image_orientation),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)
        
        slice_thickness = SeriesNumericFeature.from_values(
            get_tag_value(ds, "SliceThickness", None) for ds in datasets
        )

        spacing_between_slices = SeriesNumericFeature.from_values(
            get_tag_value(ds, "SpacingBetweenSlices", None) for ds in datasets
        )

        pixel_spacing = None
        if any(hasattr(ds, "PixelSpacing") for ds in datasets):
            try:
                pixel_spacing_x = SeriesNumericFeature.from_values(
                    get_tag_value(ds, "PixelSpacing", [None, None])[0] if hasattr(ds, "PixelSpacing") else None for ds in datasets # type: ignore
                )
                pixel_spacing_y = SeriesNumericFeature.from_values(
                    get_tag_value(ds, "PixelSpacing", [None, None])[1] if hasattr(ds, "PixelSpacing") else None for ds in datasets # type: ignore
                )
                pixel_spacing = (pixel_spacing_x, pixel_spacing_y)
            except TypeError:
                pixel_spacing = None

        image_orientation = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ImageOrientationPatient", None) for ds in datasets
        )

        return cls(
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            pixel_spacing=pixel_spacing,
            image_orientation=image_orientation,
        )
        
    def flatten(self) -> dict[str, Optional[float | tuple[float, float] | str]]:
        pixel_spacing_value = None
        if (self.pixel_spacing and self.pixel_spacing[0] and self.pixel_spacing[1] and
            self.pixel_spacing[0].value is not None and self.pixel_spacing[1].value is not None):
            pixel_spacing_value = (self.pixel_spacing[0].value, self.pixel_spacing[1].value)
        
        return {
            "slice_thickness": self.slice_thickness.value if self.slice_thickness else None,
            "spacing_between_slices": self.spacing_between_slices.value if self.spacing_between_slices else None,
            "pixel_spacing": pixel_spacing_value,
            "image_orientation": self.image_orientation.value if self.image_orientation else None,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["slice_thickness", "spacing_between_slices", "pixel_spacing", "image_orientation"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
class EncodingFeatures(BaseModel):
    phase_encoding_direction: Optional[SeriesCategoricalFeature] = None
    phase_encoding_axis: Optional[SeriesCategoricalFeature] = None
    phase_encoding_polarity: Optional[SeriesCategoricalFeature] = None  # Siemens private tag for direction polarity
    echo_spacing: Optional[SeriesNumericFeature] = None
    # is_epi: Optional[SeriesBooleanFeature] = None
    
    parallel_reduction_factor_in_plane: Optional[SeriesNumericFeature] = None
    parallel_reduction_factor_out_of_plane: Optional[SeriesNumericFeature] = None
    multiband_factor: Optional[SeriesNumericFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Encoding Features",
            [
                ("Phase encoding direction", self.phase_encoding_direction),
                ("Phase encoding axis", self.phase_encoding_axis),
                ("Phase encoding polarity", self.phase_encoding_polarity),
                ("Echo spacing", self.echo_spacing),
                # ("Is EPI", self.is_epi),
                ("Parallel reduction factor (in-plane)", self.parallel_reduction_factor_in_plane),
                ("Parallel reduction factor (out-of-plane)", self.parallel_reduction_factor_out_of_plane),
                ("Multiband factor", self.multiband_factor),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)
        phase_encoding_direction = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "InPlanePhaseEncodingDirection", None) for ds in datasets
        )
        phase_encoding_axis = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "PhaseEncodingAxis", None) for ds in datasets
        )
        echo_spacing = SeriesNumericFeature.from_values(
            get_tag_value(ds, "EchoSpacing", None) for ds in datasets
        )
        phase_encoding_polarity = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "PhaseEncodingDirection", None)  # Siemens private tag
            for ds in datasets
        )

        
        # Robust EPI detection (multi-vendor)
        is_epi_flags = []
        for ds in datasets:
            seq_name = (get_tag_value(ds, "SequenceName", "") or "").lower()
            scanning_seq = normalize_category(get_tag_value(ds, "ScanningSequence", None))
            seq_variant = normalize_category(get_tag_value(ds, "SequenceVariant", None))
            vendor = normalize_category(get_tag_value(ds, "Manufacturer", None))
            if vendor is None:
                vendor = "N/A"
            
        #     # Vendor-agnostic EPI signals
        #     is_epi_seq = (
        #         "ep" in seq_name or                    # ep2d, ep3d, epi, ep_bold, etc.
        #         "echo planar" in seq_name or           # full name
        #         scanning_seq == "ep" or                # DICOM standard
        #         "epi" in seq_name                      # generic
        #     )
            
        #     # Vendor-specific patterns (fallback)
        #     vendor_specific = (
        #         ("ep2d" in seq_name) or                # Siemens
        #         ("ep" in seq_name and "ge" in vendor.lower()) or  # GE
        #         ("epifmri" in seq_name) or             # Philips common
        #         ("ep" in seq_name and "philips" in vendor.lower())
        #     )
            
        #     is_epi_flags.append(is_epi_seq or vendor_specific)
        
        # is_epi = SeriesBooleanFeature.from_values(is_epi_flags)
        
        parallel_reduction_factor = SeriesNumericFeature.from_values(
            get_tag_value(ds, "ParallelReductionFactorInPlane", None)
            for ds in datasets
        )
        
        parallel_reduction_factor_out_of_plane = SeriesNumericFeature.from_values(
            get_tag_value(ds, "ParallelReductionFactorOutOfPlane", None)
            for ds in datasets
        )

        multiband_factor = SeriesNumericFeature.from_values(
            get_tag_value(ds, "MultibandAccelerationFactor", None)
            for ds in datasets
        )
        
        return cls(
            phase_encoding_direction=phase_encoding_direction,
            phase_encoding_axis=phase_encoding_axis,
            phase_encoding_polarity=phase_encoding_polarity,
            echo_spacing=echo_spacing,
            # is_epi=is_epi,
            parallel_reduction_factor_in_plane=parallel_reduction_factor,
            parallel_reduction_factor_out_of_plane=parallel_reduction_factor_out_of_plane,
            multiband_factor=multiband_factor,
        )
        
    def flatten(self) -> dict[str, Optional[str | float | bool]]:
        return {
            "phase_encoding_direction": self.phase_encoding_direction.value if self.phase_encoding_direction else None,
            "phase_encoding_axis": self.phase_encoding_axis.value if self.phase_encoding_axis else None,
            "phase_encoding_polarity": self.phase_encoding_polarity.value if self.phase_encoding_polarity else None,
            "echo_spacing": self.echo_spacing.value if self.echo_spacing else None,
            # "is_epi": self.is_epi.value if self.is_epi else None,
            "parallel_reduction_factor_in_plane": self.parallel_reduction_factor_in_plane.value if self.parallel_reduction_factor_in_plane else None,
            "parallel_reduction_factor_out_of_plane": self.parallel_reduction_factor_out_of_plane.value if self.parallel_reduction_factor_out_of_plane else None,
            "multiband_factor": self.multiband_factor.value if self.multiband_factor else None,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["phase_encoding_direction", "phase_encoding_axis", "phase_encoding_polarity", "echo_spacing", "parallel_reduction_factor_in_plane", "parallel_reduction_factor_out_of_plane", "multiband_factor"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]
    
class ContrastFeatures(BaseModel):
    contrast_agent: Optional[SeriesCategoricalFeature] = None
    injection_time: Optional[SeriesNumericFeature] = None
    signal_intensity_shift: Optional[SeriesNumericFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Contrast Features",
            [
                ("Contrast agent", self.contrast_agent),
                ("Injection time", self.injection_time),
                ("Signal intensity shift", self.signal_intensity_shift),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)
        contrast_agent = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ContrastBolusAgent", None) for ds in datasets
        )
        injection_time = SeriesNumericFeature.from_values(
            get_tag_value(ds, "ContrastBolusInjectionTime", None) for ds in datasets
        )
        # Placeholder logic for signal intensity shift; would need to analyze pixel data across instances to determine this properly
        signal_intensity_shift = None #* Future implementation: Analyze pixel intensity changes across timepoints to infer contrast enhancement patterns, which can be a strong indicator of contrast usage even when metadata is incomplete or inconsistent.
        
        return cls(
            contrast_agent=contrast_agent,
            injection_time=injection_time,
            signal_intensity_shift=signal_intensity_shift,
        )
        
    def flatten(self) -> dict[str, Optional[str | float]]:
        return {
            "contrast_agent": self.contrast_agent.value if self.contrast_agent else None,
            "injection_time": self.injection_time.value if self.injection_time else None,
            "signal_intensity_shift": self.signal_intensity_shift.value if self.signal_intensity_shift else None,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["contrast_agent", "injection_time", "signal_intensity_shift"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]

class ImageTypeFeature(BaseModel):
    """Structured parsing of ImageType multi-value field"""
    is_original: Optional[SeriesBooleanFeature] = None  # Position 0
    is_primary: Optional[SeriesBooleanFeature] = None   # Position 1
    
    # Modifiers (position 2+)
    is_localizer: Optional[SeriesBooleanFeature] = None
    is_mpr: Optional[SeriesBooleanFeature] = None
    is_mip: Optional[SeriesBooleanFeature] = None
    is_projection: Optional[SeriesBooleanFeature] = None
    is_reformatted: Optional[SeriesBooleanFeature] = None
    
    # Sequence-type indicators
    has_angio: Optional[SeriesBooleanFeature] = None
    has_diffusion: Optional[SeriesBooleanFeature] = None
    has_perfusion: Optional[SeriesBooleanFeature] = None
    
    # Derived map indicators
    is_adc: Optional[SeriesBooleanFeature] = None
    is_fa: Optional[SeriesBooleanFeature] = None
    is_trace: Optional[SeriesBooleanFeature] = None
    is_cbf: Optional[SeriesBooleanFeature] = None
    is_cbv: Optional[SeriesBooleanFeature] = None
    
    # Part entity
    is_magnitude: Optional[SeriesBooleanFeature] = None
    is_phase: Optional[SeriesBooleanFeature] = None
    is_real: Optional[SeriesBooleanFeature] = None
    is_imaginary: Optional[SeriesBooleanFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Image Type Features",
            [
                ("Original", self.is_original),
                ("Primary", self.is_primary),
                ("Localizer", self.is_localizer),
                ("MPR", self.is_mpr),
                ("MIP", self.is_mip),
                ("Projection", self.is_projection),
                ("Reformatted", self.is_reformatted),
                ("Angio", self.has_angio),
                ("Diffusion", self.has_diffusion),
                ("Perfusion", self.has_perfusion),
                ("ADC", self.is_adc),
                ("FA", self.is_fa),
                ("Trace", self.is_trace),
                ("CBF", self.is_cbf),
                ("CBV", self.is_cbv),
                ("Magnitude", self.is_magnitude),
                ("Phase", self.is_phase),
                ("Real", self.is_real),
                ("Imaginary", self.is_imaginary),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        """Parse ImageType field with positional and keyword semantics"""
        image_types = []
        for ds in datasets:
            it = get_tag_value(ds, "ImageType", [])
            if isinstance(it, MultiValue):
                it = [str(v).upper() for v in it]
            elif isinstance(it, str):
                it = [s.strip().upper() for s in it.split("\\")]
            else:
                it = []
            image_types.append(it)
        
        # Position-based
        is_original = SeriesBooleanFeature.from_values(
            [it[0] == "ORIGINAL" if it else None for it in image_types]
        )
        is_primary = SeriesBooleanFeature.from_values(
            [it[1] == "PRIMARY" if len(it) > 1 else None for it in image_types]
        )
        
        # Keyword-based (any position)
        is_localizer = SeriesBooleanFeature.from_values(
            [any("LOCALIZER" in token for token in it) for it in image_types]
        )
        is_mpr = SeriesBooleanFeature.from_values(
            [any("MPR" in token for token in it) for it in image_types]
        )
        is_mip = SeriesBooleanFeature.from_values(
            [any("MIP" in token or "MAXIMUM" in token for token in it) for it in image_types]
        )
        is_projection = SeriesBooleanFeature.from_values(
            [any("PROJECTION" in token for token in it) for it in image_types]
        )
        is_reformatted = SeriesBooleanFeature.from_values(
            [any("REFORMATTED" in token for token in it) for it in image_types]
        )
        has_angio = SeriesBooleanFeature.from_values(
            [any("ANGIO" in token for token in it) for it in image_types]
        )
        has_diffusion = SeriesBooleanFeature.from_values(
            [any("DIFFUSION" in token for token in it) for it in image_types]
        )
        has_perfusion = SeriesBooleanFeature.from_values(
            [any("PERFUSION" in token for token in it) for it in image_types]
        )
        is_adc = SeriesBooleanFeature.from_values(
            [any("ADC" in token or "APPARENT_DIFFUSION" in token for token in it) for it in image_types]
        )
        is_fa = SeriesBooleanFeature.from_values(
            [any("FA" in token or "FRACTIONAL" in token for token in it) for it in image_types]
        )
        is_trace = SeriesBooleanFeature.from_values(
            [any("TRACE" in token for token in it) for it in image_types]
        )
        is_cbf = SeriesBooleanFeature.from_values(
            [any("CBF" in token or "CEREBRAL_BLOOD_FLOW" in token for token in it) for it in image_types]
        )
        is_cbv = SeriesBooleanFeature.from_values(
            [any("CBV" in token or "CEREBRAL_BLOOD_VOLUME" in token for token in it) for it in image_types]
        )
        
        # Part determination
        is_magnitude = SeriesBooleanFeature.from_values(
            [any(token in ("M", "MAGNITUDE") for token in it) for it in image_types]
        )
        is_phase = SeriesBooleanFeature.from_values(
            [any(token in ("P", "PHASE") for token in it) for it in image_types]
        )
        is_real = SeriesBooleanFeature.from_values(
            [any(token in ("REAL", "RE") for token in it) for it in image_types]
        )
        is_imaginary = SeriesBooleanFeature.from_values(
            [any(token in ("IMAGINARY", "IM") for token in it) for it in image_types]
        )
        
        return cls(
            is_original=is_original,
            is_primary=is_primary,
            is_localizer=is_localizer,
            is_mpr=is_mpr,
            is_mip=is_mip,
            is_projection=is_projection,
            is_reformatted=is_reformatted,
            has_angio=has_angio,
            has_diffusion=has_diffusion,
            has_perfusion=has_perfusion,
            is_adc=is_adc,
            is_fa=is_fa,
            is_trace=is_trace,
            is_cbf=is_cbf,
            is_cbv=is_cbv,
            is_magnitude=is_magnitude,
            is_phase=is_phase,
            is_real=is_real,
            is_imaginary=is_imaginary,
        )
        
    def flatten(self) -> dict[str, Optional[bool]]:
        return {
            "is_original": self.is_original.value if self.is_original else None,
            "is_primary": self.is_primary.value if self.is_primary else None,
            "is_localizer": self.is_localizer.value if self.is_localizer else None,
            "is_mpr": self.is_mpr.value if self.is_mpr else None,
            "is_mip": self.is_mip.value if self.is_mip else None,
            "is_projection": self.is_projection.value if self.is_projection else None,
            "is_reformatted": self.is_reformatted.value if self.is_reformatted else None,
            "has_angio": self.has_angio.value if self.has_angio else None,
            "has_diffusion": self.has_diffusion.value if self.has_diffusion else None,
            "has_perfusion": self.has_perfusion.value if self.has_perfusion else None,
            "is_adc": self.is_adc.value if self.is_adc else None,
            "is_fa": self.is_fa.value if self.is_fa else None,
            "is_trace": self.is_trace.value if self.is_trace else None,
            "is_cbf": self.is_cbf.value if self.is_cbf else None,
            "is_cbv": self.is_cbv.value if self.is_cbv else None,
            "is_magnitude": self.is_magnitude.value if self.is_magnitude else None,
            "is_phase": self.is_phase.value if self.is_phase else None,
            "is_real": self.is_real.value if self.is_real else None,
            "is_imaginary": self.is_imaginary.value if self.is_imaginary else None,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["is_original", "is_primary", "is_localizer", "is_mpr", "is_mip", "is_projection", "is_reformatted", "has_angio", "has_diffusion", "has_perfusion", "is_adc", "is_fa", "is_trace", "is_cbf", "is_cbv", "is_magnitude", "is_phase", "is_real", "is_imaginary"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]

class TextualMetadataFeatures(BaseModel):
    series_description: Optional[SeriesTextFeature] = None
    protocol_name: Optional[SeriesTextFeature] = None
    sequence_name: Optional[SeriesTextFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Textual Metadata Features",
            [
                (f"Series description: {self.series_description.text if self.series_description else None}", f"Valid fraction: {self.series_description.valid_fraction if self.series_description else None}"),
                (f"Protocol name: {self.protocol_name.text if self.protocol_name else None}", f"Valid fraction: {self.protocol_name.valid_fraction if self.protocol_name else None}"),
                (f"Sequence name: {self.sequence_name.text if self.sequence_name else None}", f"Valid fraction: {self.sequence_name.valid_fraction if self.sequence_name else None}"),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)
        series_description = SeriesTextFeature.from_values(
            get_tag_value(ds, "SeriesDescription", None) for ds in datasets
        )
        protocol_name = SeriesTextFeature.from_values(
            get_tag_value(ds, "ProtocolName", None) for ds in datasets
        )
        sequence_name = SeriesTextFeature.from_values(
            get_tag_value(ds, "SequenceName", None) for ds in datasets
        )
        
        return cls(
            series_description=series_description,
            protocol_name=protocol_name,
            sequence_name=sequence_name,
        )
    
    def flatten(self) -> dict[str, Optional[str]]:
        return {
            "series_description": self.series_description.text if self.series_description else None,
            "protocol_name": self.protocol_name.text if self.protocol_name else None,
            "sequence_name": self.sequence_name.text if self.sequence_name else None,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["series_description", "protocol_name", "sequence_name"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]

class AcquisitionFeatures(BaseModel):
    acquisition_time: Optional[SeriesNumericFeature] = None
    series_time: Optional[SeriesNumericFeature] = None
    acquisition_order: Optional[float] = None  # POSIX timestamp

    def __str__(self) -> str:
        return format_section(
            "Acquisition Features",
            [
                ("Acquisition time (median)", self.acquisition_time),
                ("Series time (median)", self.series_time),
                ("Acquisition order (POSIX timestamp)", self.acquisition_order),
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
        datasets = list(datasets)

        datetimes = []

        for ds in datasets:

            # Best case: AcquisitionDateTime
            dt = get_tag_value(ds, "AcquisitionDateTime", None)
            if dt:
                parsed = parse_dicom_datetime(dt)
                if parsed:
                    datetimes.append(parsed)
                continue

            # AcquisitionDate + AcquisitionTime
            date = get_tag_value(ds, "AcquisitionDate", None)
            time = get_tag_value(ds, "AcquisitionTime", None)

            if date and time:
                parsed = parse_dicom_date_time(date, time)
                if parsed:
                    datetimes.append(parsed)
                continue

            # SeriesDate + SeriesTime
            date = get_tag_value(ds, "SeriesDate", None)
            time = get_tag_value(ds, "SeriesTime", None)

            if date and time:
                parsed = parse_dicom_date_time(date, time)
                if parsed:
                    datetimes.append(parsed)
                continue

            # StudyDate + StudyTime
            date = get_tag_value(ds, "StudyDate", None)
            time = get_tag_value(ds, "StudyTime", None)

            if date and time:
                parsed = parse_dicom_date_time(date, time)
                if parsed:
                    datetimes.append(parsed)

        # Representative timestamp
        acquisition_order = None
        if datetimes:
            datetimes_sorted = sorted(datetimes)
            median_dt = datetimes_sorted[len(datetimes_sorted) // 2]
            acquisition_order = median_dt.timestamp()

        # Keep numeric features for analysis/debug
        acq_times = [
            parse_dicom_time(get_tag_value(ds, "AcquisitionTime", None))
            for ds in datasets
        ]
        acq_feature = SeriesNumericFeature.from_values(acq_times)

        series_times = [
            parse_dicom_time(get_tag_value(ds, "SeriesTime", None))
            for ds in datasets
        ]
        series_feature = SeriesNumericFeature.from_values(series_times)

        return cls(
            acquisition_time=acq_feature,
            series_time=series_feature,
            acquisition_order=acquisition_order,
        )
    
    def flatten(self) -> dict[str, Optional[float]]:
        return {
            "acquisition_time": self.acquisition_time.value if self.acquisition_time else None,
            "series_time": self.series_time.value if self.series_time else None,
            "acquisition_order": self.acquisition_order,
        }
    
    @staticmethod
    def get_column_headers(prefix: str = "") -> list[str]:
        headers = ["acquisition_time", "series_time", "acquisition_order"]
        return [f"{prefix}_{h}" if prefix else h for h in headers]

class SeriesFeatures(BaseModel):
    series_uid: str
    study_uid: str
    modality: Optional[SeriesCategoricalFeature] = None
    series_number: Optional[SeriesNumericFeature] = None

    num_instances: int
    num_unique_slices: int
    num_volumes: int

    manufacturer: Optional[SeriesCategoricalFeature] = None
    model: Optional[SeriesCategoricalFeature] = None
    field_strength: Optional[SeriesNumericFeature] = None
    
    geometry: Optional[GeometryFeatures] = None
    temporal: Optional[TemporalFeatures] = None
    diffusion: Optional[DiffusionFeatures] = None
    perfusion: Optional[PerfusionFeatures] = None
    sequence: Optional[SequenceFeatures] = None
    multi_echo: Optional[MultiEchoFeatures] = None
    spatial: Optional[SpatialFeatures] = None
    encoding: Optional[EncodingFeatures] = None
    contrast: Optional[ContrastFeatures] = None
    image_type: Optional[ImageTypeFeature] = None
    text: Optional[TextualMetadataFeatures] = None
    acquisition: Optional[AcquisitionFeatures] = None
    
    def __str__(self) -> str:
        lines = [
            "Series Features",
            "---------------",
            f"  Series UID: {self.series_uid}",
            f"  Study UID: {self.study_uid}",
            f"  Modality: {humanize_value(self.modality)}",
            f"  Number of instances: {self.num_instances}",
            f"  Unique slices: {self.num_unique_slices}",
            f"  Volumes: {self.num_volumes}",
            f"  Manufacturer: {humanize_value(self.manufacturer)}",
            f"  Model: {humanize_value(self.model)}",
            f"  Field strength: {humanize_value(self.field_strength)}",
            f"  Series number: {humanize_value(self.series_number)}",
        ]

        def add_block(label: str, obj: Any) -> None:
            if obj is None:
                lines.append(f"  {label}: None")
                return
            lines.append(f"  {label}:")
            lines.append(indent_block(str(obj), 4))

        add_block("Geometry", self.geometry)
        add_block("Temporal", self.temporal)
        add_block("Diffusion", self.diffusion)
        add_block("Perfusion", self.perfusion)
        add_block("Sequence", self.sequence)
        add_block("Multi-echo", self.multi_echo)
        add_block("Spatial", self.spatial)
        add_block("Encoding", self.encoding)
        add_block("Contrast", self.contrast)
        add_block("Image type", self.image_type)
        add_block("Text", self.text)
        add_block("Acquisition", self.acquisition)
        
        return "\n".join(lines)
    
    @staticmethod
    def get_column_headers() -> list[str]:
        return [
            "subject_id",
            "session_id",
            "series_id",
            "data"
        ]
    
    @classmethod
    def from_sqlite(cls, conn: sqlite3.Connection, subject_id: Optional[str] = None, session_id: Optional[str] = None, series_id: Optional[str] = None) -> list[Self]:
        """
        Load series features from the database for a specific subject/session/series combination.
        Returns a list of SeriesFeatures instances or raises ValueError if not found.
        """
        cursor = conn.cursor()
        query = "SELECT data FROM series_features WHERE 1=1"
        params = []
        if subject_id is not None:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if series_id is not None:
            query += " AND series_id = ?"
            params.append(series_id)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No features found for subject {subject_id}, session {session_id}, series {series_id}")
        results = []
        for row in rows:
            data_json = row[0]
            data = json.loads(data_json)
            results.append(cls.model_validate(data))
        return results
    
    @classmethod
    def from_sqlite_all(cls, conn: sqlite3.Connection) -> list[Self]:
        """
        Load all series features from the database, returning a list of SeriesFeatures instances.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM series_features")
        rows = cursor.fetchall()
        print(f"Loaded {len(rows)} series feature entries from the database")
        results = []
        for row in rows:
            data_json = row[0]
            data = json.loads(data_json)
            features = cls.model_validate(data)
            results.append(features)
        return results
    
    @classmethod
    def from_dicom_series(cls, series_dir: Path) -> Self:
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_dir}")
        
        max_workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            datasets = list(executor.map(read_dicom_header, dicom_files))

        datasets = [ds for ds in datasets if ds is not None]

        if not datasets:
            raise ValueError("No readable DICOM instances")

        # -----------------------
        # Core identifiers
        # -----------------------
        series_uid = get_tag_value(datasets[0], "SeriesInstanceUID")
        
        if not series_uid:
            raise ValueError("Missing SeriesInstanceUID in DICOM metadata")
        study_uid = get_tag_value(datasets[0], "StudyInstanceUID")
        if not study_uid:
            raise ValueError("Missing StudyInstanceUID in DICOM metadata")
        
        series_number = SeriesNumericFeature.from_values(
            get_tag_value(ds, "SeriesNumber", None) for ds in datasets
        )

        # -----------------------
        # Simple counts
        # -----------------------
        num_instances = len(datasets)

        slice_locations = [
            get_tag_value(ds, "SliceLocation", None) for ds in datasets
        ]
        num_unique_slices = len({v for v in slice_locations if v is not None})

        temporal_positions = [
            get_tag_value(ds, "TemporalPositionIdentifier", None) for ds in datasets
        ]
        num_volumes = len({v for v in temporal_positions if v is not None}) or 1

        # -----------------------
        # Manufacturer / system
        # -----------------------
        manufacturer = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "Manufacturer", None) for ds in datasets
        )

        model = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ManufacturerModelName", None) for ds in datasets
        )

        field_strength = SeriesNumericFeature.from_values(
            get_tag_value(ds, "MagneticFieldStrength", None) for ds in datasets
        )

        modality = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "Modality", None) for ds in datasets
        )
        
        # -----------------------
        # Geometry features
        # -----------------------
        geometry = GeometryFeatures.from_datasets(datasets)

        # -----------------------
        # Temporal features
        # -----------------------
        temporal = TemporalFeatures.from_datasets(datasets, geometry=geometry)

        # -----------------------
        # Diffusion features
        # -----------------------
        diffusion = DiffusionFeatures.from_datasets(datasets)
        
        # -----------------------
        # Perfusion features
        # -----------------------
        perfusion = PerfusionFeatures.from_datasets(datasets)
        
        # -----------------------
        # Sequence features
        # -----------------------
        sequence = SequenceFeatures.from_datasets(datasets)
        
        # -----------------------
        # Multi-echo features
        # -----------------------
        multi_echo = MultiEchoFeatures.from_datasets(datasets)

        # -----------------------
        # Spatial features
        # -----------------------
        spatial = SpatialFeatures.from_datasets(datasets)
        
        # -----------------------
        # Encoding features
        # -----------------------
        encoding = EncodingFeatures.from_datasets(datasets)
        
        # -----------------------
        # Contrast features
        # -----------------------
        contrast = ContrastFeatures.from_datasets(datasets)
        
        # -----------------------
        # ImageType features
        # -----------------------
        image_type = ImageTypeFeature.from_datasets(datasets)

        # -----------------------
        # Textual metadata
        # -----------------------
        text = TextualMetadataFeatures.from_datasets(datasets)
        
        # -----------------------
        # Acquisition features
        # -----------------------
        acquisition = AcquisitionFeatures.from_datasets(datasets)        

        return cls(
            series_uid=series_uid,
            study_uid=study_uid,
            modality=modality,
            series_number=series_number,
            num_instances=num_instances,
            num_unique_slices=num_unique_slices,
            num_volumes=num_volumes,
            manufacturer=manufacturer,
            model=model,
            field_strength=field_strength,
            geometry=geometry,
            temporal=temporal,
            spatial=spatial,
            diffusion=diffusion,
            image_type=image_type,
            perfusion=perfusion,
            sequence=sequence,
            multi_echo=multi_echo,
            encoding=encoding,
            contrast=contrast,
            text=text,
            acquisition=acquisition,
        )
        
    @classmethod
    def from_nifti_series(cls, nifti_path: str | Path) -> Self:
        """Alternative constructor to build SeriesFeatures from a NIfTI file with sidecar JSON metadata"""
        # json_path = Path(nifti_path).with_suffix(".json")
        # if not json_path.exists():
        #     raise ValueError(f"Expected JSON sidecar {json_path} not found for NIfTI file {nifti_path}")
        # return cls.from_json(json_path)
        
        raise NotImplementedError("NIfTI parsing not implemented yet - would require defining a standard JSON schema for the extracted features and ensuring the NIfTI sidecar JSON files are generated in the expected format during conversion from DICOM.")

    @classmethod
    def from_json(cls, json_path: str | Path) -> Self:
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)
    
    def flatten(self) -> dict[str, Optional[str | int | float | bool | tuple]]:
        """Flatten all features into a single dictionary for easy analysis"""
        flat = {
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "modality": self.modality.value if self.modality else None,
            "series_number": self.series_number.value if self.series_number else None,
            "num_instances": self.num_instances,
            "num_unique_slices": self.num_unique_slices,
            "num_volumes": self.num_volumes,
            "manufacturer": self.manufacturer.value if self.manufacturer else None,
            "model": self.model.value if self.model else None,
            "field_strength": self.field_strength.value if self.field_strength else None,
        }
        flat.update(self.geometry.flatten() if self.geometry else {})
        flat.update(self.temporal.flatten() if self.temporal else {})
        flat.update(self.diffusion.flatten() if self.diffusion else {})
        flat.update(self.perfusion.flatten() if self.perfusion else {})
        flat.update(self.sequence.flatten() if self.sequence else {})
        flat.update(self.multi_echo.flatten() if self.multi_echo else {})
        flat.update(self.spatial.flatten() if self.spatial else {})
        flat.update(self.encoding.flatten() if self.encoding else {})
        flat.update(self.contrast.flatten() if self.contrast else {})
        flat.update(self.image_type.flatten() if self.image_type else {})
        flat.update(self.text.flatten() if self.text else {})
        flat.update(self.acquisition.flatten() if self.acquisition else {})
        
        return flat
    
    def to_sqlite(self, conn: sqlite3.Connection, subject_id: str, session_id: Optional[str] = None) -> None:
        """
        Store the series features in a SQLite database, using a composite key of (subject_id, session_id, series_uid, series_description) for upsert operations.
        """
        # Insert or replace the series features as a JSON blob
        insert_query = """
INSERT INTO series_features (subject_id, session_id, series_id, series_description, data)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(subject_id, session_id, series_id, series_description) DO UPDATE SET data=excluded.data
"""
        series_description = self.text.series_description.text if self.text and self.text.series_description else None
        conn.execute(insert_query, (subject_id, session_id, self.series_uid, series_description, json.dumps(self.model_dump())))
        conn.commit()