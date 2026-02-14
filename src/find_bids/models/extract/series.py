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
from pathlib import Path, Path
from typing import Optional, Iterable, Iterator, Self, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from pydantic import BaseModel, field_validator, ConfigDict
import pydicom as dicom
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn
import numpy as np
from pydicom.multival import MultiValue

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

def extract_dicom_tokens(text: Optional[str]) -> list[str]:
    if isinstance(text, MultiValue):
        text = multivalue_to_string(text)
    if not isinstance(text, str):
        raise ValueError(f"Expected a string for text analysis, got {type(text)}")
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)
    return _TOKEN_PATTERN.findall(text)

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

class TemporalFeatures(BaseModel):
    repetition_time: Optional[SeriesNumericFeature] = None
    echo_time: Optional[SeriesNumericFeature] = None
    inversion_time: Optional[SeriesNumericFeature] = None
    flip_angle: Optional[SeriesNumericFeature] = None

    num_timepoints: Optional[int] = None
    temporal_variation: Optional[SeriesNumericFeature] = None  # seconds

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
            ],
        )
    
    @classmethod
    def from_datasets(cls, datasets: Iterable[dicom.Dataset]) -> Self:
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

        return cls(
            repetition_time=repetition_time,
            echo_time=echo_time,
            inversion_time=inversion_time,
            flip_angle=flip_angle,
            num_timepoints=num_timepoints,
            temporal_variation=temporal_variation,
        )

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

class PerfusionFeatures(BaseModel):
    perfusion_labeling_type: Optional[SeriesCategoricalFeature] = None
    bolus_arrival_time: Optional[SeriesNumericFeature] = None
    contrast_agent: Optional[SeriesCategoricalFeature] = None
    perfusion_series_type: Optional[SeriesCategoricalFeature] = None

    num_timepoints: Optional[int] = None
    temporal_spacing: Optional[SeriesNumericFeature] = None

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
        
        return cls(
            num_echoes=num_echoes,
            echo_times=unique_echo_times,
            echo_numbers=unique_echo_numbers,
        )
    
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
            pixel_spacing_x = SeriesNumericFeature.from_values(
                get_tag_value(ds, "PixelSpacing", [None, None])[0] if hasattr(ds, "PixelSpacing") else None for ds in datasets # type: ignore
            )
            pixel_spacing_y = SeriesNumericFeature.from_values(
                get_tag_value(ds, "PixelSpacing", [None, None])[1] if hasattr(ds, "PixelSpacing") else None for ds in datasets # type: ignore
            )
            pixel_spacing = (pixel_spacing_x, pixel_spacing_y)

        image_orientation = SeriesCategoricalFeature.from_values(
            get_tag_value(ds, "ImageOrientationPatient", None) for ds in datasets
        )

        return cls(
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            pixel_spacing=pixel_spacing,
            image_orientation=image_orientation,
        )
    
class EncodingFeatures(BaseModel):
    phase_encoding_direction: Optional[SeriesCategoricalFeature] = None
    phase_encoding_axis: Optional[SeriesCategoricalFeature] = None
    echo_spacing: Optional[SeriesNumericFeature] = None
    is_epi: Optional[SeriesBooleanFeature] = None
    
    def __str__(self) -> str:
        return format_section(
            "Encoding Features",
            [
                ("Phase encoding direction", self.phase_encoding_direction),
                ("Phase encoding axis", self.phase_encoding_axis),
                ("Echo spacing", self.echo_spacing),
                ("Is EPI", self.is_epi),
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
        
        # Robust EPI detection (multi-vendor)
        is_epi_flags = []
        for ds in datasets:
            seq_name = (get_tag_value(ds, "SequenceName", "") or "").lower()
            scanning_seq = normalize_category(get_tag_value(ds, "ScanningSequence", None))
            seq_variant = normalize_category(get_tag_value(ds, "SequenceVariant", None))
            vendor = normalize_category(get_tag_value(ds, "Manufacturer", None))
            if vendor is None:
                vendor = "N/A"
            
            # Vendor-agnostic EPI signals
            is_epi_seq = (
                "ep" in seq_name or                    # ep2d, ep3d, epi, ep_bold, etc.
                "echo planar" in seq_name or           # full name
                scanning_seq == "ep" or                # DICOM standard
                "epi" in seq_name                      # generic
            )
            
            # Vendor-specific patterns (fallback)
            vendor_specific = (
                ("ep2d" in seq_name) or                # Siemens
                ("ep" in seq_name and "ge" in vendor.lower()) or  # GE
                ("epifmri" in seq_name) or             # Philips common
                ("ep" in seq_name and "philips" in vendor.lower())
            )
            
            is_epi_flags.append(is_epi_seq or vendor_specific)
        
        is_epi = SeriesBooleanFeature.from_values(is_epi_flags)
        
        return cls(
            phase_encoding_direction=phase_encoding_direction,
            phase_encoding_axis=phase_encoding_axis,
            echo_spacing=echo_spacing,
            is_epi=is_epi,
        )
    
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

        return "\n".join(lines)
    
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
        # Temporal features
        # -----------------------
        temporal = TemporalFeatures.from_datasets(datasets)

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
            temporal=temporal,
            spatial=spatial,
            diffusion=diffusion,
            image_type=image_type,
            text=text,
            perfusion=perfusion,
            sequence=sequence,
            multi_echo=multi_echo,
            encoding=encoding,
            contrast=contrast,
        )

    @classmethod
    def from_json(cls, json_path: str | Path) -> Self:
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)