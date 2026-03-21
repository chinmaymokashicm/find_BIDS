"""Feature engineering and preprocessing utilities for series-level ML tables."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    safe_divide,
)

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

DEFAULT_TEXT_KEYWORDS = [
    "adc",
    "asl",
    "bold",
    "cbf",
    "cbv",
    "dwi",
    "dti",
    "epi",
    "fa",
    "fieldmap",
    "flair",
    "gre",
    "localizer",
    "magnitude",
    "mip",
    "mprage",
    "phase",
    "rest",
    "se",
    "swi",
    "t1",
    "t2",
    "trace",
]

RAW_STRUCTURED_COLUMNS = {
    "voxel_size",
    "matrix_size",
    "pixel_spacing",
    "b_values",
    "echo_times",
    "echo_numbers",
    "image_orientation",
}

MERGE_KEY_COLUMNS = ["dataset", "subject_id", "session_id", "series_id"]


def _parse_numeric_sequence(value: object, expected_len: int | None = None) -> list[float]:
    """Parse a scalar or sequence-like value into a list of floats.

    Handles tuple/list inputs and common serialized string forms, returning an empty list for missing values.
    """
    if value is None:
        return []

    if isinstance(value, (tuple, list, np.ndarray, pd.Series)):
        parsed = []
        for item in value:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        if expected_len is not None and len(parsed) < expected_len:
            parsed = parsed + [np.nan] * (expected_len - len(parsed))
        return parsed

    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return []

    text = str(value).strip().lower()
    if text in {"", "none", "nan", "null"}:
        return []

    if "\\" in text and "[" not in text and "(" not in text:
        parts = [p.strip() for p in text.split("\\") if p.strip()]
        parsed = []
        for part in parts:
            try:
                parsed.append(float(part))
            except (TypeError, ValueError):
                continue
        if expected_len is not None and len(parsed) < expected_len:
            parsed = parsed + [np.nan] * (expected_len - len(parsed))
        return parsed

    matches = _FLOAT_PATTERN.findall(text)
    parsed = []
    for match in matches:
        try:
            parsed.append(float(match))
        except (TypeError, ValueError):
            continue

    if expected_len is not None and len(parsed) < expected_len:
        parsed = parsed + [np.nan] * (expected_len - len(parsed))
    return parsed


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Coerce selected columns to numeric dtype in place.

    Non-parsable values are converted to NaN.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _series_dataframe_with_keys(series_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the input table so canonical merge keys are explicit columns.

    MultiIndex rows are reset and subject/session/series are renamed to subject_id/session_id/series_id.
    """
    df = series_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name:
        df = df.reset_index()

    rename_map = {
        "subject": "subject_id",
        "session": "session_id",
        "series": "series_id",
    }
    return df.rename(columns=rename_map)


def split_voxel_size(df: pd.DataFrame) -> pd.DataFrame:
    """Split voxel_size into voxel_x, voxel_y, and voxel_z numeric columns.

    Returns the DataFrame unchanged when voxel_size is not present.
    """
    if "voxel_size" not in df.columns:
        return df

    vox = df["voxel_size"].apply(lambda v: _parse_numeric_sequence(v, expected_len=3))
    df["voxel_x"] = vox.str[0]
    df["voxel_y"] = vox.str[1]
    df["voxel_z"] = vox.str[2]
    return df


def split_pixel_spacing(df: pd.DataFrame) -> pd.DataFrame:
    """Split pixel_spacing into pixel_spacing_x and pixel_spacing_y numeric columns.

    Returns the DataFrame unchanged when pixel_spacing is missing.
    """
    if "pixel_spacing" not in df.columns:
        return df

    spacing = df["pixel_spacing"].apply(lambda v: _parse_numeric_sequence(v, expected_len=2))
    df["pixel_spacing_x"] = spacing.str[0]
    df["pixel_spacing_y"] = spacing.str[1]
    return df


def summarize_numeric_sequence_column(df: pd.DataFrame, source_col: str, prefix: str) -> None:
    """Create count/min/max/mean summaries from a sequence-valued source column.

    Empty or missing sequences are represented with NaN summary values.
    """
    if source_col not in df.columns:
        return

    parsed = df[source_col].apply(_parse_numeric_sequence)
    df[f"{prefix}_count"] = parsed.apply(lambda x: float(len(x)) if x else np.nan)
    df[f"{prefix}_min"] = parsed.apply(lambda x: float(np.min(x)) if x else np.nan)
    df[f"{prefix}_max"] = parsed.apply(lambda x: float(np.max(x)) if x else np.nan)
    df[f"{prefix}_mean"] = parsed.apply(lambda x: float(np.mean(x)) if x else np.nan)


def voxel_volume(df: pd.DataFrame) -> None:
    """Compute voxel_volume as voxel_x * voxel_y * voxel_z.

    Skips computation if any voxel dimension column is absent.
    """
    if not all(col in df.columns for col in ["voxel_x", "voxel_y", "voxel_z"]):
        return
    df["voxel_volume"] = df["voxel_x"] * df["voxel_y"] * df["voxel_z"]


def slice_gap_ratio(df: pd.DataFrame) -> None:
    """Compute slice_gap_ratio as spacing_between_slices divided by slice_thickness.

    Uses safe_divide and no-ops when required inputs are unavailable.
    """
    if "spacing_between_slices" not in df.columns or "slice_thickness" not in df.columns:
        return
    df["slice_gap_ratio"] = safe_divide(df["spacing_between_slices"], df["slice_thickness"])


def matrix_area(df: pd.DataFrame) -> None:
    """Compute in-plane matrix area from rows and columns.

    Leaves the DataFrame unchanged if either input column is missing.
    """
    if "rows" not in df.columns or "columns" not in df.columns:
        return
    df["matrix_area"] = df["rows"] * df["columns"]


def volumes_per_second(df: pd.DataFrame) -> None:
    """Estimate acquisition rate as volumes per second.

    Assumes repetition_time is milliseconds and uses safe_divide to handle invalid divisions.
    """
    if "num_volumes" not in df.columns or "repetition_time" not in df.columns:
        return
    df["volumes_per_second"] = safe_divide(df["num_volumes"], df["repetition_time"] / 1000)


def scan_duration_seconds(df: pd.DataFrame) -> None:
    """Estimate total scan duration in seconds from num_volumes and repetition_time.

    Assumes repetition_time is stored in milliseconds.
    """
    if "num_volumes" not in df.columns or "repetition_time" not in df.columns:
        return
    df["scan_duration_seconds"] = (df["num_volumes"] * df["repetition_time"]) / 1000


def diffusion_volume_fraction(df: pd.DataFrame) -> None:
    """Compute the fraction of diffusion volumes among all volumes.

    Uses safe_divide and skips rows when required columns are absent.
    """
    if "num_diffusion_volumes" not in df.columns or "num_volumes" not in df.columns:
        return
    df["diffusion_volume_fraction"] = safe_divide(df["num_diffusion_volumes"], df["num_volumes"])


def b0_volume_fraction(df: pd.DataFrame) -> None:
    """Compute the fraction of b0 volumes among all volumes.

    Uses safe_divide and no-ops if required source columns are missing.
    """
    if "num_b0" not in df.columns or "num_volumes" not in df.columns:
        return
    df["b0_volume_fraction"] = safe_divide(df["num_b0"], df["num_volumes"])


def echo_train_efficiency(df: pd.DataFrame) -> None:
    """Compute echo_train_efficiency as echo_train_length normalized by num_volumes.

    Uses safe_divide and skips computation when source columns are unavailable.
    """
    if "echo_train_length" not in df.columns or "num_volumes" not in df.columns:
        return
    df["echo_train_efficiency"] = safe_divide(df["echo_train_length"], df["num_volumes"])


def bvals_max(df: pd.DataFrame) -> None:
    """Extract the maximum numeric b-value from b_values into bvals_max.

    Missing or unparsable entries are mapped to NaN.
    """
    if "b_values" not in df.columns:
        return

    def parse_bvals_max(bvals: object) -> float:
        parsed = _parse_numeric_sequence(bvals)
        return float(np.max(parsed)) if parsed else np.nan

    df["bvals_max"] = df["b_values"].apply(parse_bvals_max)


def tr_te_ratio(df: pd.DataFrame) -> None:
    """Compute tr_te_ratio as repetition_time divided by echo_time.

    Uses safe_divide and no-ops when either input column is missing.
    """
    if "repetition_time" not in df.columns or "echo_time" not in df.columns:
        return
    df["tr_te_ratio"] = safe_divide(df["repetition_time"], df["echo_time"])


def voxel_anisotropy(df: pd.DataFrame) -> None:
    """Compute voxel anisotropy as std/mean across voxel dimensions per row.

    Uses safe_divide to avoid invalid ratios when means are zero or missing.
    """
    if "voxel_x" not in df.columns or "voxel_y" not in df.columns or "voxel_z" not in df.columns:
        return
    dims = df[["voxel_x", "voxel_y", "voxel_z"]]
    df["voxel_anisotropy"] = safe_divide(dims.std(axis=1), dims.mean(axis=1))


def slice_per_second(df: pd.DataFrame) -> None:
    """Compute slices_per_second from num_slices and repetition_time.

    Assumes repetition_time is milliseconds and uses safe_divide for robustness.
    """
    if "num_slices" not in df.columns or "repetition_time" not in df.columns:
        return
    df["slices_per_second"] = safe_divide(df["num_slices"], df["repetition_time"] / 1000)


def add_log_features(df: pd.DataFrame) -> None:
    """Add log1p-transformed versions of selected skewed numeric features.

    Values are clipped to non-negative before transformation.
    """
    for col in [
        "repetition_time",
        "echo_time",
        "voxel_volume",
        "scan_duration_seconds",
    ]:
        if col in df.columns:
            clipped = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[f"log_{col}"] = np.log1p(clipped)


def add_missing_indicators(df: pd.DataFrame) -> None:
    """Add binary missingness indicators for numeric and engineered numeric features.

    Each generated column is named <feature>_missing.
    """
    candidate_numeric = set(NUMERIC_FEATURES) | {
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "voxel_volume",
        "slice_gap_ratio",
        "matrix_area",
        "volumes_per_second",
        "scan_duration_seconds",
        "diffusion_volume_fraction",
        "b0_volume_fraction",
        "echo_train_efficiency",
        "bvals_max",
        "tr_te_ratio",
        "voxel_anisotropy",
        "slices_per_second",
        "bvals_count",
        "bvals_min",
        "bvals_mean",
        "echo_times_count",
        "echo_times_min",
        "echo_times_max",
        "echo_times_mean",
        "echo_numbers_count",
        "echo_numbers_min",
        "echo_numbers_max",
        "echo_numbers_mean",
    }
    for col in sorted(candidate_numeric):
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)


def normalize_boolean_features(df: pd.DataFrame) -> None:
    """Normalize boolean-like feature columns into 0/1 numeric values.

    Supports common string encodings and preserves existing numeric columns via coercion.
    """
    mapping = {
        True: 1.0,
        False: 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }
    for col in BOOLEAN_FEATURES:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        normalized = df[col].astype(str).str.strip().str.lower().map(mapping)
        df[col] = normalized.astype(float)


def add_text_features(df: pd.DataFrame, keywords: list[str] | None = None) -> None:
    """Derive lightweight text features from configured text columns.

    Produces character/token counts and keyword-presence flags after normalization.
    """
    present_text_cols = [col for col in TEXT_FEATURES if col in df.columns]
    if not present_text_cols:
        return

    keywords = keywords or DEFAULT_TEXT_KEYWORDS
    cleaned = []
    for col in present_text_cols:
        series = (
            df[col]
            .astype("string")
            .fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.strip()
        )
        cleaned.append(series)

    combined = cleaned[0]
    for series in cleaned[1:]:
        combined = combined.str.cat(series, sep=" ")
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()

    df["text_char_count"] = combined.str.len().astype(float)
    df["text_token_count"] = combined.apply(lambda x: float(len(_TOKEN_PATTERN.findall(x))))

    for kw in keywords:
        df[f"kw_{kw}"] = combined.str.contains(rf"\\b{re.escape(kw)}\\b", regex=True).astype(float)


def normalize_categorical_features(df: pd.DataFrame) -> None:
    """Normalize known categorical columns to a clean lowercase representation.

    Null-like values are standardized to the literal category "missing".
    """
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.lower()
                .str.strip()
                .replace({"nan": "missing", "none": "missing", "na": "missing"})
                .fillna("missing")
            )


def collapse_rare_categories(df: pd.DataFrame, min_count: int = 20, top_k: int = 40) -> None:
    """Collapse infrequent categorical levels into a shared "rare" bucket.

    Categories are retained only if they satisfy min_count and top_k constraints.
    """
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        keep = set(counts[counts >= min_count].head(top_k).index)
        df[col] = df[col].where(df[col].isin(keep), other="rare")


def drop_unhelpful_columns(
    df: pd.DataFrame,
    drop_raw_text: bool = True,
    sparse_threshold: float = 0.98,
) -> pd.DataFrame:
    """Drop raw structured/text columns and low-value features based on simple heuristics.

    Columns that are highly sparse or constant are removed, while merge key columns are preserved.
    """
    drop_cols = {c for c in RAW_STRUCTURED_COLUMNS if c in df.columns}
    if drop_raw_text:
        drop_cols.update({c for c in TEXT_FEATURES if c in df.columns})

    for col in df.columns:
        if col in MERGE_KEY_COLUMNS:
            continue
        non_null_frac = float(df[col].notna().mean()) if len(df) else 0.0
        nunique = int(df[col].nunique(dropna=True))
        if non_null_frac <= (1.0 - sparse_threshold):
            drop_cols.add(col)
        if nunique <= 1:
            drop_cols.add(col)

    return df.drop(columns=sorted(drop_cols), errors="ignore")


def _impute_numeric_median(df: pd.DataFrame) -> None:
    """Impute missing numeric values in place using per-column medians.

    Columns with all-missing values fall back to 0.0.
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        median = df[col].median()
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)


def _one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode available categorical feature columns.

    Returns the original DataFrame unchanged if no categorical columns are present.
    """
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=np.int8)


def prepare_features_for_modeling(
    df: pd.DataFrame,
    one_hot_encode: bool = True,
    drop_raw_text: bool = True,
) -> pd.DataFrame:
    """Run end-to-end preprocessing to convert raw series-level features into model-ready tabular features.

    The pipeline includes parsing, engineered features, normalization, pruning, imputation, and optional one-hot encoding.
    """
    df = _series_dataframe_with_keys(df)

    _coerce_numeric_columns(df, list(NUMERIC_FEATURES) + ["field_strength"])

    df = split_voxel_size(df)
    df = split_pixel_spacing(df)

    summarize_numeric_sequence_column(df, source_col="b_values", prefix="bvals")
    summarize_numeric_sequence_column(df, source_col="echo_times", prefix="echo_times")
    summarize_numeric_sequence_column(df, source_col="echo_numbers", prefix="echo_numbers")

    voxel_volume(df)
    slice_gap_ratio(df)
    matrix_area(df)
    volumes_per_second(df)
    scan_duration_seconds(df)
    diffusion_volume_fraction(df)
    b0_volume_fraction(df)
    echo_train_efficiency(df)
    bvals_max(df)
    tr_te_ratio(df)
    voxel_anisotropy(df)
    slice_per_second(df)

    add_text_features(df)
    normalize_boolean_features(df)
    _coerce_numeric_columns(df, ["field_strength"])

    add_log_features(df)
    add_missing_indicators(df)
    normalize_categorical_features(df)
    collapse_rare_categories(df)

    df = drop_unhelpful_columns(df, drop_raw_text=drop_raw_text)
    _impute_numeric_median(df)

    if one_hot_encode:
        df = _one_hot_encode_categoricals(df)

    return df
