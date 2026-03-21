"""
Module for ML classification using manually-checked heuristic scores as training data.
"""
from .utils import (
    TIER1_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    TEXT_FEATURES,
    safe_divide
    )

import numpy as np
import pandas as pd

def split_voxel_size(df: pd.DataFrame) -> pd.DataFrame:
    def parse(v):
        if pd.isna(v):
            return [np.nan, np.nan, np.nan]
        parts = str(v).split("\\")
        if len(parts) != 3:
            return [np.nan, np.nan, np.nan]
        return list(map(float, parts))

    vox = df["voxel_size"].apply(parse)

    df["voxel_x"] = vox.str[0]
    df["voxel_y"] = vox.str[1]
    df["voxel_z"] = vox.str[2]

    return df

def voxel_volume(df: pd.DataFrame) -> None:
    if not all(col in df.columns for col in ["voxel_x", "voxel_y", "voxel_z"]):
        raise ValueError("DataFrame must contain 'voxel_x', 'voxel_y', and 'voxel_z' columns.")
    df["voxel_volume"] = df["voxel_x"] * df["voxel_y"] * df["voxel_z"]

def slice_gap_ratio(df: pd.DataFrame) -> None:
    if "spacing_between_slices" not in df.columns or "slice_thickness" not in df.columns:
        raise ValueError("DataFrame must contain 'spacing_between_slices' and 'slice_thickness' columns.")
    df["slice_gap_ratio"] = safe_divide(df["spacing_between_slices"], df["slice_thickness"])

def matrix_area(df: pd.DataFrame) -> None:
    if "rows" not in df.columns or "columns" not in df.columns:
        raise ValueError("DataFrame must contain 'rows' and 'columns' columns.")
    df["matrix_area"] = df["rows"] * df["columns"]

def volumes_per_second(df: pd.DataFrame) -> None:
    if "num_volumes" not in df.columns or "repetition_time" not in df.columns:
        raise ValueError("DataFrame must contain 'num_volumes' and 'repetition_time' columns.")
    df["volumes_per_second"] = df["num_volumes"] / (df["repetition_time"] / 1000)

def scan_duration_seconds(df: pd.DataFrame) -> None:
    if "num_volumes" not in df.columns or "repetition_time" not in df.columns:
        raise ValueError("DataFrame must contain 'num_volumes' and 'repetition_time' columns.")
    df["scan_duration_seconds"] = (df["num_volumes"] * df["repetition_time"]) / 1000

def diffusion_volume_fraction(df: pd.DataFrame) -> None:
    if "num_diffusion_volumes" not in df.columns or "num_volumes" not in df.columns:
        raise ValueError("DataFrame must contain 'num_diffusion_volumes' and 'num_volumes' columns.")
    df["diffusion_volume_fraction"] = df["num_diffusion_volumes"] / df["num_volumes"]

def b0_volume_fraction(df: pd.DataFrame) -> None:
    if "num_b0" not in df.columns or "num_volumes" not in df.columns:
        raise ValueError("DataFrame must contain 'num_b0' and 'num_volumes' columns.")
    df["b0_volume_fraction"] = df["num_b0"] / df["num_volumes"]

def echo_train_efficiency(df: pd.DataFrame) -> None:
    if "echo_train_length" not in df.columns or "num_volumes" not in df.columns:
        raise ValueError("DataFrame must contain 'echo_train_length' and 'num_volumes' columns.")
    df["echo_train_efficiency"] = df["echo_train_length"] / df["num_volumes"]

def bvals_max(df: pd.DataFrame) -> None:
    if "b_values" not in df.columns:
        raise ValueError("DataFrame must contain 'b_values' column.")
    def parse_bvals(bvals):
        if pd.isna(bvals):
            return np.nan
        try:
            bvals_list = list(map(float, str(bvals).split("\\")))
            return max(bvals_list)
        except Exception:
            return np.nan
    df["bvals_max"] = df["b_values"].apply(parse_bvals)
            
def tr_te_ratio(df: pd.DataFrame) -> None:
    if "repetition_time" not in df.columns or "echo_time" not in df.columns:
        raise ValueError("DataFrame must contain 'repetition_time' and 'echo_time' columns.")
    df["tr_te_ratio"] = safe_divide(df["repetition_time"], df["echo_time"])

def voxel_anisotropy(df: pd.DataFrame) -> None:
    """Calculate voxel anisotropy as the coefficient of variation of the voxel dimensions."""
    if "voxel_x" not in df.columns or "voxel_y" not in df.columns or "voxel_z" not in df.columns:
        raise ValueError("DataFrame must contain 'voxel_x', 'voxel_y', and 'voxel_z' columns.")
    df["voxel_anisotropy"] = df[["voxel_x", "voxel_y", "voxel_z"]].std(axis=1) / df[["voxel_x", "voxel_y", "voxel_z"]].mean(axis=1)
    
def slice_per_second(df: pd.DataFrame) -> None:
    if "num_slices" not in df.columns or "repetition_time" not in df.columns:
        raise ValueError("DataFrame must contain 'num_slices' and 'repetition_time' columns.")
    df["slices_per_second"] = safe_divide(df["num_slices"], df["repetition_time"] / 1000)
    
def add_log_features(df: pd.DataFrame) -> None:
    for col in [
        "repetition_time",
        "echo_time",
        "voxel_volume",
        "scan_duration_seconds",
    ]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

def add_missing_indicators(df: pd.DataFrame) -> None:
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    
def prepare_features_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    df = split_voxel_size(df)
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
    add_log_features(df)
    return df