"""Core ML pipeline workflow functions for merging labels and creating train/val/test splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engineer import MERGE_KEY_COLUMNS, _series_dataframe_with_keys, prepare_features_for_modeling

NON_FEATURE_COLUMNS = {
    "dataset",
    "subject",
    "session",
    "series",
    "subject_id",
    "session_id",
    "series_id",
    "series_description",
    "inferred_datatype",
    "datatype_confidence",
    "inferred_suffix",
    "suffix_confidence",
    "is_derived",
    "derived_confidence",
    "label",
    "min_confidence",
    "label_confidence",
    "confidence_tier",
    "split",
}


def merge_with_heuristic_scores(
    series_features_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    preprocess_series_features: bool = True,
) -> pd.DataFrame:
    """Merge series features with heuristic labels on canonical dataset/subject/session/series keys.

    Optionally preprocesses feature columns first and deduplicates heuristic rows by best min_confidence.
    """
    if preprocess_series_features:
        series_df = prepare_features_for_modeling(series_features_df)
    else:
        series_df = _series_dataframe_with_keys(series_features_df)

    heuristics_df = heuristic_scores_df.copy()
    if isinstance(heuristics_df.index, pd.MultiIndex):
        heuristics_df = heuristics_df.reset_index()
    elif heuristics_df.index.name:
        heuristics_df = heuristics_df.reset_index()

    missing_keys = [k for k in MERGE_KEY_COLUMNS if k not in series_df.columns or k not in heuristics_df.columns]
    if missing_keys:
        raise ValueError(f"Missing merge key columns: {missing_keys}")

    if "min_confidence" in heuristics_df.columns:
        heuristics_df = heuristics_df.sort_values("min_confidence", ascending=False)
    heuristics_df = heuristics_df.drop_duplicates(subset=MERGE_KEY_COLUMNS, keep="first")

    return series_df.merge(heuristics_df, on=MERGE_KEY_COLUMNS, how="inner")


def assign_confidence_tier(
    df: pd.DataFrame,
    high_threshold: float = 0.9,
    medium_threshold: float = 0.6,
) -> pd.DataFrame:
    """Compute label_confidence from datatype and suffix confidence, then assign low/medium/high tiers.

    Tiering intentionally excludes derived confidence.
    """
    result = df.copy()
    if "datatype_confidence" not in result.columns or "suffix_confidence" not in result.columns:
        raise ValueError("DataFrame must contain 'datatype_confidence' and 'suffix_confidence'.")

    datatype_conf = pd.to_numeric(result["datatype_confidence"], errors="coerce")
    suffix_conf = pd.to_numeric(result["suffix_confidence"], errors="coerce")

    combined_conf = pd.concat([datatype_conf, suffix_conf], axis=1).min(axis=1, skipna=True)
    result["label_confidence"] = combined_conf

    tiers = np.where(
        combined_conf >= high_threshold,
        "high",
        np.where(combined_conf >= medium_threshold, "medium", "low"),
    )
    result["confidence_tier"] = pd.Categorical(tiers, categories=["low", "medium", "high"], ordered=True)
    return result


def assign_random_train_val_test_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
    stratify_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Assign reproducible random train/val/test labels, with optional stratification.

    Split fractions must sum to 1.0 and the resulting split column is returned as categorical.
    """
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    result = df.copy()
    result["split"] = ""

    if stratify_columns is None:
        stratify_columns = ["confidence_tier"] if "confidence_tier" in result.columns else []

    for col in stratify_columns:
        if col not in result.columns:
            raise ValueError(f"Stratification column '{col}' not found in DataFrame.")

    rng = np.random.default_rng(random_state)
    if stratify_columns:
        grouped = result.groupby(stratify_columns, dropna=False).groups.values()
    else:
        grouped = [result.index.to_numpy()]

    for group_idx in grouped:
        group_idx = np.array(group_idx)
        shuffled = rng.permutation(group_idx)
        n = len(shuffled)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        n_test = n - n_train - n_val

        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:n_train + n_val + n_test]

        result.loc[train_idx, "split"] = "train"
        result.loc[val_idx, "split"] = "val"
        result.loc[test_idx, "split"] = "test"

    result["split"] = pd.Categorical(result["split"], categories=["train", "val", "test"], ordered=False)
    return result


def prepare_train_val_test_sets(
    series_features_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    random_state: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    high_conf_threshold: float = 0.9,
    medium_conf_threshold: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """Build merged labeled data, assign confidence tiers, and generate train/val/test subsets.

    Returns a dictionary containing the full merged table and each split as separate DataFrames.
    """
    merged = merge_with_heuristic_scores(
        series_features_df=series_features_df,
        heuristic_scores_df=heuristic_scores_df,
        preprocess_series_features=True,
    )

    merged = assign_confidence_tier(
        merged,
        high_threshold=high_conf_threshold,
        medium_threshold=medium_conf_threshold,
    )

    merged = assign_random_train_val_test_splits(
        merged,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
        stratify_columns=["confidence_tier"],
    )

    return {
        "all": merged,
        "train": merged[merged["split"] == "train"].copy(),
        "val": merged[merged["split"] == "val"].copy(),
        "test": merged[merged["split"] == "test"].copy(),
    }


def split_features_and_targets(
    df: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labeled DataFrame into feature matrix X and target DataFrame y.

    Known metadata and leakage-prone label columns are removed from X.
    """
    if target_columns is None:
        target_columns = ["inferred_datatype", "inferred_suffix"]

    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    y = df[target_columns].copy()
    drop_cols = set(NON_FEATURE_COLUMNS)
    drop_cols.update(target_columns)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y
