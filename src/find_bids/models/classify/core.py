"""Core ML pipeline workflow functions for merging labels and creating train/val/test splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .engineer import (
    MERGE_KEY_COLUMNS,
    TEXT_WORKING_COLUMN,
    _series_dataframe_with_keys,
    prepare_features_for_modeling,
)
from .pipeline_helpers import (
    append_train_fitted_field_tfidf_features,
    compute_min_confidence_sample_weights,
    drop_unknown_targets_from_training_splits,
    normalize_localizer_unknown_suffixes,
)

PREFERRED_HEURISTIC_COLUMNS = [
    "inferred_datatype",
    "datatype_confidence",
    "inferred_suffix",
    "suffix_confidence",
    "is_derived",
    "derived_confidence",
    "label",
    "min_confidence",
]

NON_FEATURE_COLUMNS = {
    "data",
    "dataset",
    "subject",
    "session",
    "series",
    "subject_id",
    "study_uid",
    "session_id",
    "series_id",
    "series_description",
    "protocol_name",
    "sequence_name",
    TEXT_WORKING_COLUMN,
    "acquisition_time",
    "series_time",
    "acquisition_order",
    "source_image_sequences",
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

IDENTIFIER_COLUMN_EXACT = {
    "study_uid",
    "series_uid",
    "sop_instance_uid",
    "study_instance_uid",
    "frame_of_reference_uid",
}

IDENTIFIER_COLUMN_SUFFIXES = (
    "_uid",
)

LEAKAGE_COLUMN_PREFIXES = (
    "inferred_",
    "datatype_",
    "suffix_",
    "derived_",
    "label_",
    "confidence_",
)

SampleWeightingMode = Literal["none", "min_confidence"]
UNKNOWN_TARGET_VALUE = "unknown"


@dataclass(frozen=True)
class PreparedSplit:
    """Container for a model-ready split.

    Supports both named access (.X/.y/.sample_weight) and legacy tuple-style
    access via unpacking or integer indexing.
    """

    X: pd.DataFrame
    y: pd.DataFrame
    sample_weight: pd.Series | None = None

    def as_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        if self.sample_weight is None:
            return (self.X, self.y)
        return (self.X, self.y, self.sample_weight)

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return len(self.as_tuple())

    def __getitem__(self, index: int):
        return self.as_tuple()[index]


def _safe_nunique(series: pd.Series) -> int:
    """Compute nunique robustly for object columns that may include unhashable list-like values."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        normalized = series.map(
            lambda v: tuple(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series)) else v
        )
        try:
            return int(normalized.nunique(dropna=True))
        except TypeError:
            return int(series.astype("string").nunique(dropna=True))


def _infer_identifier_columns(columns: pd.Index) -> set[str]:
    """Infer identifier-like columns that should never be exposed in X."""
    identifier_columns: set[str] = set()
    for col in columns:
        if col in IDENTIFIER_COLUMN_EXACT:
            identifier_columns.add(col)
            continue
        if any(col.endswith(suffix) for suffix in IDENTIFIER_COLUMN_SUFFIXES):
            identifier_columns.add(col)
            continue
    return identifier_columns


def merge_with_heuristic_scores(
    series_features_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    preprocess_series_features: bool = True,
    sparse_threshold: float = 0.95,
    preserve_engineered_features: bool = True,
    drop_raw_text: bool = True,
) -> pd.DataFrame:
    """
    Merge series features with heuristic labels on canonical dataset/subject/session/series keys.
    
    Args:
        series_features_df: DataFrame containing features for each series, indexed by dataset/subject/session/series.
        heuristic_scores_df: DataFrame containing heuristic label scores, indexed by dataset/subject/session/series or with those columns.
        preprocess_series_features: Whether to preprocess the series features for modeling pipeline (e.g. flattening, encoding) before merging. If False, raw features will be merged, which may be useful for debugging or analysis but not for modeling.
        sparse_threshold: Fractional sparsity threshold used during feature pruning when preprocessing.
        preserve_engineered_features: Whether to preserve core engineered and keyword features during pruning.
        
    Returns:
        Merged DataFrame containing features and heuristic scores for each series, ready for further processing.
    """
    if preprocess_series_features:
        series_df = prepare_features_for_modeling(
            series_features_df,
            sparse_threshold=sparse_threshold,
            preserve_engineered_features=preserve_engineered_features,
            drop_raw_text=drop_raw_text,
        )
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

    merged = series_df.merge(
        heuristics_df,
        on=MERGE_KEY_COLUMNS,
        how="inner",
        suffixes=("_features", "_heuristics"),
    )

    # Prefer heuristic labels/confidences when feature and heuristic tables overlap.
    for col in PREFERRED_HEURISTIC_COLUMNS:
        heur_col = f"{col}_heuristics"
        feat_col = f"{col}_features"
        if heur_col in merged.columns:
            merged[col] = merged[heur_col]
        elif feat_col in merged.columns:
            merged[col] = merged[feat_col]

    cols_to_drop = [
        c
        for c in merged.columns
        if c.endswith("_features") or c.endswith("_heuristics")
    ]
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)

    required_conf_cols = ["datatype_confidence", "suffix_confidence"]
    missing_conf_cols = [c for c in required_conf_cols if c not in merged.columns]
    if missing_conf_cols:
        raise ValueError(
            "Merged DataFrame is missing required confidence columns after merge: "
            f"{missing_conf_cols}. Check overlap and source inputs."
        )
    return merged


def assign_confidence_tier(
    df: pd.DataFrame,
    high_threshold: float = 0.9,
    medium_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Compute label_confidence from datatype and suffix confidence, then assign low/medium/high tiers.
    
    Args:
        df: DataFrame containing at least 'datatype_confidence' and 'suffix_confidence' columns with numeric confidence scores between 0 and 1.
        high_threshold: Minimum confidence for "high" tier.
        medium_threshold: Minimum confidence for "medium" tier (below high_threshold).
        
    Returns:
        DataFrame with new 'label_confidence' and 'confidence_tier' columns added.
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
    """
    Assign reproducible random train/val/test labels, with optional stratification. 
    Split fractions must sum to 1.0 and the resulting split column is returned as categorical.
    
    Args:
        df: DataFrame to split, indexed by dataset/subject/session/series or with those columns.
        train_fraction: Proportion of data to assign to training set.
        val_fraction: Proportion of data to assign to validation set.
        test_fraction: Proportion of data to assign to test set.
        random_state: Seed for random number generator to ensure reproducibility.
        stratify_columns: Optional list of column names to use for stratified splitting. If None, no stratification is performed. Stratification ensures that the distribution of the specified columns is approximately preserved
        
    Returns:
        DataFrame with new 'split' column containing categorical values 'train', 'val', or 'test'.
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
        groupby_keys: str | list[str] = stratify_columns[0] if len(stratify_columns) == 1 else stratify_columns
        grouped = result.groupby(groupby_keys, dropna=False, observed=False).groups.values()
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
    sparse_threshold: float = 0.95,
    preserve_engineered_features: bool = True,
    drop_raw_text: bool = True,
    add_confidence_tier: bool = True,
    stratify_by_confidence_tier: bool = True,
    stratify_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build merged labeled data, assign confidence tiers, and generate train/val/test subsets.

    Args:
        series_features_df: DataFrame containing features for each series, indexed by dataset/subject/session/series.
        heuristic_scores_df: DataFrame containing heuristic label scores, indexed by dataset/subject/session/series or with those columns.
        random_state: Seed for random number generator to ensure reproducibility of train/val/test splits.
        train_fraction: Proportion of data to assign to training set.
        val_fraction: Proportion of data to assign to validation set.
        test_fraction: Proportion of data to assign to test set.
        high_conf_threshold: Minimum confidence for "high" tier.
        medium_conf_threshold: Minimum confidence for "medium" tier (below high_threshold).
        sparse_threshold: Fractional sparsity threshold used during feature pruning when preprocessing.
        preserve_engineered_features: Whether to preserve core engineered and keyword features during pruning.
        
    Returns:
        Dictionary with keys 'all', 'train', 'val', 'test' containing DataFrames for the full merged dataset and each split subset.
    """
    merged = merge_with_heuristic_scores(
        series_features_df=series_features_df,
        heuristic_scores_df=heuristic_scores_df,
        preprocess_series_features=True,
        sparse_threshold=sparse_threshold,
        preserve_engineered_features=preserve_engineered_features,
        drop_raw_text=drop_raw_text,
    )

    if add_confidence_tier:
        merged = assign_confidence_tier(
            merged,
            high_threshold=high_conf_threshold,
            medium_threshold=medium_conf_threshold,
        )

    effective_stratify_columns = stratify_columns
    if effective_stratify_columns is None:
        if stratify_by_confidence_tier and "confidence_tier" in merged.columns:
            effective_stratify_columns = ["confidence_tier"]
        else:
            effective_stratify_columns = []

    merged = assign_random_train_val_test_splits(
        merged,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
        stratify_columns=effective_stratify_columns,
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
    feature_sparse_threshold: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labeled DataFrame into feature matrix X and target DataFrame y.

    Known metadata and leakage-prone label columns are removed from X, then X is trimmed
    using sparse-threshold and constant-column filtering for ML readiness.
    """
    if not (0.0 <= feature_sparse_threshold <= 1.0):
        raise ValueError("feature_sparse_threshold must be between 0.0 and 1.0")

    if target_columns is None:
        target_columns = ["inferred_datatype", "inferred_suffix"]

    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    y = df[target_columns].copy()
    drop_cols = set(NON_FEATURE_COLUMNS)
    drop_cols.update(_infer_identifier_columns(df.columns))
    drop_cols.update(target_columns)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Trim sparse and constant columns after merge, right before ML handoff.
    if not X.empty:
        missing_fraction = X.isna().mean()
        sparse_cols = [c for c in X.columns if float(missing_fraction[c]) >= feature_sparse_threshold]
        if sparse_cols:
            X = X.drop(columns=sparse_cols)

        constant_cols = [c for c in X.columns if _safe_nunique(X[c]) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)

    return X, y


def _find_leakage_columns(columns: pd.Index) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        if column in NON_FEATURE_COLUMNS:
            flagged.append(column)
            continue
        if column.endswith("_confidence"):
            flagged.append(column)
            continue
        if any(column.startswith(prefix) for prefix in LEAKAGE_COLUMN_PREFIXES):
            flagged.append(column)
    return sorted(set(flagged))


def _validate_feature_matrix(X: pd.DataFrame) -> None:
    leakage_columns = _find_leakage_columns(X.columns)
    if leakage_columns:
        raise ValueError(f"Leakage-prone columns present in X: {leakage_columns}")


def _align_to_training_schema(X: pd.DataFrame, training_columns: pd.Index) -> pd.DataFrame:
    aligned = X.reindex(columns=training_columns)
    missing_columns = [col for col in training_columns if col not in X.columns]
    if missing_columns:
        aligned.loc[:, missing_columns] = aligned.loc[:, missing_columns].fillna(0.0)
    return aligned


def _finalize_feature_target_splits(
    prepared_sets: dict[str, pd.DataFrame],
    target_columns: list[str] | None,
    feature_sparse_threshold: float,
) -> dict[str, PreparedSplit]:
    result: dict[str, PreparedSplit] = {}
    train_X, train_y = split_features_and_targets(
        prepared_sets["train"],
        target_columns=target_columns,
        feature_sparse_threshold=feature_sparse_threshold,
    )
    _validate_feature_matrix(train_X)
    result["train"] = PreparedSplit(X=train_X, y=train_y)

    for split, split_data in prepared_sets.items():
        if split == "train":
            continue
        X, y = split_features_and_targets(
            split_data,
            target_columns=target_columns,
            feature_sparse_threshold=feature_sparse_threshold,
        )
        X = _align_to_training_schema(X, train_X.columns)
        _validate_feature_matrix(X)
        result[split] = PreparedSplit(X=X, y=y)
    return result


def _base_prepared_sets(
    series_features_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    random_state: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    high_conf_threshold: float,
    medium_conf_threshold: float,
    sparse_threshold: float,
    preserve_engineered_features: bool,
    add_tfidf_text_features: bool,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_ngram_range: tuple[int, int],
    tfidf_analyzer: Literal["word", "char", "char_wb"],
    tfidf_sequence_max_features: int,
    add_confidence_tier: bool,
    stratify_by_confidence_tier: bool,
) -> dict[str, pd.DataFrame]:
    """Build merged/split DataFrames and optionally append train-fitted TF-IDF features."""
    heuristic_scores_df = normalize_localizer_unknown_suffixes(heuristic_scores_df)
    prepared_sets = prepare_train_val_test_sets(
        series_features_df=series_features_df,
        heuristic_scores_df=heuristic_scores_df,
        random_state=random_state,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        high_conf_threshold=high_conf_threshold,
        medium_conf_threshold=medium_conf_threshold,
        sparse_threshold=sparse_threshold,
        preserve_engineered_features=preserve_engineered_features,
        drop_raw_text=not add_tfidf_text_features,
        add_confidence_tier=add_confidence_tier,
        stratify_by_confidence_tier=stratify_by_confidence_tier,
        stratify_columns=None,
    )

    if not add_tfidf_text_features:
        return prepared_sets

    field_max_features = {
        "series_description": tfidf_max_features,
        "protocol_name": tfidf_max_features,
        "sequence_name": tfidf_sequence_max_features,
    }
    return append_train_fitted_field_tfidf_features(
        prepared_sets,
        field_max_features=field_max_features,
        min_df=tfidf_min_df,
        ngram_range=tfidf_ngram_range,
        analyzer=tfidf_analyzer,
    )


def ml_prep_pipeline(
    series_features_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    random_state: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    high_conf_threshold: float = 0.9,
    medium_conf_threshold: float = 0.6,
    sparse_threshold: float = 0.95,
    preserve_engineered_features: bool = True,
    feature_sparse_threshold: float = 0.98,
    target_columns: list[str] | None = None,
    add_tfidf_text_features: bool = True,
    tfidf_max_features: int = 150,
    tfidf_min_df: int = 2,
    tfidf_ngram_range: tuple[int, int] = (3, 5),
    tfidf_analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    tfidf_sequence_max_features: int = 60,
    sample_weighting_mode: SampleWeightingMode = "none",
    min_confidence_weight_floor: float = 0.3,
    drop_unknown_in_train_val: bool = True,
    unknown_target_value: str = UNKNOWN_TARGET_VALUE,
) -> dict[str, PreparedSplit]:
    """Single primary ML prep entry point.

    Use `sample_weighting_mode='none'` for unweighted splits and
    `sample_weighting_mode='min_confidence'` for floor-clipped confidence weights.
    """
    target_columns = target_columns or ["inferred_datatype", "inferred_suffix"]
    prepared_sets = _base_prepared_sets(
        series_features_df=series_features_df,
        heuristic_scores_df=heuristic_scores_df,
        random_state=random_state,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        high_conf_threshold=high_conf_threshold,
        medium_conf_threshold=medium_conf_threshold,
        sparse_threshold=sparse_threshold,
        preserve_engineered_features=preserve_engineered_features,
        add_tfidf_text_features=add_tfidf_text_features,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_analyzer=tfidf_analyzer,
        tfidf_sequence_max_features=tfidf_sequence_max_features,
        add_confidence_tier=True,
        stratify_by_confidence_tier=True,
    )

    if drop_unknown_in_train_val:
        prepared_sets = drop_unknown_targets_from_training_splits(
            prepared_sets,
            target_columns=target_columns,
            unknown_value=unknown_target_value,
        )

    xy_sets = _finalize_feature_target_splits(
        prepared_sets,
        target_columns=target_columns,
        feature_sparse_threshold=feature_sparse_threshold,
    )

    if sample_weighting_mode == "none":
        return xy_sets

    if sample_weighting_mode != "min_confidence":
        raise ValueError(f"Unsupported sample_weighting_mode: {sample_weighting_mode}")

    weighted_sets: dict[str, PreparedSplit] = {}
    for split, prepared_split in xy_sets.items():
        X, y = prepared_split.X, prepared_split.y
        weights = compute_min_confidence_sample_weights(
            prepared_sets[split],
            min_weight=min_confidence_weight_floor,
        ).reindex(X.index).fillna(float(min_confidence_weight_floor))
        weighted_sets[split] = PreparedSplit(X=X, y=y, sample_weight=weights)
    return weighted_sets