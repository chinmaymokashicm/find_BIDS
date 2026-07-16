"""Reusable helper blocks for the classification ML prep pipeline."""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .engineer import build_combined_text_series


def normalize_localizer_unknown_suffixes(
    heuristic_scores_df: pd.DataFrame,
    datatype_value: str = "localizer",
    unknown_suffix_value: str = "unknown",
    normalized_suffix_value: str = "localizer",
) -> pd.DataFrame:
    """Normalize localizer rows that structurally lack a suffix registry.

    For rows with datatype=localizer and suffix=unknown, rewrite the suffix to
    localizer and align suffix/min confidence with datatype confidence so ML can
    treat these as known labels rather than structural unknowns.
    """
    result = heuristic_scores_df.copy()
    required_columns = {"inferred_datatype", "inferred_suffix"}
    if not required_columns.issubset(result.columns):
        return result

    localizer_mask = (
        result["inferred_datatype"].astype("string").str.lower().eq(datatype_value.lower())
        & result["inferred_suffix"].astype("string").str.lower().eq(unknown_suffix_value.lower())
    )
    if not localizer_mask.any():
        return result

    result.loc[localizer_mask, "inferred_suffix"] = normalized_suffix_value

    if "datatype_confidence" in result.columns:
        datatype_confidence = pd.to_numeric(result.loc[localizer_mask, "datatype_confidence"], errors="coerce")
        if "suffix_confidence" in result.columns:
            result.loc[localizer_mask, "suffix_confidence"] = datatype_confidence.to_numpy()

    if "label" in result.columns:
        result.loc[localizer_mask, "label"] = (
            result.loc[localizer_mask, "inferred_datatype"].astype("string")
            + "_"
            + result.loc[localizer_mask, "inferred_suffix"].astype("string")
        )

    if "min_confidence" in result.columns:
        confidence_columns = [
            col for col in ["datatype_confidence", "suffix_confidence", "derived_confidence"] if col in result.columns
        ]
        if confidence_columns:
            recomputed_confidence = pd.concat(
                [
                    pd.to_numeric(result.loc[localizer_mask, col], errors="coerce")
                    for col in confidence_columns
                ],
                axis=1,
            ).min(axis=1, skipna=True)
            result.loc[localizer_mask, "min_confidence"] = recomputed_confidence.to_numpy()

    return result


def make_tfidf_feature_names(raw_names: list[str], prefix: str = "tfidf") -> list[str]:
    """Create deterministic, collision-safe TF-IDF feature names."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for name in raw_names:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        normalized = normalized or "token"
        candidate = f"{prefix}_{normalized}"
        count = seen.get(candidate, 0)
        seen[candidate] = count + 1
        if count:
            candidate = f"{candidate}_{count}"
        names.append(candidate)
    return names


def append_train_fitted_field_tfidf_features(
    split_frames: dict[str, pd.DataFrame],
    field_max_features: dict[str, int],
    min_df: int = 2,
    ngram_range: tuple[int, int] = (3, 5),
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
) -> dict[str, pd.DataFrame]:
    """Fit field-wise TF-IDF on train and transform all splits."""
    train_frame = split_frames.get("train")
    if train_frame is None or train_frame.empty:
        return split_frames

    updated_frames: dict[str, pd.DataFrame] = {name: frame.copy() for name, frame in split_frames.items()}
    for field, max_features in field_max_features.items():
        if field not in train_frame.columns or max_features <= 0:
            continue

        train_text = build_combined_text_series(train_frame, text_columns=[field]).fillna("")
        if not train_text.str.len().gt(0).any():
            continue

        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        try:
            vectorizer.fit(train_text)
        except ValueError:
            # Empty vocabulary can happen for tiny splits with aggressive min_df/ngrams.
            continue

        feature_names = make_tfidf_feature_names(
            vectorizer.get_feature_names_out().tolist(),
            prefix=f"tfidf_{field}",
        )

        for split_name, frame in updated_frames.items():
            split_text = build_combined_text_series(frame, text_columns=[field]).fillna("")
            matrix = vectorizer.transform(split_text)
            dense_matrix = np.asarray(matrix.todense(), dtype=np.float32)
            tfidf_df = pd.DataFrame(
                dense_matrix,
                index=frame.index,
                columns=feature_names,
                dtype=np.float32,
            )
            updated_frames[split_name] = pd.concat([frame, tfidf_df], axis=1)

    return updated_frames


def drop_unknown_targets_from_training_splits(
    prepared_sets: dict[str, pd.DataFrame],
    target_columns: list[str],
    unknown_value: str = "unknown",
    splits_to_filter: tuple[str, ...] = ("train", "val"),
) -> dict[str, pd.DataFrame]:
    """Drop rows with unknown targets from selected splits."""
    filtered: dict[str, pd.DataFrame] = {name: frame.copy() for name, frame in prepared_sets.items()}
    for split in splits_to_filter:
        if split not in filtered:
            continue
        frame = filtered[split]
        if frame.empty:
            continue

        known_mask = pd.Series(True, index=frame.index)
        for col in target_columns:
            if col not in frame.columns:
                continue
            known_mask &= frame[col].astype("string").str.lower().fillna(unknown_value) != unknown_value
        filtered[split] = frame.loc[known_mask].copy()
    return filtered


def resolve_min_confidence(df: pd.DataFrame) -> pd.Series:
    """Resolve per-row confidence from min_confidence with a safe fallback."""
    if "min_confidence" in df.columns:
        confidence = pd.to_numeric(df["min_confidence"], errors="coerce")
    elif "datatype_confidence" in df.columns and "suffix_confidence" in df.columns:
        confidence = pd.concat(
            [
                pd.to_numeric(df["datatype_confidence"], errors="coerce"),
                pd.to_numeric(df["suffix_confidence"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1, skipna=True)
    else:
        confidence = pd.Series(np.nan, index=df.index, dtype=float)
    return confidence.clip(lower=0.0, upper=1.0)


def compute_min_confidence_sample_weights(
    df: pd.DataFrame,
    min_weight: float = 0.3,
) -> pd.Series:
    """Compute sample weights as max(min_weight, min_confidence)."""
    confidence = resolve_min_confidence(df)
    weights = np.maximum(confidence.to_numpy(dtype=float), float(min_weight))
    return pd.Series(weights, index=df.index, dtype=float).fillna(float(min_weight))
