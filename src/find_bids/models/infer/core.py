"""
Inference pipeline for BIDS dataset structure inference. 
This module contains the core logic for inferring the BIDS structure of a dataset based on extracted features and metadata. 
It defines the main classes and functions for performing inference, including loading features, applying inference rules, and generating the inferred BIDS structure.
"""
from .schema import BIDSEntities, Datatype

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

import os, json, re
from enum import Enum
from typing import Optional
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, model_validator
from rich.progress import track

# def infer_bids_datatype(series: SeriesFeatures) -> str:
#     """
#     Infer BIDS datatype from SeriesFeatures.
    
#     Returns:
#         One of: "dwi", "perf", "fmap", "func", "anat", "exclude" or "unknown"
#         None indicates: excluded (localizer/derived) or insufficient evidence
#     """
    
#     # Only handle MR
#     if not series.modality or series.modality.value != "mr":
#         return Datatype.UNKNOWN.value

#     tokens = collect_tokens(series)

#     # Exclusion check (localizers, derived maps, surgical nav)
#     if should_exclude(series, tokens):
#         return Datatype.EXCLUDE.value

#     # Score each datatype
#     scores = {
#         Datatype.DWI.value:  score_dwi(series, tokens),
#         Datatype.PERF.value: score_perf(series, tokens),
#         Datatype.FMAP.value: score_fmap(series, tokens),
#         Datatype.FUNC.value: score_func(series, tokens),
#         Datatype.ANAT.value: score_anat(series, tokens),
#     }

#     # Pick best label
#     best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
#     sorted_scores = sorted(scores.values(), reverse=True)
#     second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0

#     # Require minimum evidence (lowered from previous suggestion)
#     if best_score < 2:
#         return Datatype.UNKNOWN.value

#     # Require margin over runner-up (optional, adjust as needed)
#     margin = best_score - second_best
#     if margin < 1:
#         return Datatype.UNKNOWN.value

#     return best_type