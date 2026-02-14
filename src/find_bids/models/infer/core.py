"""
Inference pipeline for BIDS dataset structure inference. 
This module contains the core logic for inferring the BIDS structure of a dataset based on extracted features and metadata. 
It defines the main classes and functions for performing inference, including loading features, applying inference rules, and generating the inferred BIDS structure.
"""
from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset

import os, json, re
from enum import Enum
from typing import Optional
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from rich.progress import track

LOCALIZER_KEYWORDS = {
    "localizer", "scout", "aahead", "aahscout", "loc", "survey", "fiducial"
}

DERIVED_MAP_KEYWORDS = {
    # Diffusion derivatives
    "fa", "adc", "trace", "avdc", "md", "fractional", "anisotropy", "apparent",
    # Perfusion derivatives  
    "cbf", "cbv", "mtt", "ttp", "tmax",
    # Other processing
    "subtraction", "sub", "moco", "mocoseries", "tractography",
    "reformat", "rfmt", "coreg", "coregistered", "normalized", "norm", "registered",
    "derived", "derivative", "proc", "processed", "analysis", "analysed", "result", "res",
    "map", "maps", "parametric", "param", "stat", "stats", "statistic", "statisticmap",
    "nav", "stealth", "surgicalnav", "dynasuite", "brainlab",  # Surgical navigation exports
}

SURGICAL_NAV_KEYWORDS = {
    "dynasuite", "surgicalnav", "navigation", "stealth", "brainlab"
}

# Patterns in SeriesDescription that indicate timestamp-stamped derived maps
DERIVED_TIMESTAMP_PATTERN = re.compile(
    r"(fa|trace|avdc|cbf|cbv):.*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2}\s+\d{4}",
    re.IGNORECASE
)

DIFFUSION_KEYWORDS = {
    "dti", "dwi", "diff", "diffusion",  # Added "dti"
    # Removed derived map keywords (fa, adc, trace) - those are in DERIVED_MAP_KEYWORDS
}

PERFUSION_KEYWORDS = {
    "asl", "pcasl", "pasl", "casl", "arterial", "spin", "labeling",
    "perfusion", "pwi", "dsc", "dce", "dcemri", "dynamic", "susceptibility",
    # Removed CBF/CBV - those are derived maps
}

FUNC_KEYWORDS = {
    "bold", "fmri", "rest", "rsfmri", "resting", "task",
    "motor", "nback", "stroop", "language", "gambling",
    "sent", "cat", "lett", "tongue", "hands", "bhands",  # Task-specific
}

FMAP_KEYWORDS = {
    "fieldmap", "b0map", "b0", "topup",
    "sefm", "pepolar", "b1map", "tb1",
}

ANAT_KEYWORDS = {
    "t1", "t1w", "mprage", "spgr", "bravo",
    "t2", "t2w", "flair", "pd", "swi", 
    "t2star", "t2*",  # Added T2*
    "gre", "cube", "3d", "magic",  # Added MAGiC
    "anat",  # Generic anatomical label
}

class BIDSEntities(BaseModel):
    """
    Represents the core BIDS entities that can be extracted from DICOM metadata. 
    These entities are crucial for organizing and structuring neuroimaging data according to the BIDS standard. 
    Each entity is optional, as not all DICOM files may contain the necessary metadata to populate these fields. 
    The `datatype` field is required, as it indicates the type of data (e.g., anat, func, dwi) and is essential for correctly categorizing the series within a BIDS dataset.
    """
    subject: Optional[str] = None
    session: Optional[str] = None
    run: Optional[str] = None
    datatype: str
    suffix: Optional[str] = None
    part: Optional[str] = None
    echo: Optional[str] = None
    inv: Optional[str] = None
    ce: Optional[str] = None
    dir: Optional[str] = None
    mt: Optional[str] = None
    acq: Optional[str] = None

class Datatype(Enum):
    ANAT = "anat"
    FUNC = "func"
    DWI = "dwi"
    FMAP = "fmap"
    PERF = "perf"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"

def collect_tokens(series: SeriesFeatures) -> set[str]:
    """Union of all text tokens from SeriesDescription / ProtocolName / SequenceName."""
    tokens: set[str] = set()
    if series.text:
        for feat in (
            series.text.series_description,
            series.text.protocol_name,
            series.text.sequence_name,
        ):
            if feat and feat.tokens:
                # tokens are already lowercased by extract_dicom_tokens
                tokens |= set(feat.tokens.keys())
    return tokens

def get_num_timepoints(series: SeriesFeatures) -> int | None:
    return series.temporal.num_timepoints if series.temporal else None


def is_epi(series: SeriesFeatures) -> bool:
    return bool(
        series.encoding
        and series.encoding.is_epi
        and series.encoding.is_epi.value is True
    )

def should_exclude(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Return True if this series should be excluded from BIDS raw dataset.
    Reasons: localizer, derived map, surgical navigation export, etc.
    """
    
    # Localizers
    if tokens & LOCALIZER_KEYWORDS:
        return True
    
    # Derived maps (FA, Trace, CBF, etc.)
    if tokens & DERIVED_MAP_KEYWORDS:
        return True
    
    # Surgical navigation exports
    if tokens & SURGICAL_NAV_KEYWORDS:
        return True
    
    # Check for timestamp-stamped derived maps in raw text
    if series.text and series.text.series_description:
        desc = series.text.series_description.text
        if DERIVED_TIMESTAMP_PATTERN.search(desc):
            return True
    
    # "ORIG:" prefix often indicates duplicates/backups
    if series.text and series.text.series_description:
        desc = series.text.series_description.text.lower()
        if desc.startswith("orig:"):
            return True
    
    return False

def score_dwi(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for DWI datatype."""
    score = 0
    diff = series.diffusion

    # Non-zero b-values -> strong diffusion evidence
    has_nonzero_bvals = bool(
        diff and diff.b_values and any(b > 50 for b in diff.b_values)
    )
    if has_nonzero_bvals:
        score += 4

    # Many diffusion directions -> strong evidence
    has_gradients = bool(
        diff
        and diff.num_diffusion_directions is not None
        and diff.num_diffusion_directions >= 6
    )
    if has_gradients:
        score += 4

    # ImageType signals
    imt = series.image_type
    if imt and imt.has_diffusion and imt.has_diffusion.value:
        score += 3

    # Text signals - "dti" or "dwi" or "diffusion"
    if tokens & DIFFUSION_KEYWORDS:
        score += 3

    # Penalty: clearly anatomical naming and no strong diffusion evidence
    if tokens & ANAT_KEYWORDS and not (has_nonzero_bvals or has_gradients):
        score -= 4

    return score


def score_perf(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for perfusion datatype."""
    score = 0
    perf = series.perfusion

    # ASL / perfusion-specific tags
    if perf:
        if perf.perfusion_labeling_type and perf.perfusion_labeling_type.value:
            score += 4
        if perf.perfusion_series_type and perf.perfusion_series_type.value:
            score += 3
        if perf.contrast_agent and perf.contrast_agent.value:
            score += 2

    # ImageType
    imt = series.image_type
    if imt and imt.has_perfusion and imt.has_perfusion.value:
        score += 4

    # Text tokens - strong perfusion cues
    if tokens & PERFUSION_KEYWORDS:
        score += 4

    # Weak evidence: many timepoints (DSC-like)
    n_tp = get_num_timepoints(series)
    if n_tp and n_tp > 40:  # DSC typically 40-100+ timepoints
        score += 1

    # EPI perfusion (ASL or DSC)
    if is_epi(series) and (tokens & PERFUSION_KEYWORDS):
        score += 1

    return score

def score_fmap(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for fieldmap datatype."""
    score = 0
    n_tp = get_num_timepoints(series)
    imt = series.image_type
    me = series.multi_echo

    # Text tokens
    if tokens & FMAP_KEYWORDS:
        score += 5

    # Phase/magnitude combinations with few timepoints
    if imt:
        is_mag = bool(imt.is_magnitude and imt.is_magnitude.value)
        is_phase = bool(imt.is_phase and imt.is_phase.value)
        if (is_mag or is_phase) and (n_tp is not None and n_tp <= 2):
            score += 4

    # Multi-echo GRE likely fieldmap if not clearly anat/func/dwi/perf
    if me and me.num_echoes and me.num_echoes > 1:
        # Only boost if not obviously anatomical
        if not (tokens & ANAT_KEYWORDS):
            score += 2

    # Penalty: many timepoints → unlikely to be a classic fmap
    if n_tp and n_tp > 10:
        score -= 3

    return score

def score_func(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for functional datatype."""
    score = 0
    n_tp = get_num_timepoints(series)
    epi = is_epi(series)

    # Text tokens - "fmri", task names, "bold", etc.
    if tokens & FUNC_KEYWORDS:
        score += 5

    # EPI with many timepoints (classic BOLD)
    if epi:
        if n_tp and n_tp >= 50:
            score += 4
        elif n_tp and n_tp >= 20:
            score += 2
        elif n_tp and n_tp >= 10:
            score += 1

    # Penalty: very few timepoints and EPI (likely fmap or dwi)
    if epi and n_tp and n_tp < 10:
        score -= 2

    return score

def score_anat(series: SeriesFeatures, tokens: set[str]) -> int:
    """Score for anatomical datatype."""
    score = 0
    n_tp = get_num_timepoints(series)
    epi = is_epi(series)

    # Text tokens - T1, T2, FLAIR, T2*, MAGiC, etc.
    if tokens & ANAT_KEYWORDS:
        score += 5

    # Single-volume non-EPI is likely anatomical
    if n_tp is not None and n_tp <= 3 and not epi:
        score += 3

    # 3D acquisition
    if series.sequence and series.sequence.mr_acquisition_type:
        if series.sequence.mr_acquisition_type.value == "3d":
            score += 1

    # High-resolution (slice thickness < 2mm often indicates anatomical)
    if series.spatial and series.spatial.slice_thickness:
        thickness = series.spatial.slice_thickness.value
        if thickness and thickness < 2.0:
            score += 1

    # Penalty: many timepoints (unlikely anatomical unless multi-echo)
    if n_tp and n_tp > 10:
        # Unless it's multi-echo anatomical (e.g., MAGiC, ME-MPRAGE)
        if not (series.multi_echo and series.multi_echo.num_echoes and series.multi_echo.num_echoes > 1):
            score -= 3

    return score

def infer_bids_datatype(series: "SeriesFeatures") -> str:
    """
    Infer BIDS datatype from SeriesFeatures.
    
    Returns:
        One of: "dwi", "perf", "fmap", "func", "anat", "exclude" or "unknown"
        None indicates: excluded (localizer/derived) or insufficient evidence
    """
    
    # Only handle MR
    if not series.modality or series.modality.value != "mr":
        return Datatype.UNKNOWN.value

    tokens = collect_tokens(series)

    # Exclusion check (localizers, derived maps, surgical nav)
    if should_exclude(series, tokens):
        return Datatype.EXCLUDE.value

    # Score each datatype
    scores = {
        Datatype.DWI.value:  score_dwi(series, tokens),
        Datatype.PERF.value: score_perf(series, tokens),
        Datatype.FMAP.value: score_fmap(series, tokens),
        Datatype.FUNC.value: score_func(series, tokens),
        Datatype.ANAT.value: score_anat(series, tokens),
    }

    # Pick best label
    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    sorted_scores = sorted(scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0

    # Require minimum evidence (lowered from previous suggestion)
    if best_score < 2:
        return Datatype.UNKNOWN.value

    # Require margin over runner-up (optional, adjust as needed)
    margin = best_score - second_best
    if margin < 1:
        return Datatype.UNKNOWN.value

    return best_type


def infer_bids_structure(dataset: Dataset, n_subjects: Optional[int] = 1) -> pd.DataFrame:
    """
    Infer the BIDS structure of a dataset by applying inference rules to each series and aggregating the results into a hierarchical structure of subjects, sessions, and series with their inferred datatypes.
    Returns a dictionary representing the inferred BIDS structure.
    """
    if dataset.subjects is None:
        return pd.DataFrame() # No subjects to infer from
    
    columns = [
        "subject_id",
        "session_id",
        "series_id",
        "bids_participant_id",
        "bids_session_id",
        "bids_datatype",
        "series_description",
        "protocol_name",
        "sequence_name"
        ]
    records = []
    for subject in track(dataset.subjects[:n_subjects], description="Inferring BIDS structure..."):
        subject_id = subject.subject_id
        if subject.sessions is None:
            continue
        for session in subject.sessions:
            session_id = session.session_id
            if session.series is None:
                continue
            for series in session.series:
                series_id = series.series_id
                features_path = dataset.features_root / subject_id / session_id / f"{series_id}.json"
                series_features = SeriesFeatures.from_json(features_path)
                bids_datatype = infer_bids_datatype(series_features)
                record = {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "series_id": series_id,
                    "bids_participant_id": subject.bids_participant_id,
                    "bids_session_id": session.bids_session_id,
                    "bids_datatype": bids_datatype,
                    "series_description": series_features.text.series_description.text if series_features.text and series_features.text.series_description else None,
                    "protocol_name": series_features.text.protocol_name.text if series_features.text and series_features.text.protocol_name else None,
                    "sequence_name": series_features.text.sequence_name.text if series_features.text and series_features.text.sequence_name else None,
                }
                records.append(record)
            
    df = pd.DataFrame.from_records(records, columns=columns)
    return df