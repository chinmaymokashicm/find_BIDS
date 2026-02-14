"""
Inference pipeline for BIDS dataset structure inference. 
This module contains the core logic for inferring the BIDS structure of a dataset based on extracted features and metadata. 
It defines the main classes and functions for performing inference, including loading features, applying inference rules, and generating the inferred BIDS structure.
"""
from ..extract.series import SeriesFeatures
from ..extract.dataset import Dataset

import os, json
from enum import Enum
from pathlib import Path

class Datatype(Enum):
    ANAT = "anat"
    FUNC = "func"
    DWI = "dwi"
    FMAP = "fmap"
    UNKNOWN = "unknown"

def is_dwi(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Determine if a series is a diffusion-weighted imaging (DWI) series based on its features and textual cues.
    Key signals:

    - DiffusionFeatures flags and b-values
    - ImageType diffusion/ADC/FA tokens
    - Text tokens (e.g., “dwi”, “diff”, “trace”, “adc”, “fa”)
    """
    if series.diffusion and series.diffusion.has_diffusion:
        return True
    if series.image_type:
        if series.image_type.has_diffusion:
            return True
        if series.image_type.is_adc or series.image_type.is_fa or series.image_type.is_trace:
            return True
    diffusion_keywords = {"dwi", "diff", "diffusion", "adc", "fa", "trace", "md", "tensor"}
    if tokens & diffusion_keywords:
        return True
    return False

def is_perf(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Determine if a series is an ASL/perfusion series based on its features and textual cues.
    
    Key signals:
    - Your PerfusionFeatures from ASL-specific tags
    - ImageType.has_perfusion
    - Protocol/series names with perfusion terms
    
    """
    # 1) Explicit ASL/perfusion tags
    if series.perfusion:
        if (series.perfusion.perfusion_labeling_type is not None or
            series.perfusion.perfusion_series_type is not None):
            return True

    # 2) ImageType explicitly marks perfusion
    if series.image_type and series.image_type.has_perfusion:
        return True

    # 3) Textual cues for ASL or perfusion/DSC
    perf_keywords = {
        "asl", "pcasl", "casl", "pasl",
        "perfusion", "pwi", "cbf", "cbv", "mtc",
        "dsc", "dcemri", "dce", "pseudocontinuous"
    }
    if tokens & perf_keywords:
        return True

    return False

def is_fmap(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Determine if a series is a fieldmap (fmap) series based on its features and textual cues.
    Fieldmaps are typically used to correct EPI in func, dwi, or perf.    
    There are two broad families:
    1. GRE magnitude/phase-diff fieldmaps
    2. Spin-echo EPI fieldmaps (blip-up/blip-down)

    Key signals:
    - Very few timepoints (often 1-2 volumes)
    - Part-type indicators (magnitude/phase)
    - Text tokens: “fieldmap”, “b0map”, “b0”, “topup”, “sefm”, “b1map”
    - Multi-echo GRE used just for mapping (e.g., n_echoes > 1, no obvious anat/func semantics)
    """
    n_tp = series.temporal.num_timepoints if series.temporal else None

    text_keywords = {
        "fieldmap", "b0map", "b0", "topup",
        "sefm", "pepolar", "field", "tb1", "b1map"
    }

    # 1) Strong textual cues
    if tokens & text_keywords:
        return True

    # 2) Phase/magnitude patterns with few volumes
    if series.image_type:
        is_mag = bool(series.image_type.is_magnitude)
        is_phase = bool(series.image_type.is_phase)
        if (is_mag or is_phase) and (n_tp is not None and n_tp <= 2):
            return True

    # 3) Multi-echo GRE fieldmaps
    if series.multi_echo and series.multi_echo.num_echoes and series.multi_echo.num_echoes > 1:
        # If there are multiple echoes, no diffusion/perfusion, and no anat keywords:
        if not is_dwi(series, tokens) and not is_perf(series, tokens):
            anat_keywords = {"t1", "t1w", "t2", "t2w", "flair", "mprage", "spgr", "memprage", "swI"}
            if not (tokens & anat_keywords):
                return True

    return False

def is_func(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Determine if a series is a functional MRI (fMRI) series based on its features and textual cues.
    
    Key signals:
    - EPI readout (encoding.is_epi true)
    - Many timepoints (e.g., ≥ 20-30)
    - No diffusion/perfusion classification already made
    - Text tokens: “bold”, “fmri”, “rest”, “task”, “nback”, etc.
    """
    if is_dwi(series, tokens) or is_perf(series, tokens) or is_fmap(series, tokens):
        return False

    n_tp = series.temporal.num_timepoints if series.temporal else None
    is_epi = bool(series.encoding and series.encoding.is_epi)

    func_keywords = {
        "bold", "fmri", "rest", "rsfmri", "task",
        "stroop", "nback", "motor", "language", "gambling"
    }

    # 1) Strong textual cues
    if tokens & func_keywords:
        return True

    # 2) EPI + many timepoints is highly suggestive of fMRI
    if is_epi and n_tp is not None and n_tp >= 20:
        return True

    return False

def is_anat(series: SeriesFeatures, tokens: set[str]) -> bool:
    """
    Determine if a series is an anatomical (anat) series based on its features and textual cues.
    
    Key signals:
    - Typically 1 timepoint (or very few)
    - Not diffusion, perfusion, fieldmap, or func
    - Non-EPI readout often (encoding.is_epi false), though some structural EPI exists
    - Strong anatomical keywords
    """
    if any([is_dwi(series, tokens),
            is_perf(series, tokens),
            is_fmap(series, tokens),
            is_func(series, tokens)]):
        return False

    n_tp = series.temporal.num_timepoints if series.temporal else None
    is_epi = bool(series.encoding and series.encoding.is_epi)

    anat_keywords = {
        "t1", "t1w", "mp-rage", "mprage", "spgr", "fse", "tfl",
        "t2", "t2w", "flair", "pd", "swi", "gre", "cube", "3d"
    }

    # 1) Strong textual cues
    if tokens & anat_keywords:
        return True

    # 2) Single-volume, non-EPI is probably structural
    if (n_tp is not None and n_tp <= 3) and not is_epi:
        return True

    return False

def infer_datatype(series: SeriesFeatures) -> Datatype:
    """
    Infer the BIDS datatype of a series based on its features.
    """
    if not series.modality or (series.modality != "MR"):
        return Datatype.UNKNOWN # Only handle MR modality for now
    
    is_epi = bool(series.encoding and series.encoding.is_epi)
    has_difusion_bvals = bool(series.diffusion and series.diffusion.has_diffusion)
    has_perfusion_imagetype = bool(series.image_type and series.image_type.has_perfusion)
    has_diffusion_imagetype = bool(series.image_type and series.image_type.has_diffusion)
    
    num_timepoints = series.temporal.num_timepoints if series.temporal else None
    num_volumes = series.num_volumes
    num_unique_slices = series.num_unique_slices
    
    tokens = set()
    if series.text:
        for txt_feat in [series.text.series_description,
                        series.text.protocol_name,
                        series.text.sequence_name]:
            if txt_feat:
                # Add both the original tokens and their lowercase versions for case-insensitive matching
                tokens |= set(t.lower() for t in txt_feat.tokens.keys()) # Use lowercase tokens for case-insensitive matching

    if is_dwi(series, tokens):
        return Datatype.DWI
    if is_perf(series, tokens):
        return Datatype.FMAP
    if is_fmap(series, tokens):
        return Datatype.FMAP
    if is_func(series, tokens):
        return Datatype.FUNC
    if is_anat(series, tokens):
        return Datatype.ANAT
    
    return Datatype.UNKNOWN


# def infer_bids_structure(dataset: Dataset) -> dict:
#     """
#     Infer the BIDS structure of a dataset by applying inference rules to each series and aggregating the results into a hierarchical structure of subjects, sessions, and series with their inferred datatypes.
#     Returns a dictionary representing the inferred BIDS structure.
#     """
#     if dataset.subjects is None:
#         return {}
    
#     bids_structure = {"subjects": []}
#     for subject in dataset.subjects:
#         subject_dict = {"subject_id": subject.subject_id, "sessions": []}
#         for session in subject.sessions or []:
#             session_dict = {"session_id": session.session_id, "series": []}
#             for series in session.series or []:
#                 datatype = infer_datatype(series)
#                 series_dict = {
#                     "series_id": series.series_id,
#                     "inferred_datatype": datatype.value
#                 }
#                 session_dict["series"].append(series_dict)
#             subject_dict["sessions"].append(session_dict)
#         bids_structure["subjects"].append(subject_dict)
    
#     return bids_structure