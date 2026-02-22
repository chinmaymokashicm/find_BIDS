import re
from enum import Enum
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat

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