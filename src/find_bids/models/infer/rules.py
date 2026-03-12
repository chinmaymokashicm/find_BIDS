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
    "subtraction", "sub", "moco", "mocoseries", "motion corrected", "motioncorrected", "motion", "corrected",
    "tractography",
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
    "fieldmap", "b0map", "topup",
    "sefm", "pepolar", "b1map", "tb1",
    "fm", "magnitude", "phase"
}

ANAT_KEYWORDS = {
    "t1", "t1w", "mprage", "spgr", "bravo",
    "t2", "t2w", "flair", "pd", "swi", 
    "t2star", "t2*",  # Added T2*
    # "gre", 
    "cube", 
    # "3d", 
    "magic",
    "anat",  # Generic anatomical label
}

B0_KEYWORDS = {
    "b0", "b0map", "b0 map", "b0ref", "b0 ref", "b0reference", "b0 reference", "reference"
}

# ===== ANATOMICAL SUFFIX KEYWORDS =====

T1_KEYWORDS = {
    "t1", "t1w",
    "mpr", "mprage", "mp-rage",
    "spgr", "fspgr",
    "bravo",            # GE
    "tfl",              # Siemens turbo flash
    "gre",              # sometimes used for T1 GRE
    "ir", "irprep",     # inversion recovery
    "t1c", "t1ce", "post", "postcontrast", "postgad",
}

T2_KEYWORDS = {
    "t2", "t2w",
    "tse", "fse", "fastspin", "fastspinecho",
    "turbo", "turboecho",
    "space",            # Siemens 3D T2
    "cube",             # GE 3D T2
    "vista",            # Philips 3D T2
}

FLAIR_KEYWORDS = {
    "flair",
    "t2flair",
    "darkfluid",
    "fluidattenuated",
    "fluid-attenuated",
    "tirm",             # turbo inversion recovery magnitude
}

PD_KEYWORDS = {
    "pd", "pdw",
    "protondensity",
    "proton-density",
}

SWI_KEYWORDS = {
    "swi",
    "susceptibility",
}

T2STAR_KEYWORDS = {
    "t2star", "t2*", "t2_star", "t2starw"
}

T1MAP_KEYWORDS = {"t1map", "t1 map", "t1 mapping", "t1-map"}
T2MAP_KEYWORDS = {"t2map", "t2 map", "t2 mapping", "t2-map"}

# Other anatomical contrasts (not mutually exclusive with T1/T2/FLAIR)
GRE_KEYWORDS = {
    "gre",
    "gradient",
    "gradientecho",
}

# ===== FUNCTIONAL SUFFIX KEYWORDS =====
BOLD_KEYWORDS = {
    "bold", "fmri", "rest", "rsfmri", "resting", "task",
    # "motor", "nback", "stroop", "language", "gambling",
    # "sent", "cat", "lett", "tongue", "hands", "bhands",  # Task-specific
}

SBREF_KEYWORDS = {
    "sbref", "singlebandref", "sbrefep2d",
    "singlebandreference", "sbrefepi", "sbrefep2d", "sbref_epi",
    "single band reference", "sbref epi", "sbref ep2d",
}
