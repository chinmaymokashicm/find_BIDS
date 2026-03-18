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

T1_KEYWORDS = {"t1", "t1w", "mpr", "mprage", "spgr", "fspgr", "bravo", "tfl", "t1c", "t1ce", "post", "postcontrast", "postgad"}
T2_KEYWORDS = {"t2", "t2w", "tse", "fse", "fastspin", "fastspinecho", "turbo", "turboecho", "space", "cube", "vista"}
FLAIR_KEYWORDS = {"flair", "t2flair", "darkfluid", "fluidattenuated", "tirm"}
PD_KEYWORDS = {"pd", "pdw", "protondensity"}
SWI_KEYWORDS = {"swi", "susceptibility"}
T2STAR_KEYWORDS = {"t2star", "t2starw"}
T1MAP_KEYWORDS = {"t1map"}
T2MAP_KEYWORDS = {"t2map"}
GRE_KEYWORDS = {"gre", "gradient", "gradientecho"}
BOLD_KEYWORDS = {"bold", "fmri", "rest", "rsfmri", "resting", "task"}
SBREF_KEYWORDS = {"sbref", "singlebandref", "sbrefep2d", "sbrefepi"}

# Coarse derived-bucket lexical cues (Ideas v3 buckets)
REFORMAT_KEYWORDS = {
    "reformat", "reformatted", "rfmt", "mpr", "mip", "projection", "proj",
    "reslice", "resampled", "resample", "recon", "reconstruction"
}

SEGMENTATION_KEYWORDS = {
    "seg", "segmentation", "mask", "brainmask", "roi", "label", "labels",
    "lesion", "tumor", "gm", "wm", "csf"
}

FUNC_PREPROC_KEYWORDS = {
    "preproc", "preprocessed", "denoise", "denoised", "smoothed", "normalized",
    "motioncorrected", "moco", "cleaned"
}

FUNC_STATMAP_KEYWORDS = {
    "stat", "stats", "statmap", "zmap", "tmap", "fmap", "beta", "cope", "contrast"
}

FUNC_SUMMARY_KEYWORDS = {
    "timeseries", "summary", "mean", "std", "variance", "tsnr", "dvars", "fd"
}

PERF_TIMING_KEYWORDS = {
    "mtt", "ttp", "tmax", "timing", "transit", "arrival", "delay"
}

FMAP_PHASELIKE_KEYWORDS = {
    "phase", "phasediff", "susceptibility", "fieldmap", "fmap", "b0", "shimming"
}