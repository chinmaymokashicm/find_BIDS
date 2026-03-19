# Inference Heuristics

This document explains the current heuristic inference stack and how scores become predictions.

## Overview

The current inference stack predicts three outputs per series:

1. Datatype
2. Suffix (or coarse derived bucket)
3. Derived status

Main modules:

- src/find_bids/models/infer/acquisition.py
- src/find_bids/models/infer/datatype.py
- src/find_bids/models/infer/suffix.py
- src/find_bids/models/infer/core.py
- src/find_bids/models/infer/utils.py

## Category Glossary

This section defines what each currently implemented category represents.

### Acquisition Family Categories

- epi: Echo-planar imaging family. Typically fast readouts used in fMRI, diffusion, and many fieldmap protocols. Example: task-rest BOLD, Ax DWI EPI, AP/PA EPI references.
- gre: Gradient-echo family (non-EPI GRE patterns), often seen in T2star-like and susceptibility-sensitive acquisitions. Example: multi-echo GRE T2star acquisition.
- se_tse: Spin-echo and turbo/fast spin-echo family, commonly used for T1w/T2w-style structural contrast. Example: Ax T2 TSE and related fast spin-echo protocols.
- other: Specialized or uncategorized acquisition families that do not strongly match EPI, GRE, or SE/TSE (for example radial, spiral, MRA/angio, SWI, MRS-like patterns). Example: TOF MRA or single-voxel spectroscopy.

### Datatype Categories

- localizer: Scout/planning scans used for positioning and workflow setup rather than core analysis. Example: 3-plane localizer (LOC) scan.
- anat: Structural/anatomical MRI intended to represent tissue morphology and anatomy. Example: Sag T1 MPRAGE or Ax T2 FLAIR.
- func: Functional time-series MRI (for example BOLD runs and SBRef references). Example: task-handmotor BOLD run plus SBRef.
- dwi: Diffusion-weighted MRI and diffusion-derived acquisitions. Example: Ax DTI b0/b1000 series.
- fmap: Field mapping and susceptibility distortion correction references. Example: dual-echo phasediff fieldmap or AP/PA EPI pair.
- perf: Perfusion MRI (for example ASL, DSC, DCE families). Example: pCASL perfusion or DSC bolus run.
- exclude: Explicitly excluded category for non-target or utility series outside intended raw BIDS outputs. Example: surgical navigation export or vendor utility map.
- unknown: Low-confidence fallback when evidence is insufficient or conflicting. Example: uncommon vendor reconstruction with ambiguous metadata.

### Suffix Categories

Suffix candidates are selected by datatype. Some are raw BIDS suffixes and some are coarse derived buckets. Examples below are illustrative series names/use-cases.

Anat suffixes:

- T1w: T1-weighted anatomical image. Example: Sag T1 MPRAGE.
- T2w: T2-weighted anatomical image. Example: Ax T2 TSE.
- FLAIR: T2-FLAIR anatomical image with fluid suppression. Example: Ax T2 FLAIR.
- PD: Proton-density-weighted anatomical image. Example: PD turbo spin-echo.
- T2starw: T2*-weighted anatomical image. Example: GRE T2star (non-SWI emphasis).
- SWI: Susceptibility-weighted imaging anatomical image. Example: SWI magnitude/phase set.
- anatParamMap: Coarse derived anatomical parametric map bucket. Example: T1 map, T2 map, or ADC-like anatomical map.
- anatReformat: Coarse derived anatomical reformat/reconstruction bucket. Example: MPR or MIP reformat derived from anat input.
- anatSegmentation: Coarse derived anatomical segmentation/mask bucket. Example: brain mask, GM/WM/CSF segmentation.
- anatOtherDerived: Coarse derived anatomical catch-all bucket. Example: derived anatomical product not fitting map/reformat/segmentation buckets.

Func suffixes:

- bold: BOLD functional time-series run. Example: task-rest run with many EPI volumes.
- sbref: Single-band reference for functional EPI runs. Example: single-volume SBRef acquired before BOLD.
- funcPreprocBold: Coarse derived preprocessed BOLD time-series bucket. Example: motion-corrected and normalized BOLD time-series.
- funcStatMap: Coarse derived functional statistic map bucket. Example: t-map, z-map, or beta map from first-level modeling.
- funcTimeseriesSummary: Coarse derived functional time-series summary bucket. Example: mean BOLD image or temporal summary metric output.
- funcSegmentationOrMask: Coarse derived functional segmentation/mask bucket. Example: activation mask or functional-space ROI mask.
- funcOtherDerived: Coarse derived functional catch-all bucket. Example: derived functional output not fitting preproc/stat/summary/mask buckets.

DWI suffixes:

- dwi: Raw diffusion-weighted acquisition. Example: multi-volume b0/b1000 diffusion series.
- dwiParamMap: Coarse derived diffusion parametric map bucket. Example: ADC, FA, MD, or trace map.
- dwiSegmentationOrMask: Coarse derived diffusion segmentation/mask bucket. Example: lesion mask or diffusion-space ROI mask.
- dwiOtherDerived: Coarse derived diffusion catch-all bucket. Example: derived diffusion output not fitting param-map or mask buckets.

Fmap suffixes:

- phasediff: Phase-difference fieldmap image. Example: dual-echo phasediff map.
- magnitude: Magnitude image associated with field mapping (single merged label for magnitude echoes). Example: magnitude image at echo 1 or echo 2.
- epi: EPI-based fieldmap/reference image (for example blip-up/blip-down style references). Example: AP/PA spin-echo EPI references.
- fieldmap: Direct field map image. Example: scanner-generated B0 field map image.
- fmapSusceptibilityOrPhaseMap: Coarse derived susceptibility/phase-map bucket. Example: unwrapped phase map or voxel-shift-like map.
- fmapReformat: Coarse derived fieldmap reformat/reconstruction bucket. Example: reformatted fieldmap preview volume.
- fmapSegmentationOrMask: Coarse derived fieldmap segmentation/mask bucket. Example: fieldmap brain mask.
- fmapOtherDerived: Coarse derived fieldmap catch-all bucket. Example: derived fieldmap output not fitting susceptibility/reformat/mask buckets.

Perf suffixes:

- asl: Arterial spin labeling perfusion acquisition. Example: pCASL label/control timeseries.
- m0scan: M0 calibration/reference scan used with ASL workflows. Example: ASL M0 calibration scan.
- dsc: Dynamic susceptibility contrast perfusion acquisition. Example: T2star bolus perfusion run.
- dce: Dynamic contrast-enhanced perfusion acquisition. Example: serial post-contrast T1 perfusion run.
- perfCbfLikeMap: Coarse derived perfusion CBF-like map bucket. Example: CBF map output.
- perfCbvLikeMap: Coarse derived perfusion CBV-like map bucket. Example: CBV map output.
- perfTimingMap: Coarse derived perfusion timing map bucket (for example transit/arrival style maps). Example: ATT, Tmax, or MTT style map.
- perfSegmentationOrRoi: Coarse derived perfusion segmentation or ROI bucket. Example: perfusion lesion ROI/mask.
- perfOtherDerived: Coarse derived perfusion catch-all bucket. Example: derived perfusion output not fitting CBF/CBV/timing/segmentation buckets.

Localizer, exclude, and unknown datatypes currently have no suffix registry in BIDS_SCHEMA.

## 1) Acquisition Family Scoring

Family scoring is a soft prior and is intentionally orthogonal to intent flags.

Implemented families:

- epi
- gre
- se_tse
- other

Examples of cues currently used:

- ScanningSequence and SequenceVariant
- Echo spacing and echo train length
- Echo time (TE) ranges
- 2D vs 3D acquisition type
- Specialized token cues for the other bucket

API:

- get_acquisition_family_scores(series, tokens) -> dict

Boolean helper:

- is_epi(series, tokens=None)

## 2) Orthogonal Intent Scoring

Intent flags are independent from family and influence datatype/suffix decisions.

Implemented intents:

- diffusion
- perfusion
- fieldmap
- localizer

Examples of cues currently used:

- Diffusion tags and b-value direction statistics
- Perfusion labeling/series type and perfusion token cues
- Fieldmap token cues, magnitude/phase combinations, echo pairing patterns
- Localizer ImageType, geometry/thickness cues, localizer tokens

API:

- get_acquisition_intent_scores(series, tokens) -> dict

Boolean helper:

- is_fieldmap_epi(series, tokens=None)
  - computed from epi family + fieldmap intent - diffusion intent

## 3) Datatype Scoring

Datatype scoring combines:

- datatype-specific rules in datatype.py
- family and intent guidance from acquisition.py

Candidate classes include:

- anat
- func
- dwi
- perf
- fmap
- localizer

Score flow:

1. Raw rules per datatype accumulate positive/negative evidence.
2. Acquisition guidance shifts scores using family and intent thresholds.
3. Scores are converted to probabilities by softmax.
4. Label confidence is top-1 minus top-2 margin.
5. If margin is too small, prediction is set to unknown.

## 4) Suffix (Or Coarse Bucket) Scoring

Suffix scoring is conditional on selected datatype.

Score flow:

1. Enumerate allowed suffixes from BIDS_SCHEMA for the chosen datatype.
2. Optionally filter by derived status:
   - raw suffixes only
   - coarse derived buckets only
3. Score each suffix/bucket with suffix-specific rules.
4. Apply acquisition family/intent guidance.
5. Convert to probabilities with softmax and select top candidate by margin.

This design supports both:

- raw BIDS suffixes (for example T1w, bold, dwi, phasediff)
- coarse derived buckets (for example dwiParamMap, perfCbfLikeMap)

## 5) Derived Status Scoring

Derived status is scored independently in infer/core.py.

Key behavior:

- score_derived(series, tokens) returns a signed score.
- score_is_derived(...) maps score to probabilities with logistic scaling.
- outputs:
  - probabilities for derived and raw
  - label in {True, False, None}
  - confidence from probability separation

This separation follows the intended architecture where derivedness is not collapsed into datatype/family.

## 6) Confidence And Uncertainty Handling

The inference stack uses confidence margins to avoid overconfident assignments:

- Datatype confidence: top-1 minus top-2 datatype probability
- Suffix confidence: top-1 minus top-2 suffix probability
- Derived confidence: absolute distance between derived and raw probability

When confidence is low, unknown/uncertain outputs are used instead of forcing a hard label.

## 7) Practical Notes

- Heuristics are interpretable and easy to debug.
- Behavior is sensitive to threshold tuning and keyword sets.
- Current implementation is metadata-driven and does not rely on voxel-level image content.
- The stack is designed to feed a future classifier and graph refinement layer, but those are not yet implemented.
