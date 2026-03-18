# find_BIDS

Heuristic-first DICOM-to-BIDS semantic inference for clinical MRI datasets.

This repository focuses on robust series-level metadata extraction and interpretable scoring to infer:

- datatype (anat, func, dwi, perf, fmap, localizer)
- suffix or coarse derived bucket
- derived status (raw vs derived)

The implementation follows the design direction in Ideas_v5.pdf, but only documents and ships what is currently in code.

## What Is Implemented Today

- DICOM dataset discovery for two common layouts:
	- Subject -> Session -> Series
	- Session -> Series (subject derived from session id)
- Series-level feature extraction from DICOM headers and series statistics.
- Heuristic inference for datatype, suffix, and derived status.
- Acquisition family scoring (EPI, GRE, SE/TSE, Other) plus orthogonal intent flags (diffusion, perfusion, fieldmap, localizer).
- Strict BIDS label/schema validation for inferred outputs.
- Annotation data structures to support human review and weak supervision workflows.
- Feature and inference export to JSON/CSV and optional SQLite caching for extracted features.

## What Is Not Implemented Yet

- No production CLI entrypoint.
- No graph-based inference layer (MRF/CRF, LBP) yet.
- No trained discriminative classifier pipeline yet (classification module is scaffolding).
- NIfTI feature extraction is not implemented.
- Automatic assignment of all secondary BIDS entities (run/echo/dir/acq/task) is partial.

## Repository Map

- src/find_bids/models/extract
	- dataset.py: dataset crawling, hierarchy modeling, feature generation orchestration.
	- series.py: DICOM tag parsing and robust feature aggregation.
- src/find_bids/models/infer
	- acquisition.py: acquisition family and intent scoring.
	- datatype.py: datatype heuristic scoring + family/intent guidance.
	- suffix.py: suffix/coarse-bucket scoring + family/intent guidance.
	- core.py: end-to-end inference objects and derived scoring.
	- schema.py: BIDS schema registry and strict validation models.
- src/find_bids/models/annotate
	- core.py: annotation models and sampling metrics support.
- src/find_bids/models/classify
	- features.py: feature table assembly helpers.
	- core.py: placeholder for future classifier training/inference.

## Installation

The project is Python-first and currently run as library code from scripts/notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most examples in this repo run with PYTHONPATH=src.

## Quick Start Example

```python
from upath import UPath

from find_bids.models.extract.dataset import Dataset
from find_bids.models.infer.core import DatasetsInference

# Build dataset index from a Subject -> Session -> Series DICOM tree
dataset = Dataset.from_dir_with_subject_level(
		dir_root=UPath("/path/to/dicom_root"),
		features_root=UPath("/path/to/features_root"),
		dtype="DICOM",
		session_subdir_path="",
		series_subdir_path="DICOM/",
)

dataset.generate_bids_ids(replace_existing=True)
dataset.to_json()
dataset.generate_features(skip_unavailable=True)

# Run heuristic inference
all_inf = DatasetsInference.from_datasets([dataset])
all_inf.to_csv("inference_results.csv")
```

## Notes On Existing Scripts

- test_generate_features.py and notebooks in the repo are research scripts and may contain site-specific paths.
- They are useful as reference workflows, but you should adapt paths and dataset assumptions locally.

## Detailed Documentation

- docs/architecture.md: concrete architecture and module responsibilities.
- docs/inference_heuristics.md: scoring logic, confidence computation, and category glossary for acquisition family, datatype, and suffix labels.
- docs/implementation_status_vs_ideas_v5.md: Ideas_v5 concept map vs current implementation.
- docs/workflows.md: practical end-to-end workflows with current code.
