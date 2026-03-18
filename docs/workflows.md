# Common Workflows

This document gives practical workflows grounded in the current codebase.

## Workflow 1: Build Dataset Index And Extract Features

Use Dataset helpers in src/find_bids/models/extract/dataset.py.

- Create a Dataset from your directory pattern.
- Generate BIDS-style subject/session IDs.
- Persist dataset metadata and series features.

Example outline:

1. Dataset.from_dir_with_subject_level(...) or Dataset.from_dir_without_subject_level(...)
2. dataset.generate_bids_ids(...)
3. dataset.to_json()
4. dataset.generate_features(skip_unavailable=True)

Outputs:

- features_root/dataset.json
- features_root/<subject>/<session>/<series>.json

## Workflow 2: Run Heuristic Inference

Use DatasetsInference in src/find_bids/models/infer/core.py.

- Load one or more Dataset objects.
- Run DatasetsInference.from_datasets(...).
- Export to CSV with per-series predictions and confidence.

Generated fields include:

- inferred_datatype
- inferred_suffix
- is_derived
- datatype_confidence
- suffix_confidence
- derived_confidence
- min_confidence

## Workflow 3: Prepare Feature Tables For Modeling

Use helpers in src/find_bids/models/classify/features.py.

- get_features_from_series(...) builds a flattened table from saved features.
- TIER1_FEATURES defines the current model feature subset.

This supports downstream training experiments, even though the classifier training module is still incomplete.

## Workflow 4: Manual Annotation And Review

Use annotation models in src/find_bids/models/annotate/core.py.

- SessionAnnotation and AllSessionsAnnotation support structured labeling.
- CSV import/export is provided.
- Metrics helpers support annotation prioritization by uncertainty/diversity proxies.

## Workflow 5: Research Scripts And Notebooks

The repository includes notebooks and scripts for experimentation:

- test_generate_features.py
- test_generate_heuristic_scores.ipynb
- test_ml.ipynb
- test_dicom.ipynb

These are not packaged as production commands and may include site-specific paths.
Adapt them for your local environment.
