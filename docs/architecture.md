# Architecture

This document describes the actual pipeline implemented in the repository today.
It is intentionally grounded in code that exists under src/find_bids.

## 1) Data Ingestion And Dataset Modeling

Implemented in src/find_bids/models/extract/dataset.py.

Primary responsibilities:

- Traverse DICOM datasets using two hierarchy patterns:
  - Subject -> Session -> Series
  - Session -> Series (subject inferred from session id prefix)
- Validate DICOM directories.
- Build in-memory models:
  - Dataset
  - Subject
  - Session
  - Series
- Generate stable BIDS-like IDs for subject/session ordering.

Key entrypoints:

- Dataset.from_dir_with_subject_level(...)
- Dataset.from_dir_without_subject_level(...)
- Dataset.generate_bids_ids(...)
- Dataset.to_json()

## 2) Series Feature Extraction

Implemented in src/find_bids/models/extract/series.py and orchestrated by Dataset.generate_features(...).

Feature groups include:

- Core metadata:
  - manufacturer, model, field strength, modality-like cues
- Geometry/spatial:
  - rows, columns, voxel size, matrix, slice thickness, orientation
- Temporal:
  - TR/TE/TI, echo train length, timepoint counts, bucketed TR/TE, 3D vs isotropic
- Sequence/encoding:
  - scanning sequence, sequence variant, scan options, MR acquisition type
  - phase encoding direction/polarity/axis, echo spacing, parallel reduction factors, multiband factor
- Diffusion and perfusion:
  - b-values, diffusion directions/volumes, perfusion labeling/type, temporal spacing
- ImageType-derived flags:
  - original/primary, localizer, reformatted/projection/mip/mpr, map-like flags (adc/fa/trace/cbf/cbv), part flags
- Text fields:
  - series description, protocol name, sequence name tokens

Outputs:

- Per-series JSON feature files under features_root
- Optional SQLite persistence
- Optional flattened CSV tables

## 3) Heuristic Inference Layer

Implemented in src/find_bids/models/infer.

### 3.1 Acquisition Families And Intent Flags

src/find_bids/models/infer/acquisition.py

Families (physics-level):

- epi
- gre
- se_tse
- other

Orthogonal intent flags:

- diffusion intent
- perfusion intent
- fieldmap intent
- localizer intent

This mirrors the Ideas_v5 design principle: family is a soft prior, while intent and derivedness are separate axes.

### 3.2 Datatype Scoring

src/find_bids/models/infer/datatype.py

- Handcrafted rule scoring per datatype class.
- Family and intent scores are injected as guidance before softmax.
- Final label confidence is margin between top two probabilities.

### 3.3 Suffix (Or Coarse Bucket) Scoring

src/find_bids/models/infer/suffix.py

- Scores suffixes within the selected datatype.
- Supports raw suffixes and datatype-specific coarse derived buckets.
- Uses family and intent guidance to resolve ambiguous cases.

### 3.4 Derived Status Scoring

src/find_bids/models/infer/core.py

- score_derived(...) builds a signed heuristic score from ImageType/text/structure cues.
- score_is_derived(...) converts that score to probability via logistic transform and emits:
  - derived/raw probabilities
  - boolean or uncertain label
  - confidence

### 3.5 End-To-End Inference Data Models

src/find_bids/models/infer/core.py

- SeriesInference
- DatasetInference
- DatasetsInference

These collect datatype/suffix/derived predictions and confidence metrics and support CSV export.

## 4) Schema And Validation Layer

Implemented in src/find_bids/models/infer/schema.py.

- Declares supported datatypes and suffixes.
- Enumerates allowed entities per suffix.
- Validates that inferred combinations are structurally valid.

This provides consistency checks over inferred labels before downstream conversion logic.

## 5) Annotation And Weak-Supervision Support

Implemented in src/find_bids/models/annotate/core.py.

- Data structures for manual labeling at series/session/global levels.
- CSV export/import pathways for annotation workflows.
- Metrics utilities for choosing sessions to annotate.

This supports the manual verification loop discussed in Ideas_v5.

## 6) Classifier Scaffolding

Implemented partially in src/find_bids/models/classify.

- features.py builds model-ready feature tables (TIER1 feature list).
- core.py is currently a placeholder.

No complete training/inference ML pipeline is present yet.

## 7) What Is Planned But Not Yet In Code

Not implemented in current codebase:

- Graph construction over session nodes and typed edges.
- Pairwise/higher-order potentials for label consistency.
- Loopy belief propagation or other graph inference backends.
- Full entity assignment pipeline (for example robust run/echo/dir/acq/task orchestration).
