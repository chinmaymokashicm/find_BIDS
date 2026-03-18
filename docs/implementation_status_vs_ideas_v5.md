# Implementation Status Vs Ideas_v5

This document maps Ideas_v5 concepts to the codebase state as of now.

## Summary

The repository currently delivers a robust metadata extraction layer and a heuristic inference engine with acquisition families, intent flags, datatype/suffix scoring, and derivedness scoring.

The classifier and graph-based global refinement phases from Ideas_v5 remain future work.

## Concept Mapping

| Ideas_v5 concept | Status | Current code location | Notes |
|---|---|---|---|
| Series-level feature representation | Implemented | src/find_bids/models/extract/series.py, src/find_bids/models/extract/dataset.py | Rich DICOM feature extraction and dataset traversal are in place. |
| Acquisition family inference (EPI, GRE, SE/TSE, Other) | Implemented | src/find_bids/models/infer/acquisition.py | Refactored to match four-family framing. |
| Orthogonal intent flags (diffusion, perfusion, fieldmap, localizer) | Implemented | src/find_bids/models/infer/acquisition.py | Intent scores are computed separately from family. |
| Family-to-datatype soft compatibility | Implemented | src/find_bids/models/infer/datatype.py | Family and intent guidance shifts datatype scores before softmax. |
| Heuristic suffix/bucket scoring | Implemented | src/find_bids/models/infer/suffix.py | Includes raw suffixes plus coarse derived buckets. |
| Derived status axis (raw vs derived) | Implemented | src/find_bids/models/infer/core.py | Separate heuristic score transformed into derived/raw probabilities. |
| Coarse derived buckets by datatype | Implemented | src/find_bids/models/infer/schema.py, src/find_bids/models/infer/suffix.py | Buckets are encoded in schema and scored in suffix logic. |
| Weak supervision and manual verification loop | Partially implemented | src/find_bids/models/annotate/core.py | Annotation models/exports exist; full UI and end-to-end active loop are pending. |
| Discriminative node classifier | Partially implemented | src/find_bids/models/classify/features.py, src/find_bids/models/classify/core.py | Feature table prep exists; model training/inference implementation is not complete. |
| Session graph construction | Not implemented | N/A | No graph object or edge-building module yet. |
| Pairwise potentials / MRF or factor graph | Not implemented | N/A | No graph potential code currently present. |
| Loopy belief propagation refinement | Not implemented | N/A | Not yet implemented. |
| Advanced entity assignment (run/echo/dir/acq/task orchestration) | Partially implemented | src/find_bids/models/infer/schema.py | Schema supports entity fields, but a full automated assignment pipeline is not complete. |

## Current Strengths

- Interpretable, modular heuristics.
- Strong metadata-driven extraction and flattening.
- Explicit handling of derived series and coarse buckets.
- Clear separation between family, intent, and derivedness.

## Current Gaps Relative To Ideas_v5

- No learned node potential model yet.
- No graph-level refinement for session consistency.
- No inference backend for message passing or constrained MAP decoding.
- No production command-line entrypoint.

## Suggested Near-Term Milestones

1. Finalize classifier training and prediction package in classify/core.py.
2. Add reproducible evaluation scripts for datatype/suffix/derived metrics.
3. Implement session graph construction and baseline pairwise compatibility factors.
4. Add a first graph inference backend (for example loopy belief propagation).
5. Expand entity assignment modules after graph refinement is stable.
