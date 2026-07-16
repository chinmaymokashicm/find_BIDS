## 3.X BIDS entity categories and inference strategy

### 3.X.1 Conceptual categories of entities

| Category                    | Examples                          | Primary information source                            | Inference stage (local vs. graph)                           | Role of graph reasoning                                                  |
|----------------------------|-----------------------------------|--------------------------------------------------------|----------------------------------------------------------------|---------------------------------------------------------------------------|
| Intrinsic series-level     | `datatype`, `suffix`, basic `echo-`, basic `dir-` | TR/TE/TI, ImageType, ScanningSequence, geometry, diffusion tags | Mainly **local ML / rules** (hierarchical classifier)         | Optional smoothing / refinement of ambiguous labels using neighborhood context |
| Contextual / study-level   | `sub-`, `ses-`, acquisition date, scanner ID      | Patient/study identifiers, StudyInstanceUID, AccessionNumber    | **Pre-graph rules** and site-specific mappings                | Provide grouping into exam graphs; not usually modified by graph itself      |
| Relational (between series)| fmap↔BOLD pairing, fmap↔DWI pairing, SBRef↔BOLD, parameter-based protocol variants (`acq-`) | Series adjacency, parameter similarity, protocol name patterns | **Graph core**: inferred from structure of the exam graph     | Assign relational entities and links (e.g., which fmap serves which runs; which clusters get distinct `acq-`) |
| Ordinal / indexing         | `run-` indices, `echo-` indices, repetition ordering | AcquisitionTime / SeriesNumber ordering, parameter near-duplicates | Shared between **local heuristics** and **graph constraints** | Enforce contiguity and uniqueness of indices; break ties when multiple plausible orderings exist |
| Derived / higher-order     | QC flags, variant group IDs, CuBIDS-style parameter groups | BIDS sidecar JSONs, post-conversion metadata                   | Post-hoc analysis on BIDS outputs                            | Can feed back as soft evidence to refine or relabel ambiguous series in later passes |

Notes:
- Some entities (e.g., `task-`) are only weakly or indirectly encoded in DICOM and may require external sources (behavioral logs, naming conventions); these are out of scope for the core probabilistic model or are handled as optional, site-specific rules.
- The graph focuses on entities that **depend on relationships among series within an exam**, not just on single-series metadata.

### 3.X.2 Pipeline ordering of entity extraction

1. **Contextual grouping and identifiers (pre-graph)**
   - Normalize and map `sub-`/`ses-` from PatientID / StudyInstanceUID / accession systems using site-specific rules.
   - Group series into exam graphs (per subject–session), which define the node sets for subsequent inference.

2. **Local intrinsic labeling (unary stage)**
   - Apply the hierarchical series classifier to predict a probability distribution over `(datatype, suffix)` for each series.
   - Where tags are reliable, derive provisional `echo-` and `dir-` directly from DICOM fields (e.g., EchoNumber, PhaseEncodingDirection equivalents), with calibrated confidences.
   - Retain the top-K candidate labels per series as the support of the unary potentials.

3. **Graph-based refinement and relational entity assignment**
   - On each exam graph, run MRF/CRF-style inference with:
     - Unary potentials from local classifier probabilities and direct tag-derived hints.
     - Pairwise / small higher-order potentials encoding:
       - compatibility (e.g., fmap labels must be compatible with the datatypes of their targets),
       - co-occurrence (e.g., BOLD runs should have appropriate supporting field maps),
       - adjacency (e.g., field maps and SBRefs tend to be near their associated runs),
       - ordering/indexing (e.g., `run-` indices are contiguous within a near-duplicate series cluster).
   - Use the resulting MAP or marginal distributions to:
     - **Refine datatype/suffix** for ambiguous series (e.g., decide between closely related contrasts by looking at protocol context),
     - **Assign relational entities**: which field map serves which BOLD/DWI run, which near-duplicate cluster gets which `acq-` label, how `run-` and `echo-` indices are ordered,
     - **Enforce relational consistency**: prevent invalid combinations (e.g., duplicate `run-` indices, orphaned field maps) and push clearly inconsistent series into an `exclude/unknown` label when necessary.

4. **Post-graph validation and feedback**
   - Materialize BIDS filenames and sidecars, run the BIDS Validator, and compute exam-level metrics (e.g., proportion of BOLD runs with matched field maps).
   - Optionally feed post-hoc information (QC metrics, parameter-group IDs) back into the model for retraining or for a subsequent correction pass on problematic exams.

This section is intended to sit under the "Proposed Method" part of the notes and make explicit:
- which entities are primarily intrinsic vs. relational,
- which are assigned by the local classifier vs. the graph,
- and what high-level responsibilities the graph has (refinement, consistency, and relational entity inference).