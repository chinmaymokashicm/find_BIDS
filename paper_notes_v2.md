# Automated DICOM→BIDS conversion with structured probabilistic inference (notes)

## 1. Introduction

- **Motivation: the gap between raw DICOM and usable BIDS**
  - MRI exams generate heterogeneous collections of DICOM series across modalities (structural, diffusion, perfusion, functional), vendors, and sites.
  - The Brain Imaging Data Structure (BIDS) has become a de‑facto standard for organizing neuroimaging data, enabling reuse, reproducibility, and automated pipelines. https://www.nature.com/articles/sdata201644
  - However, converting raw hospital DICOM archives into valid BIDS datasets still typically requires substantial manual effort, especially in multi‑site and clinical settings.

- **Limitations of current DICOM→BIDS tooling**
  - Widely used tools (HeuDiConv, dcm2bids, BIDScoin) typically expect user‑defined heuristic files or templates that encode local naming conventions and protocol patterns; these mappings are brittle under protocol drift. https://pmc.ncbi.nlm.nih.gov/articles/PMC11423922/
  - Semi‑automatic tools such as ezBIDS propose mappings but retain a human‑in‑the‑loop “propose‑and‑revise” workflow and are not designed as fully automated, constraint‑aware inference systems. https://www.nature.com/articles/s41597-024-02959-0

- **Limitations of current automatic modality/sequence classifiers**
  - Metadata‑ and image‑based classifiers reliably distinguish T1/T2/FLAIR, diffusion, and functional MRI from DICOM tags and/or voxels, often with random forests or CNNs. https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/
  - These methods usually stop at coarse sequence labels and do not produce globally consistent BIDS datatypes/suffixes/entities over an entire exam.

- **Core idea / hypothesis**
  - BIDS mapping is a **structured prediction** problem: series labels are not independent, because a session is governed by a protocol with predictable co‑occurrence patterns (e.g., BOLD runs with supporting field maps, repeated runs, echo pairs).
  - A hybrid approach that combines **local per‑series classification** with **global consistency reasoning** should reduce errors on real‑world messy DICOM.

- **Resulting problem**
  - There is currently no general, lightweight system that:
    - takes raw DICOM series as input,
    - uses primarily DICOM headers (voxels as fallback) with minimal user configuration,
    - and outputs a **BIDS‑formatted dataset** with inferred BIDS entities, while enforcing cross‑series consistency.

- **High‑level goal of this work**
  - Propose and evaluate a structured probabilistic pipeline that:
    1) extracts robust series‑level features from DICOM,
    2) predicts probabilistic label distributions per series (datatype/suffix/entities),
    3) refines predictions using an MRF/CRF‑style model to enforce protocol‑ and physics‑motivated constraints,
    4) outputs BIDS‑compliant paths/filenames + sidecar JSONs and validates with the BIDS Validator.

- **Scope and design choices**
  - Focus on common MRI modalities used in neuro‑oncology and neuroimaging: structural (T1/T2/FLAIR), diffusion (DWI/DTI), perfusion (DSC/DCE/ASL), functional (BOLD and related time‑series), and supporting field maps.
  - Prefer **interpretable constraints** and **debuggable** inference; use learned components where rules are insufficient.
  - Emphasize scalability to very large clinical archives by decomposing inference into **per‑exam graphs**.

- **Contributions (targeted)**
  - A series‑level DICOM feature extraction strategy (robust aggregation over instances).
  - A local probabilistic classifier for datatype/suffix (+ limited entity set) with calibrated outputs.
  - A global constraint‑aware inference layer (MRF/CRF / factor graph) producing **session‑consistent** BIDS labels.
  - A large‑scale evaluation on heterogeneous clinical DICOM demonstrating improved session‑level correctness and reduced manual curation.

***

## 2. Background and Related Work

### 2.1 The BIDS standard and entities

- BIDS specifies a file/directory structure, naming conventions, and metadata requirements for neuroimaging datasets. https://bids-specification.readthedocs.io/
- Key concepts relevant to this work:
  - **Datatype**: `anat`, `func`, `dwi`, `fmap`, `perf`, etc.
  - **Suffix**: `T1w`, `T2w`, `FLAIR`, `dwi`, `bold`, `asl`, etc.
  - **Entities**: `sub-`, `ses-`, `task-`, `acq-`, `dir-`, `run-`, `echo-`, etc.
- Practical emphasis for automation:
  - `sub-`/`ses-` often come from DICOM PatientID / StudyDate / accession systems and require site‑specific handling.
  - Entities such as `task-` may not be identifiable from DICOM alone; in contrast, `dir-` and `echo-` are often inferable if the relevant tags are present.

### 2.2 DICOM→BIDS conversion tools

- **HeuDiConv / ReproIn**: powerful but typically depends on user‑authored heuristics and prospective scanner naming conventions. https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/heudiconv.html
- **BIDScoin**: template‑driven mapping with GUI editing and reuse of bidsmaps; still relies on curated templates. https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.770608/full
- **ezBIDS**: web‑based workflow that proposes mappings from DICOM metadata and requires user inspection/editing; strong practical heuristics but not a formal structured inference engine. https://www.nature.com/articles/s41597-024-02959-0

### 2.3 Automatic classification of MRI series from DICOM metadata and images

- **Metadata‑based classification**: random forests on DICOM tags (TR/TE/TI, ImageType, ScanningSequence, number of images) achieve high accuracy for classifying common sequences and junk scans. https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/
- **Metadata + image models**: combining header features with CNNs on voxels improves robustness when headers are incomplete or misleading. https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/
- **Gap to BIDS**: these works typically output sequence types, not BIDS‑consistent (datatype, suffix, entity) assignments.

### 2.4 Representation learning and protocol structure

- **Self‑supervised protocol representations**: Protocol Genome proposes self‑supervised learning on structured DICOM headers (and related tasks) to learn embeddings that capture protocol structure at scale. https://arxiv.org/pdf/2509.06995.pdf
- **Graph approaches to protocol learning**: graph representation learning has been explored for learning imaging protocol structure and variation in operational contexts (protocol management / drift), suggesting graphs are a natural substrate for modeling series relationships. https://siim.org/wp-content/uploads/2023/08/using_graph_representation_l.pdf
- **Gap to BIDS**: these approaches learn structure but do not directly perform DICOM→BIDS entity assignment with explicit BIDS constraints.

***

## 3. Proposed Method (structured probabilistic inference)

### 3.1 Problem formulation

- Input: a set of DICOM series grouped by exam / study (optionally sessionized), with series‑level feature vectors derived from DICOM headers and basic acquisition statistics.
- Output: BIDS labels for each series including:
  - core: datatype + suffix,
  - selected entities where feasible: `run-`, `dir-`, `echo-`, `acq-` (optionally `task-` when protocol names are reliable).

### 3.2 Series‑level feature extraction

- Aggregate instance‑level tags into robust series descriptors (median, mode, IQR, counts), e.g.:
  - TR/TE/TI distributions,
  - number of images / number of volumes,
  - voxel size and matrix,
  - slice timing indicators,
  - orientation and position consistency,
  - ImageType patterns,
  - diffusion b‑values/b‑vectors availability,
  - vendor‑specific tags when available.

### 3.3 Local probabilistic classifier (unary potentials)

- Train a supervised model to map series features → a calibrated probability distribution over candidate BIDS labels.
- Recommended design:
  - hierarchical heads (datatype → suffix → entities) to avoid label explosion,
  - probability calibration (isotonic / temperature scaling),
  - top‑K candidate pruning per node to make global inference tractable.

### 3.4 Session graph construction

- Build a graph per exam:
  - nodes = imaging series,
  - edges encode relationships (configurable):
    - same session membership (StudyInstanceUID, or site logic),
    - acquisition adjacency (SeriesNumber / AcquisitionTime ordering),
    - similarity edges (parameter similarity / shared protocol embedding neighborhood),
    - modality compatibility edges (e.g., fmap–func, fmap–dwi).

### 3.5 Global consistency model (pairwise / higher‑order factors)

- Use an MRF/CRF / factor graph to refine per‑series predictions.
- Components:
  - **Unary potentials**: derived from the local classifier’s log‑probabilities.
  - **Pairwise potentials**: encode domain‑informed compatibility and co‑occurrence.
  - Optional **hard constraints**: enforce impossible combinations to have zero probability.

Examples of constraint families:

- **Compatibility constraints**
  - Certain suffix/entity combinations are invalid (e.g., `bold` should not co‑occur with `anat` datatype).
  - `dir-` should be consistent within a phase‑encoding group.

- **Co‑occurrence constraints**
  - BOLD runs often co‑occur with SBRef and/or supporting field maps; diffusion scans often co‑occur with b0s.

- **Ordering / adjacency constraints**
  - Field maps are frequently acquired adjacent to the runs they support; repeated runs appear as near‑duplicates in parameters.

- **Run indexing constraints**
  - `run-` should be contiguous and unique within a set of repeated acquisitions with near‑identical parameters.

- **Protocol‑variant constraints**
  - If a session splits into clusters in parameter space (CuBIDS‑style), encourage distinct `acq-` labels for distinct clusters.

### 3.6 Inference

- Perform approximate inference per exam graph (options):
  - loopy belief propagation,
  - mean‑field inference,
  - MAP decoding via ILP (integer linear programming) for “hard‑constraint heavy” variants.
- Output either:
  - MAP assignment (single best labeling), and/or
  - marginals/uncertainties to drive QC triage.

### 3.7 Practical engineering for millions of series

- Decompose into per‑exam graphs; parallelize across exams.
- Cache feature extraction and unary predictions.
- Use label pruning and sparse edges to control runtime.
- Prefer simple potentials and small label sets for early prototypes.

***

## 4. Evaluation plan

- Datasets:
  - multi‑site hospital DICOM with broad protocol drift,
  - smaller curated subset with ground‑truth BIDS mappings.

- Baselines:
  - rules/heuristics only,
  - local classifier only (no global inference),
  - existing tools in default or minimally tuned configurations (ezBIDS/BIDScoin/HeuDiConv), where comparable.

- Metrics:
  - per‑series accuracy for datatype/suffix and selected entities,
  - session‑level correctness (e.g., correct pairing of fmap↔bold runs, consistent dir-),
  - constraint satisfaction rates,
  - downstream validity: BIDS Validator pass rate, proportion requiring manual edits.

***

## 5. Limitations and failure modes (to acknowledge explicitly)

- Some entities are not identifiable from DICOM alone (notably `task-` in many settings).
- Headers may be missing/incorrect in clinical data; the model must tolerate missingness.
- Over‑strong constraints can propagate local mistakes into globally consistent but wrong labelings.
- Site‑specific idiosyncrasies (protocol naming, scanner upgrades) can cause distribution shift; consider active learning / periodic recalibration.

***

## 6. Suggested extensions

- Use self‑supervised header embeddings (Protocol Genome–style) as input features for local classifiers and for defining similarity edges. https://arxiv.org/pdf/2509.06995.pdf
- Add a human‑in‑the‑loop “uncertainty triage” mode: only surface sessions with high posterior entropy for review.
- Consider a modular factor graph: one layer for datatype/suffix, another for run-/dir- assignment, to limit label‑space explosion.
