## 1. Introduction

- **Motivation: the gap between raw DICOM and usable BIDS**
  - MRI studies generate large, heterogeneous collections of DICOM series across modalities (structural, diffusion, perfusion, functional), vendors, and sites. [nature](https://www.nature.com/articles/sdata201644)
  - The Brain Imaging Data Structure (BIDS) has become a de‑facto standard for organizing neuroimaging data, enabling reuse, reproducibility, and automated pipelines. [bids-specification.readthedocs](https://bids-specification.readthedocs.io/en/v1.6.0/01-introduction.html)
  - However, converting raw DICOM archives into valid BIDS datasets still typically requires substantial manual effort, especially in multi‑site and clinical settings.

- **Limitations of current DICOM→BIDS tooling**
  - Widely used tools (HeuDiConv, dcm2bids, BIDScoin) expect user‑defined heuristic files or configuration templates that encode local naming conventions and protocol patterns. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11423922/)
  - These mappings are brittle under protocol drift and require ongoing human maintenance.
  - Recent semi‑automatic tools (ezBIDS, BIDScoin’s bidsmapper) propose mappings but still rely on substantial user review and editing before BIDS export. [frontiersin](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.770608/full)

- **Limitations of current automatic sequence / modality classifiers**
  - Metadata‑ and image‑based sequence classifiers reliably distinguish T1/T2/FLAIR, diffusion, and functional MRI, often with random forests or CNNs on a small set of DICOM tags or voxels. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)
  - These methods typically stop at assigning coarse sequence labels (e.g., “3D T1”, “DTI”, “fMRI”), without generating BIDS datatypes, suffixes, or entity labels.

- **Resulting problem**
  - There is currently no general, lightweight system that:
    - takes raw DICOM data as input,
    - uses primarily DICOM headers (with voxels as fallback) and minimal user configuration,
    - and outputs a **BIDS‑formatted dataset** with inferred entities for common MRI modalities.

- **High‑level goal of this work**
  - Propose and evaluate a **hierarchical inference pipeline** that:
    - operates primarily on **series‑level DICOM header statistics** (central tendency and variation across instances),
    - combines **rules, heuristics, and classical machine learning** (random forests / decision trees) in a targeted way,
    - first infers BIDS **datatype** (e.g., anat, dwi, func, perf, fmap),
    - then infers BIDS **suffix** (e.g., T1w, T2w, FLAIR, dwi, bold, ASL, perf),
    - and finally predicts a **restricted set of additional BIDS entities** (e.g., `acq-`, `dir-`, `run-`) where the evidence is reliable.

- **Scope and design choices**
  - Focus on common MRI modalities used in neuro‑oncology and neuroimaging:
    - structural (T1/T2/FLAIR),
    - diffusion (DWI/DTI),
    - perfusion (DSC/DCE/ASL),
    - functional (BOLD and related time‑series).
  - Prefer **simple, interpretable rules and heuristics** grounded in acquisition physics and DICOM semantics.
  - Use classical ML only where rules are insufficient or brittle, and keep the number of learned decision points small.
  - Use voxel data only as a **fallback** when header‑based inference is ambiguous or inconsistent.

- **Contributions**
  - A **series‑level feature extraction strategy** from DICOM, aggregating instance‑level tags into robust descriptors (e.g., median TR/TE/TI, number of dynamic volumes, variation in ImagePositionPatient).
  - A **hierarchical inference framework** that:
    - first classifies modality family (datatype),
    - then refines to sequence subtype (suffix),
    - then infers a minimal set of BIDS entities.
  - A **hybrid rules+ML approach** that:
    - encodes domain knowledge as deterministic rules where possible,
    - uses small classical models to resolve residual ambiguity.
  - An empirical evaluation on heterogeneous MRI datasets demonstrating:
    - high accuracy for key BIDS entities across structural, diffusion, perfusion, and functional series,
    - robustness across scanners and sites,
    - and a clear reduction in required manual intervention compared to current semi‑automatic tools.

***

## 2. Background and Related Work

### 2.1 The BIDS standard and entities

- **Overview of BIDS**
  - BIDS specifies a file and directory structure, naming conventions, and metadata requirements for neuroimaging datasets. [nature](https://www.nature.com/articles/sdata201644)
  - Original focus on MRI (structural, diffusion, functional), extended over time to more modalities and quantitative MRI. [semanticscholar](https://www.semanticscholar.org/paper/The-brain-imaging-data-structure,-a-format-for-and-Gorgolewski-Auer/b3b5993819f5dddeeff1a99c43545d1133296881)

- **Key concepts relevant to this work**
  - **Datatype** (top‑level folders): `anat`, `func`, `dwi`, `fmap`, `perf`, etc.
  - **Suffix** (file “kind”): `T1w`, `T2w`, `FLAIR`, `PD`, `dwi`, `bold`, `asl`, `cbv`, etc.
  - **Entities** (key–value pairs in filename): `sub-`, `ses-`, `task-`, `acq-`, `dir-`, `run-`, `echo-`, among others. [bids-specification.readthedocs](https://bids-specification.readthedocs.io/en/v1.6.0/01-introduction.html)
  - For the purposes of automated DICOM→BIDS, **datatype, suffix, and a small subset of entities (sub, ses, acq, dir, run)** are the most practically inferable from headers.

- **Current recommendations and limitations**
  - BIDS documentation provides guidance on how acquisition parameters map to suffixes and entities (e.g., TR/TE/TI ranges for T1w/T2w, BOLD vs ASL perfusion), but does not prescribe automated inference algorithms. [bids-specification.readthedocs](https://bids-specification.readthedocs.io/en/v1.6.0/01-introduction.html)
  - In practice, BIDS compliance is enforced via validation tools (BIDS Validator) but the actual mapping from raw DICOM to BIDS structures remains ad hoc.

### 2.2 DICOM→BIDS conversion tools

- **HeuDiConv and ReproIn**
  - HeuDiConv uses dcm2niix plus a user‑written Python heuristic file that inspects DICOM header fields (e.g., SeriesDescription, protocol names, ImageType) to assign BIDS paths and entities. [neuroimaging-core-docs.readthedocs](https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/heudiconv.html)
  - ReproIn defines a naming convention for sequences at the scanner; if followed prospectively, HeuDiConv can generate BIDS datasets in a nearly automatic fashion for those protocols. [github](https://github.com/ReproNim/reproin)
  - Strength: flexible, powerful for well‑controlled environments.
  - Limitation: high up‑front human effort, poor portability across sites, and ongoing maintenance under protocol drift.

- **BIDScoin**
  - BIDScoin introduces a two‑step mapping:
    - `bidsmapper` scans the dataset and groups series into “source data types” using DICOM header patterns and a **template bidsmap**. [frontiersin](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.770608/full)
    - `bidseditor` allows the user to edit and approve mappings via a GUI; `bidscoiner` then applies these to convert to BIDS. [frontiersin](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.770608/full)
  - It automates series grouping and initial mapping suggestions but still assumes a curated template and human review.
  - Focus is on user‑friendliness rather than full automation.

- **ezBIDS**
  - ezBIDS provides a web‑based, semi‑automatic workflow:
    - Parses DICOM (or dcm2niix output), groups images into series, and **proposes** BIDS datatypes, suffixes, subject/session IDs, and some entities based on heuristics over headers and paths. [nature](https://www.nature.com/articles/s41597-024-02959-0)
    - Users then inspect and edit the proposed mapping in a GUI; ezBIDS generates a full BIDS dataset and runs the BIDS validator. [nature](https://www.nature.com/articles/s41597-024-02959-0)
  - Key point: ezBIDS **does learn and encode many practical heuristics**, but retains a human‑in‑the‑loop “Propose‑and‑Revise” design.

- **Post‑BIDS curation: CuBIDS**
  - CuBIDS operates on already‑BIDS datasets, clustering scans into “parameter groups” and “variant groups” based on BIDS JSON metadata (TR, TE, flip angle, resolution, etc.). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9981813/)
  - Aims to discover and label protocol variants (e.g., different TR or slice thickness) and improve downstream harmonization.
  - Demonstrates the value of parameter‑based clustering but does not address DICOM→BIDS mapping.

- **Summary of limitations**
  - Existing tools either:
    - require user‑crafted mapping rules/templates, or
    - propose mappings but rely on manual confirmation.
  - None systematically exploit a **hierarchical, modality‑aware inference pipeline** that prioritizes DICOM headers, uses classical ML sparingly, and uses voxel data only as fallback.

### 2.3 Automatic classification of MRI series from DICOM metadata and images

- **Metadata‑based sequence identification**
  - Gauriau et al. used random forests on a small set of MR‑related DICOM tags (TR, TE, TI, ScanningSequence, ImageType, number of images, etc.) to classify brain MRI series into T1, T2, FLAIR, diffusion, MRA, localizers, and others with high accuracy and sub‑millisecond inference time. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256138/)
  - Liang et al. applied a similar metadata‑learning approach in multi‑site datasets, classifying 3D T1, 2D FLAIR, PD/T2, fMRI, DTI, and “junk” scans, achieving >99.9% accuracy with random forests on header features alone. [frontiersin](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.622951/full)
  - These works demonstrate that **coarse modality and sequence type are strongly encoded in DICOM headers** and amenable to lightweight classical ML.

- **Contrast and within‑modality classification**
  - Cluceru et al. compared:
    - rule‑based classification from TE/TR/TI/Flip and related tags,
    - metadata‑only random forests,
    - image‑only CNNs,
    - and combined models, for structural contrast classification (T1, T1C, T2, FLAIR, PD, OTHER). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984597/)
  - Metadata‑only RFs achieved ~95% accuracy; combined metadata + CNN achieved ~97–98%, showing:
    - rules alone are often insufficient,
    - classical ML models on a small, interpretable feature set are robust and lightweight,
    - voxel data can complement headers when metadata are incomplete or misleading.

- **Limitations relative to BIDS**
  - These classifiers:
    - focus on **sequence labels**, not BIDS entities.
    - rarely consider perfusion subtypes (DSC vs DCE vs ASL) or detailed functional variants.
    - do not attempt to generate BIDS directory structures or filenames.

### 2.4 Representation learning and clustering from DICOM metadata and voxels

- **Protocol‑level representation learning**
  - Recent self‑supervised frameworks (e.g., “Protocol Genome”) treat DICOM headers as structured input and learn latent representations capturing acquisition protocols across multiple imaging modalities. [wjaets](https://wjaets.com/sites/default/files/fulltext_pdf/WJAETS-2023-0319.pdf)
  - These methods show the feasibility of learning **hierarchical structure in protocol space** from headers (and sometimes voxel data), but do not directly target BIDS mapping.

- **Metadata‑based clustering in BIDS space**
  - CuBIDS’s parameter and variant grouping effectively performs hierarchical clustering within each BIDS datatype using acquisition parameters (TE, TR, flip, resolution), highlighting heterogeneity and suggesting refined `acq-` labels. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9981813/)
  - Indicates that structured, parameter‑based clustering can reveal meaningful sub‑protocols within modalities.

- **Gap to be filled**
  - Existing representation learning and clustering approaches:
    - are not integrated into DICOM→BIDS conversion pipelines,
    - typically treat modality/sequence labels as given, rather than outputs,
    - and do not explicitly follow the BIDS hierarchy (datatype → suffix → entities).

***
