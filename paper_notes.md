### 1. Introduction

* Large-scale neuroimaging datasets remain difficult to organize due to inconsistent and site-specific DICOM metadata.
* The BIDS standard provides a common data model but assumes prior human-driven classification and naming.
* Existing DICOM-to-BIDS tools require user-defined heuristics or protocol knowledge, limiting scalability and reuse.
* We introduce a fully automated semantic inference framework that assigns BIDS entities directly from DICOM data without user heuristics.

### 2. Background and Related Work

#### 2.1 DICOM Metadata Variability

* DICOM headers contain semi-structured, vendor-specific, and free-text fields that vary across scanners and sites.
* Protocol and series descriptions are not standardized and often inconsistent even within a single study.

#### 2.2 BIDS and Current Conversion Tools

* BIDS defines a standardized file organization and naming scheme for neuroimaging data (Gorgolewski et al., 2016).
* Tools such as HeuDiConv and dcm2bids rely on user-written heuristics and protocol naming conventions.
* These approaches are effective for prospective studies but poorly suited for legacy or multi-site datasets.

#### 2.3 Automated Scan Classification

* Prior work has explored scan type classification using image data or limited metadata (e.g., modality detection, protocol classification).
* Existing approaches typically focus on single tasks (e.g., T1 vs T2) and do not map to full BIDS semantics.

### 3. Problem Statement

* BIDS entities vary in how directly they can be inferred from DICOM data.
* Some entities are intrinsic to a single series, while others require dataset-level context.
* There is no formal framework that distinguishes inferable, conditional, and contextual BIDS entities.
* The lack of confidence estimates and explainability limits trust in automated systems.

### 4. Conceptual Framework

#### 4.1 BIDS Entity Tiers

* We define three tiers of BIDS entities based on inferability from DICOM data.
* Tier 1 entities are always required and directly inferable (e.g., datatype, suffix).
* Tier 2 entities are conditional on modality and availability of metadata.
* Tier 3 entities are contextual and require dataset-level reasoning.

#### 4.2 Series-Level vs Dataset-Level Inference

* Series-level inference operates on individual DICOM series without external context.
* Dataset-level inference leverages relationships between series to infer subject, session, and run entities.
* This separation reflects fundamental limits of what can be inferred from DICOM headers alone.

### 5. System Overview

#### 5.1 Input Representation

* The system accepts either a single DICOM series or a collection of series forming a dataset.
* No assumptions are made about naming conventions or protocol annotations.

#### 5.2 Feature Extraction

* Features are derived from DICOM headers, including numeric acquisition parameters and categorical fields.
* Optional image-derived features can be incorporated when metadata alone is insufficient.

#### 5.3 Semantic Inference Engine

* Machine learning and rule-based components are combined to infer BIDS entities.
* Each inferred entity is assigned a confidence score.

#### 5.4 Explainability

* Inference decisions are accompanied by feature-level evidence.
* This enables auditing and human verification without manual heuristics.

### 6. Outputs

* The system produces a structured per-series semantic summary of inferred BIDS entities.
* Outputs are independent of file naming or directory structure.
* Results can be consumed by downstream BIDS converters or QC tools.

### 7. Evaluation

#### 7.1 Datasets

* Evaluation is performed on large, heterogeneous DICOM datasets, including public repositories.
* Datasets span multiple vendors, sites, and acquisition protocols.

#### 7.2 Metrics

* Accuracy of Tier 1 and Tier 2 entity inference.
* Confidence calibration and ambiguity detection.
* Generalization to unseen datasets.

### 8. Discussion

* We analyze failure modes and ambiguous cases.
* We discuss the limits of inference from DICOM headers alone.
* We position the system as a complement, not a replacement, to existing BIDS converters.

### 9. Conclusion and Future Work

* Fully automated semantic inference enables scalable, reproducible organization of legacy neuroimaging data.
* Future work includes active learning, broader modality support, and community benchmarks.
