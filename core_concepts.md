# **Probabilistic, Graph-Constrained Inference for Automated DICOM-to-BIDS Semantic Categorization**

---

## 1. Motivation

* Multi-site hospital DICOM datasets contain:

  * Vendor-specific metadata variations
  * Free-text protocol names
  * Inconsistent acquisition parameter usage
  * Derived image series mixed with raw data
* Manual curation of MRI series into BIDS:

  * Does not scale to millions of series
  * Requires domain expertise
  * Is brittle across institutions
* Pure rule-based approaches fail due to:

  * Parameter variability
  * Free-text naming heterogeneity
  * Derived/processed series ambiguity
* Pure ML approaches fail due to:

  * Lack of global context
  * Inconsistent session-level configurations
  * Ignoring physical acquisition constraints

---

## 2. Objective

* Develop a scalable, vendor-agnostic framework that:

  * Assigns core BIDS semantic labels to MRI series
  * Is robust to heterogeneous clinical DICOM data
  * Incorporates both local evidence and global study-level structure
  * Produces interpretable and publishable inference logic
* Focus on:

  * Correct semantic categorization
  * Not full reproduction of every optional BIDS entity

---

## 3. Scope of BIDS Entities

### Core Semantic Identity (Primary Target)

* Joint label:

  * `datatype + suffix`
* Examples:

  * `anat_T1w`
  * `anat_T2w`
  * `func_bold`
  * `func_sbref`
  * `dwi_dwi`
  * `fmap_epi`
  * `fmap_phasediff`

### Deterministic Entities (Rule-Based)

* `echo`
* `dir`
* `part`
* `phase encoding direction`
* 3D vs 4D
* volume count

### Relational / Contextual Entities (Graph-Level)

* `run`
* SBRef–BOLD pairing
* Fieldmap–target matching (`IntendedFor`)
* Ambiguity resolution between competing semantic labels

---

## 4. Method Overview

### Layer 1 — Deterministic Feature Extraction

* Extract structured DICOM metadata
* Normalize numeric parameters:

  * TR, TE, FlipAngle
  * Voxel spacing
  * Volume count
* Parse categorical metadata:

  * Manufacturer
  * ImageType
* Tokenize free-text fields:

  * SeriesDescription
  * ProtocolName
* Compute derived features:

  * Is4D
  * Slice count
  * Temporal length
  * Geometry signature

---

### Layer 2 — Local Supervised Classification

* Input: Feature vector per series
* Output: Probability distribution over joint semantic labels

#### Model Options:

* Gradient Boosted Trees (e.g., XGBoost)
* Random Forest
* Lightweight transformer for text + structured fusion

#### Output:

* `P_local(label | features)`

---

### Layer 3 — Probabilistic Study Graph

#### Graph Construction

* Node:

  * One node per imaging series
  * Random variable: semantic label

* Edge Types:

  * Same-session membership
  * Temporal adjacency (acquisition order)
  * Geometry similarity
  * Candidate fieldmap–bold pairing
  * SBRef–BOLD pairing
  * Mutual exclusion constraints

Graph is sparse and typed.

---

## 5. Probabilistic Model

We define a Markov Random Field (MRF):

### Nodes

* Random variable:

  * ( Y_i \in \mathcal{L} )
* ( \mathcal{L} ): finite set of joint semantic labels

---

### Node Potentials

* Derived from local classifier:

  * ( \phi_i(y_i) = P_{local}(y_i) )

---

### Edge Potentials

* Compatibility matrix:

  * ( \psi_{ij}(y_i, y_j) )
* Encodes:

  * Physical plausibility
  * Mutual exclusivity
  * Relational reinforcement

Examples:

* Encourage:

  * `fmap_epi` ↔ `func_bold`
  * `func_sbref` ↔ `func_bold`
* Penalize:

  * Multiple incompatible anatomical identities
  * Fieldmaps without valid targets

---

### Joint Distribution

[
P(Y) \propto \prod_i \phi_i(y_i) \prod_{(i,j) \in E} \psi_{ij}(y_i, y_j)
]

---

### Inference

* Loopy Belief Propagation
* Mean-field approximation
* Iterated Conditional Modes (ICM)
* Objective:

  * MAP labeling of all series in a session

---

## 6. Run Index Assignment

After semantic labels are finalized:

* For each session:

  * Group by semantic identity
  * Sort by acquisition time
  * Assign run indices deterministically

Run assignment is:

* Not a classification problem
* A session-level ordering problem

---

## 7. Training Strategy

### Labeled Data

* Expert-labeled subset of sessions
* Balanced across:

  * Vendors
  * Institutions
  * Protocol variants

### Training Steps

1. Train local classifier
2. Freeze classifier
3. Tune graph compatibility weights:

   * Grid search
   * Cross-validation
   * Optional gradient-based learning

---

## 8. Advantages of Hybrid Approach

* Robust to noisy free-text naming
* Enforces physics-based constraints
* Reduces impossible label combinations
* Interpretable:

  * Potentials explicitly encode domain knowledge
* Modular:

  * ML component can improve independently
  * Graph rules can evolve over time

---

## 9. Scalability

* Local classification:

  * O(N)
* Graph inference:

  * Per-session graph
  * Small connected components
* Scales to millions of series via session partitioning

---

## 10. Deliverables

### Algorithmic Contribution

* Structured probabilistic inference for DICOM semantic labeling
* Hybrid ML + graphical model architecture

### Software Contribution

* Python package:

  * Feature extraction module
  * Pretrained classifier weights
  * Graph inference engine
  * Configurable compatibility matrices

### Reproducibility

* Publish:

  * Trained model weights
  * Graph configuration
  * Label taxonomy
  * Evaluation benchmarks

---

## 11. Evaluation Metrics

* Per-series semantic accuracy
* Session-level consistency accuracy
* Fieldmap-target matching accuracy
* Cross-vendor generalization
* Performance under noisy metadata

---

## 12. Positioning

This work is:

* Not merely a DICOM parser
* Not merely a BIDS renaming tool
* A structured semantic inference system for heterogeneous clinical MRI datasets

---

## 13. Core Contribution Statement

* Introduces a hybrid probabilistic framework combining supervised feature-based classification with study-level graphical consistency modeling.
* Demonstrates improved robustness over rule-based and local-only ML approaches.
* Enables scalable, vendor-agnostic semantic categorization of large-scale clinical DICOM repositories.

---

If you want, next we can:

* Formalize the mathematical section more rigorously
* Define a minimal publishable label taxonomy
* Or outline experiments for a methods paper

You’re now very close to something that can stand up in peer review.
