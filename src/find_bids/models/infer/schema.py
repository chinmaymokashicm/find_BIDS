"""
bids_schema.py

Strict BIDS schema enforcement for series-level inference.

Encodes:
- Valid datatypes
- Valid suffixes per datatype
- Valid entities per suffix
- Valid values per entity
- Validation logic
"""

from dataclasses import dataclass
import math, re

from enum import Enum
from typing import Optional, Self
from pydantic import BaseModel, model_validator, Field


# ============================================================
# Core Datatypes
# ============================================================

class Datatype(str, Enum):
    ANAT = "anat"
    FUNC = "func"
    DWI = "dwi"
    FMAP = "fmap"
    PERF = "perf"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


# ============================================================
# Controlled Entity Value Enums
# ============================================================

class Part(str, Enum):
    MAG = "mag"
    PHASE = "phase"
    REAL = "real"
    IMAG = "imag"


class Direction(str, Enum):
    AP = "AP"
    PA = "PA"
    LR = "LR"
    RL = "RL"
    SI = "SI"
    IS = "IS"


class MTState(str, Enum):
    ON = "on"
    OFF = "off"


# ============================================================
# BIDS Schema Registry
# ============================================================

# datatype → suffix → allowed entities
# Entities listed here are OPTIONAL unless required by BIDS spec.
# We only encode structural validity here.
BIDS_SCHEMA: dict[Datatype, dict[str, set[str]]] = {

    Datatype.ANAT: {
        "T1w": {"acq", "ce", "rec", "run", "part", "echo", "inv"},
        "T2w": {"acq", "run", "part", "echo"},
        "FLAIR": {"acq", "run"},
        "PD": {"acq", "run"},
    },

    Datatype.FUNC: {
        "bold": {"task", "acq", "run", "echo", "part"},
        "sbref": {"task", "acq", "run"},
    },

    Datatype.DWI: {
        "dwi": {"acq", "dir", "run"},
    },

    Datatype.FMAP: {
        "phasediff": {"run"},
        "magnitude1": {"run"},
        "magnitude2": {"run"},
        "epi": {"dir", "run"},
        "fieldmap": {"run"},
    },

    Datatype.PERF: {
        "asl": {"acq", "run"},
        "m0scan": {"acq", "run"},
    },

    Datatype.EXCLUDE: {},
    Datatype.UNKNOWN: {},
}

pattern_only_alphanum = re.compile(r"^[a-zA-Z0-9]+$")

class LabelProbability(BaseModel):
    label: str = Field(..., pattern=pattern_only_alphanum)
    probability: float

class LabelProbabilities(BaseModel):
    probabilities: list[LabelProbability]
    
    @model_validator(mode="before")
    def validate_probabilities(cls, v: list[dict | LabelProbability]) -> list[LabelProbability]:
        if not isinstance(v, list):
            raise ValueError("LabelProbabilities must be initialized with a list of LabelProbability or dicts.")
        validated = []
        for item in v:
            if isinstance(item, LabelProbability):
                validated.append(item)
            elif isinstance(item, dict):
                validated.append(LabelProbability(**item))
            else:
                raise ValueError(f"Invalid item in probabilities list: {item}")
        return validated
    
    @property
    def most_probable(self) -> Optional[str]:
        if not self.probabilities:
            return None
        return max(self.probabilities, key=lambda lp: lp.probability).label
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the label distribution, as a measure of uncertainty."""
        return -sum(lp.probability * math.log(lp.probability) for lp in self.probabilities if lp.probability > 0)
    
    @property
    def margin(self) -> Optional[float]:
        """Margin between the top two label probabilities, as a measure of confidence."""
        if len(self.probabilities) < 2:
            return None
        sorted_probs = sorted(self.probabilities, key=lambda lp: lp.probability, reverse=True)
        return sorted_probs[0].probability - sorted_probs[1].probability

class BIDSEntities(BaseModel):
    """
    Strict BIDS entity container.

    This represents a *fully inferred* series-level label.
    """

    # Core identifiers
    subject: Optional[str] = None
    session: Optional[str] = None

    # Primary semantic label
    datatype: LabelProbabilities
    suffix: Optional[LabelProbabilities] = None

    # Optional BIDS entities
    task: Optional[str] = Field(default=None, pattern=pattern_only_alphanum)
    acq: Optional[str] = Field(default=None, pattern=pattern_only_alphanum)
    run: Optional[int] = Field(default=None, ge=1)
    echo: Optional[int] = Field(default=None, ge=1)
    inv: Optional[int] = Field(default=None, ge=1)
    ce: Optional[str] = Field(default=None, pattern=pattern_only_alphanum)
    rec: Optional[str] = Field(default=None, pattern=pattern_only_alphanum)
    dir: Optional[Direction] = None
    part: Optional[Part] = None
    mt: Optional[MTState] = None

    @model_validator(mode="after")
    def validate_entities(self) -> Self:
        """Validate that the combination of datatype, suffix, and entities is structurally valid according to BIDS schema."""
        if self.datatype is None or self.datatype.most_probable is None:
            raise ValueError("Datatype must have at least one probable label.")
        inferred_datatype = Datatype(self.datatype.most_probable)
        
        if inferred_datatype in {Datatype.EXCLUDE, Datatype.UNKNOWN}:
            # For EXCLUDE or UNKNOWN, we don't enforce any entity rules
            return self
        
        if self.suffix is None or self.suffix.most_probable is None:
            raise ValueError(f"Suffix must have at least one probable label for datatype {inferred_datatype}.")
        inferred_suffix = self.suffix.most_probable
        valid_suffixes = BIDS_SCHEMA.get(inferred_datatype, {})
        if inferred_suffix not in valid_suffixes:
            raise ValueError(f"Invalid suffix '{inferred_suffix}' for datatype '{inferred_datatype}'. Valid options: {valid_suffixes.keys()}")
        
        remaining_allowed_entities = valid_suffixes[inferred_suffix]
        for entity_name in self.model_fields_set or set():
            if entity_name in {"subject", "session", "datatype", "suffix"}:
                continue  # Skip core identifiers and primary labels
            if getattr(self, entity_name) is not None and entity_name not in remaining_allowed_entities:
                raise ValueError(f"Entity '{entity_name}' is not allowed for datatype '{inferred_datatype}' with suffix '{inferred_suffix}'. Allowed entities: {remaining_allowed_entities}")
        
        return self
    
    @property
    def entities(self) -> dict[str, Optional[str]]:
        """Return a dict of all entities and their values (excluding None)."""
        results = {}
        for field_name, value in self.__dict__.items():
            if isinstance(value, (str, int)):
                results[field_name] = value
            elif isinstance(value, Enum):
                results[field_name] = value.value
            elif isinstance(value, LabelProbabilities):
                # For probabilities, we return the most probable label for simplicity
                results[field_name] = value.most_probable
            else:
                raise ValueError(f"Unsupported entity type for {field_name}: {type(value)}")
                
        return results
    
    def update_probabilities(self, datatype_probs: Optional[list[dict | LabelProbability]] = None, suffix_probs: Optional[list[dict | LabelProbability]] = None):
        """Update the datatype and suffix probabilities with new evidence."""
        if datatype_probs is not None:
            self.datatype = LabelProbabilities(probabilities=[LabelProbability(**dp) if isinstance(dp, dict) else dp for dp in datatype_probs])
        if suffix_probs is not None:
            self.suffix = LabelProbabilities(probabilities=[LabelProbability(**sp) if isinstance(sp, dict) else sp for sp in suffix_probs])