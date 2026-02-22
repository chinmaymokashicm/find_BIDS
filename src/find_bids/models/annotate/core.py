"""
Prepare annotation data structures to annotate DICOM series with BIDS labels.
The idea is to have a Dockerized annotation UI that can be used to annotate series on a per-session basis.

- Aggregate series on a session-level across datasets.
- Present series within the context of the session and dataset to the user.
- Allow the user to annotate series with BIDS labels (datatype and suffix).
- Store annotations in a standardized format (e.g., JSON) for downstream use in BIDS conversion and inference model training.
"""
from pathlib import Path

from ..extract.series import SeriesFeatures
from ..infer.schema import BIDS_SCHEMA, Datatype

from typing import Optional, Any, Self

from pydantic import BaseModel, Field, model_validator
import pandas as pd

class DatatypeAnnotation(BaseModel):
    datatype: Datatype = Field(..., description="The BIDS datatype assigned to the series (e.g., 'anat', 'func').")
    confidence: Optional[int] = Field(None, description="Confidence score for the annotation (if applicable).", ge=0, le=10)
    notes: Optional[str] = Field(None, description="Additional notes or comments about the annotation.")
    
    def annotate(self, new_datatype: Datatype, confidence: Optional[int] = None, notes: Optional[str] = None) -> Self:
        """Return a new DatatypeAnnotation with updated values."""
        return self.model_copy(update={
            "datatype": new_datatype,
            "confidence": confidence if confidence is not None else self.confidence,
            "notes": notes if notes is not None else self.notes
        })
    
class SuffixAnnotation(BaseModel):
    suffix: str = Field(..., description="The BIDS suffix assigned to the series (e.g., 'T1w', 'bold').")
    confidence: Optional[int] = Field(None, description="Confidence score for the annotation (if applicable).", ge=0, le=10)
    notes: Optional[str] = Field(None, description="Additional notes or comments about the annotation.")

    def annotate(self, new_suffix: str, confidence: Optional[int] = None, notes: Optional[str] = None) -> Self:
        """Return a new SuffixAnnotation with updated values."""
        return self.model_copy(update={
            "suffix": new_suffix,
            "confidence": confidence if confidence is not None else self.confidence,
            "notes": notes if notes is not None else self.notes
        })

class SeriesAnnotation(BaseModel):
    features: SeriesFeatures = Field(..., description="Extracted features from the DICOM series.")
    datatype: DatatypeAnnotation = Field(..., description="Annotation for the BIDS datatype of the series.")
    suffix: Optional[SuffixAnnotation] = Field(None, description="Annotation for the BIDS suffix of the series, if applicable.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about the series annotation.")
    
    @model_validator(mode="before")
    def validate_suffix(cls, values: Any) -> Any:
        """Validate that if a suffix is provided, it is compatible with the annotated datatype."""
        datatype_annotation: Datatype = values.get('datatype')
        if not isinstance(datatype_annotation, DatatypeAnnotation):
            raise ValueError("Datatype annotation must be provided and must be of type DatatypeAnnotation.")
        suffix_annotation: SuffixAnnotation = values.get('suffix')
        if suffix_annotation is not None and not isinstance(suffix_annotation, SuffixAnnotation):
            raise ValueError("Suffix annotation must be of type SuffixAnnotation if provided.")
        
        if datatype_annotation.value in {Datatype.EXCLUDE, Datatype.UNKNOWN}:
            return values
        
        if suffix_annotation is not None and datatype_annotation is not None:
            valid_suffixes = BIDS_SCHEMA.get(datatype_annotation.datatype, {})
            if suffix_annotation.suffix not in valid_suffixes:
                raise ValueError(f"Invalid suffix '{suffix_annotation.suffix}' for datatype '{datatype_annotation.datatype}'. Valid options: {valid_suffixes.keys()}")
        
        return values
    
    def set_datatype(self, new_datatype: Datatype, confidence: Optional[int] = None, notes: Optional[str] = None) -> Self:
        """Return a new SeriesAnnotation with an updated datatype annotation."""
        new_datatype_annotation = self.datatype.annotate(new_datatype, confidence, notes)
        return self.model_copy(update={"datatype": new_datatype_annotation})
    
    def set_suffix(self, new_suffix: str, confidence: Optional[int] = None, notes: Optional[str] = None) -> Self:
        """Return a new SeriesAnnotation with an updated suffix annotation."""
        if self.datatype.datatype in {Datatype.EXCLUDE, Datatype.UNKNOWN}:
            raise ValueError(f"Cannot set suffix for datatype '{self.datatype.datatype}' as it does not allow suffixes.")
        new_suffix_annotation = self.suffix.annotate(new_suffix, confidence, notes) if self.suffix else SuffixAnnotation(suffix=new_suffix, confidence=confidence, notes=notes)
        return self.model_copy(update={"suffix": new_suffix_annotation})
    
    def add_notes(self, additional_notes: str) -> Self:
        """Return a new SeriesAnnotation with additional notes appended to existing notes."""
        combined_notes = f"{self.notes}\n{additional_notes}" if self.notes else additional_notes
        return self.model_copy(update={"notes": combined_notes})

class SessionAnnotation(BaseModel):
    subject: str = Field(..., description="Subject identifier for the session.")
    session: Optional[str] = Field(None, description="Session identifier (if applicable).")
    series_annotations: list[SeriesAnnotation] = Field(..., description="List of annotations for each series in the session.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about the session annotation.")
    is_annotated: bool = Field(False, description="Flag indicating whether the session has been annotated.")
    
    def mark_annotated(self) -> Self:
        """Return a new SessionAnnotation marked as annotated."""
        return self.model_copy(update={"is_annotated": True})
    
class AllSessionsAnnotation(BaseModel):
    sessions: list[SessionAnnotation] = Field(..., description="List of session annotations for the dataset.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about all annotations.")
    
    @property
    def is_fully_annotated(self) -> bool:
        """Check if all sessions in the dataset have been annotated."""
        return all(session.is_annotated for session in self.sessions)
    
    def get_unannotated_sessions(self) -> list[SessionAnnotation]:
        """Return a list of sessions that have not yet been annotated."""
        return [session for session in self.sessions if not session.is_annotated]
    
    def annotate_session(self, subject: str, session: Optional[str], new_session_annotation: SessionAnnotation) -> Self:
        """Return a new AllSessionsAnnotation with an updated annotation for a specific session."""
        updated_sessions = []
        for existing_session in self.sessions:
            if existing_session.subject == subject and existing_session.session == session:
                updated_sessions.append(new_session_annotation)
            else:
                updated_sessions.append(existing_session)
        return self.model_copy(update={"sessions": updated_sessions})
    
    def to_csv(self, file_path: str) -> None:
        """Export all session annotations to a CSV file."""
        rows = []
        for session in self.sessions:
            for series_annotation in session.series_annotations:
                row = {
                    "subject": session.subject,
                    "session": session.session,
                    "datatype": series_annotation.datatype.datatype.value,
                    "datatype_confidence": series_annotation.datatype.confidence,
                    "datatype_notes": series_annotation.datatype.notes,
                    "suffix": series_annotation.suffix.suffix if series_annotation.suffix else None,
                    "suffix_confidence": series_annotation.suffix.confidence if series_annotation.suffix else None,
                    "suffix_notes": series_annotation.suffix.notes if series_annotation.suffix else None,
                    "series_notes": series_annotation.notes
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
    