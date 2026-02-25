"""
Prepare annotation data structures to annotate DICOM series with BIDS labels.
The idea is to have a Dockerized annotation UI that can be used to annotate series on a per-session basis.

- Aggregate series on a session-level across datasets.
- Present series within the context of the session and dataset to the user.
- Allow the user to annotate series with BIDS labels (datatype and suffix).
- Store annotations in a standardized format (e.g., JSON) for downstream use in BIDS conversion and inference model training.
"""
from pathlib import Path
from collections import Counter
import random, json

from ..extract.series import SeriesFeatures
from ..infer.schema import BIDS_SCHEMA, Datatype
from ..infer.core import infer_bids_datatype

from typing import Optional, Any, Self
import sqlite3

from pydantic import BaseModel, Field, model_validator, EmailStr
import pandas as pd
import numpy as np

def initialize_annotations_metrics_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the SQLite database for storing key metrics, for efficient querying when sampling for annotations."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS series_annotations (
            subject_id TEXT NOT NULL,
            session_id TEXT,
            protocol_score REAL,
            inferred_datatype_counts JSON,
            inferred_datatype_entropy REAL,
            class_balance_score REAL,
            is_annotated INTEGER DEFAULT 0,
            PRIMARY KEY (subject_id, session_id)
        )
    """)
    conn.commit()
    return conn

class AboutAnnotator(BaseModel):
    name: str = Field(..., description="Name of the annotator.")
    email: EmailStr = Field(..., description="Email address of the annotator.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about the annotator.")

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
    inferred_datatype: Optional[Datatype] = Field(None, description="Inferred BIDS datatype based on features (for reference).")
    datatype: DatatypeAnnotation = Field(..., description="Annotation for the BIDS datatype of the series.")
    suffix: Optional[SuffixAnnotation] = Field(None, description="Annotation for the BIDS suffix of the series, if applicable.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about the series annotation.")
    
    @model_validator(mode="before")
    def validate_suffix(cls, values: Any) -> Any:
        """Validate that if a suffix is provided, it is compatible with the annotated datatype."""
        datatype_annotation: DatatypeAnnotation = values.get('datatype')
        if not isinstance(datatype_annotation, DatatypeAnnotation):
            raise ValueError("Datatype annotation must be provided and must be of type DatatypeAnnotation.")
        suffix_annotation: SuffixAnnotation = values.get('suffix')
        if suffix_annotation is not None and not isinstance(suffix_annotation, SuffixAnnotation):
            raise ValueError("Suffix annotation must be of type SuffixAnnotation if provided.")
        
        if datatype_annotation.datatype in {Datatype.EXCLUDE, Datatype.UNKNOWN}:
            return values
        
        if suffix_annotation is not None and datatype_annotation is not None:
            valid_suffixes = BIDS_SCHEMA.get(datatype_annotation.datatype, {})
            if suffix_annotation.suffix not in valid_suffixes:
                raise ValueError(f"Invalid suffix '{suffix_annotation.suffix}' for datatype '{datatype_annotation.datatype}'. Valid options: {valid_suffixes.keys()}")
        
        return values
    
    @model_validator(mode="after")
    def infer_datatype(self) -> Self:
        """Infer the BIDS datatype based on the series features."""
        self.inferred_datatype = Datatype(infer_bids_datatype(self.features)) if self.features else None
        return self
    
    @property
    def protocol_fingerprint(self) -> str:
        """Generate a fingerprint string based on key features of the series for quick reference."""
        tokens = []
        if self.features.manufacturer and self.features.manufacturer.value:
            tokens.append(self.features.manufacturer.value)
        if self.features.model and self.features.model.value:
            tokens.append(self.features.model.value)
        if self.features.text:
            for field in [self.features.text.series_description, self.features.text.protocol_name, self.features.text.sequence_name]:
                if field and field.text:
                    tokens.append(field.text)
        return "|".join(tokens) if tokens else "unknown_protocol"
    
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
    
    @property
    def inferred_datatype_entropy(self) -> Optional[float]:
        """Calculate the entropy of the inferred datatype probabilities across all series in the session."""
        datatypes = [series.inferred_datatype for series in self.series_annotations if series.inferred_datatype is not None]
        if not datatypes:
            return None
        datatype_counts = pd.Series(datatypes).value_counts(normalize=True)
        entropy = -np.sum(datatype_counts * np.log2(datatype_counts))
        return entropy
    
    @property
    def protocol_signature(self) -> str:
        """Generate a signature string for the session based on the protocol fingerprints of its series."""
        fingerprints = [series.protocol_fingerprint for series in self.series_annotations]
        unique_fingerprints = set(fingerprints)
        signature = "|".join(sorted(unique_fingerprints))
        return signature
    
    @classmethod
    def from_session_features(cls, series_annotations: list[SeriesAnnotation], subject: str, session: Optional[str] = None, notes: Optional[str] = None) -> Self:
        """Create a SessionAnnotation instance from a list of SeriesAnnotation instances."""
        return cls(subject=subject, session=session, series_annotations=series_annotations, notes=notes, is_annotated=False)
    
    @classmethod
    def from_csv(cls, file_path: str | Path, series_features: list[SeriesFeatures], subject: str, session: Optional[str] = None) -> Self:
        """Load series annotations for a session from a CSV file and create a SessionAnnotation instance."""
        file_path = Path(file_path)
        df = pd.read_csv(file_path)
        session_df = df[(df['subject'] == subject) & (df['session'] == session)]
        series_annotations = []
        for _, row in session_df.iterrows():
            # Identify the series features that match the row's series description (this is a simplification and may need a more robust matching strategy in practice)
            matching_features: Optional[SeriesFeatures] = next((features for features in series_features if features.text and features.text.series_description == row['series_description']), None)
            if not matching_features:
                print(f"Warning: No matching series features found for row with series description '{row['series_description']}' in subject '{subject}', session '{session}'. Skipping this annotation.")
                continue
            
            datatype_annotation = DatatypeAnnotation(
                datatype=Datatype(row['datatype']),
                confidence=row.get('datatype_confidence'),
                notes=row.get('datatype_notes')
            )
            suffix_annotation = SuffixAnnotation(
                suffix=row['suffix'],
                confidence=row.get('suffix_confidence'),
                notes=row.get('suffix_notes')
            ) if pd.notna(row.get('suffix')) else None
            series_annotation = SeriesAnnotation(
                features=matching_features,
                inferred_datatype=None,
                datatype=datatype_annotation,
                suffix=suffix_annotation,
                notes=row.get('series_notes')
            )
            series_annotations.append(series_annotation)
        return cls(subject=subject, session=session, series_annotations=series_annotations, notes=None, is_annotated=True)
    
    def annotate(self, series_annotations: list[SeriesAnnotation], notes: Optional[str] = None) -> Self:
        """Return a new SessionAnnotation with updated series annotations and marked as annotated."""
        combined_notes = f"{self.notes}\n{notes}" if self.notes and notes else notes if notes else self.notes
        return self.model_copy(update={"series_annotations": series_annotations, "notes": combined_notes, "is_annotated": True})
    
    def save_to_csv(self, file_path: str | Path) -> None:
        """Append annotated series annotations for the session to a CSV file."""
        file_path = Path(file_path)
        rows = []
        for series_annotation in self.series_annotations:
            row = {
                "subject": self.subject,
                "session": self.session,
                "series_description": series_annotation.features.text.series_description if series_annotation.features and series_annotation.features.text else None,
                "datatype": series_annotation.datatype.datatype.value,
                "datatype_confidence": series_annotation.datatype.confidence,
                "datatype_notes": series_annotation.datatype.notes,
                "suffix": series_annotation.suffix.suffix if series_annotation.suffix else None,
                "suffix_confidence": series_annotation.suffix.confidence if series_annotation.suffix else None,
                "suffix_notes": series_annotation.suffix.notes if series_annotation.suffix else None,
                "series_notes": series_annotation.notes,
                "is_annotated": self.is_annotated
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if file_path.exists():
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
    
class AllSessionsAnnotation(BaseModel):
    sessions: list[SessionAnnotation] = Field(..., description="List of session annotations for the dataset.")
    notes: Optional[str] = Field(None, description="Additional notes or comments about all annotations.")
    annotator: Optional[AboutAnnotator] = Field(None, description="Information about the annotator.")
    
    @property
    def is_fully_annotated(self) -> bool:
        """Check if all sessions in the dataset have been annotated."""
        return all(session.is_annotated for session in self.sessions)
    
    @property
    def annotated(self) -> list[SessionAnnotation]:
        """Return a list of sessions that have been annotated."""
        return [session for session in self.sessions if session.is_annotated]
    
    @property
    def unannotated(self) -> list[SessionAnnotation]:
        """Return a list of sessions that have not been annotated."""
        return [session for session in self.sessions if not session.is_annotated]
    
    @property
    def global_datatype_distribution(self) -> dict[Datatype, int]:
        """Calculate the distribution of annotated datatypes across all sessions."""
        counter = Counter()
        for session in self.annotated:
            for series in session.series_annotations:
                if series.datatype and series.datatype.datatype:
                    counter[series.datatype.datatype] += 1
        return dict(counter)
    
    @property
    def global_protocol_distribution(self) -> dict[str, int]:
        """Calculate the distribution of protocol signatures across all sessions."""
        counter = Counter()
        for session in self.annotated:
            signature = session.protocol_signature
            counter[signature] += 1
        return dict(counter)
    
    @classmethod
    def from_sessions(cls, sessions: list[SessionAnnotation], notes: Optional[str] = None, annotator: Optional[AboutAnnotator] = None) -> Self:
        """Create an AllSessionsAnnotation instance from a list of SessionAnnotation instances."""
        return cls(sessions=sessions, notes=notes, annotator=annotator)
    
    @classmethod
    def from_series_features(cls, series_features_dict: dict[str, dict[str, dict[str, SeriesFeatures]]], notes: Optional[str] = None, annotator: Optional[AboutAnnotator] = None) -> Self:
        """Create an AllSessionsAnnotation instance by grouping SeriesFeatures into sessions and initializing SessionAnnotations."""
        sessions = []
        for subject_id, sessions_dict in series_features_dict.items():
            for session_id, series_dict in sessions_dict.items():
                series_annotations = []
                for _, features in series_dict.items():
                    datatype_annotation = DatatypeAnnotation(datatype=Datatype.UNKNOWN, confidence=None, notes="Inferred from features")
                    series_annotation = SeriesAnnotation(features=features, datatype=datatype_annotation, inferred_datatype=None, suffix=None, notes=None)
                    series_annotations.append(series_annotation)
                session_annotation = SessionAnnotation(subject=subject_id, session=session_id, series_annotations=series_annotations, notes=None, is_annotated=False)
                sessions.append(session_annotation)
        return cls(sessions=sessions, notes=notes, annotator=annotator)
    
    def get_protocol_score(self, session: SessionAnnotation) -> float:
        """Inverse frequency score for the session's protocol signature based on the global protocol distribution."""
        distribution = self.global_protocol_distribution
        count = distribution.get(session.protocol_signature, 0)
        total = sum(distribution.values())
        return 1 / (count + 1) if total > 0 else 0
    
    def get_class_balance_score(self, session: SessionAnnotation) -> float:
        """
        Calculate a class balance score for the session based on the distribution of datatypes in its series compared to the global distribution.
        Boost sessions containing underrepresented datatypes and penalize those with overrepresented datatypes.
        """
        global_distribution = self.global_datatype_distribution
        
        # Inverse-frequency scoring for datatypes in the session
        weights = {series.datatype.datatype: 1 / (global_distribution.get(series.datatype.datatype, 0) + 1) for series in session.series_annotations if series.datatype and series.datatype.datatype}
        if not weights:
            return 0.0
        
        inferred_datatypes = [series.inferred_datatype for series in session.series_annotations if series.inferred_datatype is not None]
        if not inferred_datatypes:
            return 0.0
        
        counts = Counter(inferred_datatypes)
        score = sum(weights.get(datatype, 0) * (counts[datatype] / len(inferred_datatypes)) for datatype in counts)
        return score
    
    # def get_next_session(
    #     self,
    #     w_entropy: float = 0.5,
    #     w_protocol: float = 0.3,
    #     w_class_balance: float = 0.2,
    #     w_random: float = 0.0
    # ) -> Optional[SessionAnnotation]:
    #     """Select the next session to annotate based on a combined score of inferred datatype entropy, protocol rarity, and class balance."""
    #     if not self.unannotated:
    #         return None
        
    #     session_scores = []
    #     for session in self.unannotated:
    #         entropy_score = session.inferred_datatype_entropy or 0
    #         protocol_score = self.get_protocol_score(session)
    #         class_balance_score = self.get_class_balance_score(session)
    #         random_score = random.random() if w_random > 0 else 0
            
    #         combined_score = (w_entropy * entropy_score) + (w_protocol * protocol_score) + (w_class_balance * class_balance_score) + (w_random * random_score)
    #         session_scores.append((session, combined_score))
        
    #     session_scores.sort(key=lambda x: x[1], reverse=True)
    #     return session_scores[0][0] if session_scores else None
    
    def annotate_session(self, subject: str, session: Optional[str], new_session_annotation: SessionAnnotation) -> Self:
        """Return a new AllSessionsAnnotation with an updated annotation for a specific session."""
        updated_sessions = []
        for existing_session in self.sessions:
            if existing_session.subject == subject and existing_session.session == session:
                updated_sessions.append(new_session_annotation)
            else:
                updated_sessions.append(existing_session)
        return self.model_copy(update={"sessions": updated_sessions})
    
    def reset_annotation(self, subject: str, session: Optional[str]) -> Self:
        """Return a new AllSessionsAnnotation with the annotation for a specific session reset (marked as unannotated)."""
        updated_sessions = []
        for existing_session in self.sessions:
            if existing_session.subject == subject and existing_session.session == session:
                reset_session = existing_session.model_copy(update={"is_annotated": False, "series_annotations": [], "notes": None})
                updated_sessions.append(reset_session)
            else:
                updated_sessions.append(existing_session)
        return self.model_copy(update={"sessions": updated_sessions})
    
    # !Do not use this method in the annotation UI, as it will overwrite existing annotations without warning! 
    # !This is intended for use in testing or if you want to reset all annotations and start over.
    def to_csv(self, file_path: str | Path) -> None:
        """Export annotated session annotations to a CSV file."""
        file_path = Path(file_path)
        rows = []
        for session in self.annotated:
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
        
    def export_annotation_metrics_to_sqlite(self, conn: sqlite3.Connection) -> None:
        """Export key metrics for each session to a SQLite database for efficient querying when sampling for annotations."""
        cursor = conn.cursor()
        for session in self.sessions:
            protocol_score = self.get_protocol_score(session)
            class_balance_score = self.get_class_balance_score(session)
            inferred_datatype_counts = {series.inferred_datatype.value: 0 for series in session.series_annotations if series.inferred_datatype is not None}
            for series in session.series_annotations:
                if series.inferred_datatype is not None:
                    inferred_datatype_counts[series.inferred_datatype.value] += 1
            cursor.execute("""
                INSERT INTO series_annotations (subject_id, session_id, inferred_datatype_counts, protocol_score, inferred_datatype_entropy, class_balance_score, is_annotated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id, session_id) DO UPDATE SET
                    inferred_datatype_counts=excluded.inferred_datatype_counts,
                    protocol_score=excluded.protocol_score,
                    inferred_datatype_entropy=excluded.inferred_datatype_entropy,
                    class_balance_score=excluded.class_balance_score,
                    is_annotated=excluded.is_annotated
            """, (
                session.subject,
                session.session,
                json.dumps(inferred_datatype_counts) if inferred_datatype_counts is not None else None,
                protocol_score,
                session.inferred_datatype_entropy,
                class_balance_score,
                1 if session.is_annotated else 0
            ))
        conn.commit()
        
def get_next_session_for_annotation(
    conn: sqlite3.Connection,
    w_entropy: float = 0.5,
    w_protocol: float = 0.3,
    w_class_balance: float = 0.2,
    w_random: float = 0.0
) -> tuple[Optional[str], Optional[str]]:
    """
    Select the next unannotated session to annotate based on a combined score of inferred datatype entropy, protocol rarity, and class balance.
    Formula:
    combined_score = (w_entropy * inferred_datatype_entropy) + (w_protocol * protocol_score) + (w_class_balance * class_balance_score) + (w_random * random_score)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subject_id, session_id, inferred_datatype_counts, inferred_datatype_entropy, protocol_score, class_balance_score
        FROM series_annotations
        WHERE (protocol_score IS NOT NULL AND inferred_datatype_entropy IS NOT NULL AND class_balance_score IS NOT NULL AND is_annotated = 0)
        ORDER BY ((? * inferred_datatype_entropy) + (? * protocol_score) + (? * class_balance_score) + (? * RANDOM())) DESC
        LIMIT 1
    """, (w_entropy, w_protocol, w_class_balance, w_random))
    
    result = cursor.fetchone()
    if result:
        subject_id, session_id, _, _, _, _ = result
        return (subject_id, session_id)
    return (None, None)

def mark_session_as_annotated(conn: sqlite3.Connection, subject_id: str, session_id: Optional[str]) -> None:
    """Mark a specific session as annotated in the database."""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE series_annotations
        SET is_annotated = 1
        WHERE subject_id = ? AND session_id = ?
    """, (subject_id, session_id))
    conn.commit()