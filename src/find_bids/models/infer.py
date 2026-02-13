from typing import Optional

from pydantic import BaseModel

class BIDSEntities(BaseModel):
    """
    Represents the core BIDS entities that can be extracted from DICOM metadata. 
    These entities are crucial for organizing and structuring neuroimaging data according to the BIDS standard. 
    Each entity is optional, as not all DICOM files may contain the necessary metadata to populate these fields. 
    The `datatype` field is required, as it indicates the type of data (e.g., anat, func, dwi) and is essential for correctly categorizing the series within a BIDS dataset.
    """
    subject: Optional[str] = None
    session: Optional[str] = None
    run: Optional[str] = None
    datatype: str
    suffix: Optional[str] = None
    part: Optional[str] = None
    echo: Optional[str] = None
    inv: Optional[str] = None
    ce: Optional[str] = None
    dir: Optional[str] = None
    mt: Optional[str] = None
    acq: Optional[str] = None