"""
Core data models for RDF-StarBase.

Defines the fundamental structures for RDF-Star triples, quoted triples,
and provenance metadata.
"""

from datetime import datetime, timezone
from typing import Optional, Any, Union
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, ConfigDict


def _utc_now() -> datetime:
    """Get current UTC time in a timezone-aware way."""
    return datetime.now(timezone.utc)


class ProvenanceContext(BaseModel):
    """
    Provenance metadata for an assertion.
    
    Tracks who made the assertion, when, how, and with what confidence.
    """
    model_config = ConfigDict(frozen=True)
    
    source: str = Field(description="System, API, or person that asserted this")
    timestamp: datetime = Field(default_factory=_utc_now, description="When the assertion was made")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0,1]")
    process: Optional[str] = Field(default=None, description="Process or method that generated this")
    version: Optional[str] = Field(default=None, description="Version of the asserting system")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class Triple(BaseModel):
    """
    A standard RDF triple: subject-predicate-object.
    
    Forms the foundation of knowledge representation in RDF-StarBase.
    """
    model_config = ConfigDict(frozen=True)
    
    subject: str = Field(description="Subject URI or blank node")
    predicate: str = Field(description="Predicate URI")
    object: Union[str, int, float, bool] = Field(description="Object URI, literal, or value")
    graph: Optional[str] = Field(default=None, description="Named graph (optional quad)")
    
    def __str__(self) -> str:
        graph_str = f" [{self.graph}]" if self.graph else ""
        return f"<{self.subject}> <{self.predicate}> {self._format_object()}{graph_str}"
    
    def _format_object(self) -> str:
        if isinstance(self.object, str) and self.object.startswith("http"):
            return f"<{self.object}>"
        elif isinstance(self.object, str):
            return f'"{self.object}"'
        return str(self.object)


class QuotedTriple(BaseModel):
    """
    An RDF-Star quoted triple: a triple that can be used as a subject or object.
    
    This is the key innovation of RDF-Star - making statements about statements.
    """
    model_config = ConfigDict(frozen=True)
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this quoted triple")
    triple: Triple = Field(description="The quoted triple")
    provenance: ProvenanceContext = Field(description="Provenance of this assertion")
    
    def __str__(self) -> str:
        return f"<<{self.triple}>> [from: {self.provenance.source}]"


class Assertion(BaseModel):
    """
    A complete assertion: triple + provenance.
    
    This is the atomic unit of knowledge in RDF-StarBase.
    """
    model_config = ConfigDict(frozen=True)
    
    id: UUID = Field(default_factory=uuid4, description="Unique assertion ID")
    triple: Triple = Field(description="The asserted triple")
    provenance: ProvenanceContext = Field(description="Who/when/how this was asserted")
    superseded_by: Optional[UUID] = Field(default=None, description="ID of assertion that supersedes this")
    deprecated: bool = Field(default=False, description="Whether this assertion is deprecated")
    
    def __str__(self) -> str:
        status = " [DEPRECATED]" if self.deprecated else ""
        return f"{self.triple} (by {self.provenance.source} at {self.provenance.timestamp}){status}"
