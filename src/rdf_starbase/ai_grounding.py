"""
AI Grounding API

A specialized API layer designed for AI/LLM consumption, providing:
- Structured fact retrieval with provenance for RAG (Retrieval-Augmented Generation)
- Claim verification against the knowledge base
- Entity context with full provenance chain
- Inference materialization with attribution

This API is separate from the visualization API (/graph/*) because:
1. Different response formats (facts+citations vs nodes+edges)
2. Different filtering needs (confidence thresholds, freshness)
3. Different latency requirements (sub-100ms for tool calls)
4. Different auth model (API keys for agents vs sessions for users)

Endpoints:
- POST /ai/query - Structured fact retrieval for grounding
- POST /ai/verify - Verify if a claim is supported
- GET /ai/context/{iri} - All facts about an entity
- POST /ai/materialize - Trigger reasoning and persist inferences
- GET /ai/inferences - List materialized inferences
"""

from datetime import datetime, timedelta
from typing import Any, Optional, Union, List
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import polars as pl

from rdf_starbase import TripleStore, execute_sparql


# =============================================================================
# Pydantic Models for AI Grounding API
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Pre-defined confidence thresholds for AI consumption."""
    HIGH = "high"        # >= 0.9
    MEDIUM = "medium"    # >= 0.7
    LOW = "low"          # >= 0.5
    ANY = "any"          # >= 0.0
    
    def to_threshold(self) -> float:
        return {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5,
            "any": 0.0,
        }[self.value]


class FactWithProvenance(BaseModel):
    """A single fact with full provenance chain."""
    subject: str
    predicate: str
    object: Union[str, int, float, bool]
    source: str
    confidence: float
    timestamp: str
    process: Optional[str] = None
    is_inferred: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "http://example.org/customer/123",
                "predicate": "http://xmlns.com/foaf/0.1/name",
                "object": "Alice Johnson",
                "source": "CRM_System",
                "confidence": 0.95,
                "timestamp": "2026-01-15T10:30:00Z",
                "process": "api_sync",
                "is_inferred": False,
            }
        }


class Citation(BaseModel):
    """Citation information for attribution."""
    fact_hash: str = Field(..., description="Unique identifier for the fact")
    source: str = Field(..., description="Originating system or person")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    timestamp: str = Field(..., description="When the assertion was made")
    retrieval_time: str = Field(..., description="When this was retrieved")


class GroundedFact(BaseModel):
    """A fact with its citation for AI grounding."""
    subject: str
    predicate: str
    object: Union[str, int, float, bool]
    citation: Citation
    
    def to_natural_language(self) -> str:
        """Convert to natural language assertion."""
        pred_label = self.predicate.split("/")[-1].split("#")[-1]
        return f"{self.subject} {pred_label} {self.object}"


# Request/Response Models

class AIQueryRequest(BaseModel):
    """Request for structured fact retrieval."""
    subject: Optional[str] = Field(None, description="Filter by subject IRI")
    predicate: Optional[str] = Field(None, description="Filter by predicate IRI")
    object: Optional[str] = Field(None, description="Filter by object value")
    sources: Optional[List[str]] = Field(None, description="Filter to specific sources")
    min_confidence: Optional[ConfidenceLevel] = Field(
        ConfidenceLevel.MEDIUM,
        description="Minimum confidence threshold"
    )
    max_age_days: Optional[int] = Field(
        None,
        description="Maximum age of facts in days (freshness filter)"
    )
    include_inferred: bool = Field(
        True,
        description="Include inferred triples from reasoning"
    )
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")


class AIQueryResponse(BaseModel):
    """Response with grounded facts for AI consumption."""
    facts: List[GroundedFact]
    total_count: int
    filtered_count: int
    confidence_threshold: float
    retrieval_timestamp: str
    sources_used: List[str]


class ClaimVerificationRequest(BaseModel):
    """Request to verify a claim against the knowledge base."""
    subject: str = Field(..., description="Subject of the claim")
    predicate: str = Field(..., description="Predicate of the claim")
    expected_object: Optional[str] = Field(
        None, 
        description="Expected object value (if checking specific value)"
    )
    min_confidence: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Minimum confidence for supporting evidence"
    )


class ClaimVerificationResponse(BaseModel):
    """Response indicating whether a claim is supported."""
    claim_supported: bool = Field(..., description="Whether the claim is supported")
    confidence: Optional[float] = Field(
        None, 
        description="Confidence of the best supporting fact"
    )
    supporting_facts: List[GroundedFact] = Field(
        default_factory=list,
        description="Facts that support the claim"
    )
    contradicting_facts: List[GroundedFact] = Field(
        default_factory=list,
        description="Facts that contradict the claim"
    )
    has_conflicts: bool = Field(
        False,
        description="Whether there are conflicting assertions"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation for the AI on how to use this information"
    )


class EntityContextResponse(BaseModel):
    """Full context about an entity for grounding."""
    entity: str
    facts: List[GroundedFact]
    related_entities: List[str]
    sources: List[str]
    confidence_summary: dict
    retrieval_timestamp: str


class MaterializeRequest(BaseModel):
    """Request to materialize inferences."""
    enable_rdfs: bool = Field(True, description="Apply RDFS entailment rules")
    enable_owl: bool = Field(True, description="Apply OWL 2 RL entailment rules")
    max_iterations: int = Field(100, ge=1, le=1000, description="Max reasoning iterations")


class MaterializeResponse(BaseModel):
    """Response from materialization."""
    success: bool
    iterations: int
    triples_inferred: int
    rdfs_inferences: int
    owl_inferences: int
    breakdown: dict


# =============================================================================
# Helper Functions
# =============================================================================

def dataframe_to_grounded_facts(
    df: pl.DataFrame,
    retrieval_time: datetime,
) -> List[GroundedFact]:
    """Convert DataFrame rows to GroundedFact objects."""
    facts = []
    
    for row in df.iter_rows(named=True):
        # Create a unique hash for the fact
        fact_hash = f"{row['subject']}|{row['predicate']}|{row['object']}|{row['source']}"
        import hashlib
        hash_id = hashlib.sha256(fact_hash.encode()).hexdigest()[:12]
        
        timestamp = row.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp) if timestamp else retrieval_time.isoformat()
        
        citation = Citation(
            fact_hash=hash_id,
            source=row.get("source", "unknown"),
            confidence=row.get("confidence", 1.0),
            timestamp=timestamp_str,
            retrieval_time=retrieval_time.isoformat(),
        )
        
        facts.append(GroundedFact(
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            citation=citation,
        ))
    
    return facts


# =============================================================================
# AI Grounding Router
# =============================================================================

def create_ai_router(store: TripleStore) -> APIRouter:
    """
    Create the AI Grounding API router.
    
    Args:
        store: TripleStore instance to query
        
    Returns:
        FastAPI APIRouter with AI grounding endpoints
    """
    router = APIRouter(prefix="/ai", tags=["AI Grounding"])
    
    # =========================================================================
    # POST /ai/query - Structured Fact Retrieval
    # =========================================================================
    
    @router.post(
        "/query",
        response_model=AIQueryResponse,
        summary="Query facts for AI grounding",
        description="""
        Retrieve facts from the knowledge base with provenance for AI grounding.
        
        Use this endpoint when your AI needs to:
        - Ground responses in verified facts
        - Retrieve information with confidence scores
        - Get citations for attribution
        
        The response includes full provenance chains and citation information
        suitable for RAG (Retrieval-Augmented Generation) pipelines.
        """,
    )
    async def ai_query(request: AIQueryRequest) -> AIQueryResponse:
        retrieval_time = datetime.utcnow()
        
        # Build filters
        confidence_threshold = request.min_confidence.to_threshold()
        
        # Use optimized query path with filter pushdown for performance
        # This avoids materializing the entire DataFrame
        if hasattr(store, 'get_triples_optimized'):
            df = store.get_triples_optimized(
                subject=request.subject,
                predicate=request.predicate,
                obj=request.object,
                min_confidence=confidence_threshold,
                limit=request.limit * 2 if request.sources or request.max_age_days else request.limit,
            )
        else:
            # Fallback for older store implementations
            df = store.get_triples(
                subject=request.subject,
                predicate=request.predicate,
                obj=request.object,
                min_confidence=confidence_threshold,
            )
        
        total_count = len(df)
        
        # Apply source filter
        if request.sources:
            df = df.filter(pl.col("source").is_in(request.sources))
        
        # Apply freshness filter
        if request.max_age_days:
            cutoff = datetime.utcnow() - timedelta(days=request.max_age_days)
            df = df.filter(pl.col("timestamp") >= cutoff)
        
        # Filter out inferred if requested
        if not request.include_inferred and "process" in df.columns:
            df = df.filter(
                (pl.col("process").is_null()) | 
                (pl.col("process") != "reasoner")
            )
        
        # Apply limit
        df = df.head(request.limit)
        
        # Convert to grounded facts
        facts = dataframe_to_grounded_facts(df, retrieval_time)
        
        # Get unique sources
        sources_used = df["source"].unique().to_list() if len(df) > 0 else []
        
        return AIQueryResponse(
            facts=facts,
            total_count=total_count,
            filtered_count=len(facts),
            confidence_threshold=confidence_threshold,
            retrieval_timestamp=retrieval_time.isoformat(),
            sources_used=sources_used,
        )
    
    # =========================================================================
    # POST /ai/verify - Claim Verification
    # =========================================================================
    
    @router.post(
        "/verify",
        response_model=ClaimVerificationResponse,
        summary="Verify a claim against the knowledge base",
        description="""
        Check if a claim is supported by the knowledge base.
        
        Use this endpoint when your AI needs to:
        - Verify a statement before including it in a response
        - Check for contradictions in the knowledge base
        - Get supporting evidence for a claim
        
        The response indicates whether the claim is supported, provides
        supporting/contradicting evidence, and gives a recommendation.
        """,
    )
    async def ai_verify(request: ClaimVerificationRequest) -> ClaimVerificationResponse:
        retrieval_time = datetime.utcnow()
        confidence_threshold = request.min_confidence.to_threshold()
        
        # Use optimized query path - verify always has subject+predicate filters
        if hasattr(store, 'get_triples_optimized'):
            df = store.get_triples_optimized(
                subject=request.subject,
                predicate=request.predicate,
                min_confidence=confidence_threshold,
            )
        else:
            df = store.get_triples(
                subject=request.subject,
                predicate=request.predicate,
                min_confidence=confidence_threshold,
            )
        
        if len(df) == 0:
            return ClaimVerificationResponse(
                claim_supported=False,
                confidence=None,
                supporting_facts=[],
                contradicting_facts=[],
                has_conflicts=False,
                recommendation="No facts found for this subject-predicate pair. "
                              "The AI should not make claims about this topic or "
                              "clearly state that information is not available.",
            )
        
        # Convert all to grounded facts
        all_facts = dataframe_to_grounded_facts(df, retrieval_time)
        
        # Check if expected object matches
        if request.expected_object:
            supporting = []
            contradicting = []
            
            for fact in all_facts:
                if str(fact.object) == str(request.expected_object):
                    supporting.append(fact)
                else:
                    contradicting.append(fact)
            
            has_conflicts = len(supporting) > 0 and len(contradicting) > 0
            best_confidence = max(
                (f.citation.confidence for f in supporting), 
                default=None
            )
            
            if supporting:
                if has_conflicts:
                    recommendation = (
                        f"The claim is supported by {len(supporting)} source(s) but "
                        f"contradicted by {len(contradicting)} source(s). "
                        "The AI should present this as contested information with "
                        "sources for both perspectives."
                    )
                else:
                    recommendation = (
                        f"The claim is supported by {len(supporting)} source(s) with "
                        f"confidence up to {best_confidence:.0%}. "
                        "The AI can confidently state this fact with attribution."
                    )
            else:
                recommendation = (
                    f"The claim is NOT supported. {len(contradicting)} source(s) "
                    "report different values. "
                    "The AI should NOT make this claim and instead report "
                    "what the knowledge base actually contains."
                )
            
            return ClaimVerificationResponse(
                claim_supported=len(supporting) > 0,
                confidence=best_confidence,
                supporting_facts=supporting,
                contradicting_facts=contradicting,
                has_conflicts=has_conflicts,
                recommendation=recommendation,
            )
        else:
            # No specific value expected - just return what we have
            best_confidence = max(
                (f.citation.confidence for f in all_facts),
                default=None
            )
            unique_values = len(set(str(f.object) for f in all_facts))
            has_conflicts = unique_values > 1
            
            if has_conflicts:
                recommendation = (
                    f"Multiple values found ({unique_values} distinct) from different sources. "
                    "The AI should acknowledge the competing claims and cite sources."
                )
            else:
                recommendation = (
                    f"Single consistent value found across {len(all_facts)} source(s). "
                    "The AI can state this fact with confidence."
                )
            
            return ClaimVerificationResponse(
                claim_supported=True,
                confidence=best_confidence,
                supporting_facts=all_facts,
                contradicting_facts=[],
                has_conflicts=has_conflicts,
                recommendation=recommendation,
            )
    
    # =========================================================================
    # GET /ai/context/{iri} - Entity Context
    # =========================================================================
    
    @router.get(
        "/context/{iri:path}",
        response_model=EntityContextResponse,
        summary="Get full context for an entity",
        description="""
        Retrieve all known facts about an entity with full provenance.
        
        Use this endpoint when your AI needs to:
        - Understand everything known about a specific entity
        - Get a complete picture before answering questions
        - Gather context for entity-centric responses
        
        Returns all facts where the entity appears as subject or object,
        along with confidence summaries and related entity links.
        """,
    )
    async def ai_context(
        iri: str,
        min_confidence: ConfidenceLevel = Query(
            ConfidenceLevel.LOW,
            description="Minimum confidence threshold"
        ),
        include_incoming: bool = Query(
            True,
            description="Include facts where entity is the object"
        ),
        limit: int = Query(100, ge=1, le=500, description="Maximum facts to return"),
    ) -> EntityContextResponse:
        import urllib.parse
        entity = urllib.parse.unquote(iri)
        retrieval_time = datetime.utcnow()
        confidence_threshold = min_confidence.to_threshold()
        
        # Use optimized query path with filter pushdown for performance
        if hasattr(store, 'get_triples_optimized'):
            # Get outgoing facts (entity as subject)
            df_out = store.get_triples_optimized(
                subject=entity,
                min_confidence=confidence_threshold,
                limit=limit,
            )
            
            # Get incoming facts (entity as object) if requested
            if include_incoming:
                df_in = store.get_triples_optimized(
                    obj=entity,
                    min_confidence=confidence_threshold,
                    limit=limit,
                )
                df = pl.concat([df_out, df_in]).unique()
            else:
                df = df_out
        else:
            # Fallback for older store implementations
            df_out = store.get_triples(
                subject=entity,
                min_confidence=confidence_threshold,
            )
            
            if include_incoming:
                df_in = store.get_triples(
                    obj=entity,
                    min_confidence=confidence_threshold,
                )
                df = pl.concat([df_out, df_in]).unique()
            else:
                df = df_out
        
        df = df.head(limit)
        
        # Convert to grounded facts
        facts = dataframe_to_grounded_facts(df, retrieval_time)
        
        # Find related entities (other URIs in the facts)
        related = set()
        for fact in facts:
            if fact.subject != entity and fact.subject.startswith("http"):
                related.add(fact.subject)
            obj_str = str(fact.object)
            if obj_str != entity and obj_str.startswith("http"):
                related.add(obj_str)
        
        # Get unique sources
        sources = list(set(f.citation.source for f in facts))
        
        # Confidence summary
        confidences = [f.citation.confidence for f in facts]
        conf_summary = {
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0,
            "avg": sum(confidences) / len(confidences) if confidences else 0,
            "high_confidence_count": sum(1 for c in confidences if c >= 0.9),
            "medium_confidence_count": sum(1 for c in confidences if 0.7 <= c < 0.9),
            "low_confidence_count": sum(1 for c in confidences if c < 0.7),
        }
        
        return EntityContextResponse(
            entity=entity,
            facts=facts,
            related_entities=list(related)[:20],  # Limit related entities
            sources=sources,
            confidence_summary=conf_summary,
            retrieval_timestamp=retrieval_time.isoformat(),
        )
    
    # =========================================================================
    # POST /ai/materialize - Inference Materialization
    # =========================================================================
    
    @router.post(
        "/materialize",
        response_model=MaterializeResponse,
        summary="Materialize inferences from reasoning",
        description="""
        Run the reasoning engine and persist inferred triples.
        
        This executes RDFS and OWL 2 RL forward-chaining inference,
        materializing entailments into the store with provenance:
        - source: "reasoner"
        - confidence: 1.0 (logical entailment)
        - process: "inference_engine"
        
        Materialized inferences can then be queried like any other facts,
        with the `is_inferred` flag indicating their origin.
        """,
    )
    async def ai_materialize(request: MaterializeRequest) -> MaterializeResponse:
        try:
            from rdf_starbase.storage.reasoner import RDFSReasoner
            from rdf_starbase.storage.terms import TermDict
            from rdf_starbase.storage.facts import FactStore
            
            # Check if store uses new storage layer
            if hasattr(store, "_term_dict") and hasattr(store, "_fact_store"):
                reasoner = RDFSReasoner(
                    term_dict=store._term_dict,
                    fact_store=store._fact_store,
                    max_iterations=request.max_iterations,
                    enable_owl=request.enable_owl,
                )
                
                stats = reasoner.reason()
                
                return MaterializeResponse(
                    success=True,
                    iterations=stats.iterations,
                    triples_inferred=stats.triples_inferred,
                    rdfs_inferences=(
                        stats.rdfs2_inferences + stats.rdfs3_inferences +
                        stats.rdfs5_inferences + stats.rdfs7_inferences +
                        stats.rdfs9_inferences + stats.rdfs11_inferences
                    ),
                    owl_inferences=(
                        stats.owl_same_as_inferences + 
                        stats.owl_equivalent_class_inferences +
                        stats.owl_equivalent_property_inferences +
                        stats.owl_inverse_of_inferences +
                        stats.owl_transitive_inferences +
                        stats.owl_symmetric_inferences +
                        stats.owl_functional_inferences +
                        stats.owl_inverse_functional_inferences +
                        stats.owl_has_value_inferences
                    ),
                    breakdown={
                        "rdfs2_domain": stats.rdfs2_inferences,
                        "rdfs3_range": stats.rdfs3_inferences,
                        "rdfs5_subPropertyOf_transitivity": stats.rdfs5_inferences,
                        "rdfs7_property_inheritance": stats.rdfs7_inferences,
                        "rdfs9_type_inheritance": stats.rdfs9_inferences,
                        "rdfs11_subClassOf_transitivity": stats.rdfs11_inferences,
                        "owl_sameAs": stats.owl_same_as_inferences,
                        "owl_equivalentClass": stats.owl_equivalent_class_inferences,
                        "owl_equivalentProperty": stats.owl_equivalent_property_inferences,
                        "owl_inverseOf": stats.owl_inverse_of_inferences,
                        "owl_transitive": stats.owl_transitive_inferences,
                        "owl_symmetric": stats.owl_symmetric_inferences,
                        "owl_functional": stats.owl_functional_inferences,
                        "owl_inverseFunctional": stats.owl_inverse_functional_inferences,
                        "owl_hasValue": stats.owl_has_value_inferences,
                    },
                )
            else:
                # Legacy store - reasoning not available
                raise HTTPException(
                    status_code=501,
                    detail="Reasoning requires the new storage layer. "
                           "Use TripleStore with FactStore backend.",
                )
                
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Reasoning engine not available: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Materialization failed: {str(e)}",
            )
    
    # =========================================================================
    # GET /ai/inferences - List Inferred Facts
    # =========================================================================
    
    @router.get(
        "/inferences",
        summary="List materialized inferences",
        description="""
        Get facts that were inferred by the reasoning engine.
        
        These are triples that were not explicitly asserted but were
        derived through RDFS/OWL entailment rules.
        """,
    )
    async def ai_inferences(
        limit: int = Query(100, ge=1, le=1000),
    ):
        retrieval_time = datetime.utcnow()
        
        # Get triples with process='reasoner' or source='reasoner'
        df = store.get_triples()
        
        # Filter for inferred triples
        if "process" in df.columns:
            df = df.filter(
                (pl.col("process") == "reasoner") |
                (pl.col("process") == "inference_engine") |
                (pl.col("source") == "reasoner")
            )
        elif "source" in df.columns:
            df = df.filter(pl.col("source") == "reasoner")
        else:
            return {"count": 0, "inferences": [], "message": "No inference markers found"}
        
        df = df.head(limit)
        
        facts = dataframe_to_grounded_facts(df, retrieval_time)
        
        return {
            "count": len(facts),
            "inferences": [f.model_dump() for f in facts],
            "retrieval_timestamp": retrieval_time.isoformat(),
        }
    
    # =========================================================================
    # GET /ai/health - AI API Health Check
    # =========================================================================
    
    @router.get(
        "/health",
        summary="AI API health check",
        description="Check if the AI Grounding API is operational.",
    )
    async def ai_health():
        """Health check for AI Grounding API."""
        try:
            # Quick store check
            stats = store.stats()
            return {
                "status": "healthy",
                "api": "ai_grounding",
                "version": "1.0.0",
                "store_stats": {
                    "total_triples": stats.get("total_triples", 0),
                    "unique_subjects": stats.get("unique_subjects", 0),
                },
                "capabilities": [
                    "query",
                    "verify", 
                    "context",
                    "materialize",
                    "inferences",
                ],
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
            }
    
    return router


def integrate_ai_router(app, store: TripleStore):
    """
    Integrate the AI Grounding router into an existing FastAPI app.
    
    Args:
        app: FastAPI application
        store: TripleStore instance
    """
    router = create_ai_router(store)
    app.include_router(router)
