"""
RDF-StarBase Web API

FastAPI-based REST API for querying and managing the knowledge graph.
Provides endpoints for:
- SPARQL queries
- Triple management
- Provenance inspection
- Competing claims analysis
- Source registry
"""

from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import polars as pl

from rdf_starbase import (
    TripleStore,
    ProvenanceContext,
    AssertionRegistry,
    SourceType,
    SourceStatus,
    execute_sparql,
    parse_query,
)


# Pydantic models for API
class ProvenanceInput(BaseModel):
    """Provenance context for adding triples."""
    source: str = Field(..., description="Source system or person")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    process: Optional[str] = None
    
    def to_context(self) -> ProvenanceContext:
        return ProvenanceContext(
            source=self.source,
            confidence=self.confidence,
            process=self.process,
        )


class TripleInput(BaseModel):
    """Input for adding a triple."""
    subject: str
    predicate: str
    object: Union[str, int, float, bool]
    provenance: ProvenanceInput
    graph: Optional[str] = None


class SPARQLQuery(BaseModel):
    """SPARQL query request."""
    query: str = Field(..., description="SPARQL-Star query string")


class SourceInput(BaseModel):
    """Input for registering a source."""
    name: str
    source_type: str = Field(..., description="One of: dataset, api, mapping, process, manual")
    uri: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    sync_frequency: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


def dataframe_to_records(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert Polars DataFrame to list of dicts for JSON serialization."""
    records = []
    for row in df.iter_rows(named=True):
        record = {}
        for k, v in row.items():
            if isinstance(v, datetime):
                record[k] = v.isoformat()
            elif v is None:
                record[k] = None
            else:
                record[k] = v
        records.append(record)
    return records


def create_app(store: Optional[TripleStore] = None, registry: Optional[AssertionRegistry] = None) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        store: Optional TripleStore instance (creates new if not provided)
        registry: Optional AssertionRegistry instance
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="RDF-StarBase API",
        description="A blazingly fast RDF★ database with native provenance tracking",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # State
    app.state.store = store or TripleStore()
    app.state.registry = registry or AssertionRegistry()
    
    # ==========================================================================
    # Health & Info
    # ==========================================================================
    
    @app.get("/", tags=["Info"])
    async def root():
        """API root with basic info."""
        return {
            "name": "RDF-StarBase",
            "version": "0.1.0",
            "description": "A blazingly fast RDF★ database with native provenance tracking",
            "docs": "/docs",
        }
    
    @app.get("/health", tags=["Info"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/stats", tags=["Info"])
    async def stats():
        """Get store and registry statistics."""
        return {
            "store": app.state.store.stats(),
            "registry": app.state.registry.get_stats(),
        }
    
    # ==========================================================================
    # Triples
    # ==========================================================================
    
    @app.post("/triples", tags=["Triples"])
    async def add_triple(triple: TripleInput):
        """Add a triple with provenance to the store."""
        try:
            assertion_id = app.state.store.add_triple(
                subject=triple.subject,
                predicate=triple.predicate,
                obj=triple.object,
                provenance=triple.provenance.to_context(),
                graph=triple.graph,
            )
            return {"assertion_id": str(assertion_id)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/triples", tags=["Triples"])
    async def get_triples(
        subject: Optional[str] = Query(None, description="Filter by subject"),
        predicate: Optional[str] = Query(None, description="Filter by predicate"),
        object: Optional[str] = Query(None, description="Filter by object"),
        source: Optional[str] = Query(None, description="Filter by source"),
        min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
        limit: int = Query(100, ge=1, le=10000, description="Maximum results"),
    ):
        """Query triples with optional filters."""
        df = app.state.store.get_triples(
            subject=subject,
            predicate=predicate,
            obj=object,
            source=source,
            min_confidence=min_confidence,
        )
        
        df = df.head(limit)
        
        return {
            "count": len(df),
            "triples": dataframe_to_records(df),
        }
    
    @app.get("/triples/{subject_encoded:path}/claims", tags=["Triples"])
    async def get_competing_claims(
        subject_encoded: str,
        predicate: str = Query(..., description="Predicate to check for conflicts"),
    ):
        """Get competing claims for a subject-predicate pair."""
        # URL decode the subject
        import urllib.parse
        subject = urllib.parse.unquote(subject_encoded)
        
        df = app.state.store.get_competing_claims(subject, predicate)
        
        if len(df) == 0:
            return {"count": 0, "has_conflicts": False, "claims": []}
        
        unique_values = df["object"].n_unique()
        
        return {
            "count": len(df),
            "has_conflicts": unique_values > 1,
            "unique_values": unique_values,
            "claims": dataframe_to_records(df),
        }
    
    @app.get("/triples/{subject_encoded:path}/timeline", tags=["Triples"])
    async def get_provenance_timeline(
        subject_encoded: str,
        predicate: str = Query(..., description="Predicate for timeline"),
    ):
        """Get provenance timeline for a subject-predicate pair."""
        import urllib.parse
        subject = urllib.parse.unquote(subject_encoded)
        
        df = app.state.store.get_provenance_timeline(subject, predicate)
        
        return {
            "count": len(df),
            "timeline": dataframe_to_records(df),
        }
    
    # ==========================================================================
    # SPARQL
    # ==========================================================================
    
    @app.post("/sparql", tags=["SPARQL"])
    async def execute_sparql_query(request: SPARQLQuery):
        """Execute a SPARQL-Star query."""
        try:
            result = execute_sparql(app.state.store, request.query)
            
            if isinstance(result, bool):
                # ASK query
                return {"type": "ask", "result": result}
            elif isinstance(result, pl.DataFrame):
                # SELECT query
                return {
                    "type": "select",
                    "count": len(result),
                    "columns": result.columns,
                    "results": dataframe_to_records(result),
                }
            else:
                return {"type": "unknown", "result": str(result)}
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")
    
    @app.post("/sparql/parse", tags=["SPARQL"])
    async def parse_sparql(request: SPARQLQuery):
        """Parse a SPARQL query and return the AST structure."""
        try:
            ast = parse_query(request.query)
            
            return {
                "type": type(ast).__name__,
                "prefixes": ast.prefixes,
                "pattern_count": len(ast.where.patterns) if ast.where else 0,
                "filter_count": len(ast.where.filters) if ast.where else 0,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")
    
    # ==========================================================================
    # Registry
    # ==========================================================================
    
    @app.post("/sources", tags=["Registry"])
    async def register_source(source: SourceInput):
        """Register a new data source."""
        try:
            src_type = SourceType(source.source_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type. Must be one of: {[t.value for t in SourceType]}"
            )
        
        registered = app.state.registry.register_source(
            name=source.name,
            source_type=src_type,
            uri=source.uri,
            description=source.description,
            owner=source.owner,
            sync_frequency=source.sync_frequency,
            tags=source.tags,
        )
        
        return {
            "id": str(registered.id),
            "name": registered.name,
            "source_type": registered.source_type.value,
        }
    
    @app.get("/sources", tags=["Registry"])
    async def get_sources(
        source_type: Optional[str] = Query(None, description="Filter by type"),
        owner: Optional[str] = Query(None, description="Filter by owner"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
    ):
        """List registered sources with optional filters."""
        kwargs = {}
        
        if source_type:
            try:
                kwargs["source_type"] = SourceType(source_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")
        
        if owner:
            kwargs["owner"] = owner
        if tag:
            kwargs["tag"] = tag
        
        sources = app.state.registry.get_sources(**kwargs)
        
        return {
            "count": len(sources),
            "sources": [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "source_type": s.source_type.value,
                    "uri": s.uri,
                    "status": s.status.value,
                    "owner": s.owner,
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "tags": s.tags,
                }
                for s in sources
            ],
        }
    
    @app.get("/sources/{source_id}", tags=["Registry"])
    async def get_source(source_id: str):
        """Get details of a specific source."""
        try:
            uid = UUID(source_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        
        source = app.state.registry.get_source(uid)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return {
            "id": str(source.id),
            "name": source.name,
            "source_type": source.source_type.value,
            "uri": source.uri,
            "description": source.description,
            "status": source.status.value,
            "created_at": source.created_at.isoformat(),
            "last_sync": source.last_sync.isoformat() if source.last_sync else None,
            "owner": source.owner,
            "sync_frequency": source.sync_frequency,
            "tags": source.tags,
        }
    
    @app.get("/sources/{source_id}/syncs", tags=["Registry"])
    async def get_sync_history(
        source_id: str,
        limit: int = Query(20, ge=1, le=100),
    ):
        """Get sync history for a source."""
        try:
            uid = UUID(source_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        
        history = app.state.registry.get_sync_history(uid, limit=limit)
        
        return {
            "count": len(history),
            "syncs": dataframe_to_records(history),
        }
    
    # ==========================================================================
    # Graph Visualization Data
    # ==========================================================================
    
    @app.get("/graph/nodes", tags=["Visualization"])
    async def get_graph_nodes(
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Get unique nodes (subjects and objects) for graph visualization."""
        df = app.state.store._df
        
        subjects = df["subject"].unique().to_list()[:limit]
        objects = df.filter(
            pl.col("object_type") == "uri"
        )["object"].unique().to_list()[:limit]
        
        all_nodes = list(set(subjects + objects))[:limit]
        
        return {
            "count": len(all_nodes),
            "nodes": [{"id": n, "label": n.split("/")[-1]} for n in all_nodes],
        }
    
    @app.get("/graph/edges", tags=["Visualization"])
    async def get_graph_edges(
        limit: int = Query(500, ge=1, le=5000),
    ):
        """Get edges (triples) for graph visualization."""
        df = app.state.store._df.head(limit)
        
        # Only include edges where target is also a URI node (not literals)
        # This makes the graph visualizable
        df_uri_objects = df.filter(pl.col("object_type") == "uri")
        
        edges = []
        for row in df_uri_objects.iter_rows(named=True):
            edges.append({
                "source": row["subject"],
                "target": row["object"],
                "predicate": row["predicate"],
                "label": row["predicate"].split("/")[-1],
                "confidence": row["confidence"],
                "provenance_source": row["source"],
            })
        
        return {
            "count": len(edges),
            "edges": edges,
        }
    
    @app.get("/graph/subgraph/{node_encoded:path}", tags=["Visualization"])
    async def get_subgraph(
        node_encoded: str,
        depth: int = Query(1, ge=1, le=3, description="Traversal depth"),
    ):
        """Get subgraph around a specific node."""
        import urllib.parse
        node = urllib.parse.unquote(node_encoded)
        
        # Get triples where node is subject or object
        df = app.state.store._df
        
        outgoing = df.filter(pl.col("subject") == node)
        incoming = df.filter(pl.col("object") == node)
        
        related = pl.concat([outgoing, incoming]).unique()
        
        nodes = set()
        edges = []
        
        for row in related.iter_rows(named=True):
            nodes.add(row["subject"])
            # Only add object as a node if it's a URI, not a literal
            if row["object_type"] == "uri":
                nodes.add(row["object"])
                edges.append({
                    "source": row["subject"],
                    "target": row["object"],
                    "predicate": row["predicate"],
                    "confidence": row["confidence"],
                })
        
        return {
            "center": node,
            "nodes": [{"id": n, "label": n.split("/")[-1] if "/" in n else n} for n in nodes],
            "edges": edges,
        }
    
    return app


# Default app instance for running directly
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rdf_starbase.web:app", host="0.0.0.0", port=8000, reload=True)
