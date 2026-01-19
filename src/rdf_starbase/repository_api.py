"""
Repository Management API Router.

Provides REST endpoints for managing multiple repositories:
- Create/delete repositories
- List repositories
- Get repository info
- Scoped SPARQL queries per repository
"""

from pathlib import Path
from typing import Optional, Union
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import polars as pl

from rdf_starbase.repositories import RepositoryManager, RepositoryInfo
from rdf_starbase import execute_sparql


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateRepositoryRequest(BaseModel):
    """Request to create a new repository."""
    name: str = Field(..., description="Unique repository name (alphanumeric, hyphens, underscores)")
    description: str = Field(default="", description="Human-readable description")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class UpdateRepositoryRequest(BaseModel):
    """Request to update repository metadata."""
    description: Optional[str] = Field(None, description="New description")
    tags: Optional[list[str]] = Field(None, description="New tags")


class RenameRepositoryRequest(BaseModel):
    """Request to rename a repository."""
    new_name: str = Field(..., description="New repository name")


class SPARQLQueryRequest(BaseModel):
    """SPARQL query for a specific repository."""
    query: str = Field(..., description="SPARQL-Star query string")


class RepositoryResponse(BaseModel):
    """Response containing repository info."""
    name: str
    description: str
    tags: list[str]
    created_at: str
    triple_count: int
    subject_count: int
    predicate_count: int
    
    @classmethod
    def from_info(cls, info: RepositoryInfo) -> "RepositoryResponse":
        return cls(
            name=info.name,
            description=info.description,
            tags=info.tags,
            created_at=info.created_at.isoformat(),
            triple_count=info.triple_count,
            subject_count=info.subject_count,
            predicate_count=info.predicate_count,
        )


def dataframe_to_records(df: pl.DataFrame) -> list[dict]:
    """Convert Polars DataFrame to list of dicts for JSON serialization."""
    from datetime import datetime
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


def create_repository_router(
    workspace_path: Optional[str | Path] = None
) -> tuple[APIRouter, RepositoryManager]:
    """
    Create the repository management API router.
    
    Args:
        workspace_path: Path to store repositories (default: ./data/repositories)
        
    Returns:
        Tuple of (router, manager)
    """
    # Default workspace path
    if workspace_path is None:
        workspace_path = os.environ.get(
            "RDFSTARBASE_REPOSITORY_PATH",
            "./data/repositories"
        )
    
    manager = RepositoryManager(workspace_path)
    router = APIRouter(prefix="/repositories", tags=["Repositories"])
    
    # =========================================================================
    # Repository CRUD
    # =========================================================================
    
    @router.get("")
    async def list_repositories():
        """List all repositories."""
        repos = manager.list_repositories()
        return {
            "count": len(repos),
            "repositories": [RepositoryResponse.from_info(r).model_dump() for r in repos]
        }
    
    @router.post("")
    async def create_repository(request: CreateRepositoryRequest):
        """Create a new repository."""
        try:
            info = manager.create(
                name=request.name,
                description=request.description,
                tags=request.tags,
            )
            # Auto-save after creation
            manager.save(request.name)
            return {
                "success": True,
                "message": f"Repository '{request.name}' created",
                "repository": RepositoryResponse.from_info(info).model_dump()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/{name}")
    async def get_repository(name: str):
        """Get repository info."""
        try:
            info = manager.get_info(name)
            return RepositoryResponse.from_info(info).model_dump()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.patch("/{name}")
    async def update_repository(name: str, request: UpdateRepositoryRequest):
        """Update repository metadata."""
        try:
            info = manager.update_info(
                name=name,
                description=request.description,
                tags=request.tags,
            )
            return RepositoryResponse.from_info(info).model_dump()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.delete("/{name}")
    async def delete_repository(
        name: str,
        force: bool = Query(False, description="Force delete even if repository has data")
    ):
        """Delete a repository."""
        try:
            manager.delete(name, force=force)
            return {
                "success": True,
                "message": f"Repository '{name}' deleted"
            }
        except ValueError as e:
            if "does not exist" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.post("/{name}/rename")
    async def rename_repository(name: str, request: RenameRepositoryRequest):
        """Rename a repository."""
        try:
            info = manager.rename(name, request.new_name)
            return {
                "success": True,
                "message": f"Repository renamed from '{name}' to '{request.new_name}'",
                "repository": RepositoryResponse.from_info(info).model_dump()
            }
        except ValueError as e:
            if "does not exist" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    # =========================================================================
    # Repository SPARQL
    # =========================================================================
    
    @router.post("/{name}/sparql")
    async def repository_sparql(name: str, request: SPARQLQueryRequest):
        """Execute a SPARQL query against a specific repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            result = execute_sparql(store, request.query)
            
            if isinstance(result, bool):
                # ASK query
                return {"type": "ask", "result": result}
            elif isinstance(result, dict):
                # UPDATE operation
                # Auto-save after update
                manager.save(name)
                return {
                    "type": "update",
                    "operation": result.get("operation", "unknown"),
                    "count": result.get("count", 0),
                    "success": True,
                }
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
    
    # =========================================================================
    # Repository Triple Management
    # =========================================================================
    
    @router.get("/{name}/triples")
    async def get_repository_triples(
        name: str,
        subject: Optional[str] = Query(None, description="Filter by subject"),
        predicate: Optional[str] = Query(None, description="Filter by predicate"),
        object: Optional[str] = Query(None, description="Filter by object"),
        limit: int = Query(100, ge=1, le=10000, description="Maximum results"),
    ):
        """Get triples from a specific repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        df = store.get_triples(
            subject=subject,
            predicate=predicate,
            obj=object,
        )
        
        df = df.head(limit)
        
        return {
            "count": len(df),
            "triples": dataframe_to_records(df),
        }
    
    @router.post("/{name}/triples/batch")
    async def add_repository_triples_batch(
        name: str,
        triples: list[dict]
    ):
        """Add multiple triples to a specific repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            count = store.add_triples_batch(triples)
            # Auto-save after batch insert
            manager.save(name)
            return {
                "success": True,
                "count": count,
                "message": f"Added {count} triples to repository '{name}'",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # =========================================================================
    # Repository Stats
    # =========================================================================
    
    @router.get("/{name}/stats")
    async def get_repository_stats(name: str):
        """Get detailed statistics for a repository."""
        try:
            store = manager.get_store(name)
            info = manager.get_info(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        return {
            "name": name,
            "description": info.description,
            "created_at": info.created_at.isoformat(),
            "stats": store.stats(),
        }
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    @router.post("/{name}/save")
    async def save_repository(name: str):
        """Explicitly save a repository to disk."""
        try:
            manager.save(name)
            return {
                "success": True,
                "message": f"Repository '{name}' saved"
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.post("/save-all")
    async def save_all_repositories():
        """Save all loaded repositories to disk."""
        manager.save_all()
        return {
            "success": True,
            "message": "All repositories saved"
        }
    
    # =========================================================================
    # Example Datasets
    # =========================================================================
    
    @router.get("/examples/datasets")
    async def list_example_datasets():
        """List available example datasets."""
        return {
            "datasets": [
                {
                    "id": "movies",
                    "name": "Movies & Directors",
                    "description": "Sample movie data with directors, actors, and genres. Great for learning SPARQL.",
                    "triple_count": 45,
                    "tags": ["movies", "entertainment", "schema.org"]
                },
                {
                    "id": "techcorp",
                    "name": "TechCorp Customer Service",
                    "description": "Customer service scenario with tickets, products, and customer data. Includes conflicting data from multiple sources.",
                    "triple_count": 35,
                    "tags": ["enterprise", "CRM", "support"]
                },
                {
                    "id": "knowledge_graph",
                    "name": "Simple Knowledge Graph",
                    "description": "Basic knowledge graph with people, organizations, and relationships.",
                    "triple_count": 28,
                    "tags": ["people", "organizations", "relationships"]
                },
                {
                    "id": "rdf_star_demo",
                    "name": "RDF-Star Demo",
                    "description": "Demonstrates RDF-Star features with quoted triples, annotations, and provenance metadata.",
                    "triple_count": 22,
                    "tags": ["rdf-star", "annotations", "provenance"]
                }
            ]
        }
    
    @router.post("/{name}/load-example/{dataset_id}")
    async def load_example_dataset(name: str, dataset_id: str):
        """Load an example dataset into a repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        datasets = {
            "movies": get_movies_dataset_triples,
            "techcorp": get_techcorp_dataset_triples,
            "knowledge_graph": get_knowledge_graph_dataset_triples,
            "rdf_star_demo": get_rdf_star_demo_dataset_triples,
        }
        
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")
        
        triples = datasets[dataset_id]()
        
        try:
            count = store.add_triples_batch(triples)
            manager.save(name)
            return {
                "success": True,
                "dataset": dataset_id,
                "message": f"Loaded example dataset '{dataset_id}' into repository '{name}'",
                "stats": store.stats()
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    return router, manager


# =============================================================================
# Example Datasets with Provenance
# =============================================================================

def get_movies_dataset_triples() -> list[dict]:
    """Movies & Directors dataset with varied sources and confidence."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    SCHEMA = "http://schema.org/"
    EX = "http://example.org/"
    
    triples = []
    
    # Helper to add triple with provenance
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Directors - from Wikipedia (high confidence)
    add(f"{EX}person/nolan", RDF_TYPE, f"{SCHEMA}Person", "Wikipedia", 0.98)
    add(f"{EX}person/nolan", f"{SCHEMA}name", "Christopher Nolan", "Wikipedia", 0.99)
    add(f"{EX}person/nolan", f"{SCHEMA}birthDate", "1970-07-30", "Wikipedia", 0.95)
    add(f"{EX}person/nolan", f"{SCHEMA}nationality", "British-American", "Wikipedia", 0.92)
    
    add(f"{EX}person/spielberg", RDF_TYPE, f"{SCHEMA}Person", "Wikipedia", 0.98)
    add(f"{EX}person/spielberg", f"{SCHEMA}name", "Steven Spielberg", "Wikipedia", 0.99)
    add(f"{EX}person/spielberg", f"{SCHEMA}birthDate", "1946-12-18", "Wikipedia", 0.97)
    
    add(f"{EX}person/greta", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.94)
    add(f"{EX}person/greta", f"{SCHEMA}name", "Greta Gerwig", "IMDb", 0.96)
    add(f"{EX}person/greta", f"{SCHEMA}birthDate", "1983-08-04", "IMDb", 0.90)
    
    # Actors - from IMDb (good confidence)
    add(f"{EX}person/dicaprio", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.97)
    add(f"{EX}person/dicaprio", f"{SCHEMA}name", "Leonardo DiCaprio", "IMDb", 0.99)
    add(f"{EX}person/dicaprio", f"{SCHEMA}birthDate", "1974-11-11", "IMDb", 0.95)
    
    add(f"{EX}person/cillian", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.96)
    add(f"{EX}person/cillian", f"{SCHEMA}name", "Cillian Murphy", "IMDb", 0.98)
    
    add(f"{EX}person/margot", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.97)
    add(f"{EX}person/margot", f"{SCHEMA}name", "Margot Robbie", "IMDb", 0.99)
    
    # Movies - mixed sources
    # Inception - from multiple sources
    add(f"{EX}movie/inception", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}name", "Inception", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}datePublished", "2010", "IMDb", 0.98)
    add(f"{EX}movie/inception", f"{SCHEMA}director", f"{EX}person/nolan", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}actor", f"{EX}person/dicaprio", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}genre", "Sci-Fi", "RottenTomatoes", 0.85)
    add(f"{EX}movie/inception", f"{SCHEMA}duration", "PT2H28M", "IMDb", 0.97)
    
    # Oppenheimer - recent film, high confidence
    add(f"{EX}movie/oppenheimer", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}name", "Oppenheimer", "IMDb", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}datePublished", "2023", "BoxOfficeMojo", 0.98)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}director", f"{EX}person/nolan", "Wikipedia", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}actor", f"{EX}person/cillian", "IMDb", 0.98)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}genre", "Drama", "RottenTomatoes", 0.88)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}genre", "Biography", "Wikipedia", 0.82)  # Multiple genres!
    
    # Interstellar
    add(f"{EX}movie/interstellar", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}name", "Interstellar", "IMDb", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}datePublished", "2014", "IMDb", 0.98)
    add(f"{EX}movie/interstellar", f"{SCHEMA}director", f"{EX}person/nolan", "Wikipedia", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}genre", "Sci-Fi", "IMDb", 0.92)
    
    # Jurassic Park - classic film
    add(f"{EX}movie/jurassic_park", RDF_TYPE, f"{SCHEMA}Movie", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}name", "Jurassic Park", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}datePublished", "1993", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}director", f"{EX}person/spielberg", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}genre", "Adventure", "IMDb", 0.90)
    
    # Barbie - from entertainment news (slightly lower confidence)
    add(f"{EX}movie/barbie", RDF_TYPE, f"{SCHEMA}Movie", "BoxOfficeMojo", 0.97)
    add(f"{EX}movie/barbie", f"{SCHEMA}name", "Barbie", "BoxOfficeMojo", 0.99)
    add(f"{EX}movie/barbie", f"{SCHEMA}datePublished", "2023", "BoxOfficeMojo", 0.98)
    add(f"{EX}movie/barbie", f"{SCHEMA}director", f"{EX}person/greta", "IMDb", 0.97)
    add(f"{EX}movie/barbie", f"{SCHEMA}actor", f"{EX}person/margot", "IMDb", 0.98)
    add(f"{EX}movie/barbie", f"{SCHEMA}genre", "Comedy", "RottenTomatoes", 0.80)
    
    return triples


def get_techcorp_dataset_triples() -> list[dict]:
    """TechCorp Customer Service dataset with conflicting data from multiple sources."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    TC = "http://techcorp.com/"
    FOAF = "http://xmlns.com/foaf/0.1/"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Customer C001 - Alice Johnson (CONFLICTING DATA from different systems!)
    add(f"{TC}customer/C001", RDF_TYPE, f"{TC}Customer", "CRM_System", 0.99)
    add(f"{TC}customer/C001", f"{FOAF}name", "Alice Johnson", "CRM_System", 0.98)
    add(f"{TC}customer/C001", f"{TC}email", "alice@example.com", "CRM_System", 0.95)
    add(f"{TC}customer/C001", f"{TC}tier", "Premium", "CRM_System", 0.90)
    add(f"{TC}customer/C001", f"{TC}since", "2020-01-15", "CRM_System", 0.99)
    
    # CONFLICT: Billing system says different tier!
    add(f"{TC}customer/C001", f"{TC}tier", "Enterprise", "Billing_System", 0.85)
    # CONFLICT: Support portal has different email
    add(f"{TC}customer/C001", f"{TC}email", "a.johnson@corp.example.com", "Support_Portal", 0.75)
    
    # Customer C002 - Bob Smith
    add(f"{TC}customer/C002", RDF_TYPE, f"{TC}Customer", "CRM_System", 0.99)
    add(f"{TC}customer/C002", f"{FOAF}name", "Bob Smith", "CRM_System", 0.97)
    add(f"{TC}customer/C002", f"{TC}email", "bob@example.com", "CRM_System", 0.96)
    add(f"{TC}customer/C002", f"{TC}tier", "Standard", "CRM_System", 0.92)
    add(f"{TC}customer/C002", f"{TC}since", "2021-06-20", "CRM_System", 0.99)
    
    # Customer C003 - Carol White (from legacy system)
    add(f"{TC}customer/C003", RDF_TYPE, f"{TC}Customer", "Legacy_DB", 0.88)
    add(f"{TC}customer/C003", f"{FOAF}name", "Carol White", "Legacy_DB", 0.85)
    add(f"{TC}customer/C003", f"{TC}email", "carol@example.com", "Legacy_DB", 0.80)
    add(f"{TC}customer/C003", f"{TC}tier", "Enterprise", "Billing_System", 0.95)
    
    # Products - from Product Catalog (high confidence)
    add(f"{TC}product/P001", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P001", f"{RDFS}label", "CloudSync Pro", "Product_Catalog", 0.99)
    add(f"{TC}product/P001", f"{TC}category", "Software", "Product_Catalog", 0.98)
    add(f"{TC}product/P001", f"{TC}price", "299.99", "Product_Catalog", 0.97)
    # CONFLICT: Sales team has different price!
    add(f"{TC}product/P001", f"{TC}price", "279.99", "Sales_Team", 0.70)
    
    add(f"{TC}product/P002", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P002", f"{RDFS}label", "DataVault", "Product_Catalog", 0.99)
    add(f"{TC}product/P002", f"{TC}category", "Storage", "Product_Catalog", 0.98)
    add(f"{TC}product/P002", f"{TC}price", "499.99", "Product_Catalog", 0.97)
    
    add(f"{TC}product/P003", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P003", f"{RDFS}label", "SecureNet", "Product_Catalog", 0.99)
    add(f"{TC}product/P003", f"{TC}category", "Security", "Product_Catalog", 0.98)
    add(f"{TC}product/P003", f"{TC}price", "199.99", "Product_Catalog", 0.97)
    
    # Support Tickets - from different support channels
    add(f"{TC}ticket/T001", RDF_TYPE, f"{TC}SupportTicket", "Support_Portal", 0.99)
    add(f"{TC}ticket/T001", f"{TC}customer", f"{TC}customer/C001", "Support_Portal", 0.99)
    add(f"{TC}ticket/T001", f"{TC}product", f"{TC}product/P001", "Support_Portal", 0.98)
    add(f"{TC}ticket/T001", f"{TC}status", "Open", "Support_Portal", 0.95)
    add(f"{TC}ticket/T001", f"{TC}priority", "High", "Support_Portal", 0.90)
    add(f"{TC}ticket/T001", f"{TC}description", "Sync failing intermittently", "Support_Portal", 0.99)
    
    add(f"{TC}ticket/T002", RDF_TYPE, f"{TC}SupportTicket", "Email_Integration", 0.95)
    add(f"{TC}ticket/T002", f"{TC}customer", f"{TC}customer/C002", "Email_Integration", 0.92)
    add(f"{TC}ticket/T002", f"{TC}product", f"{TC}product/P002", "Email_Integration", 0.90)
    add(f"{TC}ticket/T002", f"{TC}status", "Resolved", "Support_Portal", 0.98)
    add(f"{TC}ticket/T002", f"{TC}priority", "Medium", "Email_Integration", 0.85)
    add(f"{TC}ticket/T002", f"{TC}description", "Storage quota question", "Email_Integration", 0.88)
    
    add(f"{TC}ticket/T003", RDF_TYPE, f"{TC}SupportTicket", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}customer", f"{TC}customer/C003", "Security_Ops", 0.98)
    add(f"{TC}ticket/T003", f"{TC}product", f"{TC}product/P003", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}status", "Open", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}priority", "Critical", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}description", "Security alert investigation", "Security_Ops", 0.99)
    
    return triples


def get_knowledge_graph_dataset_triples() -> list[dict]:
    """Simple Knowledge Graph with people and organizations."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    FOAF = "http://xmlns.com/foaf/0.1/"
    ORG = "http://www.w3.org/ns/org#"
    EX = "http://example.org/"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # People - from LinkedIn (professional network)
    add(f"{EX}person/jane", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.95)
    add(f"{EX}person/jane", f"{FOAF}name", "Jane Doe", "LinkedIn", 0.98)
    add(f"{EX}person/jane", f"{FOAF}age", "32", "LinkedIn", 0.75)  # Age less reliable
    add(f"{EX}person/jane", f"{FOAF}knows", f"{EX}person/john", "LinkedIn", 0.90)
    add(f"{EX}person/jane", f"{FOAF}knows", f"{EX}person/alice", "LinkedIn", 0.92)
    add(f"{EX}person/jane", f"{ORG}memberOf", f"{EX}org/acme", "LinkedIn", 0.97)
    
    add(f"{EX}person/john", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.95)
    add(f"{EX}person/john", f"{FOAF}name", "John Smith", "LinkedIn", 0.97)
    add(f"{EX}person/john", f"{FOAF}age", "28", "Facebook", 0.70)  # Different source
    add(f"{EX}person/john", f"{FOAF}knows", f"{EX}person/jane", "LinkedIn", 0.90)
    add(f"{EX}person/john", f"{ORG}memberOf", f"{EX}org/globex", "CompanyWebsite", 0.99)
    
    add(f"{EX}person/alice", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.96)
    add(f"{EX}person/alice", f"{FOAF}name", "Alice Chen", "LinkedIn", 0.98)
    add(f"{EX}person/alice", f"{FOAF}age", "35", "LinkedIn", 0.72)
    add(f"{EX}person/alice", f"{FOAF}knows", f"{EX}person/jane", "LinkedIn", 0.92)
    add(f"{EX}person/alice", f"{FOAF}knows", f"{EX}person/bob", "Email_Analysis", 0.65)
    add(f"{EX}person/alice", f"{ORG}headOf", f"{EX}org/acme", "CompanyWebsite", 0.99)
    
    add(f"{EX}person/bob", RDF_TYPE, f"{FOAF}Person", "HR_System", 0.98)
    add(f"{EX}person/bob", f"{FOAF}name", "Bob Williams", "HR_System", 0.99)
    add(f"{EX}person/bob", f"{FOAF}age", "42", "HR_System", 0.95)
    add(f"{EX}person/bob", f"{ORG}memberOf", f"{EX}org/initech", "HR_System", 0.98)
    
    # Organizations - from company registries
    add(f"{EX}org/acme", RDF_TYPE, f"{ORG}Organization", "SEC_Filings", 0.99)
    add(f"{EX}org/acme", f"{FOAF}name", "Acme Corp", "SEC_Filings", 0.99)
    add(f"{EX}org/acme", f"{ORG}hasSite", f"{EX}location/sf", "CompanyWebsite", 0.95)
    add(f"{EX}org/acme", f"{EX}industry", "Technology", "CrunchBase", 0.88)
    
    add(f"{EX}org/globex", RDF_TYPE, f"{ORG}Organization", "SEC_Filings", 0.99)
    add(f"{EX}org/globex", f"{FOAF}name", "Globex Inc", "SEC_Filings", 0.99)
    add(f"{EX}org/globex", f"{ORG}hasSite", f"{EX}location/nyc", "CompanyWebsite", 0.96)
    add(f"{EX}org/globex", f"{EX}industry", "Finance", "Bloomberg", 0.95)
    
    add(f"{EX}org/initech", RDF_TYPE, f"{ORG}Organization", "State_Registry", 0.97)
    add(f"{EX}org/initech", f"{FOAF}name", "Initech", "State_Registry", 0.98)
    add(f"{EX}org/initech", f"{ORG}hasSite", f"{EX}location/austin", "GoogleMaps", 0.85)
    add(f"{EX}org/initech", f"{EX}industry", "Software", "LinkedIn", 0.80)
    
    # Locations - from geographic databases
    add(f"{EX}location/sf", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/sf", f"{FOAF}name", "San Francisco", "GeoNames", 0.99)
    add(f"{EX}location/sf", f"{EX}country", "USA", "GeoNames", 0.99)
    
    add(f"{EX}location/nyc", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/nyc", f"{FOAF}name", "New York City", "GeoNames", 0.99)
    add(f"{EX}location/nyc", f"{EX}country", "USA", "GeoNames", 0.99)
    
    add(f"{EX}location/austin", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/austin", f"{FOAF}name", "Austin", "GeoNames", 0.99)
    add(f"{EX}location/austin", f"{EX}country", "USA", "GeoNames", 0.99)
    
    return triples


def get_rdf_star_demo_dataset_triples() -> list[dict]:
    """RDF-Star Demo showing statement-level metadata."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    EX = "http://example.org/"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Employee data from HR System (high confidence)
    add(f"{EX}alice", f"{EX}name", "Alice", "HR_System", 0.99)
    add(f"{EX}alice", f"{EX}worksAt", f"{EX}acme", "HR_System", 0.98)
    add(f"{EX}alice", f"{EX}department", "Engineering", "HR_System", 0.97)
    add(f"{EX}alice", f"{EX}startDate", "2020-03-15", "HR_System", 0.99)
    
    # Salary from Payroll (very high confidence, sensitive)
    add(f"{EX}alice", f"{EX}salary", "95000", "Payroll_System", 0.999)
    
    # Manager relationship
    add(f"{EX}bob", f"{EX}name", "Bob", "HR_System", 0.99)
    add(f"{EX}bob", f"{EX}worksAt", f"{EX}acme", "HR_System", 0.98)
    add(f"{EX}bob", f"{EX}manages", f"{EX}alice", "HR_System", 0.95)
    add(f"{EX}bob", f"{EX}title", "Engineering Manager", "HR_System", 0.97)
    
    # Company info from different sources
    add(f"{EX}acme", f"{EX}name", "Acme Corporation", "SEC_Filings", 0.99)
    add(f"{EX}acme", f"{EX}founded", "2010", "CrunchBase", 0.92)
    add(f"{EX}acme", f"{EX}headquarters", "San Francisco", "CompanyWebsite", 0.95)
    add(f"{EX}acme", f"{EX}employees", "500", "LinkedIn", 0.75)  # Less reliable
    add(f"{EX}acme", f"{EX}employees", "487", "SEC_Filings", 0.98)  # Conflicting!
    
    # Performance reviews (varying confidence)
    add(f"{EX}alice", f"{EX}performanceRating", "Exceeds Expectations", "Performance_System", 0.90)
    add(f"{EX}alice", f"{EX}lastReview", "2024-12-01", "Performance_System", 0.99)
    
    # Skills from different sources
    add(f"{EX}alice", f"{EX}skill", "Python", "LinkedIn", 0.85)
    add(f"{EX}alice", f"{EX}skill", "Machine Learning", "LinkedIn", 0.80)
    add(f"{EX}alice", f"{EX}skill", "Distributed Systems", "Manager_Assessment", 0.92)
    
    # Project assignments
    add(f"{EX}project/alpha", f"{EX}name", "Project Alpha", "Jira", 0.99)
    add(f"{EX}project/alpha", f"{EX}status", "Active", "Jira", 0.98)
    add(f"{EX}project/alpha", f"{EX}lead", f"{EX}alice", "Jira", 0.97)
    add(f"{EX}alice", f"{EX}assignedTo", f"{EX}project/alpha", "Jira", 0.96)
    add(f"{EX}bob", f"{EX}sponsors", f"{EX}project/alpha", "Executive_Dashboard", 0.88)
    
    return triples


# Keep old SPARQL functions for backwards compatibility (unused)
def get_movies_dataset() -> str:
    """Deprecated: Use get_movies_dataset_triples instead."""
    return ""

def get_techcorp_dataset() -> str:
    """Deprecated: Use get_techcorp_dataset_triples instead."""
    return ""

def get_knowledge_graph_dataset() -> str:
    """Deprecated: Use get_knowledge_graph_dataset_triples instead."""
    return ""

def get_rdf_star_demo_dataset() -> str:
    """Deprecated: Use get_rdf_star_demo_dataset_triples instead."""
    return ""
