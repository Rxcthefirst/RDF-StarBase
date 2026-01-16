#!/usr/bin/env python
"""
Start the RDF-StarBase API server with demo data preloaded.
This gives users something to explore immediately.
"""

import uvicorn
from rdf_starbase import TripleStore, ProvenanceContext, AssertionRegistry, SourceType
from rdf_starbase.web import create_app


def create_demo_data():
    """Create a store and registry with interesting demo data."""
    store = TripleStore()
    registry = AssertionRegistry()
    
    # Register data sources
    imdb = registry.register_source(
        name="IMDB",
        source_type=SourceType.API,
        uri="https://api.imdb.com/v1",
        owner="media-team",
        description="Internet Movie Database - ratings and metadata",
        tags=["movies", "ratings", "entertainment"]
    )
    
    wiki = registry.register_source(
        name="Wikipedia",
        source_type=SourceType.DATASET,
        uri="https://dumps.wikimedia.org",
        owner="data-team",
        description="Wikipedia knowledge base exports",
        tags=["general", "encyclopedia", "facts"]
    )
    
    rotten = registry.register_source(
        name="RottenTomatoes",
        source_type=SourceType.API,
        uri="https://api.rottentomatoes.com",
        owner="media-team",
        description="Rotten Tomatoes critic and audience scores",
        tags=["movies", "reviews", "critics"]
    )
    
    # Create provenance contexts
    imdb_prov = ProvenanceContext(source="IMDB", confidence=0.95, process="api_sync")
    wiki_prov = ProvenanceContext(source="Wikipedia", confidence=0.85, process="dump_import")
    rotten_prov = ProvenanceContext(source="RottenTomatoes", confidence=0.90, process="api_sync")
    
    # =========================================================================
    # Movies - with competing claims!
    # =========================================================================
    
    # Inception
    store.add_triple("http://example.org/movie/inception", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Movie", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/name", "Inception", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/datePublished", "2010", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/director", "http://example.org/person/nolan", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/genre", "Sci-Fi", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/genre", "Thriller", wiki_prov)
    # Competing ratings!
    store.add_triple("http://example.org/movie/inception", "http://schema.org/aggregateRating", "8.8", imdb_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/aggregateRating", "87%", rotten_prov)
    
    # The Dark Knight
    store.add_triple("http://example.org/movie/dark-knight", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Movie", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/name", "The Dark Knight", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/datePublished", "2008", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/director", "http://example.org/person/nolan", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/genre", "Action", imdb_prov)
    # Competing ratings!
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/aggregateRating", "9.0", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/aggregateRating", "94%", rotten_prov)
    
    # Interstellar
    store.add_triple("http://example.org/movie/interstellar", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Movie", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/name", "Interstellar", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/datePublished", "2014", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/director", "http://example.org/person/nolan", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/duration", "169 min", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/duration", "2h 49m", wiki_prov)  # Same thing, different format!
    
    # =========================================================================
    # People
    # =========================================================================
    
    # Christopher Nolan
    store.add_triple("http://example.org/person/nolan", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Person", wiki_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/name", "Christopher Nolan", wiki_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/birthDate", "1970-07-30", wiki_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/nationality", "British-American", wiki_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/jobTitle", "Director", imdb_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/jobTitle", "Producer", imdb_prov)
    store.add_triple("http://example.org/person/nolan", "http://schema.org/jobTitle", "Screenwriter", imdb_prov)
    
    # Leonardo DiCaprio
    store.add_triple("http://example.org/person/dicaprio", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Person", wiki_prov)
    store.add_triple("http://example.org/person/dicaprio", "http://schema.org/name", "Leonardo DiCaprio", wiki_prov)
    store.add_triple("http://example.org/person/dicaprio", "http://schema.org/birthDate", "1974-11-11", wiki_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/actor", "http://example.org/person/dicaprio", imdb_prov)
    
    # Heath Ledger
    store.add_triple("http://example.org/person/ledger", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Person", wiki_prov)
    store.add_triple("http://example.org/person/ledger", "http://schema.org/name", "Heath Ledger", wiki_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/actor", "http://example.org/person/ledger", imdb_prov)
    
    # =========================================================================
    # Companies
    # =========================================================================
    
    store.add_triple("http://example.org/company/warner", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://schema.org/Organization", wiki_prov)
    store.add_triple("http://example.org/company/warner", "http://schema.org/name", "Warner Bros.", wiki_prov)
    store.add_triple("http://example.org/movie/inception", "http://schema.org/productionCompany", "http://example.org/company/warner", imdb_prov)
    store.add_triple("http://example.org/movie/dark-knight", "http://schema.org/productionCompany", "http://example.org/company/warner", imdb_prov)
    store.add_triple("http://example.org/movie/interstellar", "http://schema.org/productionCompany", "http://example.org/company/warner", imdb_prov)
    
    print(f"✅ Loaded {len(store._df)} assertions from {registry.get_sources().__len__()} sources")
    print(f"   - Unique subjects: {store._df['subject'].n_unique()}")
    print(f"   - Unique predicates: {store._df['predicate'].n_unique()}")
    print(f"   - Sources: {', '.join(s.name for s in registry.get_sources())}")
    
    return store, registry


def main():
    """Run the demo server."""
    print("🚀 Starting RDF-StarBase Demo Server")
    print("=" * 50)
    
    store, registry = create_demo_data()
    
    print()
    print("📊 API Endpoints:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc:      http://localhost:8000/redoc")
    print("   - Stats:      http://localhost:8000/stats")
    print()
    print("🌐 Frontend:     http://localhost:3000")
    print("   (Run 'npm run dev' in frontend/ directory)")
    print()
    print("=" * 50)
    
    app = create_app(store=store, registry=registry)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
