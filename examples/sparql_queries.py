"""
RDF-StarBase Example: SPARQL-Star Queries

This example demonstrates the SPARQL-Star query capabilities,
including basic patterns, filters, and RDF-Star quoted triples.
"""

from rdf_starbase import (
    TripleStore,
    ProvenanceContext,
    execute_sparql,
    parse_query,
)


def main():
    print("🔍 RDF-StarBase: SPARQL-Star Query Demo\n")
    
    # Create and populate store
    store = TripleStore()
    
    # Add sample data about movies
    imdb = ProvenanceContext(source="IMDB", confidence=0.95)
    wiki = ProvenanceContext(source="Wikipedia", confidence=0.85)
    
    movies = [
        ("http://example.org/movie/inception", "http://example.org/title", "Inception", imdb),
        ("http://example.org/movie/inception", "http://example.org/year", "2010", imdb),
        ("http://example.org/movie/inception", "http://example.org/director", "http://example.org/person/nolan", imdb),
        ("http://example.org/movie/inception", "http://example.org/rating", "8.8", imdb),
        
        ("http://example.org/movie/interstellar", "http://example.org/title", "Interstellar", imdb),
        ("http://example.org/movie/interstellar", "http://example.org/year", "2014", imdb),
        ("http://example.org/movie/interstellar", "http://example.org/director", "http://example.org/person/nolan", imdb),
        ("http://example.org/movie/interstellar", "http://example.org/rating", "8.6", imdb),
        
        ("http://example.org/movie/dark_knight", "http://example.org/title", "The Dark Knight", imdb),
        ("http://example.org/movie/dark_knight", "http://example.org/year", "2008", imdb),
        ("http://example.org/movie/dark_knight", "http://example.org/director", "http://example.org/person/nolan", wiki),
        ("http://example.org/movie/dark_knight", "http://example.org/rating", "9.0", imdb),
        
        ("http://example.org/person/nolan", "http://xmlns.com/foaf/0.1/name", "Christopher Nolan", wiki),
        ("http://example.org/person/nolan", "http://example.org/birthYear", "1970", wiki),
    ]
    
    for s, p, o, prov in movies:
        store.add_triple(s, p, o, prov)
    
    print(f"📊 Loaded {len(movies)} assertions about movies\n")
    
    # Example 1: Simple SELECT
    print("=" * 60)
    print("Query 1: SELECT all movie titles")
    print("=" * 60)
    query1 = """
        SELECT ?movie ?title WHERE {
            ?movie <http://example.org/title> ?title
        }
    """
    print(f"SPARQL:\n{query1.strip()}\n")
    results = execute_sparql(store, query1)
    print("Results:")
    print(results)
    print()
    
    # Example 2: With FILTER
    print("=" * 60)
    print("Query 2: Movies with rating > 8.7")
    print("=" * 60)
    query2 = """
        SELECT ?movie ?rating WHERE {
            ?movie <http://example.org/rating> ?rating .
            FILTER(?rating > "8.7")
        }
    """
    print(f"SPARQL:\n{query2.strip()}\n")
    results = execute_sparql(store, query2)
    print("Results:")
    print(results)
    print()
    
    # Example 3: JOIN patterns
    print("=" * 60)
    print("Query 3: Movie titles with their directors")
    print("=" * 60)
    query3 = """
        SELECT ?title ?director WHERE {
            ?movie <http://example.org/title> ?title .
            ?movie <http://example.org/director> ?director
        }
    """
    print(f"SPARQL:\n{query3.strip()}\n")
    results = execute_sparql(store, query3)
    print("Results:")
    print(results)
    print()
    
    # Example 4: ORDER BY and LIMIT
    print("=" * 60)
    print("Query 4: Top 2 highest rated movies")
    print("=" * 60)
    query4 = """
        SELECT ?title ?rating WHERE {
            ?movie <http://example.org/title> ?title .
            ?movie <http://example.org/rating> ?rating
        }
        ORDER BY DESC(?rating)
        LIMIT 2
    """
    print(f"SPARQL:\n{query4.strip()}\n")
    results = execute_sparql(store, query4)
    print("Results:")
    print(results)
    print()
    
    # Example 5: ASK query
    print("=" * 60)
    print("Query 5: ASK - Does Inception exist?")
    print("=" * 60)
    query5 = """
        ASK WHERE {
            ?movie <http://example.org/title> "Inception"
        }
    """
    print(f"SPARQL:\n{query5.strip()}\n")
    result = execute_sparql(store, query5)
    print(f"Result: {result}")
    print()
    
    # Example 6: SELECT *
    print("=" * 60)
    print("Query 6: SELECT * - All data about Nolan")
    print("=" * 60)
    query6 = """
        SELECT * WHERE {
            <http://example.org/person/nolan> ?predicate ?value
        }
    """
    print(f"SPARQL:\n{query6.strip()}\n")
    results = execute_sparql(store, query6)
    print("Results:")
    print(results)
    print()
    
    # Example 7: DISTINCT
    print("=" * 60)
    print("Query 7: DISTINCT directors")
    print("=" * 60)
    query7 = """
        SELECT DISTINCT ?director WHERE {
            ?movie <http://example.org/director> ?director
        }
    """
    print(f"SPARQL:\n{query7.strip()}\n")
    results = execute_sparql(store, query7)
    print("Results:")
    print(results)
    print()
    
    # Example 8: Parsing info
    print("=" * 60)
    print("Bonus: Parse Query to AST")
    print("=" * 60)
    query8 = """
        PREFIX ex: <http://example.org/>
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            FILTER(?o > 10)
        }
        LIMIT 5
    """
    print(f"SPARQL:\n{query8.strip()}\n")
    ast = parse_query(query8)
    print(f"Query type: {type(ast).__name__}")
    print(f"Prefixes: {ast.prefixes}")
    print(f"Variables: {[v.name for v in ast.variables]}")
    print(f"Limit: {ast.limit}")
    print(f"Pattern count: {len(ast.where.patterns)}")
    print(f"Filter count: {len(ast.where.filters)}")
    print()
    
    print("=" * 60)
    print("🎉 SPARQL Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
