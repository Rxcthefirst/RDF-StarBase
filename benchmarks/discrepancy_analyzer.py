"""
Triple count discrepancy analyzer.

Investigates why different RDF stores report different triple counts for the same file.

Common causes:
1. OWL imports - GraphDB may follow owl:imports and load external ontologies
2. Blank node handling - Some parsers consolidate, others expand
3. RDF-Star/reification - Different handling of quoted triples
4. Inference/reasoning - Some stores materialize inferred triples
5. Parser strictness - Some parsers reject malformed triples
"""
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_with_rdflib(file_path: str) -> dict:
    """Analyze file using rdflib (pure Python, reference parser)."""
    try:
        import rdflib
        from rdflib import OWL, RDF
    except ImportError:
        return {"error": "rdflib not installed"}
    
    g = rdflib.Graph()
    g.parse(file_path, format="turtle")
    
    analysis = {
        "parser": "rdflib",
        "total_triples": len(g),
        "unique_subjects": len(set(g.subjects())),
        "unique_predicates": len(set(g.predicates())),
        "unique_objects": len(set(g.objects())),
    }
    
    # Count blank nodes
    blank_subjects = sum(1 for s in g.subjects() if isinstance(s, rdflib.BNode))
    blank_objects = sum(1 for _, _, o in g if isinstance(o, rdflib.BNode))
    analysis["blank_node_subjects"] = len(set(s for s in g.subjects() if isinstance(s, rdflib.BNode)))
    analysis["blank_node_triples"] = sum(1 for s, _, o in g if isinstance(s, rdflib.BNode) or isinstance(o, rdflib.BNode))
    
    # Check for OWL imports
    owl_imports = list(g.objects(predicate=OWL.imports))
    analysis["owl_imports"] = [str(i) for i in owl_imports]
    analysis["owl_import_count"] = len(owl_imports)
    
    # Check for reification (rdf:Statement patterns)
    reification_count = len(list(g.subjects(RDF.type, RDF.Statement)))
    analysis["reification_statements"] = reification_count
    
    # Predicate distribution (top 20)
    pred_counts = Counter(str(p) for _, p, _ in g)
    analysis["top_predicates"] = dict(pred_counts.most_common(20))
    
    # Namespace usage
    namespaces = {}
    for prefix, ns in g.namespaces():
        namespaces[prefix] = str(ns)
    analysis["namespaces"] = namespaces
    
    # Check for RDF lists (rdf:first, rdf:rest patterns)
    rdf_list_triples = sum(1 for _, p, _ in g if str(p) in [str(RDF.first), str(RDF.rest)])
    analysis["rdf_list_triples"] = rdf_list_triples
    
    # OWL construct patterns
    owl_patterns = {
        "owl:Class": len(list(g.subjects(RDF.type, OWL.Class))),
        "owl:ObjectProperty": len(list(g.subjects(RDF.type, OWL.ObjectProperty))),
        "owl:DatatypeProperty": len(list(g.subjects(RDF.type, OWL.DatatypeProperty))),
        "owl:Restriction": len(list(g.subjects(RDF.type, OWL.Restriction))),
        "owl:AnnotationProperty": len(list(g.subjects(RDF.type, OWL.AnnotationProperty))),
    }
    analysis["owl_constructs"] = owl_patterns
    
    return analysis


def analyze_with_oxigraph(file_path: str) -> dict:
    """Analyze file using Oxigraph (Rust parser, used by RDF-StarBase)."""
    try:
        from pyoxigraph import Store, RdfFormat
    except ImportError:
        return {"error": "pyoxigraph not installed"}
    
    store = Store()
    with open(file_path, 'rb') as f:
        store.load(f, RdfFormat.TURTLE, base_iri="http://example.org/")
    
    triples = list(store)
    
    analysis = {
        "parser": "oxigraph",
        "total_triples": len(triples),
        "unique_subjects": len(set(t.subject for t in triples)),
        "unique_predicates": len(set(t.predicate for t in triples)),
        "unique_objects": len(set(t.object for t in triples)),
    }
    
    # Count blank nodes
    from pyoxigraph import BlankNode
    blank_subjects = sum(1 for t in triples if isinstance(t.subject, BlankNode))
    blank_objects = sum(1 for t in triples if isinstance(t.object, BlankNode))
    analysis["blank_node_subjects"] = len(set(t.subject for t in triples if isinstance(t.subject, BlankNode)))
    analysis["blank_node_triples"] = sum(1 for t in triples if isinstance(t.subject, BlankNode) or isinstance(t.object, BlankNode))
    
    return analysis


def query_virtuoso(graph: str = "http://benchmark.example.org/") -> dict:
    """Query Virtuoso for triple analysis."""
    endpoint = "http://localhost:8890/sparql"
    
    def run_query(query: str):
        try:
            req = urllib.request.Request(
                f"{endpoint}?query={urllib.parse.quote(query)}&format=json"
            )
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read())["results"]["bindings"]
        except Exception as e:
            return {"error": str(e)}
    
    analysis = {"store": "Virtuoso"}
    
    # Total count
    result = run_query(f"SELECT (COUNT(*) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
    if isinstance(result, dict):
        return result
    analysis["total_triples"] = int(result[0]["c"]["value"]) if result else 0
    
    # Unique subjects
    result = run_query(f"SELECT (COUNT(DISTINCT ?s) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
    analysis["unique_subjects"] = int(result[0]["c"]["value"]) if result else 0
    
    # Unique predicates
    result = run_query(f"SELECT (COUNT(DISTINCT ?p) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
    analysis["unique_predicates"] = int(result[0]["c"]["value"]) if result else 0
    
    # Blank node subjects
    result = run_query(f"SELECT (COUNT(*) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o FILTER(isBlank(?s)) }} }}")
    analysis["blank_node_subject_triples"] = int(result[0]["c"]["value"]) if result else 0
    
    # OWL imports
    result = run_query(f"SELECT ?import WHERE {{ GRAPH <{graph}> {{ ?s <http://www.w3.org/2002/07/owl#imports> ?import }} }}")
    analysis["owl_imports"] = [r["import"]["value"] for r in result] if result else []
    
    # Top predicates
    result = run_query(f"""
        SELECT ?p (COUNT(*) as ?c) WHERE {{ 
            GRAPH <{graph}> {{ ?s ?p ?o }} 
        }} 
        GROUP BY ?p ORDER BY DESC(?c) LIMIT 20
    """)
    if result and not isinstance(result, dict):
        analysis["top_predicates"] = {r["p"]["value"]: int(r["c"]["value"]) for r in result}
    
    return analysis


def query_graphdb(repo: str = "graphdb-benchmark") -> dict:
    """Query GraphDB for triple analysis."""
    endpoint = f"http://localhost:7200/repositories/{repo}"
    
    def run_query(query: str):
        try:
            req = urllib.request.Request(
                f"{endpoint}?query={urllib.parse.quote(query)}"
            )
            req.add_header("Accept", "application/sparql-results+json")
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read())["results"]["bindings"]
        except Exception as e:
            return {"error": str(e)}
    
    analysis = {"store": "GraphDB"}
    
    # Total count (default graph - GraphDB merges into default)
    result = run_query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    if isinstance(result, dict):
        return result
    analysis["total_triples"] = int(result[0]["c"]["value"]) if result else 0
    
    # Unique subjects
    result = run_query("SELECT (COUNT(DISTINCT ?s) as ?c) WHERE { ?s ?p ?o }")
    analysis["unique_subjects"] = int(result[0]["c"]["value"]) if result else 0
    
    # Unique predicates
    result = run_query("SELECT (COUNT(DISTINCT ?p) as ?c) WHERE { ?s ?p ?o }")
    analysis["unique_predicates"] = int(result[0]["c"]["value"]) if result else 0
    
    # Blank node subjects
    result = run_query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o FILTER(isBlank(?s)) }")
    analysis["blank_node_subject_triples"] = int(result[0]["c"]["value"]) if result else 0
    
    # OWL imports
    result = run_query("SELECT ?import WHERE { ?s <http://www.w3.org/2002/07/owl#imports> ?import }")
    analysis["owl_imports"] = [r["import"]["value"] for r in result] if result else []
    
    # Check for inferred triples (GraphDB specific)
    result = run_query("""
        SELECT (COUNT(*) as ?c) WHERE { 
            ?s ?p ?o 
            FILTER EXISTS { ?s a ?type }
        }
    """)
    analysis["triples_with_typed_subjects"] = int(result[0]["c"]["value"]) if result else 0
    
    # Top predicates
    result = run_query("""
        SELECT ?p (COUNT(*) as ?c) WHERE { 
            ?s ?p ?o 
        } 
        GROUP BY ?p ORDER BY DESC(?c) LIMIT 20
    """)
    if result and not isinstance(result, dict):
        analysis["top_predicates"] = {r["p"]["value"]: int(r["c"]["value"]) for r in result}
    
    # Check named graphs
    result = run_query("SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")
    analysis["named_graphs"] = [r["g"]["value"] for r in result] if result else []
    
    return analysis


def compare_analyses(rdflib_analysis: dict, oxigraph_analysis: dict, 
                     virtuoso_analysis: dict, graphdb_analysis: dict):
    """Compare analyses and identify discrepancy sources."""
    
    print("\n" + "=" * 80)
    print("TRIPLE COUNT DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    # Summary table
    print("\n### Summary ###")
    print(f"{'Parser/Store':<20} {'Total Triples':>15} {'Unique Subj':>12} {'Unique Pred':>12}")
    print("-" * 60)
    
    for name, analysis in [
        ("rdflib", rdflib_analysis),
        ("oxigraph", oxigraph_analysis),
        ("Virtuoso", virtuoso_analysis),
        ("GraphDB", graphdb_analysis)
    ]:
        if "error" in analysis:
            print(f"{name:<20} ERROR: {analysis['error']}")
        else:
            print(f"{name:<20} {analysis.get('total_triples', 'N/A'):>15,} "
                  f"{analysis.get('unique_subjects', 'N/A'):>12,} "
                  f"{analysis.get('unique_predicates', 'N/A'):>12}")
    
    # OWL imports analysis
    print("\n### OWL Imports ###")
    for name, analysis in [("rdflib", rdflib_analysis), ("Virtuoso", virtuoso_analysis), ("GraphDB", graphdb_analysis)]:
        if "owl_imports" in analysis:
            imports = analysis["owl_imports"]
            if imports:
                print(f"{name}: {len(imports)} imports found:")
                for imp in imports[:5]:
                    print(f"  - {imp}")
                if len(imports) > 5:
                    print(f"  ... and {len(imports) - 5} more")
            else:
                print(f"{name}: No owl:imports found")
    
    # Blank node analysis
    print("\n### Blank Node Analysis ###")
    for name, analysis in [("rdflib", rdflib_analysis), ("oxigraph", oxigraph_analysis)]:
        if "blank_node_triples" in analysis:
            print(f"{name}: {analysis['blank_node_triples']:,} triples involve blank nodes")
            print(f"        {analysis.get('blank_node_subjects', 'N/A')} unique blank node subjects")
    
    # Predicate comparison
    print("\n### Top Predicates Comparison ###")
    all_preds = set()
    for analysis in [rdflib_analysis, virtuoso_analysis, graphdb_analysis]:
        if "top_predicates" in analysis:
            all_preds.update(analysis["top_predicates"].keys())
    
    # Show predicates with significant differences
    print(f"{'Predicate':<60} {'rdflib':>10} {'Virtuoso':>10} {'GraphDB':>10}")
    print("-" * 95)
    
    for pred in sorted(all_preds)[:15]:
        rdflib_count = rdflib_analysis.get("top_predicates", {}).get(pred, 0)
        virtuoso_count = virtuoso_analysis.get("top_predicates", {}).get(pred, 0)
        graphdb_count = graphdb_analysis.get("top_predicates", {}).get(pred, 0)
        
        # Shorten predicate for display
        short_pred = pred.split("#")[-1] if "#" in pred else pred.split("/")[-1]
        if len(short_pred) > 55:
            short_pred = short_pred[:52] + "..."
        
        print(f"{short_pred:<60} {rdflib_count:>10,} {virtuoso_count:>10,} {graphdb_count:>10,}")
    
    # Identify likely causes
    print("\n### Likely Discrepancy Causes ###")
    
    rdflib_count = rdflib_analysis.get("total_triples", 0)
    virtuoso_count = virtuoso_analysis.get("total_triples", 0)
    graphdb_count = graphdb_analysis.get("total_triples", 0)
    
    if graphdb_count > rdflib_count * 1.5:
        print("⚠️  GraphDB has significantly MORE triples than source file:")
        print("   - Likely cause: OWL imports being followed and loaded")
        print("   - Likely cause: RDFS/OWL inference materializing additional triples")
        print("   - Check: GraphDB repository reasoning settings")
    
    if virtuoso_count < rdflib_count * 0.8:
        print("⚠️  Virtuoso has significantly FEWER triples than source file:")
        print("   - Likely cause: Strict parser rejecting some triples")
        print("   - Likely cause: Blank node consolidation")
        print("   - Likely cause: Bulk loader configuration (check ld_dir patterns)")
    
    if "owl_imports" in rdflib_analysis and rdflib_analysis["owl_imports"]:
        print(f"⚠️  File contains {len(rdflib_analysis['owl_imports'])} owl:imports declarations")
        print("   - GraphDB likely followed these imports")
        print("   - Virtuoso/RDF-StarBase likely ignored them")


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/FIBO_cleaned.ttl"
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Analyzing: {file_path}")
    print(f"File size: {Path(file_path).stat().st_size:,} bytes")
    
    print("\n--- Analyzing with rdflib ---")
    rdflib_analysis = analyze_with_rdflib(file_path)
    
    print("--- Analyzing with Oxigraph ---")
    oxigraph_analysis = analyze_with_oxigraph(file_path)
    
    print("--- Querying Virtuoso ---")
    virtuoso_analysis = query_virtuoso()
    
    print("--- Querying GraphDB ---")
    graphdb_analysis = query_graphdb()
    
    compare_analyses(rdflib_analysis, oxigraph_analysis, virtuoso_analysis, graphdb_analysis)
    
    # Detailed dump for debugging
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS (JSON)")
    print("=" * 80)
    print("\n### rdflib ###")
    print(json.dumps({k: v for k, v in rdflib_analysis.items() if k != "namespaces"}, indent=2, default=str))
    
    return {
        "rdflib": rdflib_analysis,
        "oxigraph": oxigraph_analysis,
        "virtuoso": virtuoso_analysis,
        "graphdb": graphdb_analysis
    }


if __name__ == "__main__":
    main()
