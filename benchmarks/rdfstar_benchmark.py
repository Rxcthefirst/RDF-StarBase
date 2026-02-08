"""
RDF-Star Specific Benchmark

Tests RDF-StarBase's RDF-Star capabilities with annotated fact workloads.
Simulates AI fact grounding use case with confidence scores and sources.

Usage:
    python benchmarks/rdfstar_benchmark.py
    python benchmarks/rdfstar_benchmark.py --facts 100000
"""

import time
import sys
import os
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_rdfstar_data(num_facts: int, output_file: str):
    """Generate RDF-Star test data simulating AI fact extraction."""
    print(f"Generating {num_facts:,} annotated facts to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Sources for facts
    sources = ["GPT-4", "Claude", "Gemini", "Wikipedia", "DBpedia", "Wikidata", "Manual"]
    
    # Entity types
    entity_types = ["Person", "Organization", "Location", "Event", "Concept"]
    
    # Predicates
    predicates = [
        "bornIn", "diedIn", "worksAt", "locatedIn", "foundedBy",
        "hasRole", "memberOf", "relatedTo", "partOf", "createdBy"
    ]
    
    with open(output_file, 'w') as f:
        # Prefixes
        f.write("@prefix ex: <http://example.org/> .\n")
        f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        f.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
        f.write("@prefix prov: <http://www.w3.org/ns/prov#> .\n\n")
        
        for i in range(num_facts):
            # Generate a fact
            subj_type = random.choice(entity_types)
            subj_id = random.randint(0, num_facts // 10)
            pred = random.choice(predicates)
            obj_type = random.choice(entity_types)
            obj_id = random.randint(0, num_facts // 10)
            
            subject = f"ex:{subj_type}_{subj_id}"
            predicate = f"ex:{pred}"
            obj = f"ex:{obj_type}_{obj_id}"
            
            # Generate annotations
            confidence = round(random.uniform(0.5, 1.0), 2)
            source = random.choice(sources)
            timestamp = f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00Z"
            
            # Write RDF-Star triple with annotations
            f.write(f"<<{subject} {predicate} {obj}>>\n")
            f.write(f"    ex:confidence {confidence} ;\n")
            f.write(f"    prov:wasAttributedTo \"{source}\" ;\n")
            f.write(f"    prov:generatedAtTime \"{timestamp}\"^^xsd:dateTime .\n\n")
            
            # Also write the base fact (asserted)
            f.write(f"{subject} {predicate} {obj} .\n")
            f.write(f"{subject} rdf:type ex:{subj_type} .\n\n")
            
            if (i + 1) % 50000 == 0:
                print(f"  Generated {i + 1:,} facts...")
    
    print(f"  Done: {output_file}")


def run_rdfstar_benchmark(data_file: str, iterations: int = 10):
    """Run RDF-Star specific benchmarks."""
    from rdf_starbase import TripleStore, execute_sparql
    from bulk_loader import bulk_load_turtle_oneshot
    
    print("=" * 70)
    print("RDF-Star Workload Benchmark")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    store = TripleStore()
    t0 = time.time()
    count = bulk_load_turtle_oneshot(store, data_file)
    load_time = time.time() - t0
    print(f"Loaded {count:,} triples in {load_time:.2f}s ({count/load_time:,.0f} t/s)")
    print()
    
    results = {}
    
    # Warmup
    execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    
    # Q1: Total count
    print("Q1: Total triple count...")
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
        times.append(time.time() - t0)
    results["total_count"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000}
    print(f"    {results['total_count']['min']:.2f}ms min, {results['total_count']['avg']:.2f}ms avg")
    
    # Q2: Confidence filter (using blank node pattern since Oxigraph stores quoted triples as BNodes)
    print("Q2: Confidence filter (annotation query)...")
    q2 = """
    SELECT ?qt ?conf WHERE {
        ?qt <http://example.org/confidence> ?conf .
    } LIMIT 10000
    """
    # Warmup
    execute_sparql(store, q2)
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, q2)
        times.append(time.time() - t0)
    results["confidence_filter"] = {
        "min": min(times)*1000, 
        "avg": sum(times)/len(times)*1000,
        "rows": len(result)
    }
    print(f"    {results['confidence_filter']['min']:.2f}ms min ({results['confidence_filter']['rows']:,} rows)")
    
    # Q3: Source aggregation
    print("Q3: Source aggregation (GROUP BY on annotations)...")
    q3 = """
    SELECT ?source (COUNT(*) as ?count) WHERE {
        ?qt <http://www.w3.org/ns/prov#wasAttributedTo> ?source .
    } GROUP BY ?source
    """
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, q3)
        times.append(time.time() - t0)
    results["source_aggregation"] = {
        "min": min(times)*1000, 
        "avg": sum(times)/len(times)*1000,
        "rows": len(result)
    }
    print(f"    {results['source_aggregation']['min']:.2f}ms min ({results['source_aggregation']['rows']} sources)")
    
    # Q4: Annotation join (finding quoted triples with both confidence and source)
    print("Q4: Multi-annotation join...")
    q4 = """
    SELECT ?qt ?conf ?source WHERE {
        ?qt <http://example.org/confidence> ?conf .
        ?qt <http://www.w3.org/ns/prov#wasAttributedTo> ?source .
    } LIMIT 5000
    """
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, q4)
        times.append(time.time() - t0)
    results["annotation_join"] = {
        "min": min(times)*1000, 
        "avg": sum(times)/len(times)*1000,
        "rows": len(result)
    }
    print(f"    {results['annotation_join']['min']:.2f}ms min ({results['annotation_join']['rows']:,} rows)")
    
    # Q5: Type pattern (regular pattern)
    print("Q5: Type pattern (find all Person entities)...")
    q5 = """
    SELECT ?person WHERE {
        ?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
    }
    """
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, q5)
        times.append(time.time() - t0)
    results["type_pattern"] = {
        "min": min(times)*1000, 
        "avg": sum(times)/len(times)*1000,
        "rows": len(result)
    }
    print(f"    {results['type_pattern']['min']:.2f}ms min ({results['type_pattern']['rows']:,} rows)")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: RDF-Star Workload Performance")
    print("=" * 70)
    print()
    print(f"{'Query':<35} {'Min (ms)':>12} {'Avg (ms)':>12}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['min']:>12.2f} {r['avg']:>12.2f}")
    
    print()
    print("Key Observations:")
    print(f"  - Dataset: {count:,} triples")
    print(f"  - Load rate: {count/load_time:,.0f} triples/sec")
    print(f"  - RDF-Star data stored via Oxigraph parser (quoted triples as BNodes)")
    print(f"  - Annotation queries work via pattern matching on BNode subjects")
    print(f"  - Aggregation on annotations supported via GROUP BY")
    print()
    print("Limitation: SPARQL-Star <<s p o>> syntax not yet supported in executor.")
    print("            Annotations queryable via their blank node references.")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RDF-Star Benchmark")
    parser.add_argument("--facts", type=int, default=50000, help="Number of annotated facts to generate")
    parser.add_argument("--data-file", type=str, default=None, help="Use existing data file instead of generating")
    args = parser.parse_args()
    
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = f"data/sample/rdfstar_benchmark_{args.facts // 1000}k.ttl"
        if not os.path.exists(data_file):
            generate_rdfstar_data(args.facts, data_file)
    
    run_rdfstar_benchmark(data_file)
