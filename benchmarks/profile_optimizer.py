"""
Performance profiling for RDF-StarBase.

Identifies bottlenecks in:
1. Pattern join execution 
2. Bulk loading
3. Index usage
"""
import time
import sys
import cProfile
import pstats
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rdf_starbase import TripleStore, execute_sparql
from bulk_loader import bulk_load_turtle_oneshot


def profile_query(store, query, name):
    """Profile a query and return timing breakdown."""
    # Warmup
    execute_sparql(store, query)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(3):
        result = execute_sparql(store, query)
    
    profiler.disable()
    
    # Get stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print(f"\n=== {name} ===")
    print(stream.getvalue())
    
    return result


def analyze_bottlenecks():
    """Profile pattern matching to identify bottlenecks."""
    print("Loading benchmark data...")
    store = TripleStore()
    count = bulk_load_turtle_oneshot(store, "data/sample/benchmark_10M.ttl")
    print(f"Loaded {count:,} triples\n")
    
    print("=" * 80)
    print("PROFILING QUERIES")
    print("=" * 80)
    
    # Test queries
    queries = {
        "Q1_count": "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }",
        
        "Q2_type_scan": """
            SELECT (COUNT(*) as ?c) WHERE { 
                ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 
                   <http://benchmark.example.org/GraduateStudent> 
            }
        """,
        
        "Q3_2hop_join": """
            SELECT (COUNT(*) as ?c) WHERE {
                ?student <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 
                         <http://benchmark.example.org/GraduateStudent> .
                ?student <http://benchmark.example.org/advisor> ?advisor .
            }
        """,
        
        "Q4_3hop_join": """
            SELECT (COUNT(*) as ?c) WHERE {
                ?student <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 
                         <http://benchmark.example.org/GraduateStudent> .
                ?student <http://benchmark.example.org/advisor> ?advisor .
                ?advisor <http://benchmark.example.org/worksFor> ?dept .
            }
        """,
    }
    
    for name, query in queries.items():
        profile_query(store, query, name)


def analyze_join_strategy():
    """Analyze join execution in detail."""
    print("\n" + "=" * 80)
    print("JOIN STRATEGY ANALYSIS")
    print("=" * 80)
    
    store = TripleStore()
    count = bulk_load_turtle_oneshot(store, "data/sample/benchmark_10M.ttl")
    
    # Get raw DataFrames for manual analysis
    df = store._df
    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {df.columns}")
    print(f"Schema: {df.schema}")
    
    # Cardinality analysis
    print("\n### Cardinality Analysis ###")
    for col in ["subject", "predicate", "object"]:
        unique = df[col].n_unique()
        print(f"  {col}: {unique:,} unique values ({100*unique/len(df):.1f}% selectivity)")
    
    # Predicate distribution
    print("\n### Top Predicates ###")
    pred_counts = df.group_by("predicate").len().sort("len", descending=True).head(10)
    print(pred_counts)
    
    # Test different join strategies
    print("\n### Join Strategy Comparison ###")
    import polars as pl
    
    # Pattern 1: Type lookup
    type_pred = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    grad_student = "http://benchmark.example.org/GraduateStudent"
    
    t0 = time.time()
    pattern1_df = df.lazy().filter(
        (pl.col("predicate") == type_pred) & 
        (pl.col("object") == grad_student)
    ).collect()
    t1 = time.time()
    print(f"\n  Pattern 1 (type filter): {len(pattern1_df):,} rows in {(t1-t0)*1000:.2f}ms")
    
    # Pattern 2: Advisor relationship
    advisor_pred = "http://benchmark.example.org/advisor"
    
    t0 = time.time()
    pattern2_df = df.lazy().filter(
        pl.col("predicate") == advisor_pred
    ).collect()
    t1 = time.time()
    print(f"  Pattern 2 (advisor filter): {len(pattern2_df):,} rows in {(t1-t0)*1000:.2f}ms")
    
    # Join: Sequential Polars
    t0 = time.time()
    joined = pattern1_df.rename({"subject": "student"}).select(["student"]).join(
        pattern2_df.rename({"subject": "student", "object": "advisor"}).select(["student", "advisor"]),
        on="student",
        how="inner"
    )
    t1 = time.time()
    print(f"  Polars join: {len(joined):,} rows in {(t1-t0)*1000:.2f}ms")
    
    # Join: DuckDB SQL
    try:
        import duckdb
        conn = duckdb.connect(":memory:")
        conn.register("df", df.to_arrow())
        
        sql = f"""
        SELECT COUNT(*) as c FROM (
            SELECT t1.subject as student, t2.object as advisor
            FROM df t1
            JOIN df t2 ON t1.subject = t2.subject
            WHERE t1.predicate = '{type_pred}'
              AND t1.object = '{grad_student}'
              AND t2.predicate = '{advisor_pred}'
        )
        """
        
        t0 = time.time()
        result = conn.execute(sql).fetchone()
        t1 = time.time()
        print(f"  DuckDB join: {result[0]:,} rows in {(t1-t0)*1000:.2f}ms")
        
    except Exception as e:
        print(f"  DuckDB error: {e}")


def analyze_loading():
    """Analyze bulk loading performance."""
    print("\n" + "=" * 80)
    print("LOADING ANALYSIS")
    print("=" * 80)
    
    file_path = "data/sample/benchmark_10M.ttl"
    
    # Profile loading
    profiler = cProfile.Profile()
    profiler.enable()
    
    store = TripleStore()
    count = bulk_load_turtle_oneshot(store, file_path)
    
    profiler.disable()
    
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print(f"\nLoaded {count:,} triples")
    print("\n### Loading Profile ###")
    print(stream.getvalue())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--bottlenecks", action="store_true", help="Profile query bottlenecks")
    parser.add_argument("--joins", action="store_true", help="Analyze join strategies")
    parser.add_argument("--loading", action="store_true", help="Analyze loading")
    args = parser.parse_args()
    
    if args.all or (not args.bottlenecks and not args.joins and not args.loading):
        analyze_join_strategy()
        # analyze_bottlenecks()  # Very verbose
        # analyze_loading()
    elif args.bottlenecks:
        analyze_bottlenecks()
    elif args.joins:
        analyze_join_strategy()
    elif args.loading:
        analyze_loading()
