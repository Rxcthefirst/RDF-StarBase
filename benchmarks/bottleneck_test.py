"""
Identify the bottleneck in term interning.
"""

import time
import polars as pl
from datetime import datetime, timezone

import sys
sys.path.insert(0, "src")

from rdf_starbase.storage.terms import TermDict


def test_intern_bottleneck(n: int):
    """Test where the time goes in term interning."""
    print(f"\n{'='*60}")
    print(f"Term Interning Analysis: {n:,} terms")
    print('='*60)
    
    term_dict = TermDict()
    
    # Test 1: Pure string generation
    start = time.perf_counter()
    strings = [f"value_{i}" for i in range(n)]
    t1 = time.perf_counter() - start
    print(f"  String generation: {t1:.2f}s ({n/t1:,.0f}/sec)")
    
    # Test 2: Dict lookup (simulating interning)
    start = time.perf_counter()
    cache = {}
    ids = []
    for s in strings:
        if s not in cache:
            cache[s] = len(cache)
        ids.append(cache[s])
    t2 = time.perf_counter() - start
    print(f"  Dict lookup (Python): {t2:.2f}s ({n/t2:,.0f}/sec)")
    
    # Test 3: Actual intern_literal
    start = time.perf_counter()
    term_ids = [term_dict.intern_literal(s) for s in strings]
    t3 = time.perf_counter() - start
    print(f"  term_dict.intern_literal: {t3:.2f}s ({n/t3:,.0f}/sec)")
    
    # Test 4: Polars string column (for reference)
    start = time.perf_counter()
    df = pl.DataFrame({"s": strings})
    t4 = time.perf_counter() - start
    print(f"  Polars DataFrame from list: {t4:.2f}s ({n/t4:,.0f}/sec)")
    
    # Test 5: Polars with categorical (dictionary encoding)
    start = time.perf_counter()
    df2 = pl.DataFrame({"s": strings}).with_columns(pl.col("s").cast(pl.Categorical))
    t5 = time.perf_counter() - start
    print(f"  Polars Categorical encoding: {t5:.2f}s ({n/t5:,.0f}/sec)")


def test_polars_native_path(n: int):
    """What if we used Polars for dictionary encoding instead of Python dicts?"""
    print(f"\n{'='*60}")
    print(f"Polars-Native Dictionary Encoding: {n:,} triples")
    print('='*60)
    
    # Generate raw string data
    start = time.perf_counter()
    subjects = [f"http://example.org/entity/{i % 100000}" for i in range(n)]
    predicates = [f"http://example.org/pred/{i % 50}" for i in range(n)]
    objects = [f"value_{i}" for i in range(n)]
    gen_time = time.perf_counter() - start
    print(f"  String generation: {gen_time:.2f}s")
    
    # Create DataFrame with strings
    start = time.perf_counter()
    df = pl.DataFrame({
        "s": subjects,
        "p": predicates,
        "o": objects,
    })
    df_time = time.perf_counter() - start
    print(f"  DataFrame creation: {df_time:.2f}s")
    
    # Let Polars do categorical encoding
    start = time.perf_counter()
    df_cat = df.with_columns([
        pl.col("s").cast(pl.Categorical).alias("s_cat"),
        pl.col("p").cast(pl.Categorical).alias("p_cat"),
        pl.col("o").cast(pl.Categorical).alias("o_cat"),
    ])
    cat_time = time.perf_counter() - start
    print(f"  Categorical encoding: {cat_time:.2f}s")
    
    total = gen_time + df_time + cat_time
    print(f"  TOTAL: {total:.2f}s ({n/total:,.0f} triples/sec)")
    print(f"  Memory (strings): {df.estimated_size('mb'):.1f} MB")
    print(f"  Memory (categorical): {df_cat.estimated_size('mb'):.1f} MB")
    
    # Query performance
    start = time.perf_counter()
    result = df_cat.filter(pl.col("s") == "http://example.org/entity/42")
    q_time = time.perf_counter() - start
    print(f"  Filter query: {q_time*1000:.2f}ms ({len(result)} rows)")


if __name__ == "__main__":
    test_intern_bottleneck(1_000_000)
    test_polars_native_path(10_000_000)
