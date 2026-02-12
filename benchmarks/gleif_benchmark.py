"""
GLEIF-Calibrated Benchmark Suite for RDF-StarBase
==================================================

7 query tiers modeled after finance-industry latency expectations,
run against the GLEIF L2 dataset (26.8M triples, ~464K relationships).

Each query is executed N times with warm-up, then p50/p95/p99 latencies
are computed and scored against Gold / Silver / Bronze thresholds.

Tiers
-----
1. Entity Profile   — 1-hop star lookup on a single entity
2. Name Search      — literal scan with FILTER/CONTAINS
3. Direct Parent    — single join (child → parent)
4. Parent Chain     — bounded multi-hop traversal (child → direct → ultimate)
5. Portfolio View   — VALUES batch (25–100 entities)
6. Risk Aggregation — GROUP BY on filtered subsets
7. Provenance       — metadata / registration details
"""

from __future__ import annotations

import sys, time, statistics, json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdf_starbase.store import TripleStore
from rdf_starbase.sparql.executor import execute_sparql

# ============================================================================
# Configuration
# ============================================================================
STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "repositories" / "test2" / "store.parquet"
WARMUP_RUNS = 3
BENCH_RUNS  = 20

# Latency thresholds (milliseconds)
THRESHOLDS = {
    "T1: Entity Profile":    {"gold": 40,   "silver": 120,  "bronze": 300},
    "T2: Name Search":       {"gold": 100,  "silver": 300,  "bronze": 1500},
    "T3: Direct Parent":     {"gold": 60,   "silver": 180,  "bronze": 500},
    "T4: Parent Chain":      {"gold": 150,  "silver": 600,  "bronze": 2500},
    "T5: Portfolio View":    {"gold": 250,  "silver": 900,  "bronze": 3000},
    "T6: Risk Aggregation":  {"gold": 800,  "silver": 3000, "bronze": 12000},
    "T7: Provenance":        {"gold": 120,  "silver": 400,  "bronze": 1500},
}


# ============================================================================
# Discover sample entities from the loaded store
# ============================================================================
def discover_entities(store: TripleStore) -> dict:
    """Run discovery queries to find real entity URIs for benchmarks."""
    entities = {}

    # A relationship record with child + parent
    df = execute_sparql(store, """
        SELECT ?rec ?child ?parent WHERE {
            ?rec <https://www.gleif.org/ontology/L2/hasChild> ?child .
            ?rec <https://www.gleif.org/ontology/L2/hasParent> ?parent .
        } LIMIT 5
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    if rows:
        entities['rel_records'] = rows
        entities['child_lei'] = _strip(rows[0].get('child', ''))
        entities['parent_lei'] = _strip(rows[0].get('parent', ''))
        entities['rel_rec'] = _strip(rows[0].get('rec', ''))

    # DirectConsolidation — find a child with BOTH direct and ultimate parent
    df = execute_sparql(store, """
        SELECT ?dc ?child ?parent WHERE {
            ?dc a <https://www.gleif.org/ontology/L2/DirectConsolidation> .
            ?dc <https://www.gleif.org/ontology/L2/hasChild> ?child .
            ?dc <https://www.gleif.org/ontology/L2/hasParent> ?parent .
        } LIMIT 5
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    if rows:
        entities['dc_child'] = _strip(rows[0].get('child', ''))
        entities['dc_parent'] = _strip(rows[0].get('parent', ''))
        entities['dc_rec'] = _strip(rows[0].get('dc', ''))

    # DirectConsolidation subjects (for entity profile)
    df = execute_sparql(store, """
        SELECT ?s WHERE {
            ?s a <https://www.gleif.org/ontology/L2/DirectConsolidation> .
        } LIMIT 30
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    entities['dc_subjects'] = [_strip(r['s']) for r in rows]

    # BIC tags for name search
    df = execute_sparql(store, """
        SELECT ?s ?tag WHERE {
            ?s <https://www.gleif.org/ontology/Base/tag> ?tag .
        } LIMIT 10
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    if rows:
        entities['tags'] = [_strip(r['tag']) for r in rows]
        entities['tag_subjects'] = [_strip(r['s']) for r in rows]

    # Managing LOU
    df = execute_sparql(store, """
        SELECT ?s ?lou WHERE {
            ?s <https://www.gleif.org/ontology/L1/hasManagingLOU> ?lou .
        } LIMIT 5
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    if rows:
        entities['lou'] = _strip(rows[0].get('lou', ''))

    # Portfolio batch: grab 25 distinct child LEIs
    df = execute_sparql(store, """
        SELECT DISTINCT ?child WHERE {
            ?rec <https://www.gleif.org/ontology/L2/hasChild> ?child .
        } LIMIT 25
    """)
    rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
    entities['portfolio_leis'] = [_strip(r['child']) for r in rows]

    return entities


def _strip(val: str) -> str:
    """Strip angle brackets from N-Triples IRI notation."""
    if val.startswith('<') and val.endswith('>'):
        return val[1:-1]
    # Strip surrounding quotes from literals
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    # Handle typed literals like "value"^^<type>
    if val.startswith('"'):
        idx = val.find('"', 1)
        if idx > 0:
            return val[1:idx]
    return val


# ============================================================================
# Query Templates
# ============================================================================
def build_queries(ent: dict) -> list[dict]:
    """Build the 7-tier query set from discovered entities."""
    queries = []

    # ---- T1: Entity Profile (star pattern on a single entity) ----
    entity_uri = ent.get('rel_rec', '')
    if entity_uri:
        queries.append({
            "tier": "T1: Entity Profile",
            "name": "Relationship record profile",
            "query": f"SELECT ?p ?o WHERE {{ <{entity_uri}> ?p ?o }}",
        })

    if ent.get('dc_subjects'):
        dc = ent['dc_subjects'][0]
        queries.append({
            "tier": "T1: Entity Profile",
            "name": "DirectConsolidation profile",
            "query": f"SELECT ?p ?o WHERE {{ <{dc}> ?p ?o }}",
        })

    # ---- T2: Name Search (literal scan with FILTER) ----
    if ent.get('tags'):
        tag_prefix = ent['tags'][0][:4]  # e.g. "AAAA"
        queries.append({
            "tier": "T2: Name Search",
            "name": f"BIC tag prefix search '{tag_prefix}'",
            "query": f"""
                SELECT ?s ?tag WHERE {{
                    ?s <https://www.gleif.org/ontology/Base/tag> ?tag .
                    FILTER(STRSTARTS(?tag, "{tag_prefix}"))
                }} LIMIT 50
            """,
        })

    # Broader text search
    queries.append({
        "tier": "T2: Name Search",
        "name": "Relationship status search",
        "query": """
            SELECT ?s ?status WHERE {
                ?s <https://www.gleif.org/ontology/L2/hasRelationshipStatus> ?status .
            } LIMIT 100
        """,
    })

    # ---- T3: Direct Parent (single join) ----
    queries.append({
        "tier": "T3: Direct Parent",
        "name": "Child → parent lookup",
        "query": """
            SELECT ?child ?parent WHERE {
                ?rec <https://www.gleif.org/ontology/L2/hasChild> ?child .
                ?rec <https://www.gleif.org/ontology/L2/hasParent> ?parent .
            } LIMIT 100
        """,
    })

    if ent.get('child_lei'):
        child = ent['child_lei']
        queries.append({
            "tier": "T3: Direct Parent",
            "name": "Specific child → parent",
            "query": f"""
                SELECT ?parent WHERE {{
                    ?rec <https://www.gleif.org/ontology/L2/hasChild> <{child}> .
                    ?rec <https://www.gleif.org/ontology/L2/hasParent> ?parent .
                }}
            """,
        })

    # ---- T4: Parent Chain (multi-hop: child → direct consolidation → ultimate consolidation) ----
    if ent.get('dc_child'):
        child = ent['dc_child']
        queries.append({
            "tier": "T4: Parent Chain",
            "name": "Child → direct + ultimate parent chain",
            "query": f"""
                SELECT ?directParent ?ultParent WHERE {{
                    ?r1 a <https://www.gleif.org/ontology/L2/DirectConsolidation> .
                    ?r1 <https://www.gleif.org/ontology/L2/hasChild> <{child}> .
                    ?r1 <https://www.gleif.org/ontology/L2/hasParent> ?directParent .
                    OPTIONAL {{
                        ?r2 a <https://www.gleif.org/ontology/L2/UltimateConsolidation> .
                        ?r2 <https://www.gleif.org/ontology/L2/hasChild> <{child}> .
                        ?r2 <https://www.gleif.org/ontology/L2/hasParent> ?ultParent .
                    }}
                }}
            """,
        })

    queries.append({
        "tier": "T4: Parent Chain",
        "name": "Direct + ultimate parent join (sample 50)",
        "query": """
            SELECT ?child ?directParent ?ultParent WHERE {
                ?r1 a <https://www.gleif.org/ontology/L2/DirectConsolidation> .
                ?r1 <https://www.gleif.org/ontology/L2/hasChild> ?child .
                ?r1 <https://www.gleif.org/ontology/L2/hasParent> ?directParent .
                OPTIONAL {
                    ?r2 a <https://www.gleif.org/ontology/L2/UltimateConsolidation> .
                    ?r2 <https://www.gleif.org/ontology/L2/hasChild> ?child .
                    ?r2 <https://www.gleif.org/ontology/L2/hasParent> ?ultParent .
                }
            } LIMIT 50
        """,
    })

    # ---- T5: Portfolio View (VALUES batch) ----
    leis = ent.get('portfolio_leis', [])[:25]
    if leis:
        values_block = " ".join(f"(<{lei}>)" for lei in leis)
        queries.append({
            "tier": "T5: Portfolio View",
            "name": f"Portfolio batch ({len(leis)} entities)",
            "query": f"""
                SELECT ?lei ?p ?o WHERE {{
                    VALUES (?lei) {{ {values_block} }}
                    ?rec <https://www.gleif.org/ontology/L2/hasChild> ?lei .
                    ?rec ?p ?o .
                }} LIMIT 500
            """,
        })

    # Larger portfolio
    leis50 = ent.get('portfolio_leis', [])
    if len(leis50) >= 10:
        values_block = " ".join(f"(<{lei}>)" for lei in leis50)
        queries.append({
            "tier": "T5: Portfolio View",
            "name": f"Portfolio with status ({len(leis50)} entities)",
            "query": f"""
                SELECT ?lei ?parent ?status WHERE {{
                    VALUES (?lei) {{ {values_block} }}
                    ?rec <https://www.gleif.org/ontology/L2/hasChild> ?lei .
                    ?rec <https://www.gleif.org/ontology/L2/hasParent> ?parent .
                    ?rec <https://www.gleif.org/ontology/L2/hasRelationshipStatus> ?status .
                }}
            """,
        })

    # ---- T6: Risk Aggregation (GROUP BY) ----
    queries.append({
        "tier": "T6: Risk Aggregation",
        "name": "Count relationships by type",
        "query": """
            SELECT ?type (COUNT(?s) AS ?count) WHERE {
                ?s a ?type .
            } GROUP BY ?type
        """,
    })

    queries.append({
        "tier": "T6: Risk Aggregation",
        "name": "Count by relationship status",
        "query": """
            SELECT ?status (COUNT(?rec) AS ?count) WHERE {
                ?rec <https://www.gleif.org/ontology/L2/hasRelationshipStatus> ?status .
            } GROUP BY ?status
        """,
    })

    queries.append({
        "tier": "T6: Risk Aggregation",
        "name": "Count by validation source",
        "query": """
            SELECT ?src (COUNT(?rec) AS ?count) WHERE {
                ?rec <https://www.gleif.org/ontology/L2/hasValidationSources> ?src .
            } GROUP BY ?src
        """,
    })

    queries.append({
        "tier": "T6: Risk Aggregation",
        "name": "Count by managing LOU",
        "query": """
            SELECT ?lou (COUNT(?rec) AS ?count) WHERE {
                ?rec <https://www.gleif.org/ontology/L1/hasManagingLOU> ?lou .
            } GROUP BY ?lou
        """,
    })

    # ---- T7: Provenance / as-of ----
    queries.append({
        "tier": "T7: Provenance",
        "name": "Registration dates (LIMIT 100)",
        "query": """
            SELECT ?rec ?regDate ?updateDate WHERE {
                ?rec <https://www.gleif.org/ontology/Base/hasInitialRegistrationDate> ?regDate .
                ?rec <https://www.gleif.org/ontology/Base/hasLastUpdateDate> ?updateDate .
            } LIMIT 100
        """,
    })

    queries.append({
        "tier": "T7: Provenance",
        "name": "Validation documents for records",
        "query": """
            SELECT ?rec ?doc WHERE {
                ?rec <https://www.gleif.org/ontology/Base/hasValidationDocuments> ?doc .
            } LIMIT 100
        """,
    })

    if ent.get('rel_rec'):
        rec = ent['rel_rec']
        queries.append({
            "tier": "T7: Provenance",
            "name": "Single record provenance",
            "query": f"""
                SELECT ?p ?o WHERE {{
                    <{rec}> ?p ?o .
                }}
            """,
        })

    return queries


# ============================================================================
# Benchmark Runner
# ============================================================================
def run_benchmark(store: TripleStore, queries: list[dict],
                  warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> list[dict]:
    """Execute each query with warmup + measured runs; return timing results."""
    results = []

    for qi, q in enumerate(queries, 1):
        tier = q["tier"]
        name = q["name"]
        query = q["query"]

        # Warm-up
        for _ in range(warmup):
            try:
                execute_sparql(store, query)
            except Exception:
                pass

        # Measured runs
        timings = []
        row_count = 0
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                df = execute_sparql(store, query)
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                timings.append(elapsed)
                if hasattr(df, 'height'):
                    row_count = df.height
                elif isinstance(df, dict) and 'results' in df:
                    row_count = len(df['results'])
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                timings.append(elapsed)
                if not row_count:
                    row_count = f"ERROR: {exc}"

        timings.sort()
        p50 = timings[len(timings) // 2]
        p95 = timings[int(len(timings) * 0.95)]
        p99 = timings[int(len(timings) * 0.99)]

        thresholds = THRESHOLDS.get(tier, {})
        medal = score_medal(p50, thresholds)

        results.append({
            "tier": tier,
            "name": name,
            "rows": row_count,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "medal": medal,
            "runs": runs,
        })

        icon = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉", "Miss": "❌"}.get(medal, "?")
        print(f"  [{qi:2d}/{len(queries)}] {icon} {tier} | {name}")
        print(f"         p50={p50:7.1f}ms  p95={p95:7.1f}ms  p99={p99:7.1f}ms  rows={row_count}")

    return results


def score_medal(p50_ms: float, thresholds: dict) -> str:
    if not thresholds:
        return "N/A"
    if p50_ms <= thresholds["gold"]:
        return "Gold"
    elif p50_ms <= thresholds["silver"]:
        return "Silver"
    elif p50_ms <= thresholds["bronze"]:
        return "Bronze"
    else:
        return "Miss"


# ============================================================================
# Report Generator
# ============================================================================
def generate_report(results: list[dict], store_info: dict) -> str:
    """Generate a Markdown scoreboard report."""
    lines = []
    lines.append("# GLEIF Benchmark Scoreboard")
    lines.append("")
    lines.append(f"**Dataset**: GLEIF L2 — {store_info.get('triples', '?'):,} triples, "
                 f"{store_info.get('subjects', '?'):,} subjects, "
                 f"{store_info.get('predicates', '?'):,} predicates")
    lines.append(f"**Runs**: {results[0]['runs']} measured iterations (+ {WARMUP_RUNS} warmup)")
    lines.append(f"**Engine**: RDF-StarBase native SPARQL executor (integer-first)")
    lines.append("")

    # Summary medals
    medals = {"Gold": 0, "Silver": 0, "Bronze": 0, "Miss": 0}
    for r in results:
        medals[r["medal"]] = medals.get(r["medal"], 0) + 1

    lines.append(f"## Summary: 🥇 {medals['Gold']} Gold | 🥈 {medals['Silver']} Silver | 🥉 {medals['Bronze']} Bronze | ❌ {medals['Miss']} Miss")
    lines.append("")

    # Table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| # | Medal | Tier | Query | Rows | p50 (ms) | p95 (ms) | p99 (ms) | Target |")
    lines.append("|---|-------|------|-------|------|----------|----------|----------|--------|")

    for i, r in enumerate(results, 1):
        icon = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉", "Miss": "❌"}.get(r["medal"], "?")
        thresh = THRESHOLDS.get(r["tier"], {})
        target_str = f"<{thresh.get('gold', '?')}ms" if thresh else "—"
        rows_str = str(r["rows"]) if isinstance(r["rows"], int) else str(r["rows"])
        lines.append(
            f"| {i} | {icon} | {r['tier']} | {r['name']} | "
            f"{rows_str} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['p99_ms']:.1f} | {target_str} |"
        )

    lines.append("")

    # Tier breakdown
    lines.append("## Tier Thresholds")
    lines.append("")
    lines.append("| Tier | Gold | Silver | Bronze |")
    lines.append("|------|------|--------|--------|")
    for tier, t in THRESHOLDS.items():
        lines.append(f"| {tier} | <{t['gold']}ms | {t['gold']}–{t['silver']}ms | {t['silver']}–{t['bronze']}ms |")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  GLEIF Benchmark Suite — RDF-StarBase")
    print("=" * 70)

    # Load store
    print(f"\nLoading store from {STORE_PATH} ...")
    t0 = time.perf_counter()
    store = TripleStore.load(STORE_PATH)
    load_time = time.perf_counter() - t0
    print(f"  Store loaded in {load_time:.1f}s")

    triple_count = len(store._fact_store)
    term_count = len(store._term_dict)
    print(f"  {triple_count:,} triples, {term_count:,} terms")

    # Discover entities
    print("\nDiscovering sample entities ...")
    t0 = time.perf_counter()
    entities = discover_entities(store)
    print(f"  Discovery done in {time.perf_counter() - t0:.2f}s")
    print(f"  Found: {len(entities.get('dc_subjects', []))} DC subjects, "
          f"{len(entities.get('portfolio_leis', []))} portfolio LEIs, "
          f"{len(entities.get('tags', []))} tags")

    # Build queries
    queries = build_queries(entities)
    print(f"\n{len(queries)} benchmark queries across 7 tiers")

    # Run benchmark
    print(f"\nRunning benchmarks ({BENCH_RUNS} runs each, {WARMUP_RUNS} warmup)...\n")
    results = run_benchmark(store, queries)

    # Store info
    store_info = {
        "triples": triple_count,
        "subjects": len(set()),  # We'll get this from stats
        "predicates": 47,
    }
    # Quick count for subjects
    try:
        df = execute_sparql(store, "SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE { ?s ?p ?o }")
        rows = df.to_dicts() if hasattr(df, 'to_dicts') else df.get('results', [])
        if rows:
            val = rows[0].get('c', '0')
            if isinstance(val, str):
                val = val.strip('"').split('"')[0]
            store_info["subjects"] = int(val) if str(val).isdigit() else 0
    except Exception:
        store_info["subjects"] = 0

    # Generate report
    report = generate_report(results, store_info)
    report_path = Path(__file__).parent / "GLEIF_BENCHMARK_RESULTS.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n{'=' * 70}")
    print(f"  Report written to {report_path}")
    print(f"{'=' * 70}\n")
    print(report)

    # Also save raw JSON
    json_path = Path(__file__).parent / "gleif_benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
