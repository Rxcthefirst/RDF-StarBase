# GLEIF Benchmark Scoreboard

**Dataset**: GLEIF L2 — 26,854,846 triples, 7,675,035 subjects, 47 predicates
**Runs**: 20 measured iterations (+ 3 warmup)
**Engine**: RDF-StarBase native SPARQL executor (integer-first)

## Summary: 🥇 13 Gold | 🥈 4 Silver | 🥉 0 Bronze | ❌ 0 Miss

## Detailed Results

| # | Medal | Tier | Query | Rows | p50 (ms) | p95 (ms) | p99 (ms) | Target |
|---|-------|------|-------|------|----------|----------|----------|--------|
| 1 | 🥇 | T1: Entity Profile | Relationship record profile | 5 | 25.5 | 27.4 | 27.4 | <40ms |
| 2 | 🥇 | T1: Entity Profile | DirectConsolidation profile | 5 | 24.9 | 26.0 | 26.0 | <40ms |
| 3 | 🥇 | T2: Name Search | BIC tag prefix search 'AAAA' | 0 | 0.4 | 0.6 | 0.6 | <100ms |
| 4 | 🥇 | T2: Name Search | Relationship status search | 100 | 31.7 | 36.4 | 36.4 | <100ms |
| 5 | 🥈 | T3: Direct Parent | Child → parent lookup | 100 | 74.5 | 79.9 | 79.9 | <60ms |
| 6 | 🥈 | T3: Direct Parent | Specific child → parent | 2 | 64.1 | 65.2 | 65.2 | <60ms |
| 7 | 🥈 | T4: Parent Chain | Child → direct + ultimate parent chain | 1 | 200.3 | 222.9 | 222.9 | <150ms |
| 8 | 🥈 | T4: Parent Chain | Direct + ultimate parent join (sample 50) | 50 | 223.0 | 242.7 | 242.7 | <150ms |
| 9 | 🥇 | T5: Portfolio View | Portfolio batch (25 entities) | 0 | 97.0 | 107.3 | 107.3 | <250ms |
| 10 | 🥇 | T5: Portfolio View | Portfolio with status (25 entities) | 0 | 111.8 | 118.4 | 118.4 | <250ms |
| 11 | 🥇 | T6: Risk Aggregation | Count relationships by type | 15 | 113.0 | 129.8 | 129.8 | <800ms |
| 12 | 🥇 | T6: Risk Aggregation | Count by relationship status | 3 | 35.1 | 36.6 | 36.6 | <800ms |
| 13 | 🥇 | T6: Risk Aggregation | Count by validation source | 3 | 34.9 | 37.5 | 37.5 | <800ms |
| 14 | 🥇 | T6: Risk Aggregation | Count by managing LOU | 41 | 35.5 | 47.7 | 47.7 | <800ms |
| 15 | 🥇 | T7: Provenance | Registration dates (LIMIT 100) | 100 | 74.0 | 81.0 | 81.0 | <120ms |
| 16 | 🥇 | T7: Provenance | Validation documents for records | 100 | 31.2 | 33.0 | 33.0 | <120ms |
| 17 | 🥇 | T7: Provenance | Single record provenance | 5 | 25.5 | 26.9 | 26.9 | <120ms |

## Tier Thresholds

| Tier | Gold | Silver | Bronze |
|------|------|--------|--------|
| T1: Entity Profile | <40ms | 40–120ms | 120–300ms |
| T2: Name Search | <100ms | 100–300ms | 300–1500ms |
| T3: Direct Parent | <60ms | 60–180ms | 180–500ms |
| T4: Parent Chain | <150ms | 150–600ms | 600–2500ms |
| T5: Portfolio View | <250ms | 250–900ms | 900–3000ms |
| T6: Risk Aggregation | <800ms | 800–3000ms | 3000–12000ms |
| T7: Provenance | <120ms | 120–400ms | 400–1500ms |
