# World-Class Inference Layer Architecture

**Status:** Design Document  
**Version:** 2.0  
**Date:** February 2026

---

## 1. Goals

Build a world-class inference layer that:

1. **Complete OWL 2 RL coverage** - All OWL 2 RL entailment rules
2. **RDF-Star aware** - Confidence propagation, provenance tracking for inferences
3. **High performance** - Columnar/vectorized rule application
4. **Incremental** - Efficient updates without full re-materialization
5. **Explainable** - Justification/provenance for each inferred fact
6. **Extensible** - Custom rules via SPARQL or Datalog-style syntax

---

## 2. Current State Assessment

### 2.1 What's Implemented

| Category | Rule | Description | Status |
|----------|------|-------------|--------|
| **RDFS** | rdfs2 | Domain inference | ✅ |
| | rdfs3 | Range inference | ✅ |
| | rdfs5 | subPropertyOf transitivity | ✅ |
| | rdfs7 | Property inheritance | ✅ |
| | rdfs9 | Type inheritance | ✅ |
| | rdfs11 | subClassOf transitivity | ✅ |
| **OWL** | owl:sameAs | Symmetry + transitivity | ✅ |
| | owl:equivalentClass | Mutual subClassOf | ✅ |
| | owl:equivalentProperty | Mutual subPropertyOf | ✅ |
| | owl:inverseOf | Property inversion | ✅ |
| | owl:TransitiveProperty | Transitive closure | ✅ |
| | owl:SymmetricProperty | Symmetry | ✅ |
| | owl:FunctionalProperty | Unique value → sameAs | ✅ |
| | owl:InverseFunctionalProperty | Unique subject → sameAs | ✅ |
| | owl:hasValue | Value restrictions | ✅ |

### 2.2 What's Missing

| Category | Rule | Description | Priority |
|----------|------|-------------|----------|
| **OWL 2 RL** | owl:allValuesFrom | Universal restrictions | HIGH |
| | owl:someValuesFrom | Existential (complete) | HIGH |
| | owl:unionOf | Class union | MEDIUM |
| | owl:intersectionOf | Class intersection | MEDIUM |
| | owl:complementOf | Class complement | MEDIUM |
| | owl:disjointWith | Disjoint classes | MEDIUM |
| | owl:propertyDisjointWith | Disjoint properties | LOW |
| | owl:differentFrom | Different individuals | MEDIUM |
| | owl:AllDifferent | All different | LOW |
| | owl:propertyChainAxiom | Property chains | HIGH |
| | owl:ReflexiveProperty | Reflexivity | LOW |
| | owl:IrreflexiveProperty | Irreflexivity | LOW |
| | owl:AsymmetricProperty | Asymmetry | LOW |
| **RDF-Star** | Confidence propagation | Combined confidence | HIGH |
| | Provenance tracking | Source for inferences | HIGH |
| | Annotation inheritance | Propagate metadata | MEDIUM |
| **Advanced** | Incremental reasoning | Delta updates | HIGH |
| | Explanation/justification | Why was this inferred? | HIGH |
| | Custom rules | User-defined rules | MEDIUM |
| | Parallel execution | Multi-threaded rules | MEDIUM |

---

## 3. Architecture Design

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Inference API                            │
│  (TripleStore.reason(), /ai/materialize endpoint)           │
├─────────────────────────────────────────────────────────────┤
│                  Reasoning Controller                        │
│  - Profile detection (RDFS, OWL-RL, custom)                 │
│  - Incremental vs full materialization                      │
│  - Stats and progress tracking                              │
├─────────────────────────────────────────────────────────────┤
│                    Rule Engine                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ RDFS     │  │ OWL-RL   │  │ RDF-Star │  │ Custom   │    │
│  │ Rules    │  │ Rules    │  │ Rules    │  │ Rules    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│                Explanation Tracker                           │
│  - Justification graph                                       │
│  - Proof trees                                              │
├─────────────────────────────────────────────────────────────┤
│              Integer-Based Fact Store                        │
│  - FactFlags.INFERRED for materialized facts                │
│  - Provenance columns (source, process)                     │
│  - RDF-Star confidence tracking                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Rule Representation

Each rule should be represented as a structured object:

```python
@dataclass
class InferenceRule:
    """Represents a single inference rule."""
    id: str                           # e.g., "rdfs9", "owl-hasValue"
    name: str                         # Human-readable name
    profile: str                      # "rdfs", "owl-rl", "custom"
    priority: int                     # Execution order (lower = earlier)
    
    # Polars-based pattern matching
    antecedent_patterns: List[TriplePattern]  # IF these patterns match
    consequent_template: TripleTemplate       # THEN generate this
    
    # For explanation
    description: str
    spec_reference: str               # W3C spec URL
```

### 3.3 Incremental Reasoning

Key insight: Track which facts are "new" since last reasoning, only apply rules to new facts.

```python
class IncrementalReasoner:
    def __init__(self, fact_store: FactStore):
        self._fact_store = fact_store
        self._last_txn_id: int = 0  # Track last processed transaction
    
    def reason_incremental(self) -> ReasoningStats:
        """Only process facts added since last reasoning."""
        # Get new facts since _last_txn_id
        new_facts = self._fact_store.scan_facts().filter(
            pl.col("txn") > self._last_txn_id
        )
        
        # Apply rules only to new facts
        # But check against all existing facts for joins
        ...
        
        # Update watermark
        self._last_txn_id = self._fact_store.current_txn_id
```

### 3.4 RDF-Star Inference Extensions

#### 3.4.1 Confidence Propagation

When inferring a new fact from multiple source facts, combine confidences:

```python
def combine_confidence(
    confidences: List[float],
    method: str = "min"  # "min", "product", "average"
) -> float:
    """Combine confidences for inferred facts."""
    if method == "min":
        return min(confidences)
    elif method == "product":
        return math.prod(confidences)
    elif method == "average":
        return sum(confidences) / len(confidences)
```

For example:
- `<<:Alice :type :Person>> :confidence 0.9`
- `<<:Person :subClassOf :Agent>> :confidence 0.95`
- Inferred: `<<:Alice :type :Agent>> :confidence min(0.9, 0.95) = 0.9`

#### 3.4.2 Inference Provenance

Each inferred fact should track its justification:

```python
@dataclass
class InferenceJustification:
    """Tracks why a fact was inferred."""
    rule_id: str                      # Which rule produced this
    antecedent_facts: List[FactId]   # Which facts triggered the rule
    timestamp: datetime
    confidence: float
```

Store as RDF-Star annotation:
```turtle
<<:Alice :type :Agent>> 
    :inferredBy "rdfs9" ;
    :basedOn <<:Alice :type :Person>>, <<:Person :subClassOf :Agent>> ;
    :confidence 0.9 ;
    :inferredAt "2026-02-07T10:30:00Z" .
```

### 3.5 Explanation API

```python
class ExplanationService:
    """Explain why a fact exists (asserted or inferred)."""
    
    def explain(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> Explanation:
        """
        Returns an explanation for a fact.
        
        Returns:
            Explanation with:
            - is_asserted: bool
            - is_inferred: bool
            - rules_applied: List[str]
            - supporting_facts: List[Triple]
            - proof_tree: ProofNode (recursive)
        """
```

---

## 4. Implementation Plan

### Phase 1: Complete OWL 2 RL (Week 1)

1. **Property chains** (owl:propertyChainAxiom)
   - Support chains like: `owl:topObjectProperty owl:propertyChainAxiom (foaf:knows foaf:knows)`
   
2. **allValuesFrom / someValuesFrom** (complete)
   - `(x type C) + (C allValuesFrom(p, D)) => forall y: (x p y) implies (y type D)`
   
3. **Union/Intersection** 
   - `(C unionOf (C1 C2)) + (x type C1) => (x type C)`
   - `(C intersectionOf (C1 C2)) + (x type C1) + (x type C2) => (x type C)`

4. **Disjointness**
   - `(C1 disjointWith C2) + (x type C1) + (x type C2) => inconsistency`

### Phase 2: RDF-Star Extensions (Week 2)

1. **Confidence propagation** - Auto-compute confidence for inferred facts
2. **Inference provenance** - Track rule and source facts
3. **Explanation API** - Query why a fact was inferred

### Phase 3: Advanced Features (Week 3)

1. **Incremental reasoning** - Only process deltas
2. **Custom rules** - SPARQL-based rule definitions
3. **Parallel execution** - Multi-threaded rule application

### Phase 4: Performance & Polish (Week 4)

1. **Benchmarks** - Compare with Jena, GraphDB, Stardog
2. **Profile auto-detection** - Determine minimal rule set needed
3. **Memory optimization** - Streaming for large datasets

---

## 5. API Design

### 5.1 TripleStore Integration

```python
class TripleStore:
    def reason(
        self,
        profile: str = "owl-rl",  # "rdfs", "owl-rl", "custom"
        incremental: bool = True,
        confidence_method: str = "min",
        track_provenance: bool = True,
        max_iterations: int = 100,
    ) -> ReasoningStats:
        """
        Materialize inferences.
        
        Args:
            profile: Which rule set to apply
            incremental: Only process new facts since last reasoning
            confidence_method: How to combine confidences ("min", "product", "average")
            track_provenance: Store inference justifications as RDF-Star
            max_iterations: Max fixed-point iterations
        """
```

### 5.2 Explanation API

```python
class TripleStore:
    def explain(
        self,
        subject: str,
        predicate: str,
        object_: str,
        max_depth: int = 10,
    ) -> Explanation:
        """
        Explain why a triple exists.
        
        Returns Explanation with proof tree.
        """
    
    def get_inferences_for(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
    ) -> List[InferredFact]:
        """
        Get all inferred facts matching pattern.
        
        Each InferredFact includes justification.
        """
```

### 5.3 Custom Rules API

```python
class TripleStore:
    def add_rule(
        self,
        rule_id: str,
        sparql_construct: str,  # SPARQL CONSTRUCT query defining the rule
        priority: int = 50,
    ) -> None:
        """
        Add a custom inference rule.
        
        Example:
            store.add_rule(
                "custom-uncle",
                '''
                CONSTRUCT { ?person :hasUncle ?uncle }
                WHERE {
                    ?person :hasParent ?parent .
                    ?parent :hasBrother ?uncle .
                }
                ''',
                priority=100
            )
        """
```

---

## 6. Metrics for "World Class"

| Metric | Target | Measurement |
|--------|--------|-------------|
| OWL 2 RL coverage | 100% | All OWL 2 RL rules implemented |
| Inference throughput | >100K facts/sec | Benchmark on synthetic data |
| Incremental update | <10ms for 100 new facts | Measure delta processing |
| Explanation latency | <50ms | Time to explain any fact |
| Memory efficiency | <2x base store | Overhead for inference metadata |
| Test coverage | >95% | All rules have unit tests |

---

## 7. Competitive Analysis

| Feature | RDF-StarBase (Target) | Jena | GraphDB | Stardog |
|---------|----------------------|------|---------|---------|
| RDFS | ✅ | ✅ | ✅ | ✅ |
| OWL 2 RL | ✅ (full) | ✅ | ✅ | ✅ |
| RDF-Star confidence | ✅ | ❌ | ✅ | ✅ |
| Incremental | ✅ | ⚠️ | ✅ | ✅ |
| Explanation | ✅ | ⚠️ | ✅ | ✅ |
| Custom rules | ✅ (SPARQL) | ✅ (Jena rules) | ✅ (SPIN) | ✅ |
| Embedded Python | ✅ | ❌ | ❌ | ❌ |
| Columnar perf | ✅ | ❌ | ❌ | ❌ |

---

## 8. File Structure

```
src/rdf_starbase/
├── storage/
│   ├── reasoner.py          # Core RDFS/OWL reasoner (EXISTS)
│   ├── inference/           # NEW: Modular inference engine
│   │   ├── __init__.py
│   │   ├── rules.py         # Rule definitions
│   │   ├── rdfs_rules.py    # RDFS rule implementations
│   │   ├── owl_rules.py     # OWL 2 RL rule implementations
│   │   ├── rdfstar_rules.py # RDF-Star confidence/provenance
│   │   ├── custom_rules.py  # SPARQL-based custom rules
│   │   ├── incremental.py   # Incremental reasoning
│   │   └── explanation.py   # Explanation/justification service
│   └── ...
└── ...
```

---

**Next Steps:** Begin Phase 1 - Implement missing OWL 2 RL rules.
