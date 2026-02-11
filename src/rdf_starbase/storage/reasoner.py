"""
RDFS and OWL Reasoning Engine.

Implements forward-chaining RDFS and OWL 2 RL entailment for RDF-StarBase.
Materializes inferred triples into the FactStore with INFERRED flag.

Supported RDFS entailment rules:
- rdfs2: Domain inference (x p y) + (p rdfs:domain C) => (x rdf:type C)
- rdfs3: Range inference (x p y) + (p rdfs:range C) => (y rdf:type C)
- rdfs5: Transitive subPropertyOf (p1 subProp p2) + (p2 subProp p3) => (p1 subProp p3)
- rdfs7: Property inheritance (x p1 y) + (p1 subProp p2) => (x p2 y)
- rdfs9: Type inheritance (x type C1) + (C1 subClass C2) => (x type C2)
- rdfs11: Transitive subClassOf (C1 subClass C2) + (C2 subClass C3) => (C1 subClass C3)

Supported OWL 2 RL entailment rules:
- owl:sameAs symmetry and transitivity
- owl:equivalentClass => mutual rdfs:subClassOf
- owl:equivalentProperty => mutual rdfs:subPropertyOf
- owl:inverseOf (x p y) + (p inverseOf q) => (y q x)
- owl:TransitiveProperty (x p y) + (y p z) => (x p z)
- owl:SymmetricProperty (x p y) => (y p x)
- owl:FunctionalProperty (x p y1) + (x p y2) => (y1 owl:sameAs y2)
- owl:InverseFunctionalProperty (x1 p y) + (x2 p y) => (x1 owl:sameAs x2)
- owl:hasValue + owl:onProperty class membership inference
- owl:someValuesFrom existence inference
- owl:intersectionOf class membership

Implementation approach: Forward-chaining with fixed-point iteration.
"""

from typing import Set, Tuple, Optional, List
from dataclasses import dataclass, field

import polars as pl

from rdf_starbase.storage.terms import TermDict, TermId, Term, TermKind
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID


# RDFS vocabulary IRIs
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS = "http://www.w3.org/2002/07/owl#"

RDFS_SUBCLASS_OF = RDFS_NS + "subClassOf"
RDFS_SUBPROPERTY_OF = RDFS_NS + "subPropertyOf"
RDFS_DOMAIN = RDFS_NS + "domain"
RDFS_RANGE = RDFS_NS + "range"
RDF_TYPE = RDF_NS + "type"

# OWL vocabulary IRIs
OWL_SAME_AS = OWL_NS + "sameAs"
OWL_EQUIVALENT_CLASS = OWL_NS + "equivalentClass"
OWL_EQUIVALENT_PROPERTY = OWL_NS + "equivalentProperty"
OWL_INVERSE_OF = OWL_NS + "inverseOf"
OWL_TRANSITIVE_PROPERTY = OWL_NS + "TransitiveProperty"
OWL_SYMMETRIC_PROPERTY = OWL_NS + "SymmetricProperty"
OWL_FUNCTIONAL_PROPERTY = OWL_NS + "FunctionalProperty"
OWL_INVERSE_FUNCTIONAL_PROPERTY = OWL_NS + "InverseFunctionalProperty"
OWL_HAS_VALUE = OWL_NS + "hasValue"
OWL_ON_PROPERTY = OWL_NS + "onProperty"
OWL_SOME_VALUES_FROM = OWL_NS + "someValuesFrom"
OWL_ALL_VALUES_FROM = OWL_NS + "allValuesFrom"
OWL_INTERSECTION_OF = OWL_NS + "intersectionOf"
OWL_UNION_OF = OWL_NS + "unionOf"
OWL_PROPERTY_CHAIN_AXIOM = OWL_NS + "propertyChainAxiom"
OWL_DIFFERENT_FROM = OWL_NS + "differentFrom"
RDF_FIRST = RDF_NS + "first"
RDF_REST = RDF_NS + "rest"
RDF_NIL = RDF_NS + "nil"


@dataclass
class ReasoningStats:
    """Statistics from a reasoning run."""
    iterations: int
    triples_inferred: int
    rdfs2_inferences: int  # domain
    rdfs3_inferences: int  # range
    rdfs5_inferences: int  # subPropertyOf transitivity
    rdfs7_inferences: int  # property inheritance
    rdfs9_inferences: int  # type inheritance
    rdfs11_inferences: int  # subClassOf transitivity
    # OWL statistics
    owl_same_as_inferences: int = 0
    owl_equivalent_class_inferences: int = 0
    owl_equivalent_property_inferences: int = 0
    owl_inverse_of_inferences: int = 0
    owl_transitive_inferences: int = 0
    owl_symmetric_inferences: int = 0
    owl_functional_inferences: int = 0
    owl_inverse_functional_inferences: int = 0
    owl_has_value_inferences: int = 0


class RDFSReasoner:
    """
    Forward-chaining RDFS and OWL 2 RL reasoner.
    
    Materializes RDFS and OWL entailments into the FactStore.
    Uses fixed-point iteration to compute transitive closure.
    """
    
    def __init__(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        max_iterations: int = 100,
        enable_owl: bool = True,
    ):
        """
        Initialize the reasoner.
        
        Args:
            term_dict: TermDict for term lookup
            fact_store: FactStore containing the facts
            max_iterations: Maximum iterations for fixed-point (default 100)
            enable_owl: Enable OWL reasoning (default True)
        """
        self._term_dict = term_dict
        self._fact_store = fact_store
        self._max_iterations = max_iterations
        self._enable_owl = enable_owl
        
        # Cache vocabulary term IDs
        self._vocab_ids: Optional[dict] = None
    
    def _ensure_vocab_ids(self) -> dict:
        """Ensure vocabulary term IDs are cached."""
        if self._vocab_ids is not None:
            return self._vocab_ids
        
        # Get IDs for RDFS vocabulary terms (only if they exist in data)
        vocab_terms = [
            (RDFS_SUBCLASS_OF, "subClassOf"),
            (RDFS_SUBPROPERTY_OF, "subPropertyOf"),
            (RDFS_DOMAIN, "domain"),
            (RDFS_RANGE, "range"),
        ]
        
        # OWL vocabulary terms
        owl_vocab_terms = [
            (OWL_SAME_AS, "sameAs"),
            (OWL_EQUIVALENT_CLASS, "equivalentClass"),
            (OWL_EQUIVALENT_PROPERTY, "equivalentProperty"),
            (OWL_INVERSE_OF, "inverseOf"),
            (OWL_TRANSITIVE_PROPERTY, "TransitiveProperty"),
            (OWL_SYMMETRIC_PROPERTY, "SymmetricProperty"),
            (OWL_FUNCTIONAL_PROPERTY, "FunctionalProperty"),
            (OWL_INVERSE_FUNCTIONAL_PROPERTY, "InverseFunctionalProperty"),
            (OWL_HAS_VALUE, "hasValue"),
            (OWL_ON_PROPERTY, "onProperty"),
            (OWL_SOME_VALUES_FROM, "someValuesFrom"),
            (OWL_ALL_VALUES_FROM, "allValuesFrom"),
            (OWL_INTERSECTION_OF, "intersectionOf"),
            (OWL_UNION_OF, "unionOf"),
            (OWL_PROPERTY_CHAIN_AXIOM, "propertyChainAxiom"),
            (OWL_DIFFERENT_FROM, "differentFrom"),
            (RDF_FIRST, "first"),
            (RDF_REST, "rest"),
            (RDF_NIL, "nil"),
        ]
        
        self._vocab_ids = {}
        for iri, name in vocab_terms:
            term = Term(kind=TermKind.IRI, lex=iri)
            term_id = self._term_dict.get_id(term)
            if term_id is not None:
                self._vocab_ids[name] = term_id
        
        # Add OWL vocabulary if enabled
        if self._enable_owl:
            for iri, name in owl_vocab_terms:
                term = Term(kind=TermKind.IRI, lex=iri)
                term_id = self._term_dict.get_id(term)
                if term_id is not None:
                    self._vocab_ids[name] = term_id
        
        # Check if we need to create vocabulary terms for inference output
        needs_type = (
            "domain" in self._vocab_ids or 
            "range" in self._vocab_ids or 
            "subClassOf" in self._vocab_ids or
            "hasValue" in self._vocab_ids
        )
        needs_same_as = (
            "FunctionalProperty" in self._vocab_ids or
            "InverseFunctionalProperty" in self._vocab_ids
        )
        needs_subclass = "equivalentClass" in self._vocab_ids
        needs_subprop = "equivalentProperty" in self._vocab_ids
        
        # Create vocabulary terms that will be used in inferred triples
        if needs_type:
            type_term = Term(kind=TermKind.IRI, lex=RDF_TYPE)
            self._vocab_ids["type"] = self._term_dict.get_or_create(type_term)
        else:
            type_term = Term(kind=TermKind.IRI, lex=RDF_TYPE)
            type_id = self._term_dict.get_id(type_term)
            if type_id is not None:
                self._vocab_ids["type"] = type_id
        
        if needs_same_as and "sameAs" not in self._vocab_ids:
            same_term = Term(kind=TermKind.IRI, lex=OWL_SAME_AS)
            self._vocab_ids["sameAs"] = self._term_dict.get_or_create(same_term)
        
        if needs_subclass and "subClassOf" not in self._vocab_ids:
            subclass_term = Term(kind=TermKind.IRI, lex=RDFS_SUBCLASS_OF)
            self._vocab_ids["subClassOf"] = self._term_dict.get_or_create(subclass_term)
        
        if needs_subprop and "subPropertyOf" not in self._vocab_ids:
            subprop_term = Term(kind=TermKind.IRI, lex=RDFS_SUBPROPERTY_OF)
            self._vocab_ids["subPropertyOf"] = self._term_dict.get_or_create(subprop_term)
        
        return self._vocab_ids
    
    def reason(self, graph_id: TermId = DEFAULT_GRAPH_ID) -> ReasoningStats:
        """
        Run RDFS and OWL forward-chaining inference.
        
        Materializes all entailments into the FactStore.
        
        Args:
            graph_id: Graph to reason over (default: default graph)
            
        Returns:
            ReasoningStats with counts of inferred triples
        """
        vocab = self._ensure_vocab_ids()
        
        # If no vocabulary in the data, nothing to infer
        if not vocab:
            return ReasoningStats(0, 0, 0, 0, 0, 0, 0, 0)
        
        stats = ReasoningStats(
            iterations=0,
            triples_inferred=0,
            rdfs2_inferences=0,
            rdfs3_inferences=0,
            rdfs5_inferences=0,
            rdfs7_inferences=0,
            rdfs9_inferences=0,
            rdfs11_inferences=0,
        )
        
        # Track existing facts to avoid duplicates
        existing_facts: Set[Tuple[TermId, TermId, TermId, TermId]] = set()
        df = self._fact_store.scan_facts()
        for row in df.iter_rows(named=True):
            existing_facts.add((row["g"], row["s"], row["p"], row["o"]))
        
        # Fixed-point iteration
        for iteration in range(self._max_iterations):
            stats.iterations = iteration + 1
            new_facts: List[Tuple[TermId, TermId, TermId, TermId]] = []
            
            # RDFS rules (always applied)
            new_facts.extend(self._apply_rdfs11(vocab, existing_facts, graph_id, stats))
            new_facts.extend(self._apply_rdfs5(vocab, existing_facts, graph_id, stats))
            new_facts.extend(self._apply_rdfs9(vocab, existing_facts, graph_id, stats))
            new_facts.extend(self._apply_rdfs7(vocab, existing_facts, graph_id, stats))
            new_facts.extend(self._apply_rdfs2(vocab, existing_facts, graph_id, stats))
            new_facts.extend(self._apply_rdfs3(vocab, existing_facts, graph_id, stats))
            
            # OWL rules (if enabled)
            if self._enable_owl:
                new_facts.extend(self._apply_owl_same_as(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_equivalent_class(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_equivalent_property(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_inverse_of(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_transitive(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_symmetric(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_functional(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_inverse_functional(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_has_value(vocab, existing_facts, graph_id, stats))
                # NEW OWL 2 RL rules
                new_facts.extend(self._apply_owl_all_values_from(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_some_values_from(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_union_of(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_intersection_of(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_property_chain(vocab, existing_facts, graph_id, stats))
                new_facts.extend(self._apply_owl_different_from(vocab, existing_facts, graph_id, stats))
            
            if not new_facts:
                # Fixed point reached
                break
            
            # Add new facts to store and existing set
            self._fact_store.add_facts_batch(
                new_facts,
                flags=FactFlags.INFERRED,
                process=vocab.get("type"),  # Mark with process if available
            )
            
            for fact in new_facts:
                existing_facts.add(fact)
            
            stats.triples_inferred += len(new_facts)
        
        return stats
    
    def _get_facts_with_predicate(
        self,
        predicate_id: TermId,
        graph_id: TermId,
    ) -> List[Tuple[TermId, TermId]]:
        """Get all (subject, object) pairs for a given predicate."""
        df = self._fact_store.scan_facts()
        
        filtered = df.filter(
            (pl.col("p") == predicate_id) &
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        return [
            (row["s"], row["o"])
            for row in filtered.select(["s", "o"]).iter_rows(named=True)
        ]
    
    def _apply_rdfs11(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS11: Transitive subClassOf.
        
        (C1 subClassOf C2) + (C2 subClassOf C3) => (C1 subClassOf C3)
        """
        subclass_id = vocab.get("subClassOf")
        if subclass_id is None:
            return []
        
        # Get all subClassOf facts
        subclass_pairs = self._get_facts_with_predicate(subclass_id, graph_id)
        if not subclass_pairs:
            return []
        
        # Build adjacency map: C1 -> [C2, C3, ...]
        subclass_of: dict[TermId, Set[TermId]] = {}
        for c1, c2 in subclass_pairs:
            if c1 not in subclass_of:
                subclass_of[c1] = set()
            subclass_of[c1].add(c2)
        
        # Find transitive closures
        new_facts = []
        for c1, direct_supers in subclass_of.items():
            for c2 in list(direct_supers):
                if c2 in subclass_of:
                    for c3 in subclass_of[c2]:
                        fact = (graph_id, c1, subclass_id, c3)
                        if fact not in existing and c1 != c3:
                            new_facts.append(fact)
                            stats.rdfs11_inferences += 1
        
        return new_facts
    
    def _apply_rdfs5(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS5: Transitive subPropertyOf.
        
        (p1 subProp p2) + (p2 subProp p3) => (p1 subProp p3)
        """
        subprop_id = vocab.get("subPropertyOf")
        if subprop_id is None:
            return []
        
        # Get all subPropertyOf facts
        subprop_pairs = self._get_facts_with_predicate(subprop_id, graph_id)
        if not subprop_pairs:
            return []
        
        # Build adjacency map
        subprop_of: dict[TermId, Set[TermId]] = {}
        for p1, p2 in subprop_pairs:
            if p1 not in subprop_of:
                subprop_of[p1] = set()
            subprop_of[p1].add(p2)
        
        # Find transitive closures
        new_facts = []
        for p1, direct_supers in subprop_of.items():
            for p2 in list(direct_supers):
                if p2 in subprop_of:
                    for p3 in subprop_of[p2]:
                        fact = (graph_id, p1, subprop_id, p3)
                        if fact not in existing and p1 != p3:
                            new_facts.append(fact)
                            stats.rdfs5_inferences += 1
        
        return new_facts
    
    def _apply_rdfs9(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS9: Type inheritance through subClassOf.
        
        (x rdf:type C1) + (C1 subClassOf C2) => (x rdf:type C2)
        """
        type_id = vocab.get("type")
        subclass_id = vocab.get("subClassOf")
        if type_id is None or subclass_id is None:
            return []
        
        # Get type assertions
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        if not type_pairs:
            return []
        
        # Get subClassOf hierarchy
        subclass_pairs = self._get_facts_with_predicate(subclass_id, graph_id)
        if not subclass_pairs:
            return []
        
        # Build subClassOf map: C1 -> [C2, ...]
        subclass_of: dict[TermId, Set[TermId]] = {}
        for c1, c2 in subclass_pairs:
            if c1 not in subclass_of:
                subclass_of[c1] = set()
            subclass_of[c1].add(c2)
        
        # Infer types
        new_facts = []
        for x, c1 in type_pairs:
            if c1 in subclass_of:
                for c2 in subclass_of[c1]:
                    fact = (graph_id, x, type_id, c2)
                    if fact not in existing:
                        new_facts.append(fact)
                        stats.rdfs9_inferences += 1
        
        return new_facts
    
    def _apply_rdfs7(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS7: Property inheritance through subPropertyOf.
        
        (x p1 y) + (p1 subProp p2) => (x p2 y)
        """
        subprop_id = vocab.get("subPropertyOf")
        if subprop_id is None:
            return []
        
        # Get subPropertyOf hierarchy
        subprop_pairs = self._get_facts_with_predicate(subprop_id, graph_id)
        if not subprop_pairs:
            return []
        
        # Build subPropertyOf map: p1 -> [p2, ...]
        subprop_of: dict[TermId, Set[TermId]] = {}
        for p1, p2 in subprop_pairs:
            if p1 not in subprop_of:
                subprop_of[p1] = set()
            subprop_of[p1].add(p2)
        
        # Get all facts and apply property inheritance
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        for row in filtered.iter_rows(named=True):
            p1 = row["p"]
            if p1 in subprop_of:
                for p2 in subprop_of[p1]:
                    fact = (graph_id, row["s"], p2, row["o"])
                    if fact not in existing:
                        new_facts.append(fact)
                        stats.rdfs7_inferences += 1
        
        return new_facts
    
    def _apply_rdfs2(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS2: Domain inference.
        
        (x p y) + (p rdfs:domain C) => (x rdf:type C)
        """
        domain_id = vocab.get("domain")
        type_id = vocab.get("type")
        if domain_id is None or type_id is None:
            return []
        
        # Get domain declarations
        domain_pairs = self._get_facts_with_predicate(domain_id, graph_id)
        if not domain_pairs:
            return []
        
        # Build domain map: p -> C
        domain_of: dict[TermId, TermId] = {}
        for p, c in domain_pairs:
            domain_of[p] = c
        
        # Get all facts and apply domain inference
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        for row in filtered.iter_rows(named=True):
            p = row["p"]
            if p in domain_of:
                c = domain_of[p]
                fact = (graph_id, row["s"], type_id, c)
                if fact not in existing:
                    new_facts.append(fact)
                    stats.rdfs2_inferences += 1
        
        return new_facts
    
    def _apply_rdfs3(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS3: Range inference.
        
        (x p y) + (p rdfs:range C) => (y rdf:type C)
        """
        range_id = vocab.get("range")
        type_id = vocab.get("type")
        if range_id is None or type_id is None:
            return []
        
        # Get range declarations
        range_pairs = self._get_facts_with_predicate(range_id, graph_id)
        if not range_pairs:
            return []
        
        # Build range map: p -> C
        range_of: dict[TermId, TermId] = {}
        for p, c in range_pairs:
            range_of[p] = c
        
        # Get all facts and apply range inference
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        for row in filtered.iter_rows(named=True):
            p = row["p"]
            if p in range_of:
                c = range_of[p]
                fact = (graph_id, row["o"], type_id, c)
                if fact not in existing:
                    new_facts.append(fact)
                    stats.rdfs3_inferences += 1
        
        return new_facts
    
    # =========================================================================
    # OWL Entailment Rules
    # =========================================================================
    
    def _apply_owl_same_as(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:sameAs symmetry and transitivity.
        
        Symmetry: (x sameAs y) => (y sameAs x)
        Transitivity: (x sameAs y) + (y sameAs z) => (x sameAs z)
        """
        same_as_id = vocab.get("sameAs")
        if same_as_id is None:
            return []
        
        # Get all sameAs pairs
        same_pairs = self._get_facts_with_predicate(same_as_id, graph_id)
        if not same_pairs:
            return []
        
        new_facts = []
        
        # Symmetry: (x sameAs y) => (y sameAs x)
        for x, y in same_pairs:
            fact = (graph_id, y, same_as_id, x)
            if fact not in existing and x != y:
                new_facts.append(fact)
                stats.owl_same_as_inferences += 1
        
        # Build adjacency for transitivity
        same_as_map: dict[TermId, Set[TermId]] = {}
        for x, y in same_pairs:
            if x not in same_as_map:
                same_as_map[x] = set()
            same_as_map[x].add(y)
        
        # Transitivity: (x sameAs y) + (y sameAs z) => (x sameAs z)
        for x, ys in same_as_map.items():
            for y in list(ys):
                if y in same_as_map:
                    for z in same_as_map[y]:
                        fact = (graph_id, x, same_as_id, z)
                        if fact not in existing and x != z:
                            new_facts.append(fact)
                            stats.owl_same_as_inferences += 1
        
        return new_facts
    
    def _apply_owl_equivalent_class(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:equivalentClass => mutual rdfs:subClassOf.
        
        (C1 equivalentClass C2) => (C1 subClassOf C2) + (C2 subClassOf C1)
        """
        equiv_id = vocab.get("equivalentClass")
        subclass_id = vocab.get("subClassOf")
        if equiv_id is None or subclass_id is None:
            return []
        
        equiv_pairs = self._get_facts_with_predicate(equiv_id, graph_id)
        if not equiv_pairs:
            return []
        
        new_facts = []
        for c1, c2 in equiv_pairs:
            # C1 subClassOf C2
            fact1 = (graph_id, c1, subclass_id, c2)
            if fact1 not in existing:
                new_facts.append(fact1)
                stats.owl_equivalent_class_inferences += 1
            
            # C2 subClassOf C1
            fact2 = (graph_id, c2, subclass_id, c1)
            if fact2 not in existing:
                new_facts.append(fact2)
                stats.owl_equivalent_class_inferences += 1
        
        return new_facts
    
    def _apply_owl_equivalent_property(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:equivalentProperty => mutual rdfs:subPropertyOf.
        
        (p1 equivalentProperty p2) => (p1 subPropertyOf p2) + (p2 subPropertyOf p1)
        """
        equiv_id = vocab.get("equivalentProperty")
        subprop_id = vocab.get("subPropertyOf")
        if equiv_id is None or subprop_id is None:
            return []
        
        equiv_pairs = self._get_facts_with_predicate(equiv_id, graph_id)
        if not equiv_pairs:
            return []
        
        new_facts = []
        for p1, p2 in equiv_pairs:
            # p1 subPropertyOf p2
            fact1 = (graph_id, p1, subprop_id, p2)
            if fact1 not in existing:
                new_facts.append(fact1)
                stats.owl_equivalent_property_inferences += 1
            
            # p2 subPropertyOf p1
            fact2 = (graph_id, p2, subprop_id, p1)
            if fact2 not in existing:
                new_facts.append(fact2)
                stats.owl_equivalent_property_inferences += 1
        
        return new_facts
    
    def _apply_owl_inverse_of(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:inverseOf property inversion.
        
        (x p y) + (p inverseOf q) => (y q x)
        """
        inverse_id = vocab.get("inverseOf")
        if inverse_id is None:
            return []
        
        # Get inverse declarations
        inverse_pairs = self._get_facts_with_predicate(inverse_id, graph_id)
        if not inverse_pairs:
            return []
        
        # Build inverse map: p -> q
        inverse_of: dict[TermId, TermId] = {}
        for p, q in inverse_pairs:
            inverse_of[p] = q
            # inverseOf is symmetric
            inverse_of[q] = p
        
        # Get all facts and apply inverse inference
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        for row in filtered.iter_rows(named=True):
            p = row["p"]
            if p in inverse_of:
                q = inverse_of[p]
                # (x p y) => (y q x)
                fact = (graph_id, row["o"], q, row["s"])
                if fact not in existing:
                    new_facts.append(fact)
                    stats.owl_inverse_of_inferences += 1
        
        return new_facts
    
    def _apply_owl_transitive(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:TransitiveProperty transitive closure.
        
        (p rdf:type TransitiveProperty) + (x p y) + (y p z) => (x p z)
        """
        trans_type_id = vocab.get("TransitiveProperty")
        type_id = vocab.get("type")
        if trans_type_id is None or type_id is None:
            return []
        
        # Find all transitive properties
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        transitive_props: Set[TermId] = set()
        for s, o in type_pairs:
            if o == trans_type_id:
                transitive_props.add(s)
        
        if not transitive_props:
            return []
        
        new_facts = []
        
        # For each transitive property, compute transitive closure
        for prop_id in transitive_props:
            prop_pairs = self._get_facts_with_predicate(prop_id, graph_id)
            if not prop_pairs:
                continue
            
            # Build adjacency map
            adjacency: dict[TermId, Set[TermId]] = {}
            for x, y in prop_pairs:
                if x not in adjacency:
                    adjacency[x] = set()
                adjacency[x].add(y)
            
            # Find transitive closures
            for x, ys in adjacency.items():
                for y in list(ys):
                    if y in adjacency:
                        for z in adjacency[y]:
                            fact = (graph_id, x, prop_id, z)
                            if fact not in existing and x != z:
                                new_facts.append(fact)
                                stats.owl_transitive_inferences += 1
        
        return new_facts
    
    def _apply_owl_symmetric(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:SymmetricProperty symmetry.
        
        (p rdf:type SymmetricProperty) + (x p y) => (y p x)
        """
        sym_type_id = vocab.get("SymmetricProperty")
        type_id = vocab.get("type")
        if sym_type_id is None or type_id is None:
            return []
        
        # Find all symmetric properties
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        symmetric_props: Set[TermId] = set()
        for s, o in type_pairs:
            if o == sym_type_id:
                symmetric_props.add(s)
        
        if not symmetric_props:
            return []
        
        new_facts = []
        
        # For each symmetric property, add inverse facts
        for prop_id in symmetric_props:
            prop_pairs = self._get_facts_with_predicate(prop_id, graph_id)
            for x, y in prop_pairs:
                fact = (graph_id, y, prop_id, x)
                if fact not in existing and x != y:
                    new_facts.append(fact)
                    stats.owl_symmetric_inferences += 1
        
        return new_facts
    
    def _apply_owl_functional(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:FunctionalProperty sameAs inference.
        
        (p rdf:type FunctionalProperty) + (x p y1) + (x p y2) => (y1 sameAs y2)
        """
        func_type_id = vocab.get("FunctionalProperty")
        type_id = vocab.get("type")
        same_as_id = vocab.get("sameAs")
        if func_type_id is None or type_id is None or same_as_id is None:
            return []
        
        # Find all functional properties
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        functional_props: Set[TermId] = set()
        for s, o in type_pairs:
            if o == func_type_id:
                functional_props.add(s)
        
        if not functional_props:
            return []
        
        new_facts = []
        
        # For each functional property, find conflicting values
        for prop_id in functional_props:
            prop_pairs = self._get_facts_with_predicate(prop_id, graph_id)
            
            # Group by subject
            by_subject: dict[TermId, List[TermId]] = {}
            for x, y in prop_pairs:
                if x not in by_subject:
                    by_subject[x] = []
                by_subject[x].append(y)
            
            # If multiple values, they must be sameAs
            for x, ys in by_subject.items():
                if len(ys) > 1:
                    for i, y1 in enumerate(ys):
                        for y2 in ys[i + 1:]:
                            fact = (graph_id, y1, same_as_id, y2)
                            if fact not in existing and y1 != y2:
                                new_facts.append(fact)
                                stats.owl_functional_inferences += 1
        
        return new_facts
    
    def _apply_owl_inverse_functional(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:InverseFunctionalProperty sameAs inference.
        
        (p rdf:type InverseFunctionalProperty) + (x1 p y) + (x2 p y) => (x1 sameAs x2)
        """
        inv_func_type_id = vocab.get("InverseFunctionalProperty")
        type_id = vocab.get("type")
        same_as_id = vocab.get("sameAs")
        if inv_func_type_id is None or type_id is None or same_as_id is None:
            return []
        
        # Find all inverse functional properties
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        inv_functional_props: Set[TermId] = set()
        for s, o in type_pairs:
            if o == inv_func_type_id:
                inv_functional_props.add(s)
        
        if not inv_functional_props:
            return []
        
        new_facts = []
        
        # For each inverse functional property, find conflicting subjects
        for prop_id in inv_functional_props:
            prop_pairs = self._get_facts_with_predicate(prop_id, graph_id)
            
            # Group by object
            by_object: dict[TermId, List[TermId]] = {}
            for x, y in prop_pairs:
                if y not in by_object:
                    by_object[y] = []
                by_object[y].append(x)
            
            # If multiple subjects, they must be sameAs
            for y, xs in by_object.items():
                if len(xs) > 1:
                    for i, x1 in enumerate(xs):
                        for x2 in xs[i + 1:]:
                            fact = (graph_id, x1, same_as_id, x2)
                            if fact not in existing and x1 != x2:
                                new_facts.append(fact)
                                stats.owl_inverse_functional_inferences += 1
        
        return new_facts
    
    def _apply_owl_has_value(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:hasValue restriction inference.
        
        (C owl:onProperty p) + (C owl:hasValue v) + (x rdf:type C) => (x p v)
        Also: (C owl:onProperty p) + (C owl:hasValue v) + (x p v) => (x rdf:type C)
        """
        has_value_id = vocab.get("hasValue")
        on_property_id = vocab.get("onProperty")
        type_id = vocab.get("type")
        if has_value_id is None or on_property_id is None or type_id is None:
            return []
        
        # Get hasValue and onProperty declarations
        has_value_pairs = self._get_facts_with_predicate(has_value_id, graph_id)
        on_property_pairs = self._get_facts_with_predicate(on_property_id, graph_id)
        
        if not has_value_pairs or not on_property_pairs:
            return []
        
        # Build restriction maps: C -> (p, v)
        restrictions: dict[TermId, Tuple[TermId, TermId]] = {}
        
        # Map C -> p
        c_to_prop: dict[TermId, TermId] = {}
        for c, p in on_property_pairs:
            c_to_prop[c] = p
        
        # Map C -> v and combine
        for c, v in has_value_pairs:
            if c in c_to_prop:
                restrictions[c] = (c_to_prop[c], v)
        
        if not restrictions:
            return []
        
        new_facts = []
        
        # Get type assertions
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        
        # Forward: (x type C) => (x p v)
        for x, c in type_pairs:
            if c in restrictions:
                p, v = restrictions[c]
                fact = (graph_id, x, p, v)
                if fact not in existing:
                    new_facts.append(fact)
                    stats.owl_has_value_inferences += 1
        
        # Backward: (x p v) => (x type C)
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        # Build reverse lookup: (p, v) -> C
        pv_to_class: dict[Tuple[TermId, TermId], TermId] = {}
        for c, (p, v) in restrictions.items():
            pv_to_class[(p, v)] = c
        
        for row in filtered.iter_rows(named=True):
            key = (row["p"], row["o"])
            if key in pv_to_class:
                c = pv_to_class[key]
                fact = (graph_id, row["s"], type_id, c)
                if fact not in existing:
                    new_facts.append(fact)
                    stats.owl_has_value_inferences += 1
        
        return new_facts
    
    def _apply_owl_all_values_from(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:allValuesFrom restriction inference.
        
        (C allValuesFrom D) + (C onProperty p) + (x type C) + (x p y) => (y type D)
        """
        all_values_id = vocab.get("allValuesFrom")
        on_property_id = vocab.get("onProperty")
        type_id = vocab.get("type")
        if all_values_id is None or on_property_id is None or type_id is None:
            return []
        
        # Get restrictions
        all_values_pairs = self._get_facts_with_predicate(all_values_id, graph_id)
        on_property_pairs = self._get_facts_with_predicate(on_property_id, graph_id)
        
        if not all_values_pairs or not on_property_pairs:
            return []
        
        # Build restriction map: C -> (p, D)
        c_to_prop: dict[TermId, TermId] = {}
        for c, p in on_property_pairs:
            c_to_prop[c] = p
        
        restrictions: dict[TermId, Tuple[TermId, TermId]] = {}  # C -> (p, D)
        for c, d in all_values_pairs:
            if c in c_to_prop:
                restrictions[c] = (c_to_prop[c], d)
        
        if not restrictions:
            return []
        
        # Get type and property assertions
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        
        # Build: x -> types
        x_types: dict[TermId, Set[TermId]] = {}
        for x, c in type_pairs:
            if x not in x_types:
                x_types[x] = set()
            x_types[x].add(c)
        
        # Get all property facts
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        
        # For each (x type C) where C has allValuesFrom restriction
        for x, types in x_types.items():
            for c in types:
                if c in restrictions:
                    p, d = restrictions[c]
                    # Find all (x p y) and infer (y type D)
                    prop_facts = filtered.filter(
                        (pl.col("s") == x) &
                        (pl.col("p") == p)
                    )
                    for row in prop_facts.iter_rows(named=True):
                        y = row["o"]
                        fact = (graph_id, y, type_id, d)
                        if fact not in existing:
                            new_facts.append(fact)
        
        return new_facts
    
    def _apply_owl_some_values_from(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:someValuesFrom restriction inference.
        
        (C someValuesFrom D) + (C onProperty p) + (x p y) + (y type D) => (x type C)
        """
        some_values_id = vocab.get("someValuesFrom")
        on_property_id = vocab.get("onProperty")
        type_id = vocab.get("type")
        if some_values_id is None or on_property_id is None or type_id is None:
            return []
        
        # Get restrictions
        some_values_pairs = self._get_facts_with_predicate(some_values_id, graph_id)
        on_property_pairs = self._get_facts_with_predicate(on_property_id, graph_id)
        
        if not some_values_pairs or not on_property_pairs:
            return []
        
        # Build restriction map
        c_to_prop: dict[TermId, TermId] = {}
        for c, p in on_property_pairs:
            c_to_prop[c] = p
        
        restrictions: dict[TermId, Tuple[TermId, TermId]] = {}
        for c, d in some_values_pairs:
            if c in c_to_prop:
                restrictions[c] = (c_to_prop[c], d)
        
        if not restrictions:
            return []
        
        # Build type map: y -> types
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        y_types: dict[TermId, Set[TermId]] = {}
        for y, t in type_pairs:
            if y not in y_types:
                y_types[y] = set()
            y_types[y].add(t)
        
        # Get all property facts
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        
        # For each restriction C with someValuesFrom D on property p
        for c, (p, d) in restrictions.items():
            # Find (x p y) where (y type D)
            prop_facts = filtered.filter(pl.col("p") == p)
            for row in prop_facts.iter_rows(named=True):
                x = row["s"]
                y = row["o"]
                # Check if y has type D
                if y in y_types and d in y_types[y]:
                    fact = (graph_id, x, type_id, c)
                    if fact not in existing:
                        new_facts.append(fact)
        
        return new_facts
    
    def _apply_owl_union_of(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:unionOf class expression.
        
        (C unionOf (C1 C2 ...)) + (x type Ci) => (x type C)
        """
        union_id = vocab.get("unionOf")
        type_id = vocab.get("type")
        if union_id is None or type_id is None:
            return []
        
        # Get union declarations
        union_pairs = self._get_facts_with_predicate(union_id, graph_id)
        if not union_pairs:
            return []
        
        # Parse lists to get union members
        from rdf_starbase.storage.inference.rules import RuleApplicator
        applicator = RuleApplicator(self._term_dict, self._fact_store)
        
        unions: dict[TermId, Set[TermId]] = {}  # C -> {C1, C2, ...}
        for c, list_head in union_pairs:
            members = applicator.parse_rdf_list(list_head, graph_id)
            if members:
                unions[c] = set(members)
        
        if not unions:
            return []
        
        # Get type assertions
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        
        new_facts = []
        for x, ci in type_pairs:
            # Check if ci is a member of any union
            for c, members in unions.items():
                if ci in members:
                    fact = (graph_id, x, type_id, c)
                    if fact not in existing:
                        new_facts.append(fact)
        
        return new_facts
    
    def _apply_owl_intersection_of(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:intersectionOf class expression.
        
        (C intersectionOf (C1 C2 ...)) + (x type C1) + (x type C2) + ... => (x type C)
        """
        intersection_id = vocab.get("intersectionOf")
        type_id = vocab.get("type")
        if intersection_id is None or type_id is None:
            return []
        
        # Get intersection declarations
        intersection_pairs = self._get_facts_with_predicate(intersection_id, graph_id)
        if not intersection_pairs:
            return []
        
        from rdf_starbase.storage.inference.rules import RuleApplicator
        applicator = RuleApplicator(self._term_dict, self._fact_store)
        
        intersections: dict[TermId, Set[TermId]] = {}
        for c, list_head in intersection_pairs:
            members = applicator.parse_rdf_list(list_head, graph_id)
            if members:
                intersections[c] = set(members)
        
        if not intersections:
            return []
        
        # Get all types for each individual
        type_pairs = self._get_facts_with_predicate(type_id, graph_id)
        x_types: dict[TermId, Set[TermId]] = {}
        for x, t in type_pairs:
            if x not in x_types:
                x_types[x] = set()
            x_types[x].add(t)
        
        new_facts = []
        for c, members in intersections.items():
            # Find all x that have ALL member types
            for x, types in x_types.items():
                if members.issubset(types):
                    fact = (graph_id, x, type_id, c)
                    if fact not in existing:
                        new_facts.append(fact)
        
        return new_facts
    
    def _apply_owl_property_chain(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:propertyChainAxiom inference.
        
        (p propertyChainAxiom (p1 p2)) + (x p1 y) + (y p2 z) => (x p z)
        """
        chain_id = vocab.get("propertyChainAxiom")
        if chain_id is None:
            return []
        
        # Get property chains
        chain_pairs = self._get_facts_with_predicate(chain_id, graph_id)
        if not chain_pairs:
            return []
        
        from rdf_starbase.storage.inference.rules import RuleApplicator
        applicator = RuleApplicator(self._term_dict, self._fact_store)
        
        chains: dict[TermId, List[TermId]] = {}  # p -> [p1, p2, ...]
        for p, list_head in chain_pairs:
            members = applicator.parse_rdf_list(list_head, graph_id)
            if len(members) >= 2:
                chains[p] = members
        
        if not chains:
            return []
        
        # Get all facts for property lookups
        df = self._fact_store.scan_facts()
        filtered = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        new_facts = []
        
        for p, chain in chains.items():
            if len(chain) == 2:
                # Simple 2-element chain: p1, p2
                p1, p2 = chain
                
                # Get all (x p1 y) pairs
                p1_facts = filtered.filter(pl.col("p") == p1)
                p1_dict: dict[TermId, List[TermId]] = {}  # y -> [x, ...]
                for row in p1_facts.iter_rows(named=True):
                    y = row["o"]
                    if y not in p1_dict:
                        p1_dict[y] = []
                    p1_dict[y].append(row["s"])
                
                # Get all (y p2 z) pairs and join
                p2_facts = filtered.filter(pl.col("p") == p2)
                for row in p2_facts.iter_rows(named=True):
                    y = row["s"]
                    z = row["o"]
                    if y in p1_dict:
                        for x in p1_dict[y]:
                            fact = (graph_id, x, p, z)
                            if fact not in existing:
                                new_facts.append(fact)
            elif len(chain) > 2:
                # Longer chain - recursive join (simplified for now)
                # TODO: Implement full n-ary property chain
                pass
        
        return new_facts
    
    def _apply_owl_different_from(
        self,
        vocab: dict,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: ReasoningStats,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        owl:differentFrom symmetry.
        
        (x differentFrom y) => (y differentFrom x)
        """
        diff_id = vocab.get("differentFrom")
        if diff_id is None:
            return []
        
        diff_pairs = self._get_facts_with_predicate(diff_id, graph_id)
        if not diff_pairs:
            return []
        
        new_facts = []
        for x, y in diff_pairs:
            fact = (graph_id, y, diff_id, x)
            if fact not in existing and x != y:
                new_facts.append(fact)
        
        return new_facts
    
    def get_inferred_count(self, graph_id: TermId = DEFAULT_GRAPH_ID) -> int:
        """Count the number of inferred facts in the store."""
        df = self._fact_store.scan_facts()
        return df.filter(
            (pl.col("g") == graph_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.INFERRED)).cast(pl.Boolean))
        ).height
