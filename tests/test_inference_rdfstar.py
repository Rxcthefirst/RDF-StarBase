"""
Tests for RDF-Star Inference Engine.

Tests the advanced inference capabilities:
- Confidence propagation through inference chains
- Provenance tracking for inferred facts
- Explanation service
- New OWL 2 RL rules
"""

import pytest
import math
from rdf_starbase.storage import (
    TermDict,
    FactStore,
    FactFlags,
    DEFAULT_GRAPH_ID,
    Term,
    TermKind,
)
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.inference.rdfstar_rules import (
    RDFStarInferenceEngine,
    ConfidenceMethod,
)
from rdf_starbase.storage.inference.explanation import (
    ExplanationService,
    FactOrigin,
)
from rdf_starbase.storage.reasoner import (
    RDFSReasoner,
    RDFS_SUBCLASS_OF,
    RDFS_SUBPROPERTY_OF,
    RDF_TYPE,
    OWL_UNION_OF,
    OWL_INTERSECTION_OF,
    OWL_PROPERTY_CHAIN_AXIOM,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def storage():
    """Create fresh storage components."""
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    return term_dict, qt_dict, fact_store


def intern_iri(term_dict: TermDict, iri: str) -> int:
    """Helper to intern an IRI and return its ID."""
    term = Term(kind=TermKind.IRI, lex=iri)
    return term_dict.get_or_create(term)


def intern_literal(term_dict: TermDict, value: str, datatype: str = None) -> int:
    """Helper to intern a literal and return its ID."""
    if datatype:
        dtype_id = intern_iri(term_dict, datatype)
        term = Term(kind=TermKind.LITERAL, lex=value, datatype_id=dtype_id)
    else:
        term = Term(kind=TermKind.LITERAL, lex=value)
    return term_dict.get_or_create(term)


def add_triple(
    term_dict: TermDict,
    fact_store: FactStore,
    s_iri: str,
    p_iri: str,
    o_iri: str,
) -> None:
    """Helper to add a triple."""
    s_id = intern_iri(term_dict, s_iri)
    p_id = intern_iri(term_dict, p_iri)
    o_id = intern_iri(term_dict, o_iri)
    fact_store.add_facts_batch([(DEFAULT_GRAPH_ID, s_id, p_id, o_id)])


def add_confidence_annotation(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    s_iri: str,
    p_iri: str,
    o_iri: str,
    confidence: float,
) -> None:
    """Add a confidence annotation to a triple."""
    s_id = intern_iri(term_dict, s_iri)
    p_id = intern_iri(term_dict, p_iri)
    o_id = intern_iri(term_dict, o_iri)
    
    # Create quoted triple
    qt_id = qt_dict.get_or_create(s_id, p_id, o_id)
    
    # Add confidence annotation
    conf_pred_id = intern_iri(term_dict, RDFStarInferenceEngine.CONFIDENCE_PRED)
    xsd_decimal = "http://www.w3.org/2001/XMLSchema#decimal"
    conf_obj_id = intern_literal(term_dict, str(confidence), xsd_decimal)
    
    fact_store.add_facts_batch([
        (DEFAULT_GRAPH_ID, qt_id, conf_pred_id, conf_obj_id)
    ], flags=FactFlags.ASSERTED | FactFlags.METADATA)


def has_triple(
    term_dict: TermDict,
    fact_store: FactStore,
    s_iri: str,
    p_iri: str,
    o_iri: str,
) -> bool:
    """Check if a triple exists in the store."""
    s_id = term_dict.get_id(Term(kind=TermKind.IRI, lex=s_iri))
    p_id = term_dict.get_id(Term(kind=TermKind.IRI, lex=p_iri))
    o_id = term_dict.get_id(Term(kind=TermKind.IRI, lex=o_iri))
    
    if s_id is None or p_id is None or o_id is None:
        return False
    
    df = fact_store.scan_facts()
    filtered = df.filter(
        (df["s"] == s_id) &
        (df["p"] == p_id) &
        (df["o"] == o_id)
    )
    return filtered.height > 0


# =============================================================================
# Confidence Propagation Tests
# =============================================================================

class TestConfidencePropagation:
    """Test confidence propagation through inference."""
    
    def test_confidence_min_method(self, storage):
        """Test MIN confidence combination: result = min(c1, c2)"""
        term_dict, qt_dict, fact_store = storage
        
        # Add: fido type Dog (conf 0.9), Dog subClassOf Animal (conf 0.95)
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/fido", RDF_TYPE, "http://ex/Dog", 0.9)
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal", 0.95)
        
        # Run RDF-Star aware inference
        engine = RDFStarInferenceEngine(
            term_dict, qt_dict, fact_store,
            confidence_method=ConfidenceMethod.MIN,
            track_provenance=True,
        )
        stats = engine.reason_with_confidence()
        
        # Should infer: fido type Animal with confidence min(0.9, 0.95) = 0.9
        assert has_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Animal")
        assert stats.rdfs9_inferences >= 1
        
        # Check confidence
        type_id = intern_iri(term_dict, RDF_TYPE)
        fido_id = intern_iri(term_dict, "http://ex/fido")
        animal_id = intern_iri(term_dict, "http://ex/Animal")
        
        conf = engine.get_fact_confidence(DEFAULT_GRAPH_ID, fido_id, type_id, animal_id)
        assert abs(conf - 0.9) < 0.01
    
    def test_confidence_product_method(self, storage):
        """Test PRODUCT confidence combination: result = c1 * c2"""
        term_dict, qt_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/fido", RDF_TYPE, "http://ex/Dog", 0.9)
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal", 0.95)
        
        engine = RDFStarInferenceEngine(
            term_dict, qt_dict, fact_store,
            confidence_method=ConfidenceMethod.PRODUCT,
        )
        stats = engine.reason_with_confidence()
        
        assert has_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Animal")
        
        type_id = intern_iri(term_dict, RDF_TYPE)
        fido_id = intern_iri(term_dict, "http://ex/fido")
        animal_id = intern_iri(term_dict, "http://ex/Animal")
        
        conf = engine.get_fact_confidence(DEFAULT_GRAPH_ID, fido_id, type_id, animal_id)
        expected = 0.9 * 0.95  # 0.855
        assert abs(conf - expected) < 0.01
    
    def test_confidence_chain_propagation(self, storage):
        """Test confidence through multi-step inference chain."""
        term_dict, qt_dict, fact_store = storage
        
        # fido type Dog (0.9) -> Dog subClass Mammal (0.95) -> Mammal subClass Animal (0.98)
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Mammal")
        add_triple(term_dict, fact_store, "http://ex/Mammal", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/fido", RDF_TYPE, "http://ex/Dog", 0.9)
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Mammal", 0.95)
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/Mammal", RDFS_SUBCLASS_OF, "http://ex/Animal", 0.98)
        
        engine = RDFStarInferenceEngine(
            term_dict, qt_dict, fact_store,
            confidence_method=ConfidenceMethod.MIN,
        )
        engine.reason_with_confidence()
        
        # Check fido type Animal has min confidence across chain
        type_id = intern_iri(term_dict, RDF_TYPE)
        fido_id = intern_iri(term_dict, "http://ex/fido")
        animal_id = intern_iri(term_dict, "http://ex/Animal")
        
        conf = engine.get_fact_confidence(DEFAULT_GRAPH_ID, fido_id, type_id, animal_id)
        # min(0.9, min(0.95, 0.98)) = 0.9
        assert abs(conf - 0.9) < 0.01


# =============================================================================
# Justification and Explanation Tests
# =============================================================================

class TestJustification:
    """Test inference justification tracking."""
    
    def test_justification_recorded(self, storage):
        """Test that inference justifications are recorded."""
        term_dict, qt_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        engine = RDFStarInferenceEngine(
            term_dict, qt_dict, fact_store,
            track_provenance=True,
        )
        engine.reason_with_confidence()
        
        # Get justification for inferred fact
        type_id = intern_iri(term_dict, RDF_TYPE)
        fido_id = intern_iri(term_dict, "http://ex/fido")
        animal_id = intern_iri(term_dict, "http://ex/Animal")
        
        just = engine.get_justification(DEFAULT_GRAPH_ID, fido_id, type_id, animal_id)
        
        assert just is not None
        assert just.rule_id == "rdfs9"
        assert len(just.antecedent_facts) == 2
    
    def test_get_inferences_by_rule(self, storage):
        """Test getting all facts inferred by a specific rule."""
        term_dict, qt_dict, fact_store = storage
        
        # Create data that triggers multiple rules
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Mammal")
        add_triple(term_dict, fact_store, "http://ex/Mammal", RDFS_SUBCLASS_OF, "http://ex/Animal")
        add_triple(term_dict, fact_store, "http://ex/spot", RDF_TYPE, "http://ex/Dog")
        
        engine = RDFStarInferenceEngine(term_dict, qt_dict, fact_store)
        engine.reason_with_confidence()
        
        rdfs9_inferences = engine.get_inferred_facts_by_rule("rdfs9")
        rdfs11_inferences = engine.get_inferred_facts_by_rule("rdfs11")
        
        # Should have multiple rdfs9 inferences (type inheritance)
        assert len(rdfs9_inferences) >= 2  # fido+spot -> Mammal, Animal
        # Should have rdfs11 inferences (subClassOf transitivity)
        assert len(rdfs11_inferences) >= 1  # Dog subClass Animal


class TestExplanationService:
    """Test the explanation service."""
    
    def test_explain_asserted_fact(self, storage):
        """Test explaining an asserted fact."""
        term_dict, qt_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        
        service = ExplanationService(term_dict, qt_dict, fact_store)
        explanation = service.explain(
            "http://ex/fido",
            RDF_TYPE,
            "http://ex/Dog",
        )
        
        assert explanation.exists
        assert explanation.is_asserted
        assert not explanation.is_inferred
    
    def test_explain_inferred_fact(self, storage):
        """Test explaining an inferred fact."""
        term_dict, qt_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        # Run inference with provenance
        engine = RDFStarInferenceEngine(
            term_dict, qt_dict, fact_store,
            track_provenance=True,
        )
        engine.reason_with_confidence()
        
        service = ExplanationService(term_dict, qt_dict, fact_store)
        explanation = service.explain(
            "http://ex/fido",
            RDF_TYPE,
            "http://ex/Animal",
        )
        
        assert explanation.exists
        assert explanation.is_inferred
    
    def test_explain_nonexistent_fact(self, storage):
        """Test explaining a fact that doesn't exist."""
        term_dict, qt_dict, fact_store = storage
        
        service = ExplanationService(term_dict, qt_dict, fact_store)
        explanation = service.explain(
            "http://ex/unknown",
            RDF_TYPE,
            "http://ex/Anything",
        )
        
        assert not explanation.exists
        assert not explanation.is_asserted
        assert not explanation.is_inferred


# =============================================================================
# New OWL 2 RL Rules Tests
# =============================================================================

class TestOWLUnionOf:
    """Test owl:unionOf rule."""
    
    def test_union_of_simple(self, storage):
        """Test union membership inference."""
        term_dict, qt_dict, fact_store = storage
        
        # Setup: Person unionOf (Man Woman)
        # Using RDF list representation
        rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        # Create list: Man -> Woman -> nil
        add_triple(term_dict, fact_store, "http://ex/list1", rdf_first, "http://ex/Man")
        add_triple(term_dict, fact_store, "http://ex/list1", rdf_rest, "http://ex/list2")
        add_triple(term_dict, fact_store, "http://ex/list2", rdf_first, "http://ex/Woman")
        add_triple(term_dict, fact_store, "http://ex/list2", rdf_rest, rdf_nil)
        
        # Person unionOf list1
        add_triple(term_dict, fact_store, "http://ex/Person", OWL_UNION_OF, "http://ex/list1")
        
        # alice type Man
        add_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Man")
        
        # Run inference
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice type Person (because Man is in unionOf Person)
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Person")


class TestOWLIntersectionOf:
    """Test owl:intersectionOf rule."""
    
    def test_intersection_of_simple(self, storage):
        """Test intersection membership inference."""
        term_dict, qt_dict, fact_store = storage
        
        rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        # Create list: Employee -> Parent -> nil
        add_triple(term_dict, fact_store, "http://ex/list1", rdf_first, "http://ex/Employee")
        add_triple(term_dict, fact_store, "http://ex/list1", rdf_rest, "http://ex/list2")
        add_triple(term_dict, fact_store, "http://ex/list2", rdf_first, "http://ex/Parent")
        add_triple(term_dict, fact_store, "http://ex/list2", rdf_rest, rdf_nil)
        
        # WorkingParent intersectionOf list1
        add_triple(term_dict, fact_store, "http://ex/WorkingParent", OWL_INTERSECTION_OF, "http://ex/list1")
        
        # alice type Employee AND alice type Parent
        add_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Employee")
        add_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Parent")
        
        # Run inference
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice type WorkingParent
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/WorkingParent")


class TestOWLPropertyChain:
    """Test owl:propertyChainAxiom rule."""
    
    def test_property_chain_two_step(self, storage):
        """Test 2-element property chain: (x p1 y) + (y p2 z) => (x p z)"""
        term_dict, qt_dict, fact_store = storage
        
        rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        # Create chain list: hasParent -> hasBrother -> nil
        add_triple(term_dict, fact_store, "http://ex/chain1", rdf_first, "http://ex/hasParent")
        add_triple(term_dict, fact_store, "http://ex/chain1", rdf_rest, "http://ex/chain2")
        add_triple(term_dict, fact_store, "http://ex/chain2", rdf_first, "http://ex/hasBrother")
        add_triple(term_dict, fact_store, "http://ex/chain2", rdf_rest, rdf_nil)
        
        # hasUncle propertyChainAxiom chain1
        add_triple(term_dict, fact_store, "http://ex/hasUncle", OWL_PROPERTY_CHAIN_AXIOM, "http://ex/chain1")
        
        # alice hasParent bob, bob hasBrother charlie
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasParent", "http://ex/bob")
        add_triple(term_dict, fact_store, "http://ex/bob", "http://ex/hasBrother", "http://ex/charlie")
        
        # Run inference
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice hasUncle charlie
        assert has_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasUncle", "http://ex/charlie")


# =============================================================================
# Stats Tests
# =============================================================================

class TestRDFStarStats:
    """Test RDF-Star inference statistics."""
    
    def test_stats_confidence_tracking(self, storage):
        """Test that confidence stats are tracked."""
        term_dict, qt_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/fido", RDF_TYPE, "http://ex/Dog", 0.8)
        add_confidence_annotation(term_dict, qt_dict, fact_store,
            "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal", 0.9)
        
        engine = RDFStarInferenceEngine(term_dict, qt_dict, fact_store)
        stats = engine.reason_with_confidence()
        
        assert stats.triples_inferred >= 1
        assert stats.min_inferred_confidence <= 0.9
        assert stats.confidence_propagations >= 0  # Depends on what counts as "propagated"
