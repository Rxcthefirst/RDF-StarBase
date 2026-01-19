"""
Tests for RDFS and OWL Reasoning Engine.

Tests cover:
RDFS:
- Transitive subClassOf (rdfs11)
- Transitive subPropertyOf (rdfs5)
- Type inheritance via subClassOf (rdfs9)
- Property inheritance via subPropertyOf (rdfs7)
- Domain inference (rdfs2)
- Range inference (rdfs3)

OWL:
- owl:sameAs symmetry and transitivity
- owl:equivalentClass => mutual subClassOf
- owl:equivalentProperty => mutual subPropertyOf
- owl:inverseOf property inversion
- owl:TransitiveProperty transitive closure
- owl:SymmetricProperty symmetry
- owl:FunctionalProperty sameAs inference
- owl:InverseFunctionalProperty sameAs inference
- owl:hasValue restriction inference
"""

import pytest
from rdf_starbase.storage import (
    TermDict,
    FactStore,
    FactFlags,
    DEFAULT_GRAPH_ID,
    Term,
    TermKind,
)
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.reasoner import (
    RDFSReasoner,
    ReasoningStats,
    RDFS_SUBCLASS_OF,
    RDFS_SUBPROPERTY_OF,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDF_TYPE,
    OWL_SAME_AS,
    OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY,
    OWL_INVERSE_OF,
    OWL_TRANSITIVE_PROPERTY,
    OWL_SYMMETRIC_PROPERTY,
    OWL_FUNCTIONAL_PROPERTY,
    OWL_INVERSE_FUNCTIONAL_PROPERTY,
    OWL_HAS_VALUE,
    OWL_ON_PROPERTY,
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
    return term_dict, fact_store


def intern_iri(term_dict: TermDict, iri: str) -> int:
    """Helper to intern an IRI and return its ID."""
    term = Term(kind=TermKind.IRI, lex=iri)
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
# Test: Transitive subClassOf (RDFS11)
# =============================================================================

class TestRDFS11SubClassOfTransitivity:
    """Test transitive subClassOf inference."""
    
    def test_simple_transitive_subclass(self, storage):
        """Animal -> Mammal -> Dog => Animal -> Dog"""
        term_dict, fact_store = storage
        
        # Add: Dog subClassOf Mammal, Mammal subClassOf Animal
        add_triple(term_dict, fact_store, 
                   "http://example.org/Dog", RDFS_SUBCLASS_OF, "http://example.org/Mammal")
        add_triple(term_dict, fact_store,
                   "http://example.org/Mammal", RDFS_SUBCLASS_OF, "http://example.org/Animal")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: Dog subClassOf Animal
        assert has_triple(term_dict, fact_store,
                         "http://example.org/Dog", RDFS_SUBCLASS_OF, "http://example.org/Animal")
        assert stats.rdfs11_inferences >= 1
    
    def test_three_level_hierarchy(self, storage):
        """A -> B -> C -> D should produce A -> C, A -> D, B -> D"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/A", RDFS_SUBCLASS_OF, "http://ex/B")
        add_triple(term_dict, fact_store, "http://ex/B", RDFS_SUBCLASS_OF, "http://ex/C")
        add_triple(term_dict, fact_store, "http://ex/C", RDFS_SUBCLASS_OF, "http://ex/D")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Check transitive inferences
        assert has_triple(term_dict, fact_store, "http://ex/A", RDFS_SUBCLASS_OF, "http://ex/C")
        assert has_triple(term_dict, fact_store, "http://ex/A", RDFS_SUBCLASS_OF, "http://ex/D")
        assert has_triple(term_dict, fact_store, "http://ex/B", RDFS_SUBCLASS_OF, "http://ex/D")
        assert stats.iterations >= 1


# =============================================================================
# Test: Transitive subPropertyOf (RDFS5)
# =============================================================================

class TestRDFS5SubPropertyOfTransitivity:
    """Test transitive subPropertyOf inference."""
    
    def test_simple_transitive_subproperty(self, storage):
        """hasFather subProp hasParent, hasParent subProp hasAncestor"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store,
                   "http://ex/hasFather", RDFS_SUBPROPERTY_OF, "http://ex/hasParent")
        add_triple(term_dict, fact_store,
                   "http://ex/hasParent", RDFS_SUBPROPERTY_OF, "http://ex/hasAncestor")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: hasFather subPropertyOf hasAncestor
        assert has_triple(term_dict, fact_store,
                         "http://ex/hasFather", RDFS_SUBPROPERTY_OF, "http://ex/hasAncestor")
        assert stats.rdfs5_inferences >= 1


# =============================================================================
# Test: Type inheritance via subClassOf (RDFS9)
# =============================================================================

class TestRDFS9TypeInheritance:
    """Test type inheritance through subClassOf."""
    
    def test_type_inherits_superclass(self, storage):
        """fido rdf:type Dog, Dog subClassOf Animal => fido rdf:type Animal"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store,
                   "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store,
                   "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: fido rdf:type Animal
        assert has_triple(term_dict, fact_store,
                         "http://ex/fido", RDF_TYPE, "http://ex/Animal")
        assert stats.rdfs9_inferences >= 1
    
    def test_type_inherits_multiple_levels(self, storage):
        """fido type Dog, Dog subClass Mammal subClass Animal"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Mammal")
        add_triple(term_dict, fact_store, "http://ex/Mammal", RDFS_SUBCLASS_OF, "http://ex/Animal")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: fido rdf:type Mammal AND fido rdf:type Animal
        assert has_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Mammal")
        assert has_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Animal")


# =============================================================================
# Test: Property inheritance via subPropertyOf (RDFS7)
# =============================================================================

class TestRDFS7PropertyInheritance:
    """Test property inheritance through subPropertyOf."""
    
    def test_property_inherits_superproperty(self, storage):
        """alice hasFather bob, hasFather subProp hasParent => alice hasParent bob"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store,
                   "http://ex/alice", "http://ex/hasFather", "http://ex/bob")
        add_triple(term_dict, fact_store,
                   "http://ex/hasFather", RDFS_SUBPROPERTY_OF, "http://ex/hasParent")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice hasParent bob
        assert has_triple(term_dict, fact_store,
                         "http://ex/alice", "http://ex/hasParent", "http://ex/bob")
        assert stats.rdfs7_inferences >= 1


# =============================================================================
# Test: Domain inference (RDFS2)
# =============================================================================

class TestRDFS2DomainInference:
    """Test domain inference."""
    
    def test_domain_inference(self, storage):
        """alice worksAt acme, worksAt domain Person => alice type Person"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store,
                   "http://ex/alice", "http://ex/worksAt", "http://ex/acme")
        add_triple(term_dict, fact_store,
                   "http://ex/worksAt", RDFS_DOMAIN, "http://ex/Person")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice rdf:type Person
        assert has_triple(term_dict, fact_store,
                         "http://ex/alice", RDF_TYPE, "http://ex/Person")
        assert stats.rdfs2_inferences >= 1


# =============================================================================
# Test: Range inference (RDFS3)
# =============================================================================

class TestRDFS3RangeInference:
    """Test range inference."""
    
    def test_range_inference(self, storage):
        """alice worksAt acme, worksAt range Organization => acme type Organization"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store,
                   "http://ex/alice", "http://ex/worksAt", "http://ex/acme")
        add_triple(term_dict, fact_store,
                   "http://ex/worksAt", RDFS_RANGE, "http://ex/Organization")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: acme rdf:type Organization
        assert has_triple(term_dict, fact_store,
                         "http://ex/acme", RDF_TYPE, "http://ex/Organization")
        assert stats.rdfs3_inferences >= 1


# =============================================================================
# Test: Combined reasoning scenarios
# =============================================================================

class TestCombinedReasoning:
    """Test combined reasoning scenarios."""
    
    def test_domain_with_subclass_inheritance(self, storage):
        """
        worksAt domain Person, Person subClassOf Agent
        alice worksAt acme => alice type Person, alice type Agent
        """
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/worksAt", RDFS_DOMAIN, "http://ex/Person")
        add_triple(term_dict, fact_store, "http://ex/Person", RDFS_SUBCLASS_OF, "http://ex/Agent")
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/worksAt", "http://ex/acme")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer both: alice type Person AND alice type Agent
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Person")
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Agent")
        assert stats.triples_inferred >= 2
    
    def test_property_chain_with_subproperty(self, storage):
        """
        hasFather subProp hasParent, hasParent subProp hasAncestor
        alice hasFather bob => alice hasParent bob, alice hasAncestor bob
        """
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, 
                   "http://ex/hasFather", RDFS_SUBPROPERTY_OF, "http://ex/hasParent")
        add_triple(term_dict, fact_store,
                   "http://ex/hasParent", RDFS_SUBPROPERTY_OF, "http://ex/hasAncestor")
        add_triple(term_dict, fact_store,
                   "http://ex/alice", "http://ex/hasFather", "http://ex/bob")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should infer: alice hasParent bob AND alice hasAncestor bob
        assert has_triple(term_dict, fact_store,
                         "http://ex/alice", "http://ex/hasParent", "http://ex/bob")
        assert has_triple(term_dict, fact_store,
                         "http://ex/alice", "http://ex/hasAncestor", "http://ex/bob")
    
    def test_no_duplicate_inferences(self, storage):
        """Reasoning should not create duplicate facts."""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        
        # Run reasoning twice
        stats1 = reasoner.reason()
        stats2 = reasoner.reason()
        
        # Second run should find nothing new
        assert stats2.triples_inferred == 0
    
    def test_stats_tracking(self, storage):
        """Stats should accurately track inference counts."""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Mammal")
        add_triple(term_dict, fact_store, "http://ex/Mammal", RDFS_SUBCLASS_OF, "http://ex/Animal")
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        add_triple(term_dict, fact_store, "http://ex/worksAt", RDFS_DOMAIN, "http://ex/Person")
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/worksAt", "http://ex/acme")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert stats.iterations >= 1
        assert stats.triples_inferred > 0
        assert stats.rdfs11_inferences >= 1  # Dog subClass Animal
        assert stats.rdfs9_inferences >= 1   # fido type Mammal/Animal
        assert stats.rdfs2_inferences >= 1   # alice type Person
    
    def test_inferred_count(self, storage):
        """get_inferred_count should return correct count."""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/Dog", RDFS_SUBCLASS_OF, "http://ex/Animal")
        add_triple(term_dict, fact_store, "http://ex/fido", RDF_TYPE, "http://ex/Dog")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        
        # Before reasoning
        assert reasoner.get_inferred_count() == 0
        
        # After reasoning
        reasoner.reason()
        assert reasoner.get_inferred_count() >= 1


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_no_rdfs_vocabulary(self, storage):
        """Reasoning with no RDFS vocabulary should do nothing."""
        term_dict, fact_store = storage
        
        # Add non-RDFS triples
        add_triple(term_dict, fact_store, "http://ex/a", "http://ex/b", "http://ex/c")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert stats.triples_inferred == 0
        assert stats.iterations == 0
    
    def test_empty_store(self, storage):
        """Reasoning on empty store should do nothing."""
        term_dict, fact_store = storage
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert stats.triples_inferred == 0
    
    def test_self_loop_prevention(self, storage):
        """Should not infer A subClassOf A."""
        term_dict, fact_store = storage
        
        # This shouldn't cause issues even though it's technically invalid
        add_triple(term_dict, fact_store, "http://ex/A", RDFS_SUBCLASS_OF, "http://ex/A")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Should complete without errors
        assert stats.iterations >= 1


# =============================================================================
# OWL Tests
# =============================================================================

class TestOWLSameAs:
    """Test owl:sameAs reasoning."""
    
    def test_same_as_symmetry(self, storage):
        """(x sameAs y) => (y sameAs x)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/alice", OWL_SAME_AS, "http://ex/alice2")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/alice2", OWL_SAME_AS, "http://ex/alice")
        assert stats.owl_same_as_inferences >= 1
    
    def test_same_as_transitivity(self, storage):
        """(x sameAs y) + (y sameAs z) => (x sameAs z)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/a", OWL_SAME_AS, "http://ex/b")
        add_triple(term_dict, fact_store, "http://ex/b", OWL_SAME_AS, "http://ex/c")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/a", OWL_SAME_AS, "http://ex/c")


class TestOWLEquivalentClass:
    """Test owl:equivalentClass reasoning."""
    
    def test_equivalent_class_to_subclass(self, storage):
        """(C1 equivalentClass C2) => (C1 subClassOf C2) + (C2 subClassOf C1)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/Person", OWL_EQUIVALENT_CLASS, "http://ex/Human")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/Person", RDFS_SUBCLASS_OF, "http://ex/Human")
        assert has_triple(term_dict, fact_store, "http://ex/Human", RDFS_SUBCLASS_OF, "http://ex/Person")
        assert stats.owl_equivalent_class_inferences >= 2


class TestOWLEquivalentProperty:
    """Test owl:equivalentProperty reasoning."""
    
    def test_equivalent_property_to_subproperty(self, storage):
        """(p1 equivalentProperty p2) => (p1 subPropertyOf p2) + (p2 subPropertyOf p1)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/likes", OWL_EQUIVALENT_PROPERTY, "http://ex/enjoys")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/likes", RDFS_SUBPROPERTY_OF, "http://ex/enjoys")
        assert has_triple(term_dict, fact_store, "http://ex/enjoys", RDFS_SUBPROPERTY_OF, "http://ex/likes")
        assert stats.owl_equivalent_property_inferences >= 2


class TestOWLInverseOf:
    """Test owl:inverseOf reasoning."""
    
    def test_inverse_of(self, storage):
        """(x p y) + (p inverseOf q) => (y q x)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/hasParent", OWL_INVERSE_OF, "http://ex/hasChild")
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasParent", "http://ex/bob")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/bob", "http://ex/hasChild", "http://ex/alice")
        assert stats.owl_inverse_of_inferences >= 1
    
    def test_inverse_of_bidirectional(self, storage):
        """inverseOf is symmetric: (p inverseOf q) + (x q y) => (y p x)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/hasParent", OWL_INVERSE_OF, "http://ex/hasChild")
        add_triple(term_dict, fact_store, "http://ex/bob", "http://ex/hasChild", "http://ex/alice")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasParent", "http://ex/bob")


class TestOWLTransitiveProperty:
    """Test owl:TransitiveProperty reasoning."""
    
    def test_transitive_property(self, storage):
        """(p type TransitiveProperty) + (x p y) + (y p z) => (x p z)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/ancestorOf", RDF_TYPE, OWL_TRANSITIVE_PROPERTY)
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/ancestorOf", "http://ex/bob")
        add_triple(term_dict, fact_store, "http://ex/bob", "http://ex/ancestorOf", "http://ex/charlie")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/alice", "http://ex/ancestorOf", "http://ex/charlie")
        assert stats.owl_transitive_inferences >= 1


class TestOWLSymmetricProperty:
    """Test owl:SymmetricProperty reasoning."""
    
    def test_symmetric_property(self, storage):
        """(p type SymmetricProperty) + (x p y) => (y p x)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/friendOf", RDF_TYPE, OWL_SYMMETRIC_PROPERTY)
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/friendOf", "http://ex/bob")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/bob", "http://ex/friendOf", "http://ex/alice")
        assert stats.owl_symmetric_inferences >= 1


class TestOWLFunctionalProperty:
    """Test owl:FunctionalProperty reasoning."""
    
    def test_functional_property(self, storage):
        """(p type FunctionalProperty) + (x p y1) + (x p y2) => (y1 sameAs y2)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/hasMother", RDF_TYPE, OWL_FUNCTIONAL_PROPERTY)
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasMother", "http://ex/mary")
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasMother", "http://ex/maria")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # mary and maria must be the same person
        assert has_triple(term_dict, fact_store, "http://ex/mary", OWL_SAME_AS, "http://ex/maria")
        assert stats.owl_functional_inferences >= 1


class TestOWLInverseFunctionalProperty:
    """Test owl:InverseFunctionalProperty reasoning."""
    
    def test_inverse_functional_property(self, storage):
        """(p type InverseFunctionalProperty) + (x1 p y) + (x2 p y) => (x1 sameAs x2)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/hasSSN", RDF_TYPE, OWL_INVERSE_FUNCTIONAL_PROPERTY)
        add_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasSSN", "http://ex/ssn123")
        add_triple(term_dict, fact_store, "http://ex/alice2", "http://ex/hasSSN", "http://ex/ssn123")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # alice and alice2 must be the same person
        assert has_triple(term_dict, fact_store, "http://ex/alice", OWL_SAME_AS, "http://ex/alice2")
        assert stats.owl_inverse_functional_inferences >= 1


class TestOWLHasValue:
    """Test owl:hasValue restriction reasoning."""
    
    def test_has_value_forward(self, storage):
        """(C onProperty p) + (C hasValue v) + (x type C) => (x p v)"""
        term_dict, fact_store = storage
        
        # Define a restriction class: "things that work at ACME"
        add_triple(term_dict, fact_store, "http://ex/ACMEEmployee", OWL_ON_PROPERTY, "http://ex/worksAt")
        add_triple(term_dict, fact_store, "http://ex/ACMEEmployee", OWL_HAS_VALUE, "http://ex/ACME")
        add_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/ACMEEmployee")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/alice", "http://ex/worksAt", "http://ex/ACME")
        assert stats.owl_has_value_inferences >= 1
    
    def test_has_value_backward(self, storage):
        """(C onProperty p) + (C hasValue v) + (x p v) => (x type C)"""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/RedThing", OWL_ON_PROPERTY, "http://ex/hasColor")
        add_triple(term_dict, fact_store, "http://ex/RedThing", OWL_HAS_VALUE, "http://ex/red")
        add_triple(term_dict, fact_store, "http://ex/apple", "http://ex/hasColor", "http://ex/red")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        assert has_triple(term_dict, fact_store, "http://ex/apple", RDF_TYPE, "http://ex/RedThing")


class TestOWLDisableOption:
    """Test that OWL reasoning can be disabled."""
    
    def test_disable_owl_reasoning(self, storage):
        """When enable_owl=False, OWL rules should not fire."""
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/alice", OWL_SAME_AS, "http://ex/alice2")
        
        reasoner = RDFSReasoner(term_dict, fact_store, enable_owl=False)
        stats = reasoner.reason()
        
        # Symmetry should NOT be inferred
        assert not has_triple(term_dict, fact_store, "http://ex/alice2", OWL_SAME_AS, "http://ex/alice")
        assert stats.owl_same_as_inferences == 0


class TestOWLCombined:
    """Test combined RDFS + OWL reasoning."""
    
    def test_equivalent_class_enables_type_inheritance(self, storage):
        """
        equivalentClass + type should enable full type inference.
        
        Person equivalentClass Human, Human subClassOf Agent
        alice type Person => alice type Human, alice type Agent
        """
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/Person", OWL_EQUIVALENT_CLASS, "http://ex/Human")
        add_triple(term_dict, fact_store, "http://ex/Human", RDFS_SUBCLASS_OF, "http://ex/Agent")
        add_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Person")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Person equivClass Human means alice is also type Human
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Human")
        # And Human subClassOf Agent means alice is type Agent
        assert has_triple(term_dict, fact_store, "http://ex/alice", RDF_TYPE, "http://ex/Agent")
    
    def test_inverse_with_transitive(self, storage):
        """
        Combine inverseOf with TransitiveProperty.
        
        hasAncestor inverseOf hasDescendant, hasAncestor is transitive
        """
        term_dict, fact_store = storage
        
        add_triple(term_dict, fact_store, "http://ex/hasAncestor", OWL_INVERSE_OF, "http://ex/hasDescendant")
        add_triple(term_dict, fact_store, "http://ex/hasAncestor", RDF_TYPE, OWL_TRANSITIVE_PROPERTY)
        add_triple(term_dict, fact_store, "http://ex/charlie", "http://ex/hasAncestor", "http://ex/bob")
        add_triple(term_dict, fact_store, "http://ex/bob", "http://ex/hasAncestor", "http://ex/alice")
        
        reasoner = RDFSReasoner(term_dict, fact_store)
        stats = reasoner.reason()
        
        # Transitive: charlie hasAncestor alice
        assert has_triple(term_dict, fact_store, "http://ex/charlie", "http://ex/hasAncestor", "http://ex/alice")
        # Inverse: alice hasDescendant charlie
        assert has_triple(term_dict, fact_store, "http://ex/alice", "http://ex/hasDescendant", "http://ex/charlie")
