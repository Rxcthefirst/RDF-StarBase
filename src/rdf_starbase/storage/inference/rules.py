"""
Inference Rule Definitions and Registry.

Provides the base abstractions for inference rules:
- InferenceRule: A single rule with pattern matching
- RuleRegistry: Collection of rules by profile
- InferenceProfile: Named sets of rules (RDFS, OWL-RL, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple, Dict, Optional, Callable, Any
import polars as pl

from rdf_starbase.storage.terms import TermDict, TermId, Term, TermKind
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID


class InferenceProfile(str, Enum):
    """Standard inference profiles."""
    NONE = "none"
    RDFS = "rdfs"
    OWL_RL = "owl-rl"
    OWL_RL_FULL = "owl-rl-full"  # All OWL 2 RL rules
    CUSTOM = "custom"


@dataclass
class TriplePattern:
    """
    A pattern for matching triples in the fact store.
    
    Variables are represented as ?name strings.
    Constants are TermIds.
    """
    subject: str | TermId  # Variable (?x) or constant (TermId)
    predicate: str | TermId
    object_: str | TermId
    
    def is_variable(self, component: str | TermId) -> bool:
        return isinstance(component, str) and component.startswith("?")


@dataclass
class TripleTemplate:
    """
    Template for generating inferred triples.
    
    Variables reference bindings from pattern matching.
    """
    subject: str | TermId
    predicate: str | TermId
    object_: str | TermId


@dataclass
class InferenceRule:
    """
    Represents a single inference rule.
    
    A rule has:
    - Antecedent patterns: IF these patterns all match
    - Consequent template: THEN generate this triple
    """
    id: str
    name: str
    profile: InferenceProfile
    priority: int  # Lower = earlier execution
    description: str
    spec_reference: str = ""
    
    # Implementation details
    antecedent_patterns: List[TriplePattern] = field(default_factory=list)
    consequent_template: Optional[TripleTemplate] = None
    
    # For complex rules, a custom implementation function
    custom_impl: Optional[Callable] = None


@dataclass
class InferenceResult:
    """Result of applying an inference rule."""
    rule_id: str
    new_facts: List[Tuple[TermId, TermId, TermId, TermId]]  # (g, s, p, o)
    source_facts: List[Tuple[TermId, TermId, TermId, TermId]] = field(default_factory=list)


class RuleRegistry:
    """
    Registry of inference rules organized by profile.
    
    Provides rule lookup and profile-based filtering.
    """
    
    def __init__(self):
        self._rules: Dict[str, InferenceRule] = {}
        self._by_profile: Dict[InferenceProfile, List[str]] = {
            p: [] for p in InferenceProfile
        }
    
    def register(self, rule: InferenceRule) -> None:
        """Register a rule."""
        self._rules[rule.id] = rule
        self._by_profile[rule.profile].append(rule.id)
    
    def get(self, rule_id: str) -> Optional[InferenceRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def get_rules_for_profile(
        self,
        profile: InferenceProfile,
        include_lower: bool = True,
    ) -> List[InferenceRule]:
        """
        Get all rules for a profile.
        
        Args:
            profile: The target profile
            include_lower: Include rules from lower profiles
                          (e.g., OWL-RL includes RDFS)
        """
        rule_ids: Set[str] = set()
        
        if profile == InferenceProfile.NONE:
            return []
        
        # Always include RDFS for any non-NONE profile
        if include_lower and profile != InferenceProfile.RDFS:
            rule_ids.update(self._by_profile[InferenceProfile.RDFS])
        
        rule_ids.update(self._by_profile[profile])
        
        # Sort by priority
        rules = [self._rules[rid] for rid in rule_ids if rid in self._rules]
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def list_rules(self) -> List[InferenceRule]:
        """List all registered rules."""
        return list(self._rules.values())


# =============================================================================
# OWL 2 RL Vocabulary
# =============================================================================

# Namespaces
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

# RDF vocabulary
RDF_TYPE = RDF_NS + "type"
RDF_FIRST = RDF_NS + "first"
RDF_REST = RDF_NS + "rest"
RDF_NIL = RDF_NS + "nil"

# RDFS vocabulary
RDFS_SUBCLASS_OF = RDFS_NS + "subClassOf"
RDFS_SUBPROPERTY_OF = RDFS_NS + "subPropertyOf"
RDFS_DOMAIN = RDFS_NS + "domain"
RDFS_RANGE = RDFS_NS + "range"

# OWL vocabulary - existing
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

# OWL vocabulary - NEW (for full OWL 2 RL)
OWL_ALL_VALUES_FROM = OWL_NS + "allValuesFrom"
OWL_SOME_VALUES_FROM = OWL_NS + "someValuesFrom"
OWL_UNION_OF = OWL_NS + "unionOf"
OWL_INTERSECTION_OF = OWL_NS + "intersectionOf"
OWL_COMPLEMENT_OF = OWL_NS + "complementOf"
OWL_DISJOINT_WITH = OWL_NS + "disjointWith"
OWL_PROPERTY_DISJOINT_WITH = OWL_NS + "propertyDisjointWith"
OWL_DIFFERENT_FROM = OWL_NS + "differentFrom"
OWL_ALL_DIFFERENT = OWL_NS + "AllDifferent"
OWL_DISTINCT_MEMBERS = OWL_NS + "distinctMembers"
OWL_PROPERTY_CHAIN_AXIOM = OWL_NS + "propertyChainAxiom"
OWL_REFLEXIVE_PROPERTY = OWL_NS + "ReflexiveProperty"
OWL_IRREFLEXIVE_PROPERTY = OWL_NS + "IrreflexiveProperty"
OWL_ASYMMETRIC_PROPERTY = OWL_NS + "AsymmetricProperty"
OWL_NOTHING = OWL_NS + "Nothing"
OWL_THING = OWL_NS + "Thing"
OWL_MAX_CARDINALITY = OWL_NS + "maxCardinality"
OWL_QUALIFIED_CARDINALITY = OWL_NS + "qualifiedCardinality"
OWL_MAX_QUALIFIED_CARDINALITY = OWL_NS + "maxQualifiedCardinality"
OWL_ON_CLASS = OWL_NS + "onClass"
OWL_HAS_SELF = OWL_NS + "hasSelf"
OWL_MEMBERS = OWL_NS + "members"
OWL_ONE_OF = OWL_NS + "oneOf"
OWL_SOURCE_INDIVIDUAL = OWL_NS + "sourceIndividual"
OWL_ASSERTION_PROPERTY = OWL_NS + "assertionProperty"
OWL_TARGET_VALUE = OWL_NS + "targetValue"
OWL_TARGET_INDIVIDUAL = OWL_NS + "targetIndividual"
OWL_NEGATIVE_PROPERTY_ASSERTION = OWL_NS + "NegativePropertyAssertion"


# =============================================================================
# Helper Classes for Rule Application
# =============================================================================

class RuleApplicator:
    """
    Base class for applying inference rules to the fact store.
    
    Provides common utilities for pattern matching and fact generation.
    """
    
    def __init__(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
    ):
        self._term_dict = term_dict
        self._fact_store = fact_store
        self._vocab_cache: Dict[str, Optional[TermId]] = {}
    
    def get_vocab_id(self, iri: str, create: bool = False) -> Optional[TermId]:
        """Get or create a vocabulary term ID."""
        if iri in self._vocab_cache:
            return self._vocab_cache[iri]
        
        term = Term(kind=TermKind.IRI, lex=iri)
        if create:
            term_id = self._term_dict.get_or_create(term)
        else:
            term_id = self._term_dict.get_id(term)
        
        self._vocab_cache[iri] = term_id
        return term_id
    
    def get_facts_with_predicate(
        self,
        predicate_id: TermId,
        graph_id: TermId = DEFAULT_GRAPH_ID,
    ) -> List[Tuple[TermId, TermId]]:
        """Get all (subject, object) pairs for a predicate."""
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
    
    def get_all_non_deleted_facts(
        self,
        graph_id: TermId = DEFAULT_GRAPH_ID,
    ) -> pl.DataFrame:
        """Get all non-deleted facts in a graph."""
        df = self._fact_store.scan_facts()
        return df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
    
    def fact_exists(
        self,
        g: TermId,
        s: TermId,
        p: TermId,
        o: TermId,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
    ) -> bool:
        """Check if a fact already exists."""
        return (g, s, p, o) in existing
    
    def parse_rdf_list(
        self,
        head_id: TermId,
        graph_id: TermId = DEFAULT_GRAPH_ID,
    ) -> List[TermId]:
        """
        Parse an RDF list starting at head_id.
        
        Returns list of item TermIds.
        """
        first_id = self.get_vocab_id(RDF_FIRST)
        rest_id = self.get_vocab_id(RDF_REST)
        nil_id = self.get_vocab_id(RDF_NIL)
        
        if first_id is None or rest_id is None:
            return []
        
        items: List[TermId] = []
        current = head_id
        visited: Set[TermId] = set()  # Prevent infinite loops
        
        while current != nil_id and current not in visited:
            visited.add(current)
            
            # Get first element
            first_pairs = self._fact_store.scan_facts().filter(
                (pl.col("g") == graph_id) &
                (pl.col("s") == current) &
                (pl.col("p") == first_id)
            ).select(["o"]).to_series().to_list()
            
            if first_pairs:
                items.append(first_pairs[0])
            
            # Get rest pointer
            rest_pairs = self._fact_store.scan_facts().filter(
                (pl.col("g") == graph_id) &
                (pl.col("s") == current) &
                (pl.col("p") == rest_id)
            ).select(["o"]).to_series().to_list()
            
            if rest_pairs:
                current = rest_pairs[0]
            else:
                break
        
        return items


# Global registry instance
_global_registry = RuleRegistry()


def get_global_registry() -> RuleRegistry:
    """Get the global rule registry."""
    return _global_registry
