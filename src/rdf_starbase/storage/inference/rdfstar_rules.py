"""
RDF-Star Specific Inference Rules.

Extends the base reasoner with RDF-Star specific capabilities:
1. Confidence propagation - Combine confidences for inferred facts
2. Provenance tracking - Track which rules and facts produced inferences
3. Annotation inheritance - Propagate metadata through inference chains

This is a key differentiator for RDF-StarBase.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Set, Tuple, Dict, Optional, Any
import polars as pl

from rdf_starbase.storage.terms import TermDict, TermId, Term, TermKind
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.inference.rules import (
    RuleApplicator,
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    RDFS_SUBPROPERTY_OF,
)


class ConfidenceMethod(str, Enum):
    """Methods for combining confidence values in inference."""
    MIN = "min"           # Pessimistic: min(c1, c2, ...)
    PRODUCT = "product"   # Probabilistic: c1 * c2 * ...
    AVERAGE = "average"   # Mean: (c1 + c2 + ...) / n
    WEIGHTED = "weighted" # Weighted by rule reliability
    MAX = "max"           # Optimistic: max(c1, c2, ...)


@dataclass
class InferenceJustification:
    """
    Tracks why a fact was inferred.
    
    This enables explanation queries and confidence propagation.
    """
    rule_id: str
    rule_name: str
    antecedent_facts: List[Tuple[TermId, TermId, TermId, TermId]]  # (g, s, p, o)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    iteration: int = 0


@dataclass
class RDFStarReasoningStats:
    """Extended stats for RDF-Star aware reasoning."""
    iterations: int = 0
    triples_inferred: int = 0
    
    # Standard RDFS stats
    rdfs2_inferences: int = 0  # domain
    rdfs3_inferences: int = 0  # range
    rdfs5_inferences: int = 0  # subPropertyOf transitivity
    rdfs7_inferences: int = 0  # property inheritance
    rdfs9_inferences: int = 0  # type inheritance
    rdfs11_inferences: int = 0 # subClassOf transitivity
    
    # OWL stats
    owl_same_as_inferences: int = 0
    owl_equivalent_class_inferences: int = 0
    owl_equivalent_property_inferences: int = 0
    owl_inverse_of_inferences: int = 0
    owl_transitive_inferences: int = 0
    owl_symmetric_inferences: int = 0
    owl_functional_inferences: int = 0
    owl_inverse_functional_inferences: int = 0
    owl_has_value_inferences: int = 0
    owl_all_values_from_inferences: int = 0
    owl_some_values_from_inferences: int = 0
    owl_union_inferences: int = 0
    owl_intersection_inferences: int = 0
    owl_property_chain_inferences: int = 0
    owl_disjoint_violations: int = 0
    
    # RDF-Star specific
    confidence_propagations: int = 0
    provenance_annotations: int = 0
    min_inferred_confidence: float = 1.0
    avg_inferred_confidence: float = 1.0


class RDFStarInferenceEngine:
    """
    RDF-Star aware inference engine.
    
    Extends basic RDFS/OWL reasoning with:
    - Confidence propagation: Inferred facts get combined confidence
    - Provenance tracking: Each inference tracks its justification
    - Annotation as RDF-Star: Justifications stored as << >>
    
    Example:
        <<:Alice :type :Person>> :confidence 0.9 .
        <<:Person :subClassOf :Agent>> :confidence 0.95 .
        
        After inference:
        <<:Alice :type :Agent>> 
            :confidence 0.855 ;  # 0.9 * 0.95 (product method)
            :inferredBy "rdfs9" ;
            :basedOn <<:Alice :type :Person>>, <<:Person :subClassOf :Agent>> .
    """
    
    # Namespace for inference annotations
    INFERENCE_NS = "http://rdfstarbase.io/inference#"
    CONFIDENCE_PRED = INFERENCE_NS + "confidence"
    INFERRED_BY_PRED = INFERENCE_NS + "inferredBy"
    BASED_ON_PRED = INFERENCE_NS + "basedOn"
    INFERRED_AT_PRED = INFERENCE_NS + "inferredAt"
    ITERATION_PRED = INFERENCE_NS + "iteration"
    
    def __init__(
        self,
        term_dict: TermDict,
        qt_dict: QtDict,
        fact_store: FactStore,
        confidence_method: ConfidenceMethod = ConfidenceMethod.MIN,
        track_provenance: bool = True,
        max_iterations: int = 100,
    ):
        self._term_dict = term_dict
        self._qt_dict = qt_dict
        self._fact_store = fact_store
        self._confidence_method = confidence_method
        self._track_provenance = track_provenance
        self._max_iterations = max_iterations
        
        # Applicator for common operations
        self._applicator = RuleApplicator(term_dict, fact_store)
        
        # Cache vocabulary IDs
        self._vocab_ids: Dict[str, TermId] = {}
        
        # Justification storage
        self._justifications: Dict[Tuple[TermId, TermId, TermId, TermId], InferenceJustification] = {}
        
        # Confidence cache: fact -> confidence
        self._confidence_cache: Dict[Tuple[TermId, TermId, TermId, TermId], float] = {}
    
    def _ensure_vocab_ids(self) -> Dict[str, Optional[TermId]]:
        """Ensure vocabulary term IDs are cached."""
        if self._vocab_ids:
            return self._vocab_ids
        
        vocab_terms = [
            (RDFS_SUBCLASS_OF, "subClassOf"),
            (RDFS_SUBPROPERTY_OF, "subPropertyOf"),
            (RDF_TYPE, "type"),
            (self.CONFIDENCE_PRED, "confidence"),
            (self.INFERRED_BY_PRED, "inferredBy"),
            (self.BASED_ON_PRED, "basedOn"),
            (self.INFERRED_AT_PRED, "inferredAt"),
        ]
        
        for iri, name in vocab_terms:
            term = Term(kind=TermKind.IRI, lex=iri)
            term_id = self._term_dict.get_id(term)
            if term_id is not None:
                self._vocab_ids[name] = term_id
        
        return self._vocab_ids
    
    def _get_or_create_vocab_id(self, iri: str) -> TermId:
        """Get or create a vocabulary term ID."""
        term = Term(kind=TermKind.IRI, lex=iri)
        return self._term_dict.get_or_create(term)
    
    def _build_confidence_cache(self, graph_id: TermId = DEFAULT_GRAPH_ID) -> None:
        """
        Build cache of confidence values for facts.
        
        Looks for annotations like:
        <<s p o>> :confidence 0.9 .
        """
        confidence_pred_id = self._get_or_create_vocab_id(self.CONFIDENCE_PRED)
        
        # Get all facts
        df = self._fact_store.scan_facts()
        facts = df.filter(
            (pl.col("g") == graph_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        # Look for confidence annotations
        # Subject should be a quoted triple ID
        confidence_facts = facts.filter(
            pl.col("p") == confidence_pred_id
        )
        
        for row in confidence_facts.iter_rows(named=True):
            qt_id = row["s"]
            conf_id = row["o"]
            
            # Check if subject is a quoted triple
            from rdf_starbase.storage.terms import get_term_kind, TermKind as TK
            if get_term_kind(qt_id) == TK.QUOTED_TRIPLE:
                # Get the quoted triple components
                qt = self._qt_dict.lookup(qt_id)
                if qt:
                    # QuotedTriple has .s, .p, .o attributes
                    s, p, o = qt.s, qt.p, qt.o
                    # Get confidence value from term dict
                    conf_term = self._term_dict.lookup(conf_id)
                    if conf_term and conf_term.kind == TK.LITERAL:
                        try:
                            confidence = float(conf_term.lex)
                            self._confidence_cache[(graph_id, s, p, o)] = confidence
                        except ValueError:
                            pass
        
        # Default confidence of 1.0 for asserted facts without explicit confidence
        for row in facts.iter_rows(named=True):
            key = (row["g"], row["s"], row["p"], row["o"])
            if key not in self._confidence_cache:
                # Asserted facts without annotation have confidence 1.0
                if row["flags"] & int(FactFlags.ASSERTED):
                    self._confidence_cache[key] = 1.0
    
    def combine_confidence(self, confidences: List[float]) -> float:
        """
        Combine multiple confidence values using the configured method.
        
        Args:
            confidences: List of confidence values from antecedent facts
            
        Returns:
            Combined confidence for the inferred fact
        """
        if not confidences:
            return 1.0
        
        confidences = [c for c in confidences if c is not None]
        if not confidences:
            return 1.0
        
        if self._confidence_method == ConfidenceMethod.MIN:
            return min(confidences)
        elif self._confidence_method == ConfidenceMethod.PRODUCT:
            return math.prod(confidences)
        elif self._confidence_method == ConfidenceMethod.AVERAGE:
            return sum(confidences) / len(confidences)
        elif self._confidence_method == ConfidenceMethod.MAX:
            return max(confidences)
        elif self._confidence_method == ConfidenceMethod.WEIGHTED:
            # TODO: Implement weighted combination based on rule reliability
            return min(confidences)
        else:
            return min(confidences)
    
    def get_fact_confidence(
        self,
        g: TermId,
        s: TermId,
        p: TermId,
        o: TermId,
    ) -> float:
        """Get the confidence for a fact (1.0 if not annotated)."""
        key = (g, s, p, o)
        return self._confidence_cache.get(key, 1.0)
    
    def _store_inference_annotation(
        self,
        inferred_fact: Tuple[TermId, TermId, TermId, TermId],
        justification: InferenceJustification,
        graph_id: TermId = DEFAULT_GRAPH_ID,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        Store RDF-Star annotations for an inferred fact.
        
        Creates:
        <<s p o>> :confidence 0.9 .
        <<s p o>> :inferredBy "rule_id" .
        <<s p o>> :inferredAt "timestamp" .
        
        Returns the annotation facts created.
        """
        if not self._track_provenance:
            return []
        
        g, s, p, o = inferred_fact
        annotation_facts = []
        
        # Create quoted triple for the inferred fact
        qt_id = self._qt_dict.get_or_create(s, p, o)
        
        # Confidence annotation
        conf_pred_id = self._get_or_create_vocab_id(self.CONFIDENCE_PRED)
        conf_term = Term(
            kind=TermKind.LITERAL,
            lex=str(justification.confidence),
            datatype_id=self._term_dict.get_or_create(
                Term(kind=TermKind.IRI, lex="http://www.w3.org/2001/XMLSchema#decimal")
            ),
        )
        conf_obj_id = self._term_dict.get_or_create(conf_term)
        annotation_facts.append((g, qt_id, conf_pred_id, conf_obj_id))
        
        # Inferred-by annotation
        inferred_by_pred_id = self._get_or_create_vocab_id(self.INFERRED_BY_PRED)
        rule_term = Term(kind=TermKind.LITERAL, lex=justification.rule_id)
        rule_obj_id = self._term_dict.get_or_create(rule_term)
        annotation_facts.append((g, qt_id, inferred_by_pred_id, rule_obj_id))
        
        # Timestamp annotation
        inferred_at_pred_id = self._get_or_create_vocab_id(self.INFERRED_AT_PRED)
        ts_term = Term(
            kind=TermKind.LITERAL,
            lex=justification.timestamp.isoformat(),
            datatype_id=self._term_dict.get_or_create(
                Term(kind=TermKind.IRI, lex="http://www.w3.org/2001/XMLSchema#dateTime")
            ),
        )
        ts_obj_id = self._term_dict.get_or_create(ts_term)
        annotation_facts.append((g, qt_id, inferred_at_pred_id, ts_obj_id))
        
        # Based-on annotations (link to antecedent facts as quoted triples)
        based_on_pred_id = self._get_or_create_vocab_id(self.BASED_ON_PRED)
        for ant_g, ant_s, ant_p, ant_o in justification.antecedent_facts:
            ant_qt_id = self._qt_dict.get_or_create(ant_s, ant_p, ant_o)
            annotation_facts.append((g, qt_id, based_on_pred_id, ant_qt_id))
        
        return annotation_facts
    
    def reason_with_confidence(
        self,
        graph_id: TermId = DEFAULT_GRAPH_ID,
    ) -> RDFStarReasoningStats:
        """
        Run inference with confidence propagation and provenance tracking.
        
        This wraps the basic RDFS/OWL reasoning and adds:
        1. Reads existing confidence annotations
        2. Computes combined confidence for inferred facts
        3. Stores justification annotations
        
        Returns:
            Extended stats including confidence information
        """
        from rdf_starbase.storage.reasoner import RDFSReasoner
        
        stats = RDFStarReasoningStats()
        
        # Build confidence cache from existing annotations
        self._build_confidence_cache(graph_id)
        
        # Track existing facts for duplicate detection
        existing_facts: Set[Tuple[TermId, TermId, TermId, TermId]] = set()
        df = self._fact_store.scan_facts()
        for row in df.iter_rows(named=True):
            existing_facts.add((row["g"], row["s"], row["p"], row["o"]))
        
        # Fixed-point iteration with confidence tracking
        all_annotation_facts: List[Tuple[TermId, TermId, TermId, TermId]] = []
        confidence_values: List[float] = []
        
        for iteration in range(self._max_iterations):
            stats.iterations = iteration + 1
            new_facts: List[Tuple[TermId, TermId, TermId, TermId]] = []
            new_annotations: List[Tuple[TermId, TermId, TermId, TermId]] = []
            
            # Apply each rule family with confidence tracking
            new_facts.extend(self._apply_rdfs9_with_confidence(
                existing_facts, graph_id, stats, new_annotations, iteration
            ))
            new_facts.extend(self._apply_rdfs11_with_confidence(
                existing_facts, graph_id, stats, new_annotations, iteration
            ))
            new_facts.extend(self._apply_rdfs7_with_confidence(
                existing_facts, graph_id, stats, new_annotations, iteration
            ))
            
            if not new_facts:
                # Fixed point reached
                break
            
            # Add new facts to store
            self._fact_store.add_facts_batch(
                new_facts,
                flags=FactFlags.INFERRED,
            )
            
            # Add annotation facts
            if new_annotations:
                self._fact_store.add_facts_batch(
                    new_annotations,
                    flags=FactFlags.ASSERTED | FactFlags.METADATA,
                )
                stats.provenance_annotations += len(new_annotations)
            
            # Update tracking
            for fact in new_facts:
                existing_facts.add(fact)
                conf = self._confidence_cache.get(fact, 1.0)
                confidence_values.append(conf)
            
            stats.triples_inferred += len(new_facts)
        
        # Compute confidence stats
        if confidence_values:
            stats.min_inferred_confidence = min(confidence_values)
            stats.avg_inferred_confidence = sum(confidence_values) / len(confidence_values)
            stats.confidence_propagations = len([c for c in confidence_values if c < 1.0])
        
        return stats
    
    def _apply_rdfs9_with_confidence(
        self,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: RDFStarReasoningStats,
        annotations: List[Tuple[TermId, TermId, TermId, TermId]],
        iteration: int,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS9 with confidence: (x type C1) + (C1 subClassOf C2) => (x type C2)
        
        Combined confidence = combine(conf(x type C1), conf(C1 subClassOf C2))
        """
        type_id = self._applicator.get_vocab_id(RDF_TYPE)
        subclass_id = self._applicator.get_vocab_id(RDFS_SUBCLASS_OF)
        
        if type_id is None or subclass_id is None:
            return []
        
        # Get assertions
        type_pairs = self._applicator.get_facts_with_predicate(type_id, graph_id)
        subclass_pairs = self._applicator.get_facts_with_predicate(subclass_id, graph_id)
        
        if not type_pairs or not subclass_pairs:
            return []
        
        # Build subClassOf map
        subclass_of: Dict[TermId, Set[TermId]] = {}
        for c1, c2 in subclass_pairs:
            if c1 not in subclass_of:
                subclass_of[c1] = set()
            subclass_of[c1].add(c2)
        
        new_facts = []
        for x, c1 in type_pairs:
            if c1 in subclass_of:
                for c2 in subclass_of[c1]:
                    fact = (graph_id, x, type_id, c2)
                    if fact not in existing:
                        new_facts.append(fact)
                        stats.rdfs9_inferences += 1
                        
                        # Compute combined confidence
                        conf1 = self.get_fact_confidence(graph_id, x, type_id, c1)
                        conf2 = self.get_fact_confidence(graph_id, c1, subclass_id, c2)
                        combined = self.combine_confidence([conf1, conf2])
                        self._confidence_cache[fact] = combined
                        
                        # Create justification
                        justification = InferenceJustification(
                            rule_id="rdfs9",
                            rule_name="Type inheritance via subClassOf",
                            antecedent_facts=[
                                (graph_id, x, type_id, c1),
                                (graph_id, c1, subclass_id, c2),
                            ],
                            confidence=combined,
                            iteration=iteration,
                        )
                        self._justifications[fact] = justification
                        
                        # Store annotations
                        ann_facts = self._store_inference_annotation(fact, justification, graph_id)
                        annotations.extend(ann_facts)
        
        return new_facts
    
    def _apply_rdfs11_with_confidence(
        self,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: RDFStarReasoningStats,
        annotations: List[Tuple[TermId, TermId, TermId, TermId]],
        iteration: int,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS11 with confidence: Transitive subClassOf.
        """
        subclass_id = self._applicator.get_vocab_id(RDFS_SUBCLASS_OF)
        if subclass_id is None:
            return []
        
        subclass_pairs = self._applicator.get_facts_with_predicate(subclass_id, graph_id)
        if not subclass_pairs:
            return []
        
        # Build adjacency
        subclass_of: Dict[TermId, Set[TermId]] = {}
        for c1, c2 in subclass_pairs:
            if c1 not in subclass_of:
                subclass_of[c1] = set()
            subclass_of[c1].add(c2)
        
        new_facts = []
        for c1, direct_supers in subclass_of.items():
            for c2 in list(direct_supers):
                if c2 in subclass_of:
                    for c3 in subclass_of[c2]:
                        fact = (graph_id, c1, subclass_id, c3)
                        if fact not in existing and c1 != c3:
                            new_facts.append(fact)
                            stats.rdfs11_inferences += 1
                            
                            # Confidence
                            conf1 = self.get_fact_confidence(graph_id, c1, subclass_id, c2)
                            conf2 = self.get_fact_confidence(graph_id, c2, subclass_id, c3)
                            combined = self.combine_confidence([conf1, conf2])
                            self._confidence_cache[fact] = combined
                            
                            justification = InferenceJustification(
                                rule_id="rdfs11",
                                rule_name="Transitive subClassOf",
                                antecedent_facts=[
                                    (graph_id, c1, subclass_id, c2),
                                    (graph_id, c2, subclass_id, c3),
                                ],
                                confidence=combined,
                                iteration=iteration,
                            )
                            self._justifications[fact] = justification
                            annotations.extend(
                                self._store_inference_annotation(fact, justification, graph_id)
                            )
        
        return new_facts
    
    def _apply_rdfs7_with_confidence(
        self,
        existing: Set[Tuple[TermId, TermId, TermId, TermId]],
        graph_id: TermId,
        stats: RDFStarReasoningStats,
        annotations: List[Tuple[TermId, TermId, TermId, TermId]],
        iteration: int,
    ) -> List[Tuple[TermId, TermId, TermId, TermId]]:
        """
        RDFS7 with confidence: Property inheritance via subPropertyOf.
        """
        subprop_id = self._applicator.get_vocab_id(RDFS_SUBPROPERTY_OF)
        if subprop_id is None:
            return []
        
        subprop_pairs = self._applicator.get_facts_with_predicate(subprop_id, graph_id)
        if not subprop_pairs:
            return []
        
        # Build subPropertyOf map
        subprop_of: Dict[TermId, Set[TermId]] = {}
        for p1, p2 in subprop_pairs:
            if p1 not in subprop_of:
                subprop_of[p1] = set()
            subprop_of[p1].add(p2)
        
        # Get all facts
        df = self._applicator.get_all_non_deleted_facts(graph_id)
        
        new_facts = []
        for row in df.iter_rows(named=True):
            p1 = row["p"]
            if p1 in subprop_of:
                for p2 in subprop_of[p1]:
                    fact = (graph_id, row["s"], p2, row["o"])
                    if fact not in existing:
                        new_facts.append(fact)
                        stats.rdfs7_inferences += 1
                        
                        # Confidence
                        conf1 = self.get_fact_confidence(graph_id, row["s"], p1, row["o"])
                        conf2 = self.get_fact_confidence(graph_id, p1, subprop_id, p2)
                        combined = self.combine_confidence([conf1, conf2])
                        self._confidence_cache[fact] = combined
                        
                        justification = InferenceJustification(
                            rule_id="rdfs7",
                            rule_name="Property inheritance via subPropertyOf",
                            antecedent_facts=[
                                (graph_id, row["s"], p1, row["o"]),
                                (graph_id, p1, subprop_id, p2),
                            ],
                            confidence=combined,
                            iteration=iteration,
                        )
                        self._justifications[fact] = justification
                        annotations.extend(
                            self._store_inference_annotation(fact, justification, graph_id)
                        )
        
        return new_facts
    
    def get_justification(
        self,
        g: TermId,
        s: TermId,
        p: TermId,
        o: TermId,
    ) -> Optional[InferenceJustification]:
        """Get the justification for an inferred fact."""
        return self._justifications.get((g, s, p, o))
    
    def get_inferred_facts_by_rule(
        self,
        rule_id: str,
    ) -> List[Tuple[Tuple[TermId, TermId, TermId, TermId], InferenceJustification]]:
        """Get all facts inferred by a specific rule."""
        return [
            (fact, just)
            for fact, just in self._justifications.items()
            if just.rule_id == rule_id
        ]
    
    def get_facts_below_confidence(
        self,
        threshold: float,
    ) -> List[Tuple[Tuple[TermId, TermId, TermId, TermId], float]]:
        """Get all inferred facts below a confidence threshold."""
        return [
            (fact, conf)
            for fact, conf in self._confidence_cache.items()
            if conf < threshold
        ]
