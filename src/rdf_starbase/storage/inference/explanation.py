"""
Inference Explanation and Justification Service.

Provides the ability to explain why a fact exists in the store:
- Is it asserted or inferred?
- Which rule produced it?
- What facts did it derive from?
- What is its proof tree?

This is essential for:
- Debugging ontologies
- Auditing inference results
- Understanding knowledge graph derivations
- AI explainability
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any
from enum import Enum

import polars as pl

from rdf_starbase.storage.terms import TermDict, TermId, Term, TermKind, get_term_kind
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID
from rdf_starbase.storage.quoted_triples import QtDict


class FactOrigin(str, Enum):
    """Origin of a fact in the store."""
    ASSERTED = "asserted"      # Explicitly added
    INFERRED = "inferred"      # Derived by reasoning
    UNKNOWN = "unknown"        # Cannot determine


@dataclass
class ProofNode:
    """
    A node in a proof tree.
    
    Represents one step in the derivation of a fact.
    """
    fact: Tuple[str, str, str, str]  # (graph, subject, predicate, object) as strings
    origin: FactOrigin
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    confidence: float = 1.0
    children: List["ProofNode"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fact": {
                "graph": self.fact[0],
                "subject": self.fact[1],
                "predicate": self.fact[2],
                "object": self.fact[3],
            },
            "origin": self.origin.value,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "confidence": self.confidence,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Explanation:
    """
    Complete explanation for a fact.
    
    Includes:
    - Whether the fact is asserted and/or inferred
    - The rules that produced it (if inferred)
    - The supporting facts
    - A full proof tree (recursive)
    """
    fact: Tuple[str, str, str, str]  # (graph, subject, predicate, object) as strings
    exists: bool
    is_asserted: bool
    is_inferred: bool
    rules_applied: List[str] = field(default_factory=list)
    supporting_facts: List[Tuple[str, str, str, str]] = field(default_factory=list)
    proof_tree: Optional[ProofNode] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fact": {
                "graph": self.fact[0],
                "subject": self.fact[1],
                "predicate": self.fact[2],
                "object": self.fact[3],
            },
            "exists": self.exists,
            "is_asserted": self.is_asserted,
            "is_inferred": self.is_inferred,
            "rules_applied": self.rules_applied,
            "supporting_facts": [
                {"subject": f[1], "predicate": f[2], "object": f[3]}
                for f in self.supporting_facts
            ],
            "proof_tree": self.proof_tree.to_dict() if self.proof_tree else None,
            "confidence": self.confidence,
        }


class ExplanationService:
    """
    Service for explaining facts in the knowledge graph.
    
    Queries the fact store and inference annotations to construct
    explanations and proof trees.
    """
    
    # Inference namespace (must match RDFStarInferenceEngine)
    INFERENCE_NS = "http://rdfstarbase.io/inference#"
    CONFIDENCE_PRED = INFERENCE_NS + "confidence"
    INFERRED_BY_PRED = INFERENCE_NS + "inferredBy"
    BASED_ON_PRED = INFERENCE_NS + "basedOn"
    INFERRED_AT_PRED = INFERENCE_NS + "inferredAt"
    
    def __init__(
        self,
        term_dict: TermDict,
        qt_dict: QtDict,
        fact_store: FactStore,
    ):
        self._term_dict = term_dict
        self._qt_dict = qt_dict
        self._fact_store = fact_store
    
    def _term_to_string(self, term_id: TermId) -> str:
        """Convert a term ID to its string representation."""
        term = self._term_dict.lookup(term_id)
        if term is None:
            return f"<unknown:{term_id}>"
        
        if term.kind == TermKind.IRI:
            return term.lex
        elif term.kind == TermKind.LITERAL:
            if term.lang:
                return f'"{term.lex}"@{term.lang}'
            elif term.datatype_id:
                dt_term = self._term_dict.lookup(term.datatype_id)
                if dt_term:
                    return f'"{term.lex}"^^<{dt_term.lex}>'
            return f'"{term.lex}"'
        elif term.kind == TermKind.BNODE:
            return f"_:{term.lex}"
        elif term.kind == TermKind.QUOTED_TRIPLE:
            # Get quoted triple components
            qt = self._qt_dict.lookup(term_id)
            if qt:
                s, p, o = qt.s, qt.p, qt.o
                s_str = self._term_to_string(s)
                p_str = self._term_to_string(p)
                o_str = self._term_to_string(o)
                return f"<<{s_str} {p_str} {o_str}>>"
        return f"<unknown:{term_id}>"
    
    def _find_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        graph: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a fact in the store and return its metadata."""
        # Convert strings to term IDs
        s_term = Term(kind=TermKind.IRI, lex=subject)
        p_term = Term(kind=TermKind.IRI, lex=predicate)
        
        s_id = self._term_dict.get_id(s_term)
        p_id = self._term_dict.get_id(p_term)
        
        if s_id is None or p_id is None:
            return None
        
        # Determine object type (IRI or literal)
        if object_.startswith('"'):
            # Parse literal
            if '@' in object_:
                # Language-tagged
                lex, lang = object_.rsplit('@', 1)
                lex = lex.strip('"')
                o_term = Term(kind=TermKind.LITERAL, lex=lex, lang=lang)
            elif '^^' in object_:
                # Typed literal
                lex, dtype = object_.rsplit('^^', 1)
                lex = lex.strip('"')
                dtype = dtype.strip('<>')
                dtype_term = Term(kind=TermKind.IRI, lex=dtype)
                dtype_id = self._term_dict.get_id(dtype_term)
                o_term = Term(kind=TermKind.LITERAL, lex=lex, datatype_id=dtype_id)
            else:
                # Plain literal
                o_term = Term(kind=TermKind.LITERAL, lex=object_.strip('"'))
        elif object_.startswith('_:'):
            o_term = Term(kind=TermKind.BNODE, lex=object_[2:])
        else:
            o_term = Term(kind=TermKind.IRI, lex=object_)
        
        o_id = self._term_dict.get_id(o_term)
        if o_id is None:
            return None
        
        # Find the fact
        df = self._fact_store.scan_facts()
        
        filter_cond = (
            (pl.col("s") == s_id) &
            (pl.col("p") == p_id) &
            (pl.col("o") == o_id) &
            (~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)).cast(pl.Boolean))
        )
        
        if graph:
            g_term = Term(kind=TermKind.IRI, lex=graph)
            g_id = self._term_dict.get_id(g_term)
            if g_id is not None:
                filter_cond = filter_cond & (pl.col("g") == g_id)
        
        filtered = df.filter(filter_cond)
        
        if filtered.height == 0:
            return None
        
        row = filtered.row(0, named=True)
        return {
            "g": row["g"],
            "s": row["s"],
            "p": row["p"],
            "o": row["o"],
            "flags": row["flags"],
            "confidence": row.get("confidence", 1.0),
        }
    
    def _get_inference_annotations(
        self,
        s_id: TermId,
        p_id: TermId,
        o_id: TermId,
        g_id: TermId,
    ) -> Dict[str, Any]:
        """Get inference annotations for a fact (stored as RDF-Star)."""
        # Create or get quoted triple ID for the fact
        qt_id = self._qt_dict.get_id(s_id, p_id, o_id)
        if qt_id is None:
            return {}
        
        annotations = {}
        
        # Get annotations where the quoted triple is the subject
        df = self._fact_store.scan_facts()
        ann_facts = df.filter(
            (pl.col("g") == g_id) &
            (pl.col("s") == qt_id)
        )
        
        for row in ann_facts.iter_rows(named=True):
            pred_term = self._term_dict.lookup(row["p"])
            if pred_term:
                obj_term = self._term_dict.lookup(row["o"])
                if pred_term.lex == self.INFERRED_BY_PRED and obj_term:
                    annotations["rule_id"] = obj_term.lex
                elif pred_term.lex == self.CONFIDENCE_PRED and obj_term:
                    try:
                        annotations["confidence"] = float(obj_term.lex)
                    except ValueError:
                        pass
                elif pred_term.lex == self.BASED_ON_PRED:
                    # Object is a quoted triple
                    if "based_on" not in annotations:
                        annotations["based_on"] = []
                    # Resolve the quoted triple
                    if get_term_kind(row["o"]) == TermKind.QUOTED_TRIPLE:
                        qt = self._qt_dict.lookup(row["o"])
                        if qt:
                            b_s, b_p, b_o = qt.s, qt.p, qt.o
                            annotations["based_on"].append((g_id, b_s, b_p, b_o))
        
        return annotations
    
    def explain(
        self,
        subject: str,
        predicate: str,
        object_: str,
        graph: Optional[str] = None,
        max_depth: int = 10,
    ) -> Explanation:
        """
        Explain why a fact exists (or doesn't exist).
        
        Args:
            subject: Subject IRI
            predicate: Predicate IRI
            object_: Object (IRI or literal string)
            graph: Optional graph IRI
            max_depth: Maximum depth for proof tree
            
        Returns:
            Explanation with proof tree
        """
        graph_str = graph or "default"
        fact_tuple = (graph_str, subject, predicate, object_)
        
        fact_info = self._find_fact(subject, predicate, object_, graph)
        
        if fact_info is None:
            return Explanation(
                fact=fact_tuple,
                exists=False,
                is_asserted=False,
                is_inferred=False,
            )
        
        is_asserted = bool(fact_info["flags"] & int(FactFlags.ASSERTED))
        is_inferred = bool(fact_info["flags"] & int(FactFlags.INFERRED))
        
        # Get inference annotations
        annotations = self._get_inference_annotations(
            fact_info["s"],
            fact_info["p"],
            fact_info["o"],
            fact_info["g"],
        )
        
        rules_applied = []
        if "rule_id" in annotations:
            rules_applied.append(annotations["rule_id"])
        
        supporting_facts = []
        if "based_on" in annotations:
            for g, s, p, o in annotations["based_on"]:
                s_str = self._term_to_string(s)
                p_str = self._term_to_string(p)
                o_str = self._term_to_string(o)
                supporting_facts.append((graph_str, s_str, p_str, o_str))
        
        confidence = annotations.get("confidence", fact_info.get("confidence", 1.0))
        
        # Build proof tree
        proof_tree = self._build_proof_tree(
            fact_tuple,
            fact_info,
            annotations,
            max_depth,
            visited=set(),
        )
        
        return Explanation(
            fact=fact_tuple,
            exists=True,
            is_asserted=is_asserted,
            is_inferred=is_inferred,
            rules_applied=rules_applied,
            supporting_facts=supporting_facts,
            proof_tree=proof_tree,
            confidence=confidence,
        )
    
    def _build_proof_tree(
        self,
        fact_tuple: Tuple[str, str, str, str],
        fact_info: Dict[str, Any],
        annotations: Dict[str, Any],
        max_depth: int,
        visited: Set[Tuple[TermId, TermId, TermId, TermId]],
    ) -> ProofNode:
        """Recursively build a proof tree for a fact."""
        is_asserted = bool(fact_info["flags"] & int(FactFlags.ASSERTED))
        is_inferred = bool(fact_info["flags"] & int(FactFlags.INFERRED))
        
        origin = FactOrigin.ASSERTED if is_asserted else (
            FactOrigin.INFERRED if is_inferred else FactOrigin.UNKNOWN
        )
        
        node = ProofNode(
            fact=fact_tuple,
            origin=origin,
            rule_id=annotations.get("rule_id"),
            confidence=annotations.get("confidence", 1.0),
        )
        
        # Add rule name based on ID
        rule_names = {
            "rdfs2": "Domain inference",
            "rdfs3": "Range inference",
            "rdfs5": "subPropertyOf transitivity",
            "rdfs7": "Property inheritance",
            "rdfs9": "Type inheritance via subClassOf",
            "rdfs11": "subClassOf transitivity",
            "owl-sameAs": "sameAs symmetry/transitivity",
            "owl-equivalentClass": "equivalentClass to subClassOf",
            "owl-equivalentProperty": "equivalentProperty to subPropertyOf",
            "owl-inverseOf": "inverseOf property inversion",
            "owl-transitive": "TransitiveProperty closure",
            "owl-symmetric": "SymmetricProperty symmetry",
            "owl-functional": "FunctionalProperty sameAs",
            "owl-inverseFunctional": "InverseFunctionalProperty sameAs",
            "owl-hasValue": "hasValue restriction",
        }
        if node.rule_id:
            node.rule_name = rule_names.get(node.rule_id, node.rule_id)
        
        # Don't recurse if at max depth or if asserted
        if max_depth <= 0 or origin == FactOrigin.ASSERTED:
            return node
        
        # Add children for supporting facts
        fact_key = (fact_info["g"], fact_info["s"], fact_info["p"], fact_info["o"])
        if fact_key in visited:
            return node  # Prevent cycles
        visited.add(fact_key)
        
        if "based_on" in annotations:
            for g_id, s_id, p_id, o_id in annotations["based_on"]:
                child_key = (g_id, s_id, p_id, o_id)
                if child_key in visited:
                    continue
                
                # Convert to strings
                g_str = fact_tuple[0]  # Same graph
                s_str = self._term_to_string(s_id)
                p_str = self._term_to_string(p_id)
                o_str = self._term_to_string(o_id)
                child_tuple = (g_str, s_str, p_str, o_str)
                
                # Get child fact info
                child_info = {
                    "g": g_id,
                    "s": s_id,
                    "p": p_id,
                    "o": o_id,
                    "flags": FactFlags.ASSERTED,  # Default
                    "confidence": 1.0,
                }
                
                # Try to find actual flags
                df = self._fact_store.scan_facts()
                child_df = df.filter(
                    (pl.col("g") == g_id) &
                    (pl.col("s") == s_id) &
                    (pl.col("p") == p_id) &
                    (pl.col("o") == o_id)
                )
                if child_df.height > 0:
                    row = child_df.row(0, named=True)
                    child_info["flags"] = row["flags"]
                
                # Get child annotations
                child_annotations = self._get_inference_annotations(s_id, p_id, o_id, g_id)
                
                # Recursively build child tree
                child_node = self._build_proof_tree(
                    child_tuple,
                    child_info,
                    child_annotations,
                    max_depth - 1,
                    visited,
                )
                node.children.append(child_node)
        
        return node
    
    def get_all_inferred_facts(
        self,
        graph: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all inferred facts with their explanations.
        
        Returns list of dicts with:
        - subject, predicate, object
        - rule_id
        - confidence
        """
        df = self._fact_store.scan_facts()
        
        filter_cond = (
            (pl.col("flags").cast(pl.Int32) & int(FactFlags.INFERRED)).cast(pl.Boolean)
        )
        
        if graph:
            g_term = Term(kind=TermKind.IRI, lex=graph)
            g_id = self._term_dict.get_id(g_term)
            if g_id is not None:
                filter_cond = filter_cond & (pl.col("g") == g_id)
        
        inferred = df.filter(filter_cond)
        
        results = []
        for row in inferred.iter_rows(named=True):
            # Get annotations
            annotations = self._get_inference_annotations(
                row["s"], row["p"], row["o"], row["g"]
            )
            
            results.append({
                "subject": self._term_to_string(row["s"]),
                "predicate": self._term_to_string(row["p"]),
                "object": self._term_to_string(row["o"]),
                "rule_id": annotations.get("rule_id"),
                "confidence": annotations.get("confidence", 1.0),
            })
        
        return results
    
    def get_inferences_by_rule(
        self,
        rule_id: str,
        graph: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all facts inferred by a specific rule."""
        all_inferred = self.get_all_inferred_facts(graph)
        return [f for f in all_inferred if f.get("rule_id") == rule_id]
