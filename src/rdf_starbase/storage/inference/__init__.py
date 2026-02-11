"""
Inference Engine Package.

Provides modular, extensible inference capabilities for RDF-StarBase.

Modules:
- rules: Base rule definitions and registry
- rdfs_rules: RDFS entailment rules
- owl_rules: OWL 2 RL rules
- rdfstar_rules: RDF-Star specific inference (confidence, provenance)
- incremental: Incremental reasoning support
- explanation: Inference explanation/justification
"""

from rdf_starbase.storage.inference.rules import (
    InferenceRule,
    RuleRegistry,
    InferenceProfile,
)
from rdf_starbase.storage.inference.rdfstar_rules import (
    RDFStarInferenceEngine,
    ConfidenceMethod,
)
from rdf_starbase.storage.inference.explanation import (
    Explanation,
    ExplanationService,
)

__all__ = [
    # Core
    "InferenceRule",
    "RuleRegistry",
    "InferenceProfile",
    # RDF-Star
    "RDFStarInferenceEngine",
    "ConfidenceMethod",
    # Explanation
    "Explanation",
    "ExplanationService",
]
