"""
Trust Scoring module for RDF-StarBase.

Provides configurable trust policies, confidence decay,
conflict resolution, and trust inheritance for enterprise data governance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ConflictStrategy(Enum):
    """Strategies for resolving conflicting assertions."""
    
    MOST_RECENT = "most_recent"           # Latest timestamp wins
    HIGHEST_CONFIDENCE = "highest_confidence"  # Highest trust score wins
    FIRST_WINS = "first_wins"             # First assertion wins
    MANUAL_REVIEW = "manual_review"       # Flag for human review
    MERGE = "merge"                       # Keep all values
    AVERAGE = "average"                   # Average numeric values


class DecayFunction(Enum):
    """Functions for confidence decay over time."""
    
    NONE = "none"                # No decay
    LINEAR = "linear"            # Linear decay
    EXPONENTIAL = "exponential"  # Exponential decay
    STEP = "step"                # Step function (drops after threshold)


@dataclass
class TrustPolicy:
    """Trust policy for a data source."""
    
    source_id: str
    name: str
    
    # Base confidence (0.0 to 1.0)
    base_confidence: float = 1.0
    
    # Decay settings
    decay_function: DecayFunction = DecayFunction.NONE
    decay_half_life_days: float = 30.0  # For exponential decay
    decay_rate_per_day: float = 0.01     # For linear decay
    decay_min_confidence: float = 0.1    # Floor for decayed confidence
    
    # Conflict handling
    conflict_strategy: ConflictStrategy = ConflictStrategy.HIGHEST_CONFIDENCE
    
    # Priority (higher = more authoritative in conflicts)
    priority: int = 0
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def calculate_confidence(self, assertion_time: datetime) -> float:
        """Calculate confidence for an assertion at a given time."""
        if self.decay_function == DecayFunction.NONE:
            return self.base_confidence
        
        age_days = (datetime.now() - assertion_time).total_seconds() / 86400
        
        if age_days <= 0:
            return self.base_confidence
        
        if self.decay_function == DecayFunction.LINEAR:
            decayed = self.base_confidence - (age_days * self.decay_rate_per_day)
        elif self.decay_function == DecayFunction.EXPONENTIAL:
            decayed = self.base_confidence * math.pow(0.5, age_days / self.decay_half_life_days)
        elif self.decay_function == DecayFunction.STEP:
            if age_days > self.decay_half_life_days:
                decayed = self.decay_min_confidence
            else:
                decayed = self.base_confidence
        else:
            decayed = self.base_confidence
        
        return max(decayed, self.decay_min_confidence)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "base_confidence": self.base_confidence,
            "decay_function": self.decay_function.value,
            "decay_half_life_days": self.decay_half_life_days,
            "decay_rate_per_day": self.decay_rate_per_day,
            "decay_min_confidence": self.decay_min_confidence,
            "conflict_strategy": self.conflict_strategy.value,
            "priority": self.priority,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrustPolicy":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            name=data["name"],
            base_confidence=data.get("base_confidence", 1.0),
            decay_function=DecayFunction(data.get("decay_function", "none")),
            decay_half_life_days=data.get("decay_half_life_days", 30.0),
            decay_rate_per_day=data.get("decay_rate_per_day", 0.01),
            decay_min_confidence=data.get("decay_min_confidence", 0.1),
            conflict_strategy=ConflictStrategy(data.get("conflict_strategy", "highest_confidence")),
            priority=data.get("priority", 0),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class TrustAssertion:
    """A fact with associated trust metadata."""
    
    subject: str
    predicate: str
    object: str
    graph: str | None = None
    
    # Trust metadata
    source_id: str = ""
    confidence: float = 1.0
    asserted_at: datetime = field(default_factory=datetime.now)
    
    # Computed fields
    current_confidence: float | None = None  # After decay
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "graph": self.graph,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "asserted_at": self.asserted_at.isoformat(),
            "current_confidence": self.current_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrustAssertion":
        """Create from dictionary."""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            graph=data.get("graph"),
            source_id=data.get("source_id", ""),
            confidence=data.get("confidence", 1.0),
            asserted_at=datetime.fromisoformat(data["asserted_at"]) if data.get("asserted_at") else datetime.now(),
            current_confidence=data.get("current_confidence"),
        )
    
    def triple_key(self) -> tuple[str, str, str, str | None]:
        """Get the triple key for conflict detection."""
        return (self.subject, self.predicate, self.object, self.graph)
    
    def spo_key(self) -> tuple[str, str, str | None]:
        """Get subject-predicate key for conflict detection."""
        return (self.subject, self.predicate, self.graph)


@dataclass
class Conflict:
    """A detected conflict between assertions."""
    
    conflict_id: str
    subject: str
    predicate: str
    graph: str | None
    assertions: list[TrustAssertion]
    resolved: bool = False
    resolution: TrustAssertion | None = None
    resolution_reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "graph": self.graph,
            "assertions": [a.to_dict() for a in self.assertions],
            "resolved": self.resolved,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "resolution_reason": self.resolution_reason,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class TrustPolicyManager:
    """Manages trust policies for data sources."""
    
    def __init__(self, storage_path: Path | None = None):
        """Initialize policy manager."""
        self.storage_path = storage_path
        self._policies: dict[str, TrustPolicy] = {}
        self._default_policy: TrustPolicy | None = None
        
        if storage_path and storage_path.exists():
            self._load()
    
    def set_policy(self, policy: TrustPolicy) -> None:
        """Set or update a trust policy."""
        policy.updated_at = datetime.now()
        self._policies[policy.source_id] = policy
        self._save()
    
    def get_policy(self, source_id: str) -> TrustPolicy | None:
        """Get policy for a source."""
        return self._policies.get(source_id)
    
    def get_policy_or_default(self, source_id: str) -> TrustPolicy:
        """Get policy for a source, or return default."""
        policy = self._policies.get(source_id)
        if policy:
            return policy
        if self._default_policy:
            return self._default_policy
        # Return a basic default policy
        return TrustPolicy(source_id=source_id, name=f"Default for {source_id}")
    
    def set_default_policy(self, policy: TrustPolicy) -> None:
        """Set the default policy for unknown sources."""
        self._default_policy = policy
        self._save()
    
    def list_policies(self) -> list[TrustPolicy]:
        """List all policies."""
        return list(self._policies.values())
    
    def remove_policy(self, source_id: str) -> bool:
        """Remove a policy."""
        if source_id not in self._policies:
            return False
        del self._policies[source_id]
        self._save()
        return True
    
    def calculate_confidence(
        self,
        source_id: str,
        assertion_time: datetime,
    ) -> float:
        """Calculate current confidence for an assertion."""
        policy = self.get_policy_or_default(source_id)
        return policy.calculate_confidence(assertion_time)
    
    def _save(self) -> None:
        """Persist policies."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policies": {sid: p.to_dict() for sid, p in self._policies.items()},
            "default_policy": self._default_policy.to_dict() if self._default_policy else None,
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load policies."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        self._policies = {
            sid: TrustPolicy.from_dict(pd) 
            for sid, pd in data.get("policies", {}).items()
        }
        if data.get("default_policy"):
            self._default_policy = TrustPolicy.from_dict(data["default_policy"])


class ConflictResolver:
    """Resolves conflicts between competing assertions."""
    
    def __init__(self, policy_manager: TrustPolicyManager):
        """Initialize conflict resolver."""
        self.policy_manager = policy_manager
        self._conflicts: dict[str, Conflict] = {}
        self._conflict_counter = 0
    
    def detect_conflicts(
        self,
        assertions: list[TrustAssertion],
        functional_predicates: set[str] | None = None,
    ) -> list[Conflict]:
        """Detect conflicts in a set of assertions.
        
        Args:
            assertions: List of assertions to check
            functional_predicates: Predicates that should have only one value
        """
        conflicts = []
        
        # Group by subject-predicate
        by_sp: dict[tuple, list[TrustAssertion]] = {}
        for assertion in assertions:
            key = assertion.spo_key()
            if key not in by_sp:
                by_sp[key] = []
            by_sp[key].append(assertion)
        
        # Find conflicts
        for key, group in by_sp.items():
            if len(group) <= 1:
                continue
            
            # Check if this is a functional predicate (must have single value)
            subject, predicate, graph = key
            is_functional = functional_predicates and predicate in functional_predicates
            
            # Get unique values
            unique_values = set(a.object for a in group)
            
            if len(unique_values) > 1 and is_functional:
                self._conflict_counter += 1
                conflict = Conflict(
                    conflict_id=f"conflict-{self._conflict_counter:06d}",
                    subject=subject,
                    predicate=predicate,
                    graph=graph,
                    assertions=group,
                )
                conflicts.append(conflict)
                self._conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    def resolve(
        self,
        conflict: Conflict,
        strategy: ConflictStrategy | None = None,
    ) -> TrustAssertion | None:
        """Resolve a conflict using the specified strategy.
        
        Returns the winning assertion, or None if manual review needed.
        """
        if not conflict.assertions:
            return None
        
        # Get strategy from policy if not specified
        if strategy is None:
            # Use strategy from first assertion's source policy
            source_id = conflict.assertions[0].source_id
            policy = self.policy_manager.get_policy_or_default(source_id)
            strategy = policy.conflict_strategy
        
        # Calculate current confidence for all assertions
        for assertion in conflict.assertions:
            policy = self.policy_manager.get_policy_or_default(assertion.source_id)
            assertion.current_confidence = policy.calculate_confidence(assertion.asserted_at)
        
        winner = None
        reason = ""
        
        if strategy == ConflictStrategy.MOST_RECENT:
            winner = max(conflict.assertions, key=lambda a: a.asserted_at)
            reason = "Selected most recent assertion"
        
        elif strategy == ConflictStrategy.HIGHEST_CONFIDENCE:
            winner = max(conflict.assertions, key=lambda a: a.current_confidence or 0)
            reason = f"Selected highest confidence ({winner.current_confidence:.2f})"
        
        elif strategy == ConflictStrategy.FIRST_WINS:
            winner = min(conflict.assertions, key=lambda a: a.asserted_at)
            reason = "Selected first assertion"
        
        elif strategy == ConflictStrategy.MANUAL_REVIEW:
            reason = "Flagged for manual review"
            # Don't resolve, leave for human
        
        elif strategy == ConflictStrategy.MERGE:
            # For merge, we don't pick a single winner
            reason = "Merged - keeping all values"
        
        elif strategy == ConflictStrategy.AVERAGE:
            # Only works for numeric values
            try:
                values = [float(a.object) for a in conflict.assertions]
                avg = sum(values) / len(values)
                winner = TrustAssertion(
                    subject=conflict.subject,
                    predicate=conflict.predicate,
                    object=str(avg),
                    graph=conflict.graph,
                    source_id="computed",
                    confidence=max(a.current_confidence or 0 for a in conflict.assertions),
                )
                reason = f"Averaged numeric values: {avg}"
            except ValueError:
                reason = "Cannot average non-numeric values"
        
        if winner or strategy == ConflictStrategy.MERGE:
            conflict.resolved = True
            conflict.resolution = winner
            conflict.resolution_reason = reason
            conflict.resolved_at = datetime.now()
        
        return winner
    
    def resolve_manually(
        self,
        conflict_id: str,
        winner_index: int,
        reason: str = "Manual selection",
    ) -> TrustAssertion | None:
        """Manually resolve a conflict by selecting a winner."""
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return None
        
        if winner_index < 0 or winner_index >= len(conflict.assertions):
            return None
        
        winner = conflict.assertions[winner_index]
        conflict.resolved = True
        conflict.resolution = winner
        conflict.resolution_reason = reason
        conflict.resolved_at = datetime.now()
        
        return winner
    
    def get_conflict(self, conflict_id: str) -> Conflict | None:
        """Get a conflict by ID."""
        return self._conflicts.get(conflict_id)
    
    def list_unresolved(self) -> list[Conflict]:
        """List unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.resolved]
    
    def list_all(self) -> list[Conflict]:
        """List all conflicts."""
        return list(self._conflicts.values())


class TrustInheritance:
    """Calculates trust inheritance for inferred facts."""
    
    def __init__(self, policy_manager: TrustPolicyManager):
        """Initialize trust inheritance calculator."""
        self.policy_manager = policy_manager
    
    def calculate_inherited_confidence(
        self,
        premise_confidences: list[float],
        method: str = "min",
    ) -> float:
        """Calculate confidence for an inferred fact from its premises.
        
        Args:
            premise_confidences: Confidence values of premise facts
            method: "min", "product", "average", or "weakest_link"
        """
        if not premise_confidences:
            return 0.0
        
        if method == "min":
            # Weakest premise determines confidence
            return min(premise_confidences)
        
        elif method == "product":
            # Multiply confidences (independence assumption)
            result = 1.0
            for c in premise_confidences:
                result *= c
            return result
        
        elif method == "average":
            # Average of premises
            return sum(premise_confidences) / len(premise_confidences)
        
        elif method == "weakest_link":
            # Same as min, but with a floor
            return max(min(premise_confidences), 0.1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def inherit_from_assertions(
        self,
        premises: list[TrustAssertion],
        inferred_subject: str,
        inferred_predicate: str,
        inferred_object: str,
        graph: str | None = None,
        method: str = "min",
    ) -> TrustAssertion:
        """Create an inferred assertion with inherited confidence."""
        # Calculate current confidence for all premises
        confidences = []
        for premise in premises:
            policy = self.policy_manager.get_policy_or_default(premise.source_id)
            conf = policy.calculate_confidence(premise.asserted_at)
            confidences.append(conf)
        
        inherited_confidence = self.calculate_inherited_confidence(confidences, method)
        
        # Track which sources contributed
        source_ids = list(set(p.source_id for p in premises))
        
        return TrustAssertion(
            subject=inferred_subject,
            predicate=inferred_predicate,
            object=inferred_object,
            graph=graph,
            source_id=f"inferred:{','.join(source_ids[:3])}",
            confidence=inherited_confidence,
            current_confidence=inherited_confidence,
        )


@dataclass
class TrustScore:
    """Aggregated trust score for an entity or graph."""
    
    target_id: str
    target_type: str  # "entity", "graph", "source"
    
    # Score components
    assertion_count: int = 0
    source_count: int = 0
    average_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    
    # Weighted score
    weighted_score: float = 0.0
    
    # Age metrics
    oldest_assertion: datetime | None = None
    newest_assertion: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "assertion_count": self.assertion_count,
            "source_count": self.source_count,
            "average_confidence": self.average_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "weighted_score": self.weighted_score,
            "oldest_assertion": self.oldest_assertion.isoformat() if self.oldest_assertion else None,
            "newest_assertion": self.newest_assertion.isoformat() if self.newest_assertion else None,
        }


class TrustScorer:
    """Calculates aggregate trust scores."""
    
    def __init__(self, policy_manager: TrustPolicyManager):
        """Initialize trust scorer."""
        self.policy_manager = policy_manager
    
    def score_assertions(
        self,
        target_id: str,
        target_type: str,
        assertions: list[TrustAssertion],
    ) -> TrustScore:
        """Calculate aggregate trust score for a set of assertions."""
        if not assertions:
            return TrustScore(target_id=target_id, target_type=target_type)
        
        # Calculate current confidence for all
        confidences = []
        sources = set()
        oldest = None
        newest = None
        
        for assertion in assertions:
            policy = self.policy_manager.get_policy_or_default(assertion.source_id)
            conf = policy.calculate_confidence(assertion.asserted_at)
            confidences.append(conf)
            sources.add(assertion.source_id)
            
            if oldest is None or assertion.asserted_at < oldest:
                oldest = assertion.asserted_at
            if newest is None or assertion.asserted_at > newest:
                newest = assertion.asserted_at
        
        return TrustScore(
            target_id=target_id,
            target_type=target_type,
            assertion_count=len(assertions),
            source_count=len(sources),
            average_confidence=sum(confidences) / len(confidences),
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            weighted_score=sum(confidences) / len(confidences),  # Simple average for now
            oldest_assertion=oldest,
            newest_assertion=newest,
        )
    
    def score_entity(
        self,
        entity_uri: str,
        assertions: list[TrustAssertion],
    ) -> TrustScore:
        """Calculate trust score for an entity (subject)."""
        entity_assertions = [a for a in assertions if a.subject == entity_uri]
        return self.score_assertions(entity_uri, "entity", entity_assertions)
    
    def score_graph(
        self,
        graph_uri: str,
        assertions: list[TrustAssertion],
    ) -> TrustScore:
        """Calculate trust score for a named graph."""
        graph_assertions = [a for a in assertions if a.graph == graph_uri]
        return self.score_assertions(graph_uri, "graph", graph_assertions)


# Convenience functions

def create_policy_manager(storage_path: Path | None = None) -> TrustPolicyManager:
    """Create a new trust policy manager."""
    return TrustPolicyManager(storage_path)


def create_conflict_resolver(policy_manager: TrustPolicyManager) -> ConflictResolver:
    """Create a new conflict resolver."""
    return ConflictResolver(policy_manager)


def create_trust_scorer(policy_manager: TrustPolicyManager) -> TrustScorer:
    """Create a new trust scorer."""
    return TrustScorer(policy_manager)


def create_trust_inheritance(policy_manager: TrustPolicyManager) -> TrustInheritance:
    """Create a new trust inheritance calculator."""
    return TrustInheritance(policy_manager)
