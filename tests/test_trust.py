"""Tests for the trust module - trust policies, confidence decay, conflict resolution."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import math

from rdf_starbase.storage.trust import (
    ConflictStrategy,
    DecayFunction,
    TrustPolicy,
    TrustAssertion,
    Conflict,
    TrustPolicyManager,
    ConflictResolver,
    TrustInheritance,
    TrustScore,
    TrustScorer,
    create_policy_manager,
    create_conflict_resolver,
    create_trust_scorer,
    create_trust_inheritance,
)


class TestConflictStrategy:
    """Tests for ConflictStrategy enum."""
    
    def test_strategies(self):
        assert ConflictStrategy.MOST_RECENT.value == "most_recent"
        assert ConflictStrategy.HIGHEST_CONFIDENCE.value == "highest_confidence"
        assert ConflictStrategy.MANUAL_REVIEW.value == "manual_review"


class TestDecayFunction:
    """Tests for DecayFunction enum."""
    
    def test_functions(self):
        assert DecayFunction.NONE.value == "none"
        assert DecayFunction.LINEAR.value == "linear"
        assert DecayFunction.EXPONENTIAL.value == "exponential"


class TestTrustPolicy:
    """Tests for TrustPolicy dataclass."""
    
    def test_creation(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Trusted Source",
            base_confidence=0.9,
        )
        assert policy.source_id == "src-1"
        assert policy.base_confidence == 0.9
    
    def test_no_decay(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            decay_function=DecayFunction.NONE,
        )
        
        old_time = datetime.now() - timedelta(days=365)
        confidence = policy.calculate_confidence(old_time)
        assert confidence == 1.0
    
    def test_linear_decay(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=1.0,
            decay_function=DecayFunction.LINEAR,
            decay_rate_per_day=0.1,
            decay_min_confidence=0.0,
        )
        
        # 5 days ago
        five_days_ago = datetime.now() - timedelta(days=5)
        confidence = policy.calculate_confidence(five_days_ago)
        assert abs(confidence - 0.5) < 0.01
    
    def test_exponential_decay(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=1.0,
            decay_function=DecayFunction.EXPONENTIAL,
            decay_half_life_days=30,
            decay_min_confidence=0.0,
        )
        
        # 30 days ago (half-life)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        confidence = policy.calculate_confidence(thirty_days_ago)
        assert abs(confidence - 0.5) < 0.01
    
    def test_step_decay(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=1.0,
            decay_function=DecayFunction.STEP,
            decay_half_life_days=7,  # Step threshold
            decay_min_confidence=0.2,
        )
        
        # Before threshold
        recent = datetime.now() - timedelta(days=3)
        assert policy.calculate_confidence(recent) == 1.0
        
        # After threshold
        old = datetime.now() - timedelta(days=10)
        assert policy.calculate_confidence(old) == 0.2
    
    def test_decay_floor(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=1.0,
            decay_function=DecayFunction.LINEAR,
            decay_rate_per_day=0.1,
            decay_min_confidence=0.3,
        )
        
        # Very old - should hit floor
        very_old = datetime.now() - timedelta(days=100)
        confidence = policy.calculate_confidence(very_old)
        assert confidence == 0.3
    
    def test_to_dict(self):
        policy = TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=0.8,
        )
        d = policy.to_dict()
        assert d["source_id"] == "src-1"
        assert d["base_confidence"] == 0.8
    
    def test_from_dict(self):
        data = {
            "source_id": "src-1",
            "name": "Source",
            "base_confidence": 0.9,
            "decay_function": "exponential",
            "conflict_strategy": "most_recent",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        policy = TrustPolicy.from_dict(data)
        assert policy.base_confidence == 0.9
        assert policy.decay_function == DecayFunction.EXPONENTIAL


class TestTrustAssertion:
    """Tests for TrustAssertion dataclass."""
    
    def test_creation(self):
        assertion = TrustAssertion(
            subject="http://ex.org/bob",
            predicate="http://ex.org/age",
            object="30",
            source_id="src-1",
            confidence=0.9,
        )
        assert assertion.subject == "http://ex.org/bob"
        assert assertion.confidence == 0.9
    
    def test_triple_key(self):
        assertion = TrustAssertion(
            subject="s",
            predicate="p",
            object="o",
            graph="g",
        )
        key = assertion.triple_key()
        assert key == ("s", "p", "o", "g")
    
    def test_spo_key(self):
        assertion = TrustAssertion(
            subject="s",
            predicate="p",
            object="o",
            graph="g",
        )
        key = assertion.spo_key()
        assert key == ("s", "p", "g")
    
    def test_to_dict(self):
        assertion = TrustAssertion(
            subject="s",
            predicate="p",
            object="o",
            confidence=0.8,
        )
        d = assertion.to_dict()
        assert d["subject"] == "s"
        assert d["confidence"] == 0.8


class TestConflict:
    """Tests for Conflict dataclass."""
    
    def test_creation(self):
        a1 = TrustAssertion(subject="s", predicate="p", object="o1")
        a2 = TrustAssertion(subject="s", predicate="p", object="o2")
        
        conflict = Conflict(
            conflict_id="conflict-001",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[a1, a2],
        )
        
        assert conflict.conflict_id == "conflict-001"
        assert len(conflict.assertions) == 2
        assert not conflict.resolved


class TestTrustPolicyManager:
    """Tests for TrustPolicyManager."""
    
    def test_set_policy(self):
        manager = TrustPolicyManager()
        policy = TrustPolicy(source_id="src-1", name="Source")
        
        manager.set_policy(policy)
        
        retrieved = manager.get_policy("src-1")
        assert retrieved is not None
        assert retrieved.name == "Source"
    
    def test_get_policy_or_default(self):
        manager = TrustPolicyManager()
        
        # No policy exists, should return default
        policy = manager.get_policy_or_default("unknown")
        assert policy.source_id == "unknown"
    
    def test_set_default_policy(self):
        manager = TrustPolicyManager()
        default = TrustPolicy(
            source_id="default",
            name="Default",
            base_confidence=0.5,
        )
        manager.set_default_policy(default)
        
        policy = manager.get_policy_or_default("unknown")
        assert policy.base_confidence == 0.5
    
    def test_list_policies(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="src-1", name="Source 1"))
        manager.set_policy(TrustPolicy(source_id="src-2", name="Source 2"))
        
        policies = manager.list_policies()
        assert len(policies) == 2
    
    def test_remove_policy(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="src-1", name="Source"))
        
        assert manager.remove_policy("src-1")
        assert manager.get_policy("src-1") is None
    
    def test_calculate_confidence(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(
            source_id="src-1",
            name="Source",
            base_confidence=0.8,
            decay_function=DecayFunction.NONE,
        ))
        
        confidence = manager.calculate_confidence("src-1", datetime.now())
        assert confidence == 0.8
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policies.json"
            
            manager1 = TrustPolicyManager(storage_path=path)
            manager1.set_policy(TrustPolicy(
                source_id="src-1",
                name="Persisted",
                base_confidence=0.7,
            ))
            
            manager2 = TrustPolicyManager(storage_path=path)
            policy = manager2.get_policy("src-1")
            
            assert policy is not None
            assert policy.base_confidence == 0.7


class TestConflictResolver:
    """Tests for ConflictResolver."""
    
    def test_detect_conflicts(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        assertions = [
            TrustAssertion(subject="s", predicate="age", object="30", source_id="src-1"),
            TrustAssertion(subject="s", predicate="age", object="31", source_id="src-2"),
            TrustAssertion(subject="s", predicate="name", object="Bob", source_id="src-1"),
        ]
        
        conflicts = resolver.detect_conflicts(
            assertions,
            functional_predicates={"age"},
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].predicate == "age"
    
    def test_resolve_most_recent(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(
            source_id="src-1",
            name="Source 1",
            conflict_strategy=ConflictStrategy.MOST_RECENT,
        ))
        
        resolver = ConflictResolver(manager)
        
        old = TrustAssertion(
            subject="s", predicate="p", object="old",
            source_id="src-1",
            asserted_at=datetime.now() - timedelta(days=10),
        )
        new = TrustAssertion(
            subject="s", predicate="p", object="new",
            source_id="src-1",
            asserted_at=datetime.now(),
        )
        
        conflict = Conflict(
            conflict_id="c1",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[old, new],
        )
        
        winner = resolver.resolve(conflict, ConflictStrategy.MOST_RECENT)
        
        assert winner is not None
        assert winner.object == "new"
    
    def test_resolve_highest_confidence(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="trusted", name="Trusted", base_confidence=0.9))
        manager.set_policy(TrustPolicy(source_id="untrusted", name="Untrusted", base_confidence=0.3))
        
        resolver = ConflictResolver(manager)
        
        trusted = TrustAssertion(
            subject="s", predicate="p", object="trusted_value",
            source_id="trusted",
        )
        untrusted = TrustAssertion(
            subject="s", predicate="p", object="untrusted_value",
            source_id="untrusted",
        )
        
        conflict = Conflict(
            conflict_id="c1",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[trusted, untrusted],
        )
        
        winner = resolver.resolve(conflict, ConflictStrategy.HIGHEST_CONFIDENCE)
        
        assert winner.object == "trusted_value"
    
    def test_resolve_first_wins(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        first = TrustAssertion(
            subject="s", predicate="p", object="first",
            asserted_at=datetime.now() - timedelta(days=10),
        )
        second = TrustAssertion(
            subject="s", predicate="p", object="second",
            asserted_at=datetime.now(),
        )
        
        conflict = Conflict(
            conflict_id="c1",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[first, second],
        )
        
        winner = resolver.resolve(conflict, ConflictStrategy.FIRST_WINS)
        assert winner.object == "first"
    
    def test_resolve_manual_review(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        conflict = Conflict(
            conflict_id="c1",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[
                TrustAssertion(subject="s", predicate="p", object="a"),
                TrustAssertion(subject="s", predicate="p", object="b"),
            ],
        )
        
        winner = resolver.resolve(conflict, ConflictStrategy.MANUAL_REVIEW)
        assert winner is None
        assert not conflict.resolved  # Still needs manual resolution
    
    def test_resolve_average(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        conflict = Conflict(
            conflict_id="c1",
            subject="s",
            predicate="p",
            graph=None,
            assertions=[
                TrustAssertion(subject="s", predicate="p", object="10", confidence=0.8),
                TrustAssertion(subject="s", predicate="p", object="20", confidence=0.8),
            ],
        )
        
        winner = resolver.resolve(conflict, ConflictStrategy.AVERAGE)
        assert winner is not None
        assert float(winner.object) == 15.0
    
    def test_resolve_manually(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        a1 = TrustAssertion(subject="s", predicate="p", object="value1")
        a2 = TrustAssertion(subject="s", predicate="p", object="value2")
        
        conflicts = resolver.detect_conflicts(
            [a1, a2],
            functional_predicates={"p"},
        )
        
        conflict = conflicts[0]
        winner = resolver.resolve_manually(conflict.conflict_id, winner_index=1)
        
        assert winner.object == "value2"
        assert conflict.resolved
    
    def test_list_unresolved(self):
        manager = TrustPolicyManager()
        resolver = ConflictResolver(manager)
        
        resolver.detect_conflicts([
            TrustAssertion(subject="s", predicate="p", object="a"),
            TrustAssertion(subject="s", predicate="p", object="b"),
        ], functional_predicates={"p"})
        
        unresolved = resolver.list_unresolved()
        assert len(unresolved) == 1


class TestTrustInheritance:
    """Tests for TrustInheritance."""
    
    def test_calculate_inherited_min(self):
        manager = TrustPolicyManager()
        inheritance = TrustInheritance(manager)
        
        result = inheritance.calculate_inherited_confidence(
            [0.9, 0.8, 0.7],
            method="min",
        )
        assert result == 0.7
    
    def test_calculate_inherited_product(self):
        manager = TrustPolicyManager()
        inheritance = TrustInheritance(manager)
        
        result = inheritance.calculate_inherited_confidence(
            [0.5, 0.5],
            method="product",
        )
        assert result == 0.25
    
    def test_calculate_inherited_average(self):
        manager = TrustPolicyManager()
        inheritance = TrustInheritance(manager)
        
        result = inheritance.calculate_inherited_confidence(
            [0.8, 0.6],
            method="average",
        )
        assert result == 0.7
    
    def test_inherit_from_assertions(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="src-1", name="S1", base_confidence=0.8))
        manager.set_policy(TrustPolicy(source_id="src-2", name="S2", base_confidence=0.6))
        
        inheritance = TrustInheritance(manager)
        
        premises = [
            TrustAssertion(subject="a", predicate="p", object="b", source_id="src-1"),
            TrustAssertion(subject="b", predicate="p", object="c", source_id="src-2"),
        ]
        
        inferred = inheritance.inherit_from_assertions(
            premises,
            inferred_subject="a",
            inferred_predicate="p",
            inferred_object="c",
            method="min",
        )
        
        assert inferred.source_id.startswith("inferred:")
        assert inferred.confidence == 0.6  # min of 0.8 and 0.6


class TestTrustScore:
    """Tests for TrustScore."""
    
    def test_creation(self):
        score = TrustScore(
            target_id="entity-1",
            target_type="entity",
            assertion_count=10,
            average_confidence=0.85,
        )
        assert score.target_id == "entity-1"
        assert score.average_confidence == 0.85


class TestTrustScorer:
    """Tests for TrustScorer."""
    
    def test_score_assertions(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="src-1", name="S1", base_confidence=0.9))
        
        scorer = TrustScorer(manager)
        
        assertions = [
            TrustAssertion(subject="s", predicate="p1", object="o1", source_id="src-1"),
            TrustAssertion(subject="s", predicate="p2", object="o2", source_id="src-1"),
        ]
        
        score = scorer.score_assertions("s", "entity", assertions)
        
        assert score.assertion_count == 2
        assert score.source_count == 1
        assert score.average_confidence == 0.9
    
    def test_score_entity(self):
        manager = TrustPolicyManager()
        manager.set_policy(TrustPolicy(source_id="src-1", name="S1", base_confidence=0.8))
        
        scorer = TrustScorer(manager)
        
        assertions = [
            TrustAssertion(subject="entity1", predicate="p", object="o", source_id="src-1"),
            TrustAssertion(subject="entity2", predicate="p", object="o", source_id="src-1"),
        ]
        
        score = scorer.score_entity("entity1", assertions)
        
        assert score.assertion_count == 1
        assert score.target_type == "entity"
    
    def test_score_graph(self):
        manager = TrustPolicyManager()
        scorer = TrustScorer(manager)
        
        assertions = [
            TrustAssertion(subject="s", predicate="p", object="o", graph="g1"),
            TrustAssertion(subject="s", predicate="p", object="o", graph="g2"),
        ]
        
        score = scorer.score_graph("g1", assertions)
        assert score.assertion_count == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_policy_manager(self):
        manager = create_policy_manager()
        assert isinstance(manager, TrustPolicyManager)
    
    def test_create_conflict_resolver(self):
        manager = TrustPolicyManager()
        resolver = create_conflict_resolver(manager)
        assert isinstance(resolver, ConflictResolver)
    
    def test_create_trust_scorer(self):
        manager = TrustPolicyManager()
        scorer = create_trust_scorer(manager)
        assert isinstance(scorer, TrustScorer)
    
    def test_create_trust_inheritance(self):
        manager = TrustPolicyManager()
        inheritance = create_trust_inheritance(manager)
        assert isinstance(inheritance, TrustInheritance)
