"""Tests for the storage layer."""

import pytest
import tempfile
from pathlib import Path

import polars as pl

from rdf_starbase.storage import (
    TermKind,
    TermId,
    TermDict,
    Term,
    QtDict,
    QtId,
    FactStore,
)
from rdf_starbase.storage.terms import (
    make_term_id,
    get_term_kind,
    get_term_payload,
    is_quoted_triple,
)
from rdf_starbase.storage.facts import FactFlags, DEFAULT_GRAPH_ID


class TestTermEncoding:
    """Tests for term ID encoding/decoding."""
    
    def test_make_term_id_iri(self):
        """Test creating an IRI term ID."""
        term_id = make_term_id(TermKind.IRI, 42)
        assert get_term_kind(term_id) == TermKind.IRI
        assert get_term_payload(term_id) == 42
    
    def test_make_term_id_literal(self):
        """Test creating a literal term ID."""
        term_id = make_term_id(TermKind.LITERAL, 100)
        assert get_term_kind(term_id) == TermKind.LITERAL
        assert get_term_payload(term_id) == 100
    
    def test_make_term_id_bnode(self):
        """Test creating a blank node term ID."""
        term_id = make_term_id(TermKind.BNODE, 7)
        assert get_term_kind(term_id) == TermKind.BNODE
        assert get_term_payload(term_id) == 7
    
    def test_make_term_id_quoted_triple(self):
        """Test creating a quoted triple term ID."""
        term_id = make_term_id(TermKind.QUOTED_TRIPLE, 999)
        assert get_term_kind(term_id) == TermKind.QUOTED_TRIPLE
        assert get_term_payload(term_id) == 999
        assert is_quoted_triple(term_id)
    
    def test_is_quoted_triple_false(self):
        """Test that non-QT terms are not identified as quoted triples."""
        iri_id = make_term_id(TermKind.IRI, 1)
        literal_id = make_term_id(TermKind.LITERAL, 2)
        bnode_id = make_term_id(TermKind.BNODE, 3)
        
        assert not is_quoted_triple(iri_id)
        assert not is_quoted_triple(literal_id)
        assert not is_quoted_triple(bnode_id)


class TestTerm:
    """Tests for Term class."""
    
    def test_term_iri(self):
        """Test creating an IRI term."""
        term = Term.iri("http://example.org/entity")
        assert term.kind == TermKind.IRI
        assert term.lex == "http://example.org/entity"
        assert term.datatype_id is None
        assert term.lang is None
    
    def test_term_literal(self):
        """Test creating a literal term."""
        term = Term.literal("hello", datatype_id=42)
        assert term.kind == TermKind.LITERAL
        assert term.lex == "hello"
        assert term.datatype_id == 42
    
    def test_term_bnode(self):
        """Test creating a blank node term."""
        term = Term.bnode("b0")
        assert term.kind == TermKind.BNODE
        assert term.lex == "b0"
    
    def test_term_hash_consistency(self):
        """Test that term hashing is consistent."""
        term1 = Term.iri("http://example.org/a")
        term2 = Term.iri("http://example.org/a")
        term3 = Term.iri("http://example.org/b")
        
        assert term1.compute_hash() == term2.compute_hash()
        assert term1.compute_hash() != term3.compute_hash()
    
    def test_term_equality(self):
        """Test term equality."""
        term1 = Term.literal("42", datatype_id=100)
        term2 = Term.literal("42", datatype_id=100)
        term3 = Term.literal("42", datatype_id=200)
        
        assert term1 == term2
        assert term1 != term3


class TestTermDict:
    """Tests for TermDict class."""
    
    def test_intern_iri(self):
        """Test interning an IRI."""
        td = TermDict()
        
        term_id = td.intern_iri("http://example.org/entity")
        
        assert term_id is not None
        assert get_term_kind(term_id) == TermKind.IRI
        
        # Interning again should return same ID
        term_id2 = td.intern_iri("http://example.org/entity")
        assert term_id == term_id2
    
    def test_intern_literal(self):
        """Test interning literals with type inference."""
        td = TermDict()
        
        # String literal
        str_id = td.intern_literal("hello")
        term = td.lookup(str_id)
        assert term.lex == "hello"
        assert term.datatype_id == td.xsd_string_id
        
        # Integer literal
        int_id = td.intern_literal(42)
        term = td.lookup(int_id)
        assert term.lex == "42"
        assert term.datatype_id == td.xsd_integer_id
        
        # Float literal
        float_id = td.intern_literal(3.14)
        term = td.lookup(float_id)
        assert term.lex == "3.14"
        assert term.datatype_id == td.xsd_decimal_id
    
    def test_intern_bnode(self):
        """Test interning blank nodes."""
        td = TermDict()
        
        bnode_id = td.intern_bnode("b0")
        
        assert get_term_kind(bnode_id) == TermKind.BNODE
        term = td.lookup(bnode_id)
        assert term.lex == "b0"
    
    def test_batch_intern(self):
        """Test bulk interning."""
        td = TermDict()
        
        terms = [
            Term.iri("http://example.org/a"),
            Term.iri("http://example.org/b"),
            Term.iri("http://example.org/c"),
        ]
        
        ids = td.get_or_create_batch(terms)
        
        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique
    
    def test_lookup(self):
        """Test term lookup."""
        td = TermDict()
        
        term_id = td.intern_iri("http://example.org/test")
        term = td.lookup(term_id)
        
        assert term is not None
        assert term.lex == "http://example.org/test"
    
    def test_get_lex(self):
        """Test getting lexical form."""
        td = TermDict()
        
        term_id = td.intern_iri("http://example.org/test")
        lex = td.get_lex(term_id)
        
        assert lex == "http://example.org/test"
    
    def test_contains(self):
        """Test contains check."""
        td = TermDict()
        
        term = Term.iri("http://example.org/test")
        assert not td.contains(term)
        
        td.get_or_create(term)
        assert td.contains(term)
    
    def test_count_by_kind(self):
        """Test count by kind."""
        td = TermDict()
        
        td.intern_iri("http://example.org/a")
        td.intern_iri("http://example.org/b")
        td.intern_literal("hello")
        td.intern_bnode("b0")
        
        counts = td.count_by_kind()
        
        # Note: well-known datatypes are pre-interned as IRIs
        assert counts[TermKind.IRI] >= 2
        assert counts[TermKind.LITERAL] >= 1
        assert counts[TermKind.BNODE] >= 1
    
    def test_persistence(self):
        """Test saving and loading term dictionary."""
        td = TermDict()
        
        iri_id = td.intern_iri("http://example.org/test")
        lit_id = td.intern_literal("hello", lang="en")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            td.save(path)
            
            # Load and verify
            td2 = TermDict.load(path)
            
            assert td2.get_lex(iri_id) == "http://example.org/test"
            assert td2.get_lex(lit_id) == "hello"
            
            term = td2.lookup(lit_id)
            assert term.lang == "en"


class TestQtDict:
    """Tests for QtDict class."""
    
    @pytest.fixture
    def setup(self):
        """Create term dict and qt dict."""
        td = TermDict()
        qd = QtDict(td)
        return td, qd
    
    def test_intern_quoted_triple(self, setup):
        """Test interning a quoted triple."""
        td, qd = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        qt_id = qd.get_or_create(s, p, o)
        
        assert qt_id is not None
        assert is_quoted_triple(qt_id)
        
        # Interning again should return same ID
        qt_id2 = qd.get_or_create(s, p, o)
        assert qt_id == qt_id2
    
    def test_lookup_quoted_triple(self, setup):
        """Test looking up a quoted triple."""
        td, qd = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        qt_id = qd.get_or_create(s, p, o)
        qt = qd.lookup(qt_id)
        
        assert qt is not None
        assert qt.s == s
        assert qt.p == p
        assert qt.o == o
    
    def test_expand_to_dataframe(self, setup):
        """Test expanding qt_ids to DataFrame."""
        td, qd = setup
        
        s1 = td.intern_iri("http://example.org/a")
        p1 = td.intern_iri("http://example.org/p")
        o1 = td.intern_iri("http://example.org/b")
        
        s2 = td.intern_iri("http://example.org/c")
        o2 = td.intern_iri("http://example.org/d")
        
        qt1 = qd.get_or_create(s1, p1, o1)
        qt2 = qd.get_or_create(s2, p1, o2)
        
        df = qd.expand_to_dataframe([qt1, qt2])
        
        assert len(df) == 2
        assert "qt_id" in df.columns
        assert "s" in df.columns
        assert "p" in df.columns
        assert "o" in df.columns
    
    def test_batch_intern(self, setup):
        """Test bulk interning quoted triples."""
        td, qd = setup
        
        s = td.intern_iri("http://example.org/s")
        p = td.intern_iri("http://example.org/p")
        o1 = td.intern_iri("http://example.org/o1")
        o2 = td.intern_iri("http://example.org/o2")
        o3 = td.intern_iri("http://example.org/o3")
        
        qt_ids = qd.get_or_create_batch([
            (s, p, o1),
            (s, p, o2),
            (s, p, o3),
        ])
        
        assert len(qt_ids) == 3
        assert len(set(qt_ids)) == 3  # All unique
    
    def test_persistence(self, setup):
        """Test saving and loading qt dictionary."""
        td, qd = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        qt_id = qd.get_or_create(s, p, o)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            td.save(path)
            qd.save(path)
            
            # Load and verify
            td2 = TermDict.load(path)
            qd2 = QtDict.load(path, td2)
            
            qt = qd2.lookup(qt_id)
            assert qt is not None
            assert qt.s == s
            assert qt.p == p
            assert qt.o == o


class TestFactStore:
    """Tests for FactStore class."""
    
    @pytest.fixture
    def setup(self):
        """Create term dict, qt dict, and fact store."""
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        return td, qd, fs
    
    def test_add_fact(self, setup):
        """Test adding a single fact."""
        td, qd, fs = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        txn = fs.add_fact(s, p, o)
        
        assert txn == 0
        assert len(fs) == 1
    
    def test_add_facts_batch(self, setup):
        """Test adding a batch of facts."""
        td, qd, fs = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o1 = td.intern_iri("http://example.org/bob")
        o2 = td.intern_iri("http://example.org/charlie")
        
        source = td.intern_iri("http://example.org/source1")
        
        txn = fs.add_facts_batch(
            [
                (DEFAULT_GRAPH_ID, s, p, o1),
                (DEFAULT_GRAPH_ID, s, p, o2),
            ],
            source=source,
            confidence=0.9,
        )
        
        assert len(fs) == 2
        assert fs.count_active() == 2
    
    def test_scan_facts(self, setup):
        """Test scanning facts by predicate."""
        td, qd, fs = setup
        
        s = td.intern_iri("http://example.org/alice")
        p1 = td.intern_iri("http://example.org/knows")
        p2 = td.intern_iri("http://example.org/likes")
        o = td.intern_iri("http://example.org/bob")
        
        fs.add_fact(s, p1, o)
        fs.add_fact(s, p2, o)
        
        df = fs.scan_facts(p=p1)
        assert len(df) == 1
        
        df = fs.scan_facts(p=p2)
        assert len(df) == 1
        
        df = fs.scan_facts()
        assert len(df) == 2
    
    def test_scan_facts_by_subject(self, setup):
        """Test scanning facts by subject."""
        td, qd, fs = setup
        
        s1 = td.intern_iri("http://example.org/alice")
        s2 = td.intern_iri("http://example.org/bob")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/charlie")
        
        fs.add_fact(s1, p, o)
        fs.add_fact(s2, p, o)
        
        df = fs.scan_facts_by_s(s1)
        assert len(df) == 1
    
    def test_metadata_fact_detection(self, setup):
        """Test that metadata facts are auto-detected."""
        td, qd, fs = setup
        
        # Create a base triple
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        # Quote it
        qt_id = qd.get_or_create(s, p, o)
        
        # Add metadata about the quoted triple
        prov_p = td.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
        source = td.intern_iri("http://example.org/source1")
        
        fs.add_fact(qt_id, prov_p, source)
        
        assert fs.count_metadata() == 1
    
    def test_expand_qt_metadata(self, setup):
        """Test RDF★ expansion join."""
        td, qd, fs = setup
        
        # Create some base triples
        alice = td.intern_iri("http://example.org/alice")
        knows = td.intern_iri("http://example.org/knows")
        bob = td.intern_iri("http://example.org/bob")
        charlie = td.intern_iri("http://example.org/charlie")
        
        # Quote them
        qt1 = qd.get_or_create(alice, knows, bob)
        qt2 = qd.get_or_create(alice, knows, charlie)
        
        # Add metadata
        prov_from = td.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
        source1 = td.intern_iri("http://example.org/source1")
        source2 = td.intern_iri("http://example.org/source2")
        
        fs.add_fact(qt1, prov_from, source1, confidence=0.9)
        fs.add_fact(qt2, prov_from, source2, confidence=0.8)
        
        # Expand: get all triples derived from any source
        df = fs.expand_qt_metadata(prov_from)
        
        assert len(df) == 2
        assert "base_s" in df.columns
        assert "base_p" in df.columns
        assert "base_o" in df.columns
        assert "metadata_o" in df.columns
    
    def test_soft_delete(self, setup):
        """Test soft delete."""
        td, qd, fs = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        fs.add_fact(s, p, o)
        assert fs.count_active() == 1
        
        deleted = fs.mark_deleted(s=s, p=p, o=o)
        assert deleted == 1
        assert fs.count_active() == 0
        assert len(fs) == 1  # Still in store but marked deleted
    
    def test_persistence(self, setup):
        """Test saving and loading fact store."""
        td, qd, fs = setup
        
        s = td.intern_iri("http://example.org/alice")
        p = td.intern_iri("http://example.org/knows")
        o = td.intern_iri("http://example.org/bob")
        
        fs.add_fact(s, p, o, confidence=0.95)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            td.save(path)
            qd.save(path)
            fs.save(path)
            
            # Load and verify
            td2 = TermDict.load(path)
            qd2 = QtDict.load(path, td2)
            fs2 = FactStore.load(path, td2, qd2)
            
            assert len(fs2) == 1
            df = fs2.scan_facts()
            assert len(df) == 1


class TestStorageIntegration:
    """Integration tests for the complete storage layer."""
    
    def test_full_rdf_star_workflow(self):
        """Test complete RDF★ workflow with provenance."""
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Define predicates
        rdf_type = td.intern_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        knows = td.intern_iri("http://example.org/knows")
        person = td.intern_iri("http://example.org/Person")
        prov_from = td.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
        prov_time = td.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
        confidence = td.intern_iri("http://example.org/confidence")
        
        # Create entities
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        source = td.intern_iri("http://example.org/crm-system")
        timestamp = td.intern_literal("2026-01-16T12:00:00Z")
        conf_val = td.intern_literal(0.95)
        
        # Add base triples
        fs.add_fact(alice, rdf_type, person)
        fs.add_fact(bob, rdf_type, person)
        fs.add_fact(alice, knows, bob)
        
        # Quote the "alice knows bob" triple
        qt_id = qd.get_or_create(alice, knows, bob)
        
        # Add provenance metadata about the quoted triple
        fs.add_fact(qt_id, prov_from, source)
        fs.add_fact(qt_id, prov_time, timestamp)
        fs.add_fact(qt_id, confidence, conf_val)
        
        # Verify counts
        assert fs.count_active() == 6
        assert fs.count_metadata() == 3
        
        # Verify expansion join works
        df = fs.expand_qt_metadata(prov_from)
        assert len(df) == 1
        
        row = df.row(0, named=True)
        assert row["base_s"] == alice
        assert row["base_p"] == knows
        assert row["base_o"] == bob
        assert row["metadata_o"] == source
    
    def test_high_volume_interning(self):
        """Test interning many terms for performance baseline."""
        td = TermDict()
        
        # Intern 10,000 IRIs
        for i in range(10000):
            td.intern_iri(f"http://example.org/entity/{i}")
        
        assert len(td) >= 10000
        assert td.collision_count == 0  # Should have no collisions
    
    def test_high_volume_quoted_triples(self):
        """Test interning many quoted triples."""
        td = TermDict()
        qd = QtDict(td)
        
        p = td.intern_iri("http://example.org/predicate")
        
        # Create 1000 quoted triples
        for i in range(1000):
            s = td.intern_iri(f"http://example.org/s{i}")
            o = td.intern_iri(f"http://example.org/o{i}")
            qd.get_or_create(s, p, o)
        
        assert len(qd) == 1000
        assert qd.collision_count == 0


class TestLSMStorage:
    """Tests for LSM-style predicate-partitioned storage."""
    
    def test_initialize_storage(self):
        """Test initializing a new storage directory."""
        from rdf_starbase.storage import LSMStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            storage.initialize()
            
            # Check directory structure
            assert (Path(tmpdir) / "data" / "term_dict").exists()
            assert (Path(tmpdir) / "data" / "qt_dict").exists()
            assert (Path(tmpdir) / "data" / "facts" / "base").exists()
            assert (Path(tmpdir) / "data" / "facts" / "delta").exists()
            assert (Path(tmpdir) / "data" / "meta").exists()
            
            # Check schema version file
            assert (Path(tmpdir) / "data" / "meta" / "schema_version.json").exists()
    
    def test_add_and_save_facts(self):
        """Test adding facts and saving to disk."""
        from rdf_starbase.storage import LSMStorage, DEFAULT_GRAPH_ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            
            # Add some facts
            s = storage.term_dict.intern_iri("http://example.org/alice")
            p = storage.term_dict.intern_iri("http://example.org/knows")
            o = storage.term_dict.intern_iri("http://example.org/bob")
            
            storage.add_facts_batch([
                (DEFAULT_GRAPH_ID, s, p, o),
            ])
            
            # Save
            storage.save()
            
            # Verify files exist
            assert (Path(tmpdir) / "data" / "term_dict" / "term_dict.parquet").exists()
            assert (Path(tmpdir) / "data" / "qt_dict" / "qt_dict.parquet").exists()
    
    def test_save_and_load(self):
        """Test saving and loading storage."""
        from rdf_starbase.storage import LSMStorage, DEFAULT_GRAPH_ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate storage
            storage = LSMStorage(Path(tmpdir))
            
            s = storage.term_dict.intern_iri("http://example.org/alice")
            p = storage.term_dict.intern_iri("http://example.org/knows")
            o = storage.term_dict.intern_iri("http://example.org/bob")
            
            storage.add_facts_batch([
                (DEFAULT_GRAPH_ID, s, p, o),
            ])
            storage.save()
            
            # Load and verify
            storage2 = LSMStorage.load(Path(tmpdir))
            
            # Verify term dict
            assert storage2.term_dict.get_lex(s) == "http://example.org/alice"
            assert storage2.term_dict.get_lex(p) == "http://example.org/knows"
            assert storage2.term_dict.get_lex(o) == "http://example.org/bob"
            
            # Verify facts
            assert len(storage2.fact_store) == 1
    
    def test_compaction(self):
        """Test partition compaction."""
        from rdf_starbase.storage import LSMStorage, DEFAULT_GRAPH_ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            
            # Add facts with same predicate in multiple batches
            s = storage.term_dict.intern_iri("http://example.org/alice")
            p = storage.term_dict.intern_iri("http://example.org/knows")
            o1 = storage.term_dict.intern_iri("http://example.org/bob")
            o2 = storage.term_dict.intern_iri("http://example.org/charlie")
            
            storage.add_facts_batch([(DEFAULT_GRAPH_ID, s, p, o1)])
            storage.save()
            
            storage.add_facts_batch([(DEFAULT_GRAPH_ID, s, p, o2)])
            storage.save()
            
            # Compact
            storage.compact_partition(p)
            
            # Verify stats
            stats = storage.get_partition_stats(p)
            assert stats is not None
            assert stats.row_count == 2
    
    def test_compaction_deduplication(self):
        """Test that compaction deduplicates facts."""
        from rdf_starbase.storage import LSMStorage, DEFAULT_GRAPH_ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            
            s = storage.term_dict.intern_iri("http://example.org/alice")
            p = storage.term_dict.intern_iri("http://example.org/knows")
            o = storage.term_dict.intern_iri("http://example.org/bob")
            
            # Add same fact multiple times
            storage.add_facts_batch([(DEFAULT_GRAPH_ID, s, p, o)])
            storage.save()
            storage.add_facts_batch([(DEFAULT_GRAPH_ID, s, p, o)])
            storage.save()
            
            # Compact
            storage.compact_partition(p)
            
            # Should deduplicate to 1 fact
            stats = storage.get_partition_stats(p)
            assert stats is not None
            assert stats.row_count == 1
    
    def test_scan_partition(self):
        """Test scanning a specific partition."""
        from rdf_starbase.storage import LSMStorage, DEFAULT_GRAPH_ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            
            s = storage.term_dict.intern_iri("http://example.org/alice")
            p1 = storage.term_dict.intern_iri("http://example.org/knows")
            p2 = storage.term_dict.intern_iri("http://example.org/likes")
            o = storage.term_dict.intern_iri("http://example.org/bob")
            
            storage.add_facts_batch([
                (DEFAULT_GRAPH_ID, s, p1, o),
                (DEFAULT_GRAPH_ID, s, p2, o),
            ])
            storage.save()
            
            # Scan p1 partition
            df = storage.scan_partition(p1)
            assert len(df) == 1
            
            # Scan p2 partition
            df = storage.scan_partition(p2)
            assert len(df) == 1
    
    def test_stats(self):
        """Test storage statistics."""
        from rdf_starbase.storage import LSMStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LSMStorage(Path(tmpdir))
            storage.initialize()
            
            stats = storage.stats()
            
            assert "term_dict" in stats
            assert "qt_dict" in stats
            assert "fact_store" in stats
            assert "partitions" in stats


class TestExpansionPatterns:
    """Tests for RDF★ expansion patterns (Q6-Q12 from SPARQL-Star suite)."""
    
    def _setup_test_data(self):
        """Create test storage with sample RDF★ data."""
        from rdf_starbase.storage import TermDict, QtDict, FactStore, ExpansionPatterns
        
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Create base entities
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        charlie = td.intern_iri("http://example.org/charlie")
        
        # Create predicates
        knows = td.intern_iri("http://example.org/knows")
        likes = td.intern_iri("http://example.org/likes")
        conf_pred = td.intern_iri("http://example.org/confidence")
        derived = td.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
        generated = td.intern_iri("http://www.w3.org/ns/prov#wasGeneratedBy")
        gen_time = td.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
        
        # Create sources and activities
        source1 = td.intern_iri("http://example.org/Source_001")
        source2 = td.intern_iri("http://example.org/Source_002")
        run1 = td.intern_iri("http://example.org/Run_001")
        
        # Add base facts (these create quoted triples for metadata)
        qt1 = qd.get_or_create(alice, knows, bob)
        qt2 = qd.get_or_create(alice, likes, charlie)
        qt3 = qd.get_or_create(bob, knows, charlie)
        
        # Add metadata about quoted triples
        conf_val = td.intern_literal("0.95")
        conf_val2 = td.intern_literal("0.75")
        time_val = td.intern_literal("2025-06-15T10:00:00Z")
        time_val2 = td.intern_literal("2024-06-15T10:00:00Z")
        
        g = DEFAULT_GRAPH_ID
        
        # qt1 has confidence 0.95, derived from source1, generated by run1
        # Signature: add_fact(s, p, o, g=..., ...)
        fs.add_fact(qt1, conf_pred, conf_val, g=g)
        fs.add_fact(qt1, derived, source1, g=g)
        fs.add_fact(qt1, generated, run1, g=g)
        fs.add_fact(qt1, gen_time, time_val, g=g)
        
        # qt2 has confidence 0.75, derived from source2
        fs.add_fact(qt2, conf_pred, conf_val2, g=g)
        fs.add_fact(qt2, derived, source2, g=g)
        fs.add_fact(qt2, gen_time, time_val2, g=g)
        
        # qt3 derived from source1
        fs.add_fact(qt3, derived, source1, g=g)
        
        return td, qd, fs, ExpansionPatterns(td, qd, fs)
    
    def test_q6_metadata_for_triple(self):
        """Q6: Fetch all metadata about a specific quoted triple."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q6_metadata_for_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob"
        )
        
        assert len(df) == 4  # confidence, derived, generated, generatedAtTime
        
        # Check we got the expected predicates
        predicates = set(df["mp"].to_list())
        assert "http://example.org/confidence" in predicates
        assert "http://www.w3.org/ns/prov#wasDerivedFrom" in predicates
    
    def test_q6_nonexistent_triple(self):
        """Q6 with a triple that doesn't exist."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q6_metadata_for_triple(
            "http://example.org/nobody",
            "http://example.org/knows",
            "http://example.org/nobody"
        )
        
        assert len(df) == 0
    
    def test_q7_expand_by_source(self):
        """Q7: Find all quoted triples derived from a source."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q7_expand_by_source("http://example.org/Source_001")
        
        # Should find qt1 and qt3 (both derived from Source_001)
        assert len(df) == 2
        
        # Check subjects
        subjects = set(df["s"].to_list())
        assert "http://example.org/alice" in subjects
        assert "http://example.org/bob" in subjects
    
    def test_q8_expand_by_activity(self):
        """Q8: Find all statements generated by a run."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q8_expand_by_activity("http://example.org/Run_001")
        
        # Only qt1 was generated by Run_001
        assert len(df) == 1
        assert df["s"][0] == "http://example.org/alice"
        assert df["o"][0] == "http://example.org/bob"
    
    def test_q9_filter_by_confidence(self):
        """Q9: Filter statements by confidence threshold."""
        td, qd, fs, patterns = self._setup_test_data()
        
        # High confidence (> 0.8)
        df = patterns.q9_filter_by_confidence(0.8)
        
        assert len(df) == 1
        assert df["c"][0] == 0.95
        assert df["s"][0] == "http://example.org/alice"
        
        # Lower threshold (> 0.5)
        df = patterns.q9_filter_by_confidence(0.5)
        assert len(df) == 2
    
    def test_q10_filter_by_time_range(self):
        """Q10: Filter by time range."""
        from datetime import datetime, timezone
        
        td, qd, fs, patterns = self._setup_test_data()
        
        # 2025 range - should get qt1
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        
        df = patterns.q10_filter_by_time_range(start, end)
        
        assert len(df) == 1
        assert "2025" in df["t"][0]
    
    def test_q11_count_by_source(self):
        """Q11: Count statements per source."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q11_count_by_source()
        
        # Should have 2 sources
        assert len(df) == 2
        
        # Source_001 has 2 triples, Source_002 has 1
        source1_row = df.filter(pl.col("src").str.contains("Source_001"))
        assert source1_row["n"][0] == 2
        
        source2_row = df.filter(pl.col("src").str.contains("Source_002"))
        assert source2_row["n"][0] == 1
    
    def test_q12_count_by_run(self):
        """Q12: Count statements per run."""
        td, qd, fs, patterns = self._setup_test_data()
        
        df = patterns.q12_count_by_run()
        
        # Only Run_001 exists
        assert len(df) == 1
        assert df["n"][0] == 1


class TestStorageExecutor:
    """Tests for the storage-based SPARQL executor."""
    
    def _setup_executor(self):
        """Create test executor with sample data."""
        from rdf_starbase.storage import TermDict, QtDict, FactStore, StorageExecutor
        
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Create entities and predicates
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        charlie = td.intern_iri("http://example.org/charlie")
        knows = td.intern_iri("http://example.org/knows")
        likes = td.intern_iri("http://example.org/likes")
        class_person = td.intern_iri("http://example.org/Person")
        rdf_type = td.intern_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        
        g = DEFAULT_GRAPH_ID
        
        # Add facts (signature: s, p, o, g=...)
        fs.add_fact(alice, knows, bob, g=g)
        fs.add_fact(alice, likes, charlie, g=g)
        fs.add_fact(bob, knows, charlie, g=g)
        fs.add_fact(alice, rdf_type, class_person, g=g)
        fs.add_fact(bob, rdf_type, class_person, g=g)
        
        return StorageExecutor(td, qd, fs)
    
    def test_scan_all_facts(self):
        """Test scanning all facts."""
        executor = self._setup_executor()
        
        df = executor.fact_store.scan_facts()
        assert len(df) == 5
    
    def test_term_lookup(self):
        """Test term ID lookup."""
        executor = self._setup_executor()
        
        alice_id = executor.term_dict.lookup_iri("http://example.org/alice")
        assert alice_id is not None
        
        unknown = executor.term_dict.lookup_iri("http://example.org/unknown")
        assert unknown is None
