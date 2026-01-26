"""Tests for graph explorer functionality."""
import pytest
from datetime import datetime

from rdf_starbase.storage.graph_explorer import (
    GraphStatistics,
    GraphMetadata,
    GraphInfo,
    GraphExplorer,
    explore_graphs,
    DCAT,
    PROV,
    DCT,
)


# ========== GraphStatistics Tests ==========

class TestGraphStatistics:
    def test_defaults(self):
        stats = GraphStatistics()
        assert stats.triple_count == 0
        assert stats.rdf_star_count == 0
    
    def test_to_dict(self):
        stats = GraphStatistics(
            triple_count=1000,
            subject_count=500,
            predicate_count=20,
            rdf_star_count=50
        )
        d = stats.to_dict()
        assert d["triple_count"] == 1000
        assert d["rdf_star_count"] == 50


# ========== GraphMetadata Tests ==========

class TestGraphMetadata:
    def test_defaults(self):
        metadata = GraphMetadata(graph_iri="http://example.org/graph")
        assert metadata.graph_iri == "http://example.org/graph"
        assert metadata.title is None
    
    def test_full_metadata(self):
        metadata = GraphMetadata(
            graph_iri="http://example.org/g1",
            title="My Dataset",
            description="A test dataset",
            keywords=["test", "sample"],
            creator="http://example.org/me",
            created=datetime.now()
        )
        assert metadata.title == "My Dataset"
        assert len(metadata.keywords) == 2
    
    def test_to_dict(self):
        metadata = GraphMetadata(
            graph_iri="http://example.org/g1",
            title="Test",
            version="1.0"
        )
        d = metadata.to_dict()
        assert d["title"] == "Test"
        assert d["version"] == "1.0"
    
    def test_to_rdf_triples(self):
        metadata = GraphMetadata(
            graph_iri="http://example.org/g1",
            title="My Graph",
            description="Description",
            creator="http://example.org/creator"
        )
        triples = metadata.to_rdf_triples()
        
        assert len(triples) >= 3
        predicates = [t[1] for t in triples]
        assert f"{DCT}title" in predicates
        assert f"{DCT}description" in predicates
    
    def test_keywords_as_triples(self):
        metadata = GraphMetadata(
            graph_iri="http://example.org/g1",
            keywords=["keyword1", "keyword2"]
        )
        triples = metadata.to_rdf_triples()
        
        keyword_triples = [t for t in triples if f"{DCAT}keyword" in t[1]]
        assert len(keyword_triples) == 2


# ========== GraphInfo Tests ==========

class TestGraphInfo:
    def test_creation(self):
        info = GraphInfo(iri="http://example.org/graph")
        assert info.iri == "http://example.org/graph"
        assert not info.is_default
    
    def test_default_graph(self):
        info = GraphInfo(iri="", is_default=True)
        assert info.is_default
    
    def test_to_dict(self):
        info = GraphInfo(
            iri="http://example.org/g1",
            statistics=GraphStatistics(triple_count=100),
            metadata=GraphMetadata(title="Test")
        )
        d = info.to_dict()
        assert d["iri"] == "http://example.org/g1"
        assert d["statistics"]["triple_count"] == 100
        assert d["metadata"]["title"] == "Test"


# ========== Mock Store for Testing ==========

class MockStore:
    """Mock store for testing graph explorer."""
    
    def __init__(self):
        self._graphs = {}
    
    def add(self, s, p, o, graph=None):
        graph = graph or ""
        if graph not in self._graphs:
            self._graphs[graph] = []
        self._graphs[graph].append((s, p, o))
    
    def get_graphs(self):
        return list(self._graphs.keys())
    
    def get_triples(self, graph=None):
        return self._graphs.get(graph, [])
    
    def clear_graph(self, graph):
        if graph in self._graphs:
            self._graphs[graph] = []
    
    def remove(self, s, p, o, graph=None):
        graph = graph or ""
        if graph in self._graphs:
            self._graphs[graph] = [
                t for t in self._graphs[graph]
                if not (t[0] == s and t[1] == p and (o is None or t[2] == o))
            ]


# ========== GraphExplorer Tests ==========

class TestGraphExplorer:
    def test_list_graphs_empty(self):
        store = MockStore()
        explorer = GraphExplorer(store)
        
        graphs = explorer.list_graphs()
        assert len(graphs) == 0
    
    def test_list_graphs(self):
        store = MockStore()
        store.add("s1", "p1", "o1", graph="http://g1")
        store.add("s2", "p2", "o2", graph="http://g2")
        
        explorer = GraphExplorer(store)
        graphs = explorer.list_graphs()
        
        assert len(graphs) == 2
        iris = [g.iri for g in graphs]
        assert "http://g1" in iris
        assert "http://g2" in iris
    
    def test_get_statistics(self):
        store = MockStore()
        store.add("s1", "p1", "o1", graph="http://g1")
        store.add("s2", "p1", "o2", graph="http://g1")
        store.add("s1", "p2", "o3", graph="http://g1")
        
        explorer = GraphExplorer(store)
        stats = explorer.get_statistics("http://g1")
        
        assert stats.triple_count == 3
        assert stats.subject_count == 2
        assert stats.predicate_count == 2
    
    def test_get_metadata(self):
        store = MockStore()
        graph = "http://example.org/g1"
        store.add(graph, f"{DCT}title", '"My Graph"', graph=graph)
        store.add(graph, f"{DCT}creator", "http://example.org/me", graph=graph)
        
        explorer = GraphExplorer(store)
        metadata = explorer.get_metadata(graph)
        
        assert metadata.graph_iri == graph
        assert metadata.creator == "http://example.org/me"
    
    def test_set_metadata(self):
        store = MockStore()
        graph = "http://example.org/g1"
        
        explorer = GraphExplorer(store)
        metadata = GraphMetadata(
            graph_iri=graph,
            title="New Title",
            description="New Description"
        )
        explorer.set_metadata(graph, metadata)
        
        triples = store.get_triples(graph)
        predicates = [t[1] for t in triples]
        assert f"{DCT}title" in predicates
    
    def test_drop_graph(self):
        store = MockStore()
        store.add("s1", "p1", "o1", graph="http://g1")
        store.add("s2", "p2", "o2", graph="http://g1")
        
        explorer = GraphExplorer(store)
        result = explorer.drop_graph("http://g1")
        
        assert result["success"]
        assert result["triples_removed"] == 2
        assert len(store.get_triples("http://g1")) == 0
    
    def test_copy_graph(self):
        store = MockStore()
        store.add("s1", "p1", "o1", graph="http://source")
        store.add("s2", "p2", "o2", graph="http://source")
        
        explorer = GraphExplorer(store)
        result = explorer.copy_graph("http://source", "http://target")
        
        assert result["success"]
        assert result["triples_copied"] == 2
        # 2 data triples + 2 provenance triples (dct:created, prov:wasDerivedFrom)
        target_triples = store.get_triples("http://target")
        assert len(target_triples) == 4
        # Verify data triples are copied
        data_triples = [(s, p, o) for s, p, o in target_triples 
                       if not p.startswith("http://purl.org/dc/terms/") 
                       and not p.startswith("http://www.w3.org/ns/prov#")]
        assert len(data_triples) == 2
    
    def test_get_summary(self):
        store = MockStore()
        store.add("s1", "p1", "o1", graph="http://g1")
        store.add("s2", "p2", "o2", graph="http://g2")
        
        explorer = GraphExplorer(store)
        summary = explorer.get_summary()
        
        assert summary["graph_count"] == 2
        assert summary["total_triples"] == 2


# ========== Convenience Function Tests ==========

class TestConvenienceFunctions:
    def test_explore_graphs(self):
        store = MockStore()
        explorer = explore_graphs(store)
        assert isinstance(explorer, GraphExplorer)
