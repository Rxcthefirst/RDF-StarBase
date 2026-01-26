"""
Graph Explorer for RDF-StarBase.

Provides:
- Graph list with statistics
- Graph metadata (DCAT/PROV)
- Per-graph statistics
- Replace graph action
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Well-known namespaces
DCAT = "http://www.w3.org/ns/dcat#"
PROV = "http://www.w3.org/ns/prov#"
DCT = "http://purl.org/dc/terms/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
FOAF = "http://xmlns.com/foaf/0.1/"


@dataclass
class GraphStatistics:
    """Statistics for a named graph."""
    triple_count: int = 0
    subject_count: int = 0
    predicate_count: int = 0
    object_count: int = 0
    literal_count: int = 0
    blank_node_count: int = 0
    rdf_star_count: int = 0  # Quoted triple annotations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple_count": self.triple_count,
            "subject_count": self.subject_count,
            "predicate_count": self.predicate_count,
            "object_count": self.object_count,
            "literal_count": self.literal_count,
            "blank_node_count": self.blank_node_count,
            "rdf_star_count": self.rdf_star_count
        }


@dataclass
class GraphMetadata:
    """
    Metadata for a named graph.
    
    Structured according to DCAT/PROV/DCT vocabularies.
    """
    # Identity
    graph_iri: str = ""
    
    # DCAT Dataset properties
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # DCT properties
    creator: Optional[str] = None
    publisher: Optional[str] = None
    license: Optional[str] = None
    language: Optional[str] = None
    
    # Temporal
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    issued: Optional[datetime] = None
    
    # Provenance (PROV-O)
    was_derived_from: Optional[str] = None
    was_generated_by: Optional[str] = None
    was_attributed_to: Optional[str] = None
    
    # Source info
    source_file: Optional[str] = None
    source_format: Optional[str] = None
    source_url: Optional[str] = None
    
    # Version
    version: Optional[str] = None
    version_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_iri": self.graph_iri,
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "creator": self.creator,
            "publisher": self.publisher,
            "license": self.license,
            "language": self.language,
            "created": self.created.isoformat() if self.created else None,
            "modified": self.modified.isoformat() if self.modified else None,
            "issued": self.issued.isoformat() if self.issued else None,
            "was_derived_from": self.was_derived_from,
            "was_generated_by": self.was_generated_by,
            "was_attributed_to": self.was_attributed_to,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "source_url": self.source_url,
            "version": self.version,
            "version_notes": self.version_notes
        }
    
    def to_rdf_triples(self) -> List[tuple]:
        """Convert metadata to RDF triples for storage."""
        triples = []
        graph = self.graph_iri
        
        if self.title:
            triples.append((graph, f"{DCT}title", self.title))
        if self.description:
            triples.append((graph, f"{DCT}description", self.description))
        if self.creator:
            triples.append((graph, f"{DCT}creator", self.creator))
        if self.publisher:
            triples.append((graph, f"{DCT}publisher", self.publisher))
        if self.license:
            triples.append((graph, f"{DCT}license", self.license))
        if self.created:
            triples.append((graph, f"{DCT}created", self.created.isoformat()))
        if self.modified:
            triples.append((graph, f"{DCT}modified", self.modified.isoformat()))
        if self.was_derived_from:
            triples.append((graph, f"{PROV}wasDerivedFrom", self.was_derived_from))
        if self.was_generated_by:
            triples.append((graph, f"{PROV}wasGeneratedBy", self.was_generated_by))
        if self.version:
            triples.append((graph, f"{DCAT}version", self.version))
        
        for keyword in self.keywords:
            triples.append((graph, f"{DCAT}keyword", keyword))
        
        return triples


@dataclass
class GraphInfo:
    """Complete information about a named graph."""
    iri: str
    statistics: GraphStatistics = field(default_factory=GraphStatistics)
    metadata: GraphMetadata = field(default_factory=GraphMetadata)
    is_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iri": self.iri,
            "statistics": self.statistics.to_dict(),
            "metadata": self.metadata.to_dict(),
            "is_default": self.is_default
        }


class GraphExplorer:
    """
    Explores and manages named graphs in a store.
    
    Features:
    - List all graphs with statistics
    - Get graph metadata (DCAT/PROV)
    - Calculate per-graph statistics
    - Replace graph contents
    """
    
    def __init__(self, store: Any):
        """
        Initialize graph explorer.
        
        Args:
            store: The TripleStore or similar with graph support
        """
        self.store = store
    
    def list_graphs(self, include_stats: bool = True) -> List[GraphInfo]:
        """
        List all named graphs.
        
        Args:
            include_stats: Whether to calculate statistics for each graph
        
        Returns:
            List of GraphInfo objects
        """
        graphs = []
        
        # Get all graph names
        graph_names = self._get_graph_names()
        
        for graph_name in graph_names:
            info = GraphInfo(
                iri=graph_name,
                is_default=(graph_name == "" or graph_name is None)
            )
            info.metadata.graph_iri = graph_name
            
            if include_stats:
                info.statistics = self.get_statistics(graph_name)
            
            # Try to load metadata
            info.metadata = self.get_metadata(graph_name)
            
            graphs.append(info)
        
        return graphs
    
    def _get_graph_names(self) -> List[str]:
        """Get list of all graph names from store."""
        if hasattr(self.store, 'get_graphs'):
            return list(self.store.get_graphs())
        elif hasattr(self.store, 'graphs'):
            return list(self.store.graphs())
        elif hasattr(self.store, '_facts') and hasattr(self.store._facts, 'get_graphs'):
            return list(self.store._facts.get_graphs())
        else:
            # Fallback: query for distinct graphs
            try:
                from rdf_starbase.sparql import parse_sparql, execute_sparql
                query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }"
                result = execute_sparql(parse_sparql(query), self.store)
                return [row.get('g', '') for row in result.bindings]
            except:
                return []
    
    def get_statistics(self, graph_iri: str) -> GraphStatistics:
        """
        Calculate statistics for a graph.
        
        Args:
            graph_iri: The graph IRI
        
        Returns:
            GraphStatistics with counts
        """
        stats = GraphStatistics()
        
        try:
            # Get triples in graph
            triples = self._get_graph_triples(graph_iri)
            
            subjects = set()
            predicates = set()
            objects = set()
            
            for s, p, o in triples:
                stats.triple_count += 1
                subjects.add(s)
                predicates.add(p)
                objects.add(o)
                
                # Count literals
                if isinstance(o, str) and (
                    '"' in o or 
                    o.startswith("'") or
                    '^^' in o or
                    '@' in o
                ):
                    stats.literal_count += 1
                
                # Count blank nodes
                if isinstance(s, str) and s.startswith('_:'):
                    stats.blank_node_count += 1
                if isinstance(o, str) and o.startswith('_:'):
                    stats.blank_node_count += 1
                
                # Count RDF-Star annotations
                if isinstance(s, tuple) or isinstance(o, tuple):
                    stats.rdf_star_count += 1
            
            stats.subject_count = len(subjects)
            stats.predicate_count = len(predicates)
            stats.object_count = len(objects)
            
        except Exception as e:
            logger.warning(f"Error calculating statistics for {graph_iri}: {e}")
        
        return stats
    
    def _get_graph_triples(self, graph_iri: str) -> List[tuple]:
        """Get all triples from a graph."""
        if hasattr(self.store, 'get_triples'):
            return list(self.store.get_triples(graph=graph_iri))
        elif hasattr(self.store, 'triples'):
            return list(self.store.triples((None, None, None), graph=graph_iri))
        else:
            return []
    
    def get_metadata(self, graph_iri: str) -> GraphMetadata:
        """
        Extract metadata for a graph from its contents.
        
        Looks for DCAT, PROV, DCT properties where the subject
        is the graph IRI itself.
        """
        metadata = GraphMetadata(graph_iri=graph_iri)
        
        try:
            # Query for metadata triples about this graph
            triples = self._get_graph_triples(graph_iri)
            
            for s, p, o in triples:
                # Only look at triples where subject is the graph itself
                if str(s) != graph_iri:
                    continue
                
                p_str = str(p)
                o_str = str(o)
                
                # DCT properties
                if p_str == f"{DCT}title":
                    metadata.title = self._extract_literal(o_str)
                elif p_str == f"{DCT}description":
                    metadata.description = self._extract_literal(o_str)
                elif p_str == f"{DCT}creator":
                    metadata.creator = o_str
                elif p_str == f"{DCT}publisher":
                    metadata.publisher = o_str
                elif p_str == f"{DCT}license":
                    metadata.license = o_str
                elif p_str == f"{DCT}language":
                    metadata.language = self._extract_literal(o_str)
                elif p_str == f"{DCT}created":
                    metadata.created = self._parse_datetime(o_str)
                elif p_str == f"{DCT}modified":
                    metadata.modified = self._parse_datetime(o_str)
                elif p_str == f"{DCT}issued":
                    metadata.issued = self._parse_datetime(o_str)
                
                # PROV properties
                elif p_str == f"{PROV}wasDerivedFrom":
                    metadata.was_derived_from = o_str
                elif p_str == f"{PROV}wasGeneratedBy":
                    metadata.was_generated_by = o_str
                elif p_str == f"{PROV}wasAttributedTo":
                    metadata.was_attributed_to = o_str
                
                # DCAT properties
                elif p_str == f"{DCAT}keyword":
                    metadata.keywords.append(self._extract_literal(o_str))
                elif p_str == f"{DCAT}version":
                    metadata.version = self._extract_literal(o_str)
            
        except Exception as e:
            logger.warning(f"Error extracting metadata for {graph_iri}: {e}")
        
        return metadata
    
    def _extract_literal(self, value: str) -> str:
        """Extract string value from RDF literal."""
        # Handle quoted strings
        if value.startswith('"') and '"' in value[1:]:
            # Find end quote
            end = value.find('"', 1)
            return value[1:end]
        return value
    
    def _parse_datetime(self, value: str) -> Optional[datetime]:
        """Parse datetime from RDF literal."""
        try:
            # Extract the value part
            val = self._extract_literal(value)
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except:
            return None
    
    def set_metadata(
        self,
        graph_iri: str,
        metadata: GraphMetadata
    ) -> None:
        """
        Set metadata for a graph.
        
        Stores metadata as RDF triples in the graph.
        """
        # First, remove existing metadata triples
        self._remove_metadata_triples(graph_iri)
        
        # Add new metadata triples
        triples = metadata.to_rdf_triples()
        
        for s, p, o in triples:
            if hasattr(self.store, 'add'):
                self.store.add(s, p, o, graph=graph_iri)
            elif hasattr(self.store, 'add_triple'):
                self.store.add_triple(s, p, o, graph=graph_iri)
    
    def _remove_metadata_triples(self, graph_iri: str) -> None:
        """Remove existing metadata triples for a graph."""
        # Metadata predicates to remove
        predicates = [
            f"{DCT}title", f"{DCT}description", f"{DCT}creator",
            f"{DCT}publisher", f"{DCT}license", f"{DCT}language",
            f"{DCT}created", f"{DCT}modified", f"{DCT}issued",
            f"{PROV}wasDerivedFrom", f"{PROV}wasGeneratedBy",
            f"{PROV}wasAttributedTo", f"{DCAT}keyword", f"{DCAT}version"
        ]
        
        for pred in predicates:
            if hasattr(self.store, 'remove'):
                try:
                    self.store.remove(graph_iri, pred, None, graph=graph_iri)
                except:
                    pass
    
    def replace_graph(
        self,
        graph_iri: str,
        data: str,
        format: str = "turtle",
        preserve_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Replace graph contents with new data.
        
        This is a single operation that:
        1. Optionally preserves existing metadata
        2. Clears the graph
        3. Loads new data
        4. Restores metadata if preserved
        
        Args:
            graph_iri: The graph to replace
            data: New data to load
            format: Data format (turtle, ntriples, jsonld, etc.)
            preserve_metadata: Whether to preserve existing metadata
        
        Returns:
            Result dictionary with statistics
        """
        result = {
            "graph": graph_iri,
            "success": False,
            "triples_before": 0,
            "triples_after": 0
        }
        
        try:
            # Get current stats
            before_stats = self.get_statistics(graph_iri)
            result["triples_before"] = before_stats.triple_count
            
            # Preserve metadata if requested
            metadata = None
            if preserve_metadata:
                metadata = self.get_metadata(graph_iri)
            
            # Clear the graph
            if hasattr(self.store, 'clear_graph'):
                self.store.clear_graph(graph_iri)
            elif hasattr(self.store, 'remove'):
                # Remove all triples
                for s, p, o in self._get_graph_triples(graph_iri):
                    self.store.remove(s, p, o, graph=graph_iri)
            
            # Parse and load new data
            triples_loaded = self._load_data(data, format, graph_iri)
            
            # Restore metadata
            if metadata:
                metadata.modified = datetime.now()
                self.set_metadata(graph_iri, metadata)
            
            # Get final stats
            after_stats = self.get_statistics(graph_iri)
            result["triples_after"] = after_stats.triple_count
            result["success"] = True
            result["triples_loaded"] = triples_loaded
            
        except Exception as e:
            logger.error(f"Error replacing graph {graph_iri}: {e}")
            result["error"] = str(e)
        
        return result
    
    def _load_data(self, data: str, format: str, graph_iri: str) -> int:
        """Load data into a graph."""
        count = 0
        
        # Try to use store's load method
        if hasattr(self.store, 'load_data'):
            return self.store.load_data(data, format=format, graph=graph_iri)
        
        # Parse manually
        from rdf_starbase.formats import get_parser
        parser_class = get_parser(format)
        if parser_class:
            parser = parser_class()
            doc = parser.parse(data)
            
            for triple in doc.triples:
                if hasattr(self.store, 'add'):
                    self.store.add(
                        triple.subject,
                        triple.predicate,
                        triple.object,
                        graph=graph_iri
                    )
                elif hasattr(self.store, 'add_triple'):
                    self.store.add_triple(
                        triple.subject,
                        triple.predicate,
                        triple.object,
                        graph=graph_iri
                    )
                count += 1
        
        return count
    
    def drop_graph(self, graph_iri: str) -> Dict[str, Any]:
        """
        Drop a named graph completely.
        
        Returns:
            Result dictionary
        """
        result = {
            "graph": graph_iri,
            "success": False,
            "triples_removed": 0
        }
        
        try:
            before_stats = self.get_statistics(graph_iri)
            result["triples_removed"] = before_stats.triple_count
            
            if hasattr(self.store, 'drop_graph'):
                self.store.drop_graph(graph_iri)
            elif hasattr(self.store, 'clear_graph'):
                self.store.clear_graph(graph_iri)
            else:
                # Manual removal
                for s, p, o in self._get_graph_triples(graph_iri):
                    if hasattr(self.store, 'remove'):
                        self.store.remove(s, p, o, graph=graph_iri)
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error dropping graph {graph_iri}: {e}")
            result["error"] = str(e)
        
        return result
    
    def copy_graph(
        self,
        source_iri: str,
        target_iri: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Copy a graph to a new location.
        
        Args:
            source_iri: Source graph
            target_iri: Target graph (will be cleared first)
            include_metadata: Whether to copy metadata
        
        Returns:
            Result dictionary
        """
        result = {
            "source": source_iri,
            "target": target_iri,
            "success": False,
            "triples_copied": 0
        }
        
        try:
            # Clear target if exists
            if target_iri in self._get_graph_names():
                self.drop_graph(target_iri)
            
            # Copy triples
            triples = self._get_graph_triples(source_iri)
            for s, p, o in triples:
                if hasattr(self.store, 'add'):
                    self.store.add(s, p, o, graph=target_iri)
                elif hasattr(self.store, 'add_triple'):
                    self.store.add_triple(s, p, o, graph=target_iri)
                result["triples_copied"] += 1
            
            # Copy metadata
            if include_metadata:
                metadata = self.get_metadata(source_iri)
                metadata.graph_iri = target_iri
                metadata.was_derived_from = source_iri
                metadata.created = datetime.now()
                self.set_metadata(target_iri, metadata)
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error copying graph: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all graphs."""
        graphs = self.list_graphs(include_stats=True)
        
        total_triples = sum(g.statistics.triple_count for g in graphs)
        total_subjects = sum(g.statistics.subject_count for g in graphs)
        total_rdf_star = sum(g.statistics.rdf_star_count for g in graphs)
        
        return {
            "graph_count": len(graphs),
            "total_triples": total_triples,
            "total_subjects": total_subjects,
            "total_rdf_star_annotations": total_rdf_star,
            "graphs": [g.to_dict() for g in graphs]
        }


# Convenience function
def explore_graphs(store: Any) -> GraphExplorer:
    """Create a graph explorer for a store."""
    return GraphExplorer(store)
