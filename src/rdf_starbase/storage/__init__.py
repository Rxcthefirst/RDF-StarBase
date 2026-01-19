"""
RDF-StarBase Storage Layer.

High-performance columnar storage with dictionary-encoded terms
and predicate-partitioned facts.
"""

from rdf_starbase.storage.terms import (
    TermKind,
    TermId,
    TermDict,
    Term,
)
from rdf_starbase.storage.quoted_triples import QtDict, QtId
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID
from rdf_starbase.storage.lsm import LSMStorage, PartitionStats
from rdf_starbase.storage.executor import StorageExecutor, ExpansionPatterns
from rdf_starbase.storage.reasoner import RDFSReasoner, ReasoningStats
from rdf_starbase.storage.persistence import (
    StoragePersistence,
    save_storage,
    load_storage,
)

__all__ = [
    "TermKind",
    "TermId",
    "TermDict",
    "Term",
    "QtDict",
    "QtId",
    "FactStore",
    "FactFlags",
    "DEFAULT_GRAPH_ID",
    "LSMStorage",
    "PartitionStats",
    "StorageExecutor",
    "ExpansionPatterns",
    "RDFSReasoner",
    "ReasoningStats",
    "StoragePersistence",
    "save_storage",
    "load_storage",
]
