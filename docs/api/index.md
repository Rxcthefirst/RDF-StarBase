# API Reference

## Core Classes

### TripleStore

::: rdf_starbase.TripleStore
    options:
      show_root_heading: true
      members:
        - __init__
        - add_triple
        - parse
        - query
        - save
        - load

### SPARQLExecutor

::: rdf_starbase.SPARQLExecutor
    options:
      show_root_heading: true

## AI Grounding

### Models

::: rdf_starbase.AIQueryRequest
    options:
      show_root_heading: true

::: rdf_starbase.AIQueryResponse
    options:
      show_root_heading: true

::: rdf_starbase.GroundedFact
    options:
      show_root_heading: true

## rdflib Compatibility

### Graph

::: rdf_starbase.compat.rdflib.Graph
    options:
      show_root_heading: true
      members:
        - parse
        - serialize
        - triples
        - query
        - add

### Term Types

::: rdf_starbase.compat.rdflib.URIRef

::: rdf_starbase.compat.rdflib.Literal

::: rdf_starbase.compat.rdflib.BNode

## Storage Layer

### TermDict

::: rdf_starbase.storage.TermDict
    options:
      show_root_heading: true

### FactStore

::: rdf_starbase.storage.FactStore
    options:
      show_root_heading: true
