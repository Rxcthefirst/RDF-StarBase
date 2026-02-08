# RDF-StarBase Graph Visualization

This document explains how RDF data is rendered in the graph view and the purpose of UI components.

## Schema Browser

The **Schema Browser** panel (accessible via the sidebar) shows what's in your repository:

### Classes
Lists all `rdf:type` values used in the repository with counts:
```sparql
SELECT ?class (COUNT(?s) AS ?count) 
WHERE { ?s a ?class } 
GROUP BY ?class
```

**Click a class** to insert a query pattern like `?s a <http://...Class> .`

### Properties
Lists all predicates used in the repository with counts:
```sparql
SELECT ?prop (COUNT(*) AS ?count)
WHERE { ?s ?prop ?o }
GROUP BY ?prop
```

**Click a property** to insert a pattern like `?s <http://...prop> ?o .`

### Use Cases
- **Discover data structure** - What classes and properties exist?
- **Build queries faster** - Click to insert patterns into the query editor
- **Check data distribution** - See which predicates/classes are most common

---

## Overview

The graph visualization uses D3.js force-directed layout to display RDF triples as a node-link diagram. Not all RDF data can be meaningfully visualized as a graph, so the visualization focuses on **object properties** (relationships between resources) while **datatype properties** (literal values) are shown through click interactions.

## What Gets Rendered

### Nodes

Nodes represent **RDF resources** (subjects and objects that are URIs/IRIs). Each node is displayed as:

- A circle with the local name (the part after the last `/` or `#`) as a label
- Clicking a node opens a **details panel** showing all properties with their provenance

### Edges (Links)

Edges represent **object properties** — predicates that connect two resources:

```turtle
:Alice foaf:knows :Bob .  # Rendered as edge: Alice --knows--> Bob
```

Edges are shown as:
- Directed arrows from subject to object
- Hovering shows the full predicate URI
- Clicking opens edge details with source/target information

### What Is NOT Rendered as Edges

**Datatype properties** (where the object is a literal) are not rendered as edges:

```turtle
:Alice foaf:name "Alice Smith" .      # NOT rendered as edge
:Alice foaf:age 30 .                  # NOT rendered as edge
:Alice foaf:homepage <http://...> .   # Rendered as edge (object is URI)
```

Datatype properties are accessible by:
1. Clicking on a node to open the properties panel
2. Viewing the Table view for full query results

## Interaction

### Node Interactions

| Action | Result |
|--------|--------|
| **Click** | Opens properties panel with all predicates/objects and provenance |
| **Drag** | Move node position (physics simulation adjusts) |
| **Hover** | Highlights the node |

### Edge Interactions

| Action | Result |
|--------|--------|
| **Click** | Opens edge details panel with predicate, source, and target |
| **Hover** | Shows predicate tooltip, thickens edge |

### Pan & Zoom

- **Scroll** to zoom in/out
- **Click & drag** on empty space to pan

## Properties Panel

When you click a node, the properties panel shows:

| Column | Description |
|--------|-------------|
| **Property** | The predicate (local name) |
| **Value** | The object value (local name for URIs, literal value for literals) |
| **Source** | Provenance: which source asserted this triple and confidence score |

The provenance is queried using RDF-Star syntax:

```sparql
SELECT ?p ?o ?source ?confidence WHERE {
  <node-uri> ?p ?o .
  OPTIONAL {
    << <node-uri> ?p ?o >> :source ?source .
    << <node-uri> ?p ?o >> :confidence ?confidence .
  }
}
```

## Graph Building Algorithm

Given query results with columns `?s ?p ?o`, the graph is built as:

```
for each row:
  if object is a URI (starts with http:// or urn:):
    add subject as node
    add object as node  
    add edge: subject --predicate--> object
  else:
    # Datatype property - not added to graph
    # (accessible via node click)
```

## Views Comparison

| View | Best For |
|------|----------|
| **Graph** | Exploring relationships, understanding network structure |
| **Table** | Seeing all results including literals, pagination |
| **JSON** | Debugging, programmatic access |

## Customization

The visualization uses:
- **Catppuccin Mocha** (dark theme) / **Catppuccin Latte** (light theme) colors
- Force-directed layout with collision detection
- Zoom/pan via D3.js zoom behavior
