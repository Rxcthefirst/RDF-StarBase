# RDF-StarBase — Product Manifesto

**A Product of Ontus**

## What It Is

RDF-StarBase is a native RDF★ platform for storing, querying, and visualizing assertions about data, systems, and knowledge.

**It is not just a graph of facts.**  
**It is a graph of claims.**

Every triple can carry:

- provenance
- time
- confidence
- source system
- process lineage
- policy context

This makes RDF-StarBase the semantic bedrock of the enterprise.

## Why It Exists

Modern enterprises do not lack data.  
**They lack agreement.**

They cannot answer:

- Who says this data is correct?
- When was it last materialized?
- Which system owns this claim?
- Why does this attribute exist twice?
- Which version should AI trust?

Traditional databases store values.  
Traditional catalogs store descriptions.  
**RDF-StarBase stores assertions about reality.**

## Core Philosophy

### Knowledge is contextual
Facts without provenance are guesses.

### Assertions are first-class data
Metadata is not an afterthought — it is the model.

### Disagreement is expected
Competing claims are stored, not overwritten.

### Time matters
Every assertion exists in history.

### Trust must be computable
Policies, confidence, and lineage must be queryable.

## One-Sentence Positioning

> RDF-StarBase is the foundational RDF★ system for modeling, governing, and visualizing enterprise assertions with provenance and trust.

---

## MVP Scope vs Long-Term Vision

### ✅ MVP (Hackathon / First Release)

**Goal:** Prove that RDF★ unlocks capabilities no other system offers.

#### Core Capabilities

- **Native RDF★ storage**
  - triples with embedded statements
  - no "RDF-star as an extension" hacks

- **SPARQL★ query support**
  - filter by source, time, confidence

- **Assertion Registry**
  - datasets
  - APIs
  - mappings
  - materialization runs

- **Provenance model**
  - who / when / how
  - process-generated vs human-asserted

- **Basic UI**
  - graph view
  - assertion inspection panel
  - provenance timeline

#### What You Explicitly Do Not Build Yet

- Massive horizontal scaling
- Complex reasoning engines
- AI copilots
- Full policy engines

**The MVP answers one question extremely well:**

> "What does our enterprise believe to be true — and why?"

---

### 🚀 Long-Term Vision

RDF-StarBase becomes the **semantic control plane**.

#### Future Capabilities

- Assertion conflict detection
- Trust scoring & decay
- Time-travel queries
- Semantic drift detection
- Federation across RDF-StarBase instances
- AI grounding via trusted assertions
- Governance workflows (approve / supersede / deprecate claims)

At scale, RDF-StarBase becomes:

- the source of truth about truth
- the substrate beneath AI, analytics, and integration

### Strategic Insight

> You are not competing with databases.  
> You are completing them.

---

## UI Primitives Unique to RDF-StarBase

These primitives do not exist in traditional graph tools.

### 🧱 Primitive 1: Assertion Card

Click any edge or node.

You don't see just:
```
Movie — directedBy → Director
```

You see:
- who asserted it
- source system
- timestamp
- confidence score
- generation process
- supporting evidence

> "Git blame, but for knowledge."

### 🧭 Primitive 2: Provenance Timeline

Every assertion has a history.

Visually:
- creation
- modification
- supersession
- deprecation

You can:
- scrub time
- compare states
- roll back trust

This directly supports:
- audits
- compliance
- debugging data pipelines

### ⚖️ Primitive 3: Competing Claims View

Multiple systems assert different truths?

You don't resolve it manually.  
**You see it.**

Example:
- CRM says `Customer.age = 34`
- Data Lake says `Customer.age = 36`

RDF-StarBase shows:
- both assertions
- their sources
- freshness
- trust policies

**No overwrites. No silent failures.**

### 🧠 Primitive 4: Trust Lens (Filter Layer)

Toggle graph views by:
- source system
- team
- confidence threshold
- process type
- recency

This turns the same graph into:
- an analytics view
- a governance view
- an AI-ready view

### 🧩 Primitive 5: Assertion-Aware Search

Search is not keyword-based.

You search:
- "attributes asserted by system X"
- "relationships with confidence < 0.7"
- "entities materialized last week"

This is only possible because RDF★ is native.

---

## Closing Thought

You didn't just name a product.

You identified a missing layer:

**The place where enterprises store beliefs, not just data.**

RDF-StarBase is the right name for that layer.
