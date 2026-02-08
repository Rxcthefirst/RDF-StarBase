# PhD Research Plan: Algebraic Foundations for Provenance-Aware Graph Data Management

**Candidate:** Christopher Gonzalez  
**Working Title:** *Semantic-Aligned Physical Models for Provenance-Aware Graph Analytics: Theory, Complexity, and Systems*  
**Target Duration:** 4-5 years  
**Primary Areas:** Database Theory, Knowledge Representation, Computational Complexity

---

## Executive Summary

This document outlines a PhD research trajectory building on the Semantic-Aligned Physical Data Model (SAPDM) work. The research extends from an applied systems contribution toward fundamental theoretical questions in provenance algebra, query complexity, and semantic data management.

Two primary research directions are proposed:

1. **Complexity Theory Track:** Prove novel complexity results for provenance-annotated graph queries
2. **Algebraic Extension Track:** Develop new semiring structures for richer provenance semantics

Both tracks can yield multiple publications and culminate in a unified thesis.

---

## Table of Contents

1. [Background: Semiring Provenance](#1-background-semiring-provenance)
2. [Research Direction 1: Complexity Results](#2-research-direction-1-complexity-results)
3. [Research Direction 2: Semiring Extensions](#3-research-direction-2-semiring-extensions)
4. [Proposed Timeline](#4-proposed-timeline)
5. [Target Venues](#5-target-venues)
6. [Required Coursework](#6-required-coursework)
7. [Potential Advisors and Collaborators](#7-potential-advisors-and-collaborators)
8. [Reading List](#8-reading-list)

---

## 1. Background: Semiring Provenance

### 1.1 What Is a Semiring?

A **semiring** is an algebraic structure $(K, \oplus, \otimes, \mathbf{0}, \mathbf{1})$ satisfying:

| Property | Definition |
|----------|------------|
| **Additive commutative monoid** | $(K, \oplus, \mathbf{0})$ is a commutative monoid |
| **Multiplicative monoid** | $(K, \otimes, \mathbf{1})$ is a monoid |
| **Distributivity** | $a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c)$ |
| **Annihilation** | $a \otimes \mathbf{0} = \mathbf{0} \otimes a = \mathbf{0}$ |

Unlike a ring, a semiring does not require additive inverses (no subtraction).

### 1.2 Provenance Semirings (Green et al., 2007)

The foundational insight of Green, Karvounarakis, and Tannen [GKT07] is that **database query evaluation can be generalized over semirings**:

- **Standard relational algebra:** Uses the Boolean semiring $(\{0,1\}, \lor, \land, 0, 1)$
- **Bag semantics:** Uses the natural number semiring $(\mathbb{N}, +, \times, 0, 1)$
- **Provenance tracking:** Uses polynomial semirings $\mathbb{N}[X]$ where $X$ are tuple identifiers

The key theorem: For any semiring $K$ and any positive relational algebra query $Q$, there exists a **unique provenance polynomial** that captures all derivation paths.

### 1.3 The K-Relation Model

A **K-relation** is a function $R: \text{Tuples} \to K$ mapping tuples to semiring values.

Query operations lift to K-relations:
- **Selection:** $\sigma_\theta(R)(t) = \theta(t) \otimes R(t)$
- **Projection:** $\pi_A(R)(t) = \bigoplus_{t': \pi_A(t') = t} R(t')$
- **Join:** $(R \bowtie S)(t) = R(t_R) \otimes S(t_S)$
- **Union:** $(R \cup S)(t) = R(t) \oplus S(t)$

### 1.4 Why This Matters for SAPDM

SAPDM instantiates provenance semirings for row-aligned storage:

$$K = [0,1] \times 2^{\mathcal{S}} \times \mathcal{T}$$

This is a product semiring combining:
- Fuzzy/probabilistic trust $([0,1], \max, \min, 0, 1)$
- Source tracking $(2^{\mathcal{S}}, \cup, \cup, \emptyset, \emptyset)$
- Timestamps $(\mathcal{T}, \max, \max, -\infty, -\infty)$

**Gap in current work:** We use this semiring but don't prove properties about it or explore alternatives.

---

## 2. Research Direction 1: Complexity Results

### 2.1 Overview

**Goal:** Prove novel computational complexity results for provenance-annotated queries over RDF/graph databases.

**Why this matters:** 
- Establishes theoretical foundations for query optimization
- Identifies tractable vs. intractable cases
- Provides lower bounds that justify algorithmic choices

### 2.2 Specific Research Questions

#### Question 1.1: Complexity of Provenance-Filtered BGP Evaluation

**Problem:** Given a BGP $P$, a semiring $K$, a K-annotated RDF store $\mathcal{A}$, and a threshold $\tau \in K$, compute:

$$\{(t, k) : t \in \llbracket P \rrbracket_\mathcal{A}, k = \text{prov}(t), k \geq_K \tau\}$$

**Conjecture:** For the confidence semiring with $\leq_K$ defined as $\leq$ on the trust component:
- **Data complexity:** PTIME (polynomial in $|\mathcal{A}|$)
- **Combined complexity:** NP-complete (hardness from BGP evaluation)

**Research approach:**
1. Reduce from known NP-complete graph problems (e.g., subgraph isomorphism)
2. Identify restrictions that yield polynomial-time algorithms
3. Characterize the boundary between tractable and intractable

#### Question 1.2: Complexity of RDF-Star Annotation Queries

**Problem:** Given an RDF-Star store with quoted triples and annotation patterns, what is the complexity of evaluating queries with nested statement references?

**Example query:**
```sparql
SELECT ?x ?y ?conf WHERE {
  << ?x :relatedTo ?y >> :confidence ?conf .
  FILTER(?conf > 0.9)
}
```

**Conjecture:** 
- Without nesting bounds: PSPACE-complete (arbitrary recursion over quoted triples)
- With bounded nesting depth $d$: In PTIME (fixed parameter tractable)

**Research approach:**
1. Model RDF-Star as a tree-structured database
2. Apply known results on query complexity over tree-structured data
3. Prove matching upper and lower bounds

#### Question 1.3: Complexity of Provenance Explanation

**Problem:** Given a query result $t$ and its provenance polynomial, compute a minimal explanation (a minimal set of base tuples whose removal would eliminate $t$ from the result).

**Conjecture:** This is NP-hard for general semirings, but polynomial for specific semirings (Boolean, confidence).

**Research approach:**
1. Reduce from Set Cover or Vertex Cover
2. Identify semiring properties that yield tractability
3. Develop approximation algorithms

### 2.3 Expected Contributions

| Contribution | Venue Target | Timeline |
|--------------|--------------|----------|
| Complexity of BGP + provenance filtering | PODS/ICDT | Year 2 |
| RDF-Star nested query complexity | ISWC/AAAI | Year 2-3 |
| Provenance explanation complexity | VLDB/SIGMOD | Year 3 |
| Survey/unification paper | JACM/TODS | Year 4 |

### 2.4 Required Background

- **Computational complexity theory:** P, NP, PSPACE, reductions, completeness
- **Finite model theory:** Query complexity, descriptive complexity
- **Database theory:** Data complexity vs. combined complexity, conjunctive queries

---

## 3. Research Direction 2: Semiring Extensions

### 3.1 Overview

**Goal:** Develop new semiring structures that capture richer provenance semantics than existing frameworks.

**Why this matters:**
- Current semirings don't capture temporal dynamics, uncertainty types, or policy composition
- New structures could enable novel query capabilities
- Theoretical contributions to algebra applied to data management

### 3.2 Specific Research Questions

#### Question 2.1: Temporal Provenance Semirings

**Motivation:** Current provenance tracks "where did this come from?" but not "when was this valid?" or "how did this change over time?"

**Proposed structure:** A **temporal semiring** $K_T$ that captures:
- Valid time intervals: $[\text{start}, \text{end})$
- Transaction time: when the data was recorded
- Temporal relationships: before, during, after, overlaps

**Definition (sketch):**

$$K_T = \mathcal{I} \times K_{\text{base}}$$

where $\mathcal{I}$ is the set of time intervals with Allen algebra operations and $K_{\text{base}}$ is a base provenance semiring.

**Operations:**
- $\oplus$: Union of valid intervals with provenance combination
- $\otimes$: Intersection of valid intervals (join is valid only when both sources overlap)

**Research questions:**
- What are the semiring laws for temporal composition?
- How do Allen's interval relations interact with provenance?
- What query optimizations are enabled by temporal provenance?

#### Question 2.2: Uncertainty-Typed Provenance

**Motivation:** Not all uncertainty is the same:
- **Aleatoric uncertainty:** Inherent randomness (probabilistic)
- **Epistemic uncertainty:** Lack of knowledge (possibilistic)
- **Model uncertainty:** Disagreement between sources

**Proposed structure:** A **typed uncertainty semiring** that tracks uncertainty type alongside magnitude:

$$K_U = \{(\tau, v) : \tau \in \{\text{aleatoric}, \text{epistemic}, \text{model}\}, v \in [0,1]\}$$

**Operations must respect type:**
- Aleatoric × Aleatoric → Aleatoric (probability multiplication)
- Epistemic × Epistemic → Epistemic (possibility theory)
- Mixed → requires explicit uncertainty propagation rules

**Research questions:**
- How do different uncertainty types compose under join/union?
- Can we define a semiring structure, or is a more general algebraic structure needed?
- What does "minimal provenance" mean for typed uncertainty?

#### Question 2.3: Policy-Aware Provenance Semirings

**Motivation:** In governance-critical systems, provenance must interact with access control and data policies.

**Proposed structure:** A **policy semiring** where provenance annotations include:
- Access control labels (security clearances)
- Usage policies (can this data be used for X purpose?)
- Retention requirements (how long must lineage be retained?)

**Definition (sketch):**

$$K_P = 2^{\text{Policies}} \times K_{\text{base}}$$

with lattice-based operations on policy sets.

**Key challenge:** Policy composition is non-commutative in many models (order of policy application matters). This breaks standard semiring properties.

**Research questions:**
- Can we define a "near-semiring" or "semiring-like" structure for policies?
- What query semantics arise from policy-annotated provenance?
- How does policy provenance interact with GDPR/data sovereignty?

#### Question 2.4: Semiring Homomorphisms for Provenance Abstraction

**Motivation:** Different users need different provenance views:
- Auditors: Full lineage detail
- Analysts: Confidence summaries
- Users: Simple source attribution

**Proposed contribution:** Formal framework for **provenance abstraction** via semiring homomorphisms.

**Definition:** A semiring homomorphism $h: K_1 \to K_2$ is a function preserving:
- $h(\mathbf{0}_1) = \mathbf{0}_2$
- $h(\mathbf{1}_1) = \mathbf{1}_2$
- $h(a \oplus_1 b) = h(a) \oplus_2 h(b)$
- $h(a \otimes_1 b) = h(a) \otimes_2 h(b)$

**Example homomorphisms:**
- Polynomial → Boolean: "Is there any derivation?"
- Polynomial → Count: "How many derivations?"
- Full provenance → Confidence: Summary view

**Research questions:**
- What is the lattice of provenance semirings ordered by homomorphisms?
- Which abstractions are "safe" (preserve query semantics)?
- Can we automatically derive optimal abstraction for a given query?

### 3.3 Expected Contributions

| Contribution | Venue Target | Timeline |
|--------------|--------------|----------|
| Temporal provenance semiring | VLDB/SIGMOD | Year 2 |
| Uncertainty-typed provenance | KR/IJCAI | Year 2-3 |
| Policy-aware provenance | ESORICS/CCS (security) or PODS | Year 3 |
| Semiring abstraction framework | PODS/ICDT | Year 3-4 |
| Unified monograph/thesis | PhD Thesis | Year 5 |

### 3.4 Required Background

- **Abstract algebra:** Ring theory, module theory, universal algebra
- **Lattice theory:** Complete lattices, Galois connections
- **Category theory:** Functors, natural transformations (for abstraction framework)
- **Domain-specific knowledge:** Temporal databases, uncertainty reasoning, access control models

---

## 4. Proposed Timeline

```
Year 1: Foundations
├── Q1-Q2: Coursework (complexity, algebra, database theory)
├── Q3: Literature survey, identify specific thesis questions
└── Q4: Preliminary results, workshop paper on SAPDM formalization

Year 2: First Contributions
├── Q1-Q2: Complexity results (Direction 1.1)
├── Q3: First full paper submission (PODS/ICDT)
└── Q4: Begin algebraic extensions (Direction 2.1 or 2.2)

Year 3: Core Research
├── Q1-Q2: Continue algebraic work, second paper
├── Q3: Thesis proposal defense
└── Q4: Third paper, begin integration work

Year 4: Integration and Systems
├── Q1-Q2: Implement theoretical contributions in RDF-StarBase
├── Q3: Experimental evaluation, benchmark paper
└── Q4: Begin thesis writing

Year 5: Completion
├── Q1-Q2: Complete thesis draft
├── Q3: Defense preparation
└── Q4: Defend, revisions, graduation
```

---

## 5. Target Venues

### Tier 1 (Theory-Focused)

| Venue | Focus | Acceptance Rate |
|-------|-------|-----------------|
| **PODS** | Database theory, complexity | ~20% |
| **ICDT** | Database theory, logic | ~25% |
| **LICS** | Logic in computer science | ~25% |

### Tier 1 (Systems + Theory)

| Venue | Focus | Acceptance Rate |
|-------|-------|-----------------|
| **VLDB** | Database systems and theory | ~20% |
| **SIGMOD** | Data management | ~20% |
| **ICDE** | Data engineering | ~20% |

### Tier 1 (AI/Knowledge)

| Venue | Focus | Acceptance Rate |
|-------|-------|-----------------|
| **AAAI** | Artificial intelligence | ~15% |
| **IJCAI** | AI, knowledge representation | ~15% |
| **KR** | Knowledge representation | ~25% |

### Semantic Web

| Venue | Focus | Acceptance Rate |
|-------|-------|-----------------|
| **ISWC** | Semantic web research | ~20% |
| **ESWC** | European semantic web | ~25% |
| **JWS** | Journal of Web Semantics | N/A (journal) |

### Journals

| Journal | Focus |
|---------|-------|
| **TODS** | ACM Transactions on Database Systems |
| **VLDB Journal** | Database systems and theory |
| **JACM** | Theoretical computer science |
| **JWS** | Semantic web |

---

## 6. Required Coursework

### Core (Essential)

| Course | Topics |
|--------|--------|
| **Database Theory** | Query complexity, conjunctive queries, datalog |
| **Computational Complexity** | P, NP, PSPACE, circuit complexity, reductions |
| **Abstract Algebra** | Groups, rings, modules, semirings |
| **Logic and Computation** | First-order logic, finite model theory |

### Recommended (Supportive)

| Course | Topics |
|--------|--------|
| **Category Theory** | Functors, natural transformations, adjunctions |
| **Temporal Reasoning** | Allen's interval algebra, temporal logics |
| **Uncertainty Reasoning** | Probability, possibility theory, Dempster-Shafer |
| **Formal Methods** | Type theory, proof assistants (Coq/Agda) |

---

## 7. Potential Advisors and Collaborators

### Database Theory

- **Val Tannen** (UPenn) — Co-creator of provenance semirings
- **Todd Green** (RelationalAI, formerly LogicBlox) — Provenance semirings
- **Dan Suciu** (University of Washington) — Probabilistic databases
- **Phokion Kolaitis** (UC Santa Cruz / IBM) — Database theory, logic

### Semantic Web / Knowledge Graphs

- **Olaf Hartig** (Linköping University) — RDF-Star creator
- **Aidan Hogan** (Universidad de Chile) — Knowledge graphs
- **Axel Polleres** (WU Vienna) — SPARQL, RDF

### Systems + Theory

- **Peter Boncz** (CWI Amsterdam) — Columnar databases, vectorized execution
- **Thomas Neumann** (TU Munich) — RDF-3X, cardinality estimation
- **Anastasia Ailamaki** (EPFL) — Data-intensive systems

---

## 8. Reading List

### Foundational Papers

1. **[GKT07]** T. J. Green, G. Karvounarakis, V. Tannen. "Provenance Semirings." PODS 2007.
   - *The foundational paper. Read multiple times.*

2. **[GP10]** F. Geerts, A. Poggi. "On Database Query Languages for K-Relations." Journal of Applied Logic, 2010.
   - *Extended treatment of K-relations and semiring provenance.*

3. **[CCT09]** J. Cheney, L. Chiticariu, W. C. Tan. "Provenance in Databases: Why, How, and Where." Foundations and Trends in Databases, 2009.
   - *Comprehensive survey of database provenance.*

4. **[AKG12]** Y. Amsterdamer, D. Deutch, V. Tannen. "Provenance for Aggregate Queries." PODS 2011.
   - *Extends provenance to aggregates.*

### RDF and SPARQL Theory

5. **[PAG09]** J. Pérez, M. Arenas, C. Gutierrez. "Semantics and Complexity of SPARQL." ACM TODS, 2009.
   - *Formal semantics and complexity of SPARQL.*

6. **[Har17]** O. Hartig. "RDF* and SPARQL*." ISWC 2017.
   - *Foundations of RDF-Star.*

7. **[NW10]** T. Neumann, G. Weikum. "The RDF-3X Engine." VLDB Journal, 2010.
   - *Systems paper with theoretical insights.*

### Complexity Theory

8. **[Var82]** M. Vardi. "The Complexity of Relational Query Languages." STOC 1982.
   - *Foundational complexity results for database queries.*

9. **[AHV95]** S. Abiteboul, R. Hull, V. Vianu. "Foundations of Databases." Addison-Wesley, 1995.
   - *Bible of database theory.*

10. **[Lib04]** L. Libkin. "Elements of Finite Model Theory." Springer, 2004.
    - *Finite model theory for database theorists.*

### Semirings and Algebra

11. **[Gol99]** J. S. Golan. "Semirings and Their Applications." Springer, 1999.
    - *Comprehensive reference on semirings.*

12. **[KS86]** W. Kuich, A. Salomaa. "Semirings, Automata, Languages." Springer, 1986.
    - *Semirings in formal language theory.*

### Temporal and Uncertainty

13. **[All83]** J. F. Allen. "Maintaining Knowledge about Temporal Intervals." CACM, 1983.
    - *Foundational temporal reasoning.*

14. **[SD07]** D. Suciu, N. Dalvi. "Foundations of Probabilistic Answers to Queries." SIGMOD Record, 2007.
    - *Probabilistic databases.*

### Additional Recommended

15. **[DEGV01]** E. Dantsin, T. Eiter, G. Gottlob, A. Voronkov. "Complexity and Expressive Power of Logic Programming." ACM Computing Surveys, 2001.

16. **[GGM14]** G. Gottlob, G. Greco, L. Mancini, et al. "Hypertree Decompositions." JACM, 2014.
    - *Structural decomposition methods for query complexity.*

---

## Appendix A: Semiring Cheat Sheet

### Common Semirings

| Name | Set | $\oplus$ | $\otimes$ | $\mathbf{0}$ | $\mathbf{1}$ | Use |
|------|-----|----------|-----------|--------------|--------------|-----|
| Boolean | $\{0,1\}$ | $\lor$ | $\land$ | $0$ | $1$ | Set semantics |
| Natural | $\mathbb{N}$ | $+$ | $\times$ | $0$ | $1$ | Bag semantics |
| Tropical | $\mathbb{R} \cup \{\infty\}$ | $\min$ | $+$ | $\infty$ | $0$ | Shortest path |
| Viterbi | $[0,1]$ | $\max$ | $\times$ | $0$ | $1$ | Confidence |
| Polynomial | $\mathbb{N}[X]$ | $+$ | $\times$ | $0$ | $1$ | Full provenance |
| PosBool | $\text{PosBool}(X)$ | $\lor$ | $\land$ | $\bot$ | $\top$ | Why-provenance |

### Semiring Homomorphism Examples

```
Polynomial ──→ Natural (count derivations)
    │             │
    │             ↓
    └──────→ Boolean (existence)
                 │
                 ↓
               Viterbi (confidence)
```

### Key Properties

- **Idempotent:** $a \oplus a = a$ (e.g., Boolean, max-min)
- **Zero-sum-free:** $a \oplus b = \mathbf{0} \Rightarrow a = b = \mathbf{0}$
- **Zero-divisor-free:** $a \otimes b = \mathbf{0} \Rightarrow a = \mathbf{0} \lor b = \mathbf{0}$

---

## Appendix B: Example Complexity Proof Sketch

**Claim:** BGP evaluation with confidence threshold is NP-hard (combined complexity).

**Proof sketch:**

Reduce from 3-SAT.

Given a 3-CNF formula $\phi = C_1 \land \cdots \land C_m$ over variables $x_1, \ldots, x_n$:

1. **Construct RDF store $\mathcal{A}$:**
   - For each variable $x_i$: triples for true/false assignments with confidence 1.0
   - For each clause $C_j$: triples connecting clause to satisfying literals

2. **Construct BGP $P$:**
   - Pattern matching selects one assignment per variable
   - Joins enforce consistency
   - Additional pattern checks clause satisfaction

3. **Set threshold $\tau = 1.0$:**
   - A satisfying assignment exists iff the BGP has a result with confidence $\geq \tau$

4. **Polynomial reduction:**
   - Store size: $O(n + m)$
   - BGP size: $O(n + m)$
   - Construction time: polynomial

Since 3-SAT is NP-complete and we have a polynomial reduction, BGP + confidence threshold is NP-hard. $\square$

**Note:** Full proof requires formal definition of the reduction and verification of correctness.

---

*Document Version: 1.0*  
*Created: February 2026*  
*Status: Research Planning*
