# AI Grounding API - Comprehensive Workflow Guide

> **Building Trustworthy AI with Provenance-Backed Knowledge**

This guide demonstrates how to use RDF-StarBase's AI Grounding API to build AI systems that:
- Ground responses in **verified facts** with confidence scores
- **Cite sources** for every claim
- Detect and handle **conflicting information**
- Provide **transparent provenance** chains

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Scenario: AI Customer Service Agent](#scenario-ai-customer-service-agent)
3. [Step 1: Loading Knowledge](#step-1-loading-knowledge)
4. [Step 2: Querying Facts for RAG](#step-2-querying-facts-for-rag)
5. [Step 3: Verifying Claims](#step-3-verifying-claims)
6. [Step 4: Getting Entity Context](#step-4-getting-entity-context)
7. [Step 5: Handling Conflicting Information](#step-5-handling-conflicting-information)
8. [Step 6: Materializing Inferences](#step-6-materializing-inferences)
9. [Complete Python Client](#complete-python-client)
10. [Integration with LangChain/LlamaIndex](#integration-with-langchainllamaindex)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Your AI Application                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Chatbot   │  │  RAG Agent  │  │ Fact-Check  │  │   Search    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼───────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RDF-StarBase AI Grounding API                        │
├─────────────────┬─────────────────┬───────────────┬────────────────────┤
│  POST /ai/query │ POST /ai/verify │ GET /ai/context│ POST /ai/materialize│
│  ───────────────│─────────────────│───────────────│────────────────────│
│  Structured     │ Claim           │ Entity        │ Inference          │
│  fact retrieval │ verification    │ exploration   │ materialization    │
│  with citations │ with evidence   │ with related  │ with RDFS/OWL     │
└─────────────────┴─────────────────┴───────────────┴────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Triple Store (Polars DataFrames)                       │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │ subject | predicate | object | source | confidence | timestamp     ││
│  ├────────────────────────────────────────────────────────────────────┤│
│  │ Every triple carries: WHO said it, WHEN, HOW CONFIDENT, WHICH proc ││
│  └────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Differences: AI API vs UI API

| Aspect | UI API (`/graph/*`) | AI Grounding API (`/ai/*`) |
|--------|---------------------|----------------------------|
| **Consumer** | D3.js visualization | LLM tool calls / agents |
| **Response format** | Nodes + edges for rendering | Facts + provenance + citations |
| **Query pattern** | Browsing, neighborhood | Precise fact lookup, verification |
| **Filtering** | Limit by count | Confidence threshold, freshness |
| **Latency goal** | 200ms acceptable | Sub-100ms for tool calls |

---

## Scenario: AI Customer Service Agent

Imagine building an AI agent for **TechCorp**, a software company. The agent needs to:

1. Answer questions about products, customers, and support tickets
2. **Never hallucinate** - only state facts backed by data
3. **Cite sources** when making claims
4. Handle **conflicting information** gracefully (e.g., different systems report different data)

We'll load a sample dataset and demonstrate the full workflow.

---

## Step 1: Loading Knowledge

First, let's load some example data into RDF-StarBase.

### 1.1 Start the Server

```powershell
# In one terminal
uvicorn rdf_starbase.web:app --reload
```

### 1.2 Load Sample Data via API

```python
import requests

BASE_URL = "http://localhost:8000"

# Create a named graph for customer data
requests.post(f"{BASE_URL}/sparql/update", json={
    "query": "CREATE SILENT GRAPH <http://techcorp.com/graphs/customers>"
})

# Insert sample customer data from multiple sources
requests.post(f"{BASE_URL}/sparql/update", json={
    "query": """
        INSERT DATA {
            # Customer record from CRM system
            <http://techcorp.com/customer/C001> 
                <http://schema.org/name> "Alice Johnson" ;
                <http://schema.org/email> "alice@example.com" ;
                <http://techcorp.com/tier> "Enterprise" ;
                <http://techcorp.com/accountManager> "Bob Smith" ;
                <http://techcorp.com/since> "2023-03-15" .
            
            # Product data
            <http://techcorp.com/product/CloudSuite>
                <http://schema.org/name> "CloudSuite Pro" ;
                <http://schema.org/description> "Enterprise cloud management platform" ;
                <http://schema.org/price> "299.99" .
            
            # Support ticket
            <http://techcorp.com/ticket/T-2024-001>
                <http://techcorp.com/customer> <http://techcorp.com/customer/C001> ;
                <http://techcorp.com/status> "Open" ;
                <http://techcorp.com/priority> "High" ;
                <http://schema.org/description> "SSO integration not working" .
        }
    """
})

print("✓ Sample data loaded")
```

### 1.3 Load Conflicting Data (Simulating Real-World Scenario)

In the real world, different systems often have different values. Let's add a conflicting record:

```python
# Add a triple with different source and confidence
requests.post(f"{BASE_URL}/triples", json={
    "subject": "http://techcorp.com/customer/C001",
    "predicate": "http://techcorp.com/tier",
    "object": "Premium",  # Different from "Enterprise" above!
    "provenance": {
        "source": "BillingSystem",
        "confidence": 0.85,
        "process": "nightly_sync"
    }
})

# Now we have COMPETING CLAIMS about customer tier!
print("✓ Conflicting data loaded")
```

---

## Step 2: Querying Facts for RAG

The `/ai/query` endpoint retrieves facts with full provenance for RAG pipelines.

### Request

```python
response = requests.post(f"{BASE_URL}/ai/query", json={
    "subject": "http://techcorp.com/customer/C001",
    "min_confidence": "medium",  # "high" (>=0.9), "medium" (>=0.7), "low" (>=0.5), "any"
    "max_age_days": 30,          # Only facts from last 30 days
    "include_inferred": True,    # Include reasoner-derived facts
    "limit": 50
})

facts = response.json()
```

### Response

```json
{
    "facts": [
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://schema.org/name",
            "object": "Alice Johnson",
            "citation": {
                "fact_hash": "a1b2c3d4e5f6",
                "source": "CRM_System",
                "confidence": 0.95,
                "timestamp": "2026-01-17T10:30:00Z",
                "retrieval_time": "2026-01-17T14:22:00Z"
            }
        },
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://techcorp.com/tier",
            "object": "Enterprise",
            "citation": {
                "fact_hash": "b2c3d4e5f6a1",
                "source": "CRM_System",
                "confidence": 0.90,
                "timestamp": "2026-01-17T10:30:00Z",
                "retrieval_time": "2026-01-17T14:22:00Z"
            }
        },
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://techcorp.com/tier",
            "object": "Premium",
            "citation": {
                "fact_hash": "c3d4e5f6a1b2",
                "source": "BillingSystem",
                "confidence": 0.85,
                "timestamp": "2026-01-17T11:00:00Z",
                "retrieval_time": "2026-01-17T14:22:00Z"
            }
        }
    ],
    "total_count": 6,
    "filtered_count": 6,
    "confidence_threshold": 0.7,
    "retrieval_timestamp": "2026-01-17T14:22:00Z",
    "sources_used": ["CRM_System", "BillingSystem"]
}
```

### Using in Your AI

```python
def build_context_for_llm(facts):
    """Convert facts to context string for LLM prompt."""
    context_lines = []
    for fact in facts["facts"]:
        pred_label = fact["predicate"].split("/")[-1]
        source = fact["citation"]["source"]
        conf = fact["citation"]["confidence"]
        
        context_lines.append(
            f"- {pred_label}: {fact['object']} "
            f"(source: {source}, confidence: {conf:.0%})"
        )
    
    return "\n".join(context_lines)

# Build prompt for LLM
context = build_context_for_llm(facts)
prompt = f"""
You are a customer service agent for TechCorp. Answer based ONLY on these verified facts:

{context}

User question: What tier is customer Alice Johnson on?

Important: Cite your sources. If there are conflicting facts, acknowledge them.
"""
```

---

## Step 3: Verifying Claims

Before your AI states something, use `/ai/verify` to check if it's supported.

### Verify a Specific Claim

```python
# "Is Alice Johnson's tier Enterprise?"
response = requests.post(f"{BASE_URL}/ai/verify", json={
    "subject": "http://techcorp.com/customer/C001",
    "predicate": "http://techcorp.com/tier",
    "expected_object": "Enterprise",
    "min_confidence": "medium"
})

result = response.json()
```

### Response When Claim is Contested

```json
{
    "claim_supported": true,
    "confidence": 0.90,
    "supporting_facts": [
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://techcorp.com/tier",
            "object": "Enterprise",
            "citation": {
                "source": "CRM_System",
                "confidence": 0.90
            }
        }
    ],
    "contradicting_facts": [
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://techcorp.com/tier",
            "object": "Premium",
            "citation": {
                "source": "BillingSystem",
                "confidence": 0.85
            }
        }
    ],
    "has_conflicts": true,
    "recommendation": "The claim is supported by 1 source(s) but contradicted by 1 source(s). The AI should present this as contested information with sources for both perspectives."
}
```

### Using Verification in Your AI

```python
def verify_before_stating(subject, predicate, value):
    """Verify a claim before the AI states it."""
    result = requests.post(f"{BASE_URL}/ai/verify", json={
        "subject": subject,
        "predicate": predicate,
        "expected_object": value,
        "min_confidence": "medium"
    }).json()
    
    if not result["claim_supported"]:
        return f"I cannot confirm that claim. {result['recommendation']}"
    
    if result["has_conflicts"]:
        sources_for = [f["citation"]["source"] for f in result["supporting_facts"]]
        sources_against = [f["citation"]["source"] for f in result["contradicting_facts"]]
        return (
            f"This is contested: {sources_for} say '{value}', "
            f"but {sources_against} report a different value."
        )
    
    confidence = result["confidence"]
    source = result["supporting_facts"][0]["citation"]["source"]
    return f"Yes, confirmed with {confidence:.0%} confidence (source: {source})"

# Example usage
print(verify_before_stating(
    "http://techcorp.com/customer/C001",
    "http://techcorp.com/tier",
    "Enterprise"
))
# Output: "This is contested: ['CRM_System'] say 'Enterprise', but ['BillingSystem'] report a different value."
```

---

## Step 4: Getting Entity Context

Use `/ai/context/{iri}` to get everything known about an entity.

### Request

```python
import urllib.parse

entity_iri = "http://techcorp.com/customer/C001"
encoded_iri = urllib.parse.quote(entity_iri, safe="")

response = requests.get(
    f"{BASE_URL}/ai/context/{encoded_iri}",
    params={
        "min_confidence": "low",
        "include_incoming": True,  # Also get facts where entity is the object
        "limit": 100
    }
)

context = response.json()
```

### Response

```json
{
    "entity": "http://techcorp.com/customer/C001",
    "facts": [
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://schema.org/name",
            "object": "Alice Johnson",
            "citation": { "source": "CRM_System", "confidence": 0.95 }
        },
        {
            "subject": "http://techcorp.com/customer/C001",
            "predicate": "http://schema.org/email",
            "object": "alice@example.com",
            "citation": { "source": "CRM_System", "confidence": 0.95 }
        },
        {
            "subject": "http://techcorp.com/ticket/T-2024-001",
            "predicate": "http://techcorp.com/customer",
            "object": "http://techcorp.com/customer/C001",
            "citation": { "source": "TicketSystem", "confidence": 0.99 }
        }
    ],
    "related_entities": [
        "http://techcorp.com/ticket/T-2024-001",
        "http://techcorp.com/product/CloudSuite"
    ],
    "sources": ["CRM_System", "BillingSystem", "TicketSystem"],
    "confidence_summary": {
        "min": 0.85,
        "max": 0.99,
        "avg": 0.92,
        "high_confidence_count": 5,
        "medium_confidence_count": 1,
        "low_confidence_count": 0
    },
    "retrieval_timestamp": "2026-01-17T14:30:00Z"
}
```

### Building a Rich Profile

```python
def build_entity_profile(entity_iri):
    """Build a comprehensive profile for an entity."""
    encoded = urllib.parse.quote(entity_iri, safe="")
    context = requests.get(f"{BASE_URL}/ai/context/{encoded}").json()
    
    profile = {
        "entity": entity_iri,
        "properties": {},
        "relationships": [],
        "data_quality": context["confidence_summary"]
    }
    
    for fact in context["facts"]:
        pred = fact["predicate"].split("/")[-1].split("#")[-1]
        
        if fact["subject"] == entity_iri:
            # Outgoing property
            if pred not in profile["properties"]:
                profile["properties"][pred] = []
            profile["properties"][pred].append({
                "value": fact["object"],
                "source": fact["citation"]["source"],
                "confidence": fact["citation"]["confidence"]
            })
        else:
            # Incoming relationship
            profile["relationships"].append({
                "from": fact["subject"],
                "predicate": pred,
                "source": fact["citation"]["source"]
            })
    
    return profile

profile = build_entity_profile("http://techcorp.com/customer/C001")
print(json.dumps(profile, indent=2))
```

---

## Step 5: Handling Conflicting Information

RDF-StarBase is designed for competing claims. Here's how to handle them:

### Detect Conflicts with Verification

```python
def check_for_conflicts(subject, predicate):
    """Check if there are conflicting values for a property."""
    result = requests.post(f"{BASE_URL}/ai/verify", json={
        "subject": subject,
        "predicate": predicate,
        # No expected_object - just check what's there
        "min_confidence": "any"
    }).json()
    
    if result["has_conflicts"]:
        values = {}
        for fact in result["supporting_facts"]:
            value = str(fact["object"])
            source = fact["citation"]["source"]
            conf = fact["citation"]["confidence"]
            
            if value not in values:
                values[value] = []
            values[value].append({"source": source, "confidence": conf})
        
        return {
            "has_conflicts": True,
            "values": values,
            "recommendation": result["recommendation"]
        }
    
    return {"has_conflicts": False}

conflicts = check_for_conflicts(
    "http://techcorp.com/customer/C001",
    "http://techcorp.com/tier"
)
print(json.dumps(conflicts, indent=2))
```

### Output

```json
{
    "has_conflicts": true,
    "values": {
        "Enterprise": [
            {"source": "CRM_System", "confidence": 0.90}
        ],
        "Premium": [
            {"source": "BillingSystem", "confidence": 0.85}
        ]
    },
    "recommendation": "Multiple values found (2 distinct) from different sources. The AI should acknowledge the competing claims and cite sources."
}
```

### AI Response Strategy

```python
def generate_safe_response(subject, predicate):
    """Generate a response that handles conflicts gracefully."""
    conflicts = check_for_conflicts(subject, predicate)
    
    if not conflicts["has_conflicts"]:
        # Simple case - one consistent value
        result = requests.post(f"{BASE_URL}/ai/verify", json={
            "subject": subject,
            "predicate": predicate,
            "min_confidence": "any"
        }).json()
        
        if result["supporting_facts"]:
            fact = result["supporting_facts"][0]
            return f"According to {fact['citation']['source']}, the value is {fact['object']}."
        return "No information available."
    
    # Conflict case - present both perspectives
    lines = ["There are different values reported by different systems:"]
    
    for value, sources in conflicts["values"].items():
        source_list = ", ".join([s["source"] for s in sources])
        best_conf = max(s["confidence"] for s in sources)
        lines.append(f"  • {value} (from {source_list}, {best_conf:.0%} confidence)")
    
    return "\n".join(lines)

print(generate_safe_response(
    "http://techcorp.com/customer/C001",
    "http://techcorp.com/tier"
))
```

### Output

```
There are different values reported by different systems:
  • Enterprise (from CRM_System, 90% confidence)
  • Premium (from BillingSystem, 85% confidence)
```

---

## Step 6: Materializing Inferences

Use RDFS/OWL reasoning to derive new facts.

### Define Ontology Relationships

```python
# Add some class hierarchy
requests.post(f"{BASE_URL}/sparql/update", json={
    "query": """
        INSERT DATA {
            # Define class hierarchy
            <http://techcorp.com/EnterpriseCustomer> 
                <http://www.w3.org/2000/01/rdf-schema#subClassOf> 
                <http://techcorp.com/Customer> .
            
            <http://techcorp.com/Customer>
                <http://www.w3.org/2000/01/rdf-schema#subClassOf>
                <http://schema.org/Organization> .
            
            # Classify our customer
            <http://techcorp.com/customer/C001>
                <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                <http://techcorp.com/EnterpriseCustomer> .
        }
    """
})
```

### Trigger Reasoning

```python
response = requests.post(f"{BASE_URL}/ai/materialize", json={
    "enable_rdfs": True,   # RDFS entailment (subclass, subproperty)
    "enable_owl": True,    # OWL 2 RL (sameAs, inverseOf, transitivity)
    "max_iterations": 100
})

result = response.json()
print(f"Inferred {result['triples_inferred']} new triples")
```

### Response

```json
{
    "success": true,
    "iterations": 3,
    "triples_inferred": 2,
    "rdfs_inferences": 2,
    "owl_inferences": 0,
    "breakdown": {
        "rdfs:subClassOf": 2
    }
}
```

Now `C001` is also a `Customer` and `Organization` (inferred via rdfs:subClassOf).

### Query Inferences

```python
response = requests.get(f"{BASE_URL}/ai/inferences", params={"limit": 50})
inferences = response.json()

for inf in inferences["inferences"]:
    print(f"Inferred: {inf['subject']} → {inf['predicate'].split('#')[-1]} → {inf['object']}")
```

---

## Complete Python Client

Here's a complete client class for the AI Grounding API:

```python
"""
RDF-StarBase AI Grounding Client

A Python client for grounding AI responses in verified knowledge.
"""

import requests
import urllib.parse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class GroundedFact:
    subject: str
    predicate: str
    object: Any
    source: str
    confidence: float
    timestamp: str
    fact_hash: str
    
    @property
    def predicate_label(self) -> str:
        return self.predicate.split("/")[-1].split("#")[-1]


class AIGroundingClient:
    """Client for RDF-StarBase AI Grounding API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> bool:
        """Check if the API is available."""
        try:
            r = requests.get(f"{self.base_url}/ai/health")
            return r.status_code == 200
        except:
            return False
    
    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: str = "medium",
        max_age_days: Optional[int] = None,
        limit: int = 100
    ) -> List[GroundedFact]:
        """Query facts with provenance for RAG."""
        payload = {
            "min_confidence": min_confidence,
            "limit": limit
        }
        if subject:
            payload["subject"] = subject
        if predicate:
            payload["predicate"] = predicate
        if obj:
            payload["object"] = obj
        if max_age_days:
            payload["max_age_days"] = max_age_days
        
        r = requests.post(f"{self.base_url}/ai/query", json=payload)
        r.raise_for_status()
        data = r.json()
        
        return [
            GroundedFact(
                subject=f["subject"],
                predicate=f["predicate"],
                object=f["object"],
                source=f["citation"]["source"],
                confidence=f["citation"]["confidence"],
                timestamp=f["citation"]["timestamp"],
                fact_hash=f["citation"]["fact_hash"]
            )
            for f in data["facts"]
        ]
    
    def verify_claim(
        self,
        subject: str,
        predicate: str,
        expected_value: Optional[str] = None,
        min_confidence: str = "medium"
    ) -> Dict[str, Any]:
        """Verify if a claim is supported by the knowledge base."""
        payload = {
            "subject": subject,
            "predicate": predicate,
            "min_confidence": min_confidence
        }
        if expected_value:
            payload["expected_object"] = expected_value
        
        r = requests.post(f"{self.base_url}/ai/verify", json=payload)
        r.raise_for_status()
        return r.json()
    
    def get_entity_context(
        self,
        entity_iri: str,
        min_confidence: str = "low",
        include_incoming: bool = True
    ) -> Dict[str, Any]:
        """Get full context about an entity."""
        encoded = urllib.parse.quote(entity_iri, safe="")
        r = requests.get(
            f"{self.base_url}/ai/context/{encoded}",
            params={
                "min_confidence": min_confidence,
                "include_incoming": include_incoming
            }
        )
        r.raise_for_status()
        return r.json()
    
    def build_context_string(
        self,
        subject: str,
        min_confidence: str = "medium"
    ) -> str:
        """Build a context string for LLM prompts."""
        facts = self.query_facts(subject=subject, min_confidence=min_confidence)
        
        if not facts:
            return "No verified information available."
        
        lines = []
        for fact in facts:
            lines.append(
                f"- {fact.predicate_label}: {fact.object} "
                f"[source: {fact.source}, confidence: {fact.confidence:.0%}]"
            )
        
        return "\n".join(lines)
    
    def safe_get_value(
        self,
        subject: str,
        predicate: str
    ) -> Dict[str, Any]:
        """Safely get a value, handling conflicts."""
        result = self.verify_claim(subject, predicate, min_confidence="any")
        
        if not result["supporting_facts"]:
            return {
                "found": False,
                "message": "No information available"
            }
        
        if result["has_conflicts"]:
            values = {}
            for f in result["supporting_facts"]:
                v = str(f["object"])
                if v not in values:
                    values[v] = []
                values[v].append({
                    "source": f["citation"]["source"],
                    "confidence": f["citation"]["confidence"]
                })
            return {
                "found": True,
                "has_conflicts": True,
                "values": values,
                "recommendation": result["recommendation"]
            }
        
        fact = result["supporting_facts"][0]
        return {
            "found": True,
            "has_conflicts": False,
            "value": fact["object"],
            "source": fact["citation"]["source"],
            "confidence": fact["citation"]["confidence"]
        }


# Example usage
if __name__ == "__main__":
    client = AIGroundingClient()
    
    if not client.health_check():
        print("❌ Server not available")
        exit(1)
    
    print("=" * 60)
    print("RDF-StarBase AI Grounding Client Demo")
    print("=" * 60)
    
    # Query facts
    print("\n📋 Querying facts about a customer...")
    facts = client.query_facts(
        subject="http://techcorp.com/customer/C001",
        min_confidence="medium"
    )
    for fact in facts:
        print(f"  {fact.predicate_label}: {fact.object} ({fact.source})")
    
    # Verify a claim
    print("\n✓ Verifying a claim...")
    result = client.verify_claim(
        subject="http://techcorp.com/customer/C001",
        predicate="http://schema.org/name",
        expected_value="Alice Johnson"
    )
    print(f"  Supported: {result['claim_supported']}")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Build context for LLM
    print("\n📝 Building LLM context...")
    context = client.build_context_string("http://techcorp.com/customer/C001")
    print(context)
```

---

## Integration with LangChain/LlamaIndex

### LangChain Tool

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class RDFFactQueryInput(BaseModel):
    entity: str = Field(description="The entity IRI to query facts about")


class RDFFactQueryTool(BaseTool):
    name = "rdf_fact_query"
    description = """
    Query verified facts about an entity from the knowledge base.
    Returns facts with source attribution and confidence scores.
    Use this to ground your responses in verified information.
    """
    args_schema = RDFFactQueryInput
    
    def __init__(self, client: AIGroundingClient):
        super().__init__()
        self.client = client
    
    def _run(self, entity: str) -> str:
        return self.client.build_context_string(entity)


class RDFClaimVerifyInput(BaseModel):
    subject: str = Field(description="The subject of the claim")
    predicate: str = Field(description="The predicate/property")
    value: str = Field(description="The value to verify")


class RDFClaimVerifyTool(BaseTool):
    name = "rdf_claim_verify"
    description = """
    Verify if a specific claim is supported by the knowledge base.
    Use this BEFORE stating any fact to ensure accuracy.
    Returns whether the claim is supported and any conflicts.
    """
    args_schema = RDFClaimVerifyInput
    
    def __init__(self, client: AIGroundingClient):
        super().__init__()
        self.client = client
    
    def _run(self, subject: str, predicate: str, value: str) -> str:
        result = self.client.verify_claim(subject, predicate, value)
        return result["recommendation"]
```

### LlamaIndex Retriever

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode


class RDFStarBaseRetriever(BaseRetriever):
    """Retrieve grounded facts from RDF-StarBase."""
    
    def __init__(self, client: AIGroundingClient, min_confidence: str = "medium"):
        self.client = client
        self.min_confidence = min_confidence
    
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # Extract entity from query (simplified - use NER in production)
        query = query_bundle.query_str
        
        # Query for related facts
        facts = self.client.query_facts(min_confidence=self.min_confidence)
        
        nodes = []
        for fact in facts:
            text = f"{fact.subject} {fact.predicate_label} {fact.object}"
            metadata = {
                "source": fact.source,
                "confidence": fact.confidence,
                "fact_hash": fact.fact_hash
            }
            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=fact.confidence))
        
        return nodes
```

---

## Summary

The AI Grounding API provides:

| Endpoint | Purpose | When to Use |
|----------|---------|-------------|
| `POST /ai/query` | Retrieve facts with citations | Building RAG context |
| `POST /ai/verify` | Check claim validity | Before stating facts |
| `GET /ai/context/{iri}` | Get entity profile | Understanding an entity |
| `POST /ai/materialize` | Run reasoning | Derive new facts |
| `GET /ai/inferences` | List derived facts | Audit reasoning results |

### Key Principles

1. **Never hallucinate** - Query facts before making claims
2. **Always cite** - Include source attribution
3. **Handle conflicts** - Acknowledge when data disagrees
4. **Trust scores matter** - Filter by confidence level
5. **Freshness counts** - Use `max_age_days` for time-sensitive data

---

**RDF-StarBase** — *The place where AI systems find truth, not just data.*
