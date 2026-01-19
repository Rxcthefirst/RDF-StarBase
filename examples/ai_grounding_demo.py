#!/usr/bin/env python
"""
AI Grounding API - Complete Demo

This script demonstrates the full AI Grounding workflow:
1. Load sample data (customer service scenario)
2. Query facts for RAG
3. Verify claims before stating them
4. Handle conflicting information
5. Get entity context
6. Materialize inferences

Prerequisites:
    - Start the server: uvicorn rdf_starbase.web:app --reload
    - Run this script: python examples/ai_grounding_demo.py

This demo uses a TechCorp customer service scenario with:
- Customer records from CRM
- Support tickets
- Product information
- CONFLICTING data from different systems (realistic!)
"""

import requests
import json
import urllib.parse
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://localhost:8000"

# Sample data using Turtle syntax in INSERT DATA
SAMPLE_DATA = """
PREFIX tc: <http://techcorp.com/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

INSERT DATA {
    # Customer C001
    tc:customer/C001 a tc:Customer ;
        foaf:name "Alice Johnson" ;
        tc:email "alice@example.com" ;
        tc:tier "Premium" ;
        tc:since "2020-01-15"^^xsd:date .
    
    # Customer C002
    tc:customer/C002 a tc:Customer ;
        foaf:name "Bob Smith" ;
        tc:email "bob@example.com" ;
        tc:tier "Standard" ;
        tc:since "2021-06-20"^^xsd:date .
    
    # Product P001
    tc:product/P001 a tc:Product ;
        rdfs:label "CloudSync Pro" ;
        tc:category "Software" ;
        tc:price "299.99"^^xsd:decimal .
    
    # Product P002
    tc:product/P002 a tc:Product ;
        rdfs:label "DataVault" ;
        tc:category "Storage" ;
        tc:price "499.99"^^xsd:decimal .
    
    # Support Ticket T001
    tc:ticket/T001 a tc:SupportTicket ;
        tc:customer tc:customer/C001 ;
        tc:product tc:product/P001 ;
        tc:status "Open" ;
        tc:priority "High" ;
        tc:description "Sync failing intermittently" ;
        tc:created "2024-01-10T09:30:00"^^xsd:dateTime .
    
    # Support Ticket T002
    tc:ticket/T002 a tc:SupportTicket ;
        tc:customer tc:customer/C002 ;
        tc:product tc:product/P002 ;
        tc:status "Resolved" ;
        tc:priority "Medium" ;
        tc:description "Storage quota question" ;
        tc:created "2024-01-08T14:15:00"^^xsd:dateTime ;
        tc:resolved "2024-01-09T10:00:00"^^xsd:dateTime .
}
"""


# =============================================================================
# Helper Functions
# =============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    print(f"\n--- {title} ---")


def sparql_update(query: str) -> dict:
    """Execute SPARQL UPDATE."""
    r = requests.post(f"{BASE_URL}/sparql/update", json={"query": query})
    r.raise_for_status()
    return r.json()


def sparql_query(query: str) -> dict:
    """Execute SPARQL SELECT."""
    r = requests.post(f"{BASE_URL}/sparql", json={"query": query})
    r.raise_for_status()
    return r.json()


def ai_query(subject=None, predicate=None, min_confidence="medium", limit=50):
    """Query facts via AI Grounding API."""
    payload = {"min_confidence": min_confidence, "limit": limit}
    if subject:
        payload["subject"] = subject
    if predicate:
        payload["predicate"] = predicate
    r = requests.post(f"{BASE_URL}/ai/query", json=payload)
    r.raise_for_status()
    return r.json()


def ai_verify(subject, predicate, expected_value=None):
    """Verify a claim via AI Grounding API."""
    payload = {"subject": subject, "predicate": predicate, "min_confidence": "any"}
    if expected_value:
        payload["expected_object"] = expected_value
    r = requests.post(f"{BASE_URL}/ai/verify", json=payload)
    r.raise_for_status()
    return r.json()


def ai_context(entity_iri):
    """Get entity context via AI Grounding API."""
    encoded = urllib.parse.quote(entity_iri, safe="")
    r = requests.get(f"{BASE_URL}/ai/context/{encoded}")
    r.raise_for_status()
    return r.json()


def ai_materialize():
    """Materialize inferences."""
    r = requests.post(f"{BASE_URL}/ai/materialize", json={
        "enable_rdfs": True,
        "enable_owl": True,
        "max_iterations": 100
    })
    r.raise_for_status()
    return r.json()


def add_triple_with_source(subject, predicate, obj, source, confidence):
    """Add a triple with specific source and confidence."""
    r = requests.post(f"{BASE_URL}/triples", json={
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "provenance": {
            "source": source,
            "confidence": confidence
        }
    })
    r.raise_for_status()
    return r.json()


# =============================================================================
# Demo Steps
# =============================================================================

def check_server():
    """Check if server is running."""
    try:
        r = requests.get(f"{BASE_URL}/health")
        return r.status_code == 200
    except:
        return False


def step1_load_data():
    """Step 1: Load sample data."""
    print_header("STEP 1: Loading Sample Data")
    
    print("\nLoading TechCorp customer service data...")
    result = sparql_update(SAMPLE_DATA)
    print(f"✓ Loaded data: {result}")
    
    # Add conflicting data from different sources
    print("\nAdding CONFLICTING data from different systems (realistic scenario)...")
    
    # CRM says tier is "Enterprise"
    # But Billing says tier is "Premium" (different system, different value!)
    add_triple_with_source(
        "http://techcorp.com/customer/C001",
        "http://techcorp.com/tier",
        "Premium",  # Different from "Enterprise" in main data!
        "BillingSystem",
        0.85
    )
    print("  ✓ Added BillingSystem record (tier='Premium', confidence=85%)")
    
    # Support system has different email
    add_triple_with_source(
        "http://techcorp.com/customer/C001",
        "http://schema.org/email",
        "a.johnson@megacorp.com",  # Slightly different email!
        "SupportPortal",
        0.75
    )
    print("  ✓ Added SupportPortal record (different email, confidence=75%)")
    
    print("\n✓ Data loaded with intentional conflicts for demo")


def step2_query_facts():
    """Step 2: Query facts for RAG."""
    print_header("STEP 2: Querying Facts for RAG")
    
    print("\nQuerying all facts about customer C001...")
    result = ai_query(
        subject="http://techcorp.com/customer/C001",
        min_confidence="low"  # Get everything including low-confidence
    )
    
    print(f"\nFound {result['filtered_count']} facts from sources: {result['sources_used']}")
    print("\nFacts with citations:")
    
    for fact in result["facts"]:
        pred = fact["predicate"].split("/")[-1]
        source = fact["citation"]["source"]
        conf = fact["citation"]["confidence"]
        print(f"  • {pred}: {fact['object']}")
        print(f"    └─ source: {source}, confidence: {conf:.0%}")
    
    # Show how to build LLM context
    print("\n" + "-" * 50)
    print("Context string for LLM prompt:")
    print("-" * 50)
    
    lines = []
    for fact in result["facts"]:
        pred = fact["predicate"].split("/")[-1].split("#")[-1]
        source = fact["citation"]["source"]
        conf = fact["citation"]["confidence"]
        lines.append(f"- {pred}: {fact['object']} (source: {source}, {conf:.0%})")
    
    context = "\n".join(lines)
    print(context)


def step3_verify_claims():
    """Step 3: Verify claims before stating them."""
    print_header("STEP 3: Verifying Claims Before Stating Them")
    
    # Verify a true claim
    print_section("Verifying: 'Customer name is Alice Johnson'")
    result = ai_verify(
        "http://techcorp.com/customer/C001",
        "http://xmlns.com/foaf/0.1/name",  # foaf:name
        "Alice Johnson"
    )
    print(f"  Supported: {result['claim_supported']}")
    print(f"  Confidence: {result['confidence']:.0%}" if result['confidence'] else "  Confidence: N/A")
    print(f"  Conflicts: {result['has_conflicts']}")
    print(f"  → {result['recommendation']}")
    
    # Verify a contested claim (tier)
    print_section("Verifying: 'Customer tier is Enterprise'")
    result = ai_verify(
        "http://techcorp.com/customer/C001",
        "http://techcorp.com/tier",
        "Enterprise"
    )
    print(f"  Supported: {result['claim_supported']}")
    print(f"  Confidence: {result['confidence']:.0%}" if result['confidence'] else "  Confidence: N/A")
    print(f"  Conflicts: {result['has_conflicts']}")
    
    if result['has_conflicts']:
        print("\n  ⚠️ CONFLICTING VALUES DETECTED:")
        for f in result['supporting_facts']:
            print(f"    ✓ '{f['object']}' from {f['citation']['source']}")
        for f in result['contradicting_facts']:
            print(f"    ✗ '{f['object']}' from {f['citation']['source']}")
    
    print(f"\n  → {result['recommendation']}")
    
    # Verify a false claim
    print_section("Verifying: 'Customer tier is Gold' (FALSE)")
    result = ai_verify(
        "http://techcorp.com/customer/C001",
        "http://techcorp.com/tier",
        "Gold"
    )
    print(f"  Supported: {result['claim_supported']}")
    print(f"  → {result['recommendation']}")


def step4_entity_context():
    """Step 4: Get comprehensive entity context."""
    print_header("STEP 4: Getting Entity Context")
    
    print("\nFetching full context for customer C001...")
    context = ai_context("http://techcorp.com/customer/C001")
    
    print(f"\nEntity: {context['entity']}")
    print(f"Total facts: {len(context['facts'])}")
    print(f"Sources: {context['sources']}")
    print(f"Related entities: {context['related_entities'][:5]}...")
    
    print("\nConfidence Summary:")
    summary = context['confidence_summary']
    print(f"  • High confidence (≥90%): {summary['high_confidence_count']} facts")
    print(f"  • Medium confidence (70-89%): {summary['medium_confidence_count']} facts")
    print(f"  • Low confidence (<70%): {summary['low_confidence_count']} facts")
    print(f"  • Average: {summary['avg']:.0%}")


def step5_handle_conflicts():
    """Step 5: Gracefully handle conflicting information."""
    print_header("STEP 5: Handling Conflicting Information")
    
    print("\nScenario: User asks 'What tier is Alice Johnson on?'")
    print("\nStep 1: Query facts about the tier...")
    
    result = ai_verify(
        "http://techcorp.com/customer/C001",
        "http://techcorp.com/tier"
    )
    
    if result['has_conflicts']:
        print("\n⚠️ CONFLICT DETECTED - Multiple values from different systems")
        
        # Group by value
        values = {}
        for f in result['supporting_facts']:
            v = f['object']
            if v not in values:
                values[v] = []
            values[v].append({
                'source': f['citation']['source'],
                'confidence': f['citation']['confidence']
            })
        
        print("\nCompeting claims:")
        for value, sources in values.items():
            for s in sources:
                print(f"  • '{value}' - {s['source']} ({s['confidence']:.0%} confidence)")
        
        print("\n" + "-" * 50)
        print("SAFE AI RESPONSE:")
        print("-" * 50)
        
        response = """Based on the knowledge base, there are different values reported:

"""
        for value, sources in values.items():
            source_names = [s['source'] for s in sources]
            best_conf = max(s['confidence'] for s in sources)
            response += f"• According to {', '.join(source_names)}: {value} ({best_conf:.0%} confidence)\n"
        
        response += """
Note: The CRM and Billing systems report different tier values. 
You may want to verify with the account manager which is current."""
        
        print(response)
    else:
        fact = result['supporting_facts'][0]
        print(f"\n✓ Consistent answer: {fact['object']}")
        print(f"  Source: {fact['citation']['source']}")
        print(f"  Confidence: {fact['citation']['confidence']:.0%}")


def step6_materialize_inferences():
    """Step 6: Materialize inferences with reasoning."""
    print_header("STEP 6: Materializing Inferences (RDFS/OWL Reasoning)")
    
    print("\nRunning reasoner to derive new facts...")
    try:
        result = ai_materialize()
        
        print(f"\n✓ Reasoning complete:")
        print(f"  • Iterations: {result['iterations']}")
        print(f"  • Triples inferred: {result['triples_inferred']}")
        print(f"  • RDFS inferences: {result['rdfs_inferences']}")
        print(f"  • OWL inferences: {result['owl_inferences']}")
        
        if result['triples_inferred'] > 0:
            print("\nExample: Customer C001 was typed as EnterpriseCustomer")
            print("         → Now also inferred to be a Customer (via rdfs:subClassOf)")
            print("         → And also a schema:Organization (transitive subclass)")
            
            # Query to show inferred types
            print("\nQuerying inferred types...")
            query_result = sparql_query("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?type WHERE {
                    <http://techcorp.com/customer/C001> rdf:type ?type
                }
            """)
            
            if 'results' in query_result:
                print("\nCustomer C001 types:")
                for row in query_result['results']:
                    type_uri = row.get('type', str(row))
                    type_name = type_uri.split('/')[-1].split('#')[-1]
                    print(f"  • {type_name}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 501:
            print("\n⚠️ Reasoning engine not available (requires new storage layer)")
            print("   This is a preview feature - the API endpoint exists but the")
            print("   underlying storage layer for the reasoner is still in development.")
            print("   Skipping this step...")
        else:
            print(f"\n❌ API Error: {e}")
            raise


def step7_demo_ai_conversation():
    """Step 7: Simulate an AI conversation using the API."""
    print_header("STEP 7: Simulated AI Conversation")
    
    print("\n" + "─" * 60)
    print("USER: Tell me about customer Alice Johnson")
    print("─" * 60)
    
    # Get context
    context = ai_context("http://techcorp.com/customer/C001")
    
    print("\n🤖 AI (using grounded facts):\n")
    
    # Build response from facts
    name = None
    email = None
    tier_info = []
    tickets = []
    
    for fact in context['facts']:
        pred = fact['predicate'].split('/')[-1].split('#')[-1]
        if pred == 'name' and fact['subject'].endswith('C001'):
            name = fact['object']
        elif pred == 'email' and fact['subject'].endswith('C001'):
            # Track all emails (there might be conflicts)
            email = fact['object']
        elif pred == 'tier':
            tier_info.append({
                'value': fact['object'],
                'source': fact['citation']['source'],
                'confidence': fact['citation']['confidence']
            })
        elif pred == 'customer' and 'ticket' in fact['subject']:
            tickets.append(fact['subject'].split('/')[-1])
    
    print(f"I found information about {name}:\n")
    
    # Handle tier carefully (known conflict)
    if len(tier_info) > 1:
        print("**Account Tier:** There are different values in our systems:")
        for t in tier_info:
            print(f"  - {t['value']} (per {t['source']}, {t['confidence']:.0%} confidence)")
    elif tier_info:
        print(f"**Account Tier:** {tier_info[0]['value']} (source: {tier_info[0]['source']})")
    
    if tickets:
        print(f"\n**Support Tickets:** {', '.join(tickets)}")
    
    print(f"\n**Sources consulted:** {', '.join(context['sources'])}")
    print(f"**Data quality:** {context['confidence_summary']['high_confidence_count']} high-confidence facts")
    
    # Follow-up question
    print("\n" + "─" * 60)
    print("USER: Does Alice have any open tickets?")
    print("─" * 60)
    
    # Query tickets
    result = sparql_query("""
        PREFIX tc: <http://techcorp.com/>
        SELECT ?ticket ?status ?priority ?description WHERE {
            ?ticket tc:customer <http://techcorp.com/customer/C001> ;
                    tc:status ?status ;
                    tc:priority ?priority ;
                    <http://schema.org/description> ?description .
            FILTER(?status = "Open")
        }
    """)
    
    print("\n🤖 AI:\n")
    
    if 'results' in result and result['results']:
        print("Yes, I found the following open tickets:\n")
        for ticket in result['results']:
            print(f"  📋 {ticket.get('ticket', 'Unknown').split('/')[-1]}")
            print(f"     Priority: {ticket.get('priority', 'Unknown')}")
            print(f"     Issue: {ticket.get('description', 'No description')}")
    else:
        print("I don't see any open tickets for this customer.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "     RDF-StarBase AI Grounding API - Complete Demo".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Check server
    if not check_server():
        print("\n❌ ERROR: Server not running!")
        print("\nPlease start the server first:")
        print("  uvicorn rdf_starbase.web:app --reload")
        print("\nThen run this script again.")
        return
    
    print("\n✓ Server is running at", BASE_URL)
    
    try:
        step1_load_data()
        step2_query_facts()
        step3_verify_claims()
        step4_entity_context()
        step5_handle_conflicts()
        step6_materialize_inferences()
        step7_demo_ai_conversation()
        
        print("\n" + "=" * 70)
        print("  DEMO COMPLETE!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  1. Always query facts with /ai/query before making claims")
        print("  2. Verify specific claims with /ai/verify before stating them")
        print("  3. Handle conflicts gracefully - show competing values")
        print("  4. Include source attribution in AI responses")
        print("  5. Use confidence scores to weight information")
        print("\nExplore the API at: http://localhost:8000/docs")
        print("Full guide at: docs/AI_GROUNDING_GUIDE.md")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        raise


if __name__ == "__main__":
    main()
