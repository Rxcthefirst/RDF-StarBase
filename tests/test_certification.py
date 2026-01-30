"""
Tests for the Certification and Compliance Module.

Tests W3C SPARQL 1.1, RDF-Star, and Security audit functionality.
"""

import pytest
from datetime import datetime, timedelta

from rdf_starbase.storage.certification import (
    ComplianceTestStatus,
    ComplianceTestCategory,
    Severity,
    ComplianceTestCase,
    ComplianceTestResult,
    ComplianceReport,
    SPARQL11TestSuite,
    RDFStarTestSuite,
    AuditFinding,
    SecurityAuditReport,
    SecurityAuditor,
    CertificationManager,
    run_sparql_compliance,
    run_rdfstar_compliance,
    run_security_audit,
)


# ============================================================================
# TestStatus Tests
# ============================================================================

class TestComplianceTestStatus:
    """Tests for TestStatus enum."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert ComplianceTestStatus.PASSED.value == "passed"
        assert ComplianceTestStatus.FAILED.value == "failed"
        assert ComplianceTestStatus.SKIPPED.value == "skipped"
        assert ComplianceTestStatus.ERROR.value == "error"
        assert ComplianceTestStatus.NOT_IMPLEMENTED.value == "not_implemented"
    
    def test_status_count(self):
        """Test expected number of statuses."""
        assert len(ComplianceTestStatus) == 5


# ============================================================================
# TestCategory Tests
# ============================================================================

class TestComplianceTestCategory:
    """Tests for TestCategory enum."""
    
    def test_sparql_basic_categories(self):
        """Test SPARQL basic categories."""
        assert ComplianceTestCategory.SPARQL_BASIC.value == "sparql_basic"
        assert ComplianceTestCategory.SPARQL_OPTIONAL.value == "sparql_optional"
        assert ComplianceTestCategory.SPARQL_UNION.value == "sparql_union"
        assert ComplianceTestCategory.SPARQL_FILTER.value == "sparql_filter"
    
    def test_sparql_advanced_categories(self):
        """Test SPARQL advanced categories."""
        assert ComplianceTestCategory.SPARQL_AGGREGATES.value == "sparql_aggregates"
        assert ComplianceTestCategory.SPARQL_SUBQUERIES.value == "sparql_subqueries"
        assert ComplianceTestCategory.SPARQL_PROPERTY_PATHS.value == "sparql_property_paths"
    
    def test_rdfstar_categories(self):
        """Test RDF-Star categories."""
        assert ComplianceTestCategory.RDFSTAR_SYNTAX.value == "rdfstar_syntax"
        assert ComplianceTestCategory.RDFSTAR_SEMANTICS.value == "rdfstar_semantics"
        assert ComplianceTestCategory.RDFSTAR_SPARQL.value == "rdfstar_sparql"
    
    def test_security_categories(self):
        """Test security categories."""
        assert ComplianceTestCategory.SECURITY_AUTH.value == "security_auth"
        assert ComplianceTestCategory.SECURITY_ISOLATION.value == "security_isolation"
        assert ComplianceTestCategory.SECURITY_AUDIT.value == "security_audit"


# ============================================================================
# TestCase Tests
# ============================================================================

class TestComplianceTestCase:
    """Tests for TestCase dataclass."""
    
    def test_create_test_case(self):
        """Test creating a test case."""
        tc = ComplianceTestCase(
            id="test-001",
            name="Test Case",
            category=ComplianceTestCategory.SPARQL_BASIC,
            description="Test description",
            query="SELECT * WHERE { ?s ?p ?o }",
        )
        
        assert tc.id == "test-001"
        assert tc.name == "Test Case"
        assert tc.category == ComplianceTestCategory.SPARQL_BASIC
        assert tc.description == "Test description"
        assert tc.query is not None
    
    def test_test_case_with_data(self):
        """Test case with test data."""
        tc = ComplianceTestCase(
            id="test-002",
            name="Test with Data",
            category=ComplianceTestCategory.SPARQL_BASIC,
            description="Test with data",
            data='@prefix : <http://example.org/> . :s :p :o .',
            query="SELECT * WHERE { ?s ?p ?o }",
            expected_result={"count": 1},
        )
        
        assert tc.data is not None
        assert tc.expected_result == {"count": 1}
    
    def test_test_case_with_expected_error(self):
        """Test case expecting an error."""
        tc = ComplianceTestCase(
            id="test-003",
            name="Error Test",
            category=ComplianceTestCategory.SPARQL_BASIC,
            description="Should fail",
            query="INVALID QUERY",
            expected_error="syntax error",
        )
        
        assert tc.expected_error == "syntax error"
    
    def test_test_case_tags(self):
        """Test case with tags."""
        tc = ComplianceTestCase(
            id="test-004",
            name="Tagged Test",
            category=ComplianceTestCategory.SPARQL_BASIC,
            description="With tags",
            tags=["core", "regression"],
        )
        
        assert "core" in tc.tags
        assert "regression" in tc.tags


# ============================================================================
# TestResult Tests
# ============================================================================

class TestComplianceTestResult:
    """Tests for TestResult dataclass."""
    
    def test_create_result(self):
        """Test creating a test result."""
        result = ComplianceTestResult(
            test_id="test-001",
            test_name="Test Case",
            category=ComplianceTestCategory.SPARQL_BASIC,
            status=ComplianceTestStatus.PASSED,
            duration_ms=10.5,
        )
        
        assert result.test_id == "test-001"
        assert result.status == ComplianceTestStatus.PASSED
        assert result.duration_ms == 10.5
    
    def test_result_with_error(self):
        """Test result with error message."""
        result = ComplianceTestResult(
            test_id="test-002",
            test_name="Failed Test",
            category=ComplianceTestCategory.SPARQL_BASIC,
            status=ComplianceTestStatus.FAILED,
            duration_ms=5.0,
            error_message="Expected 2 results, got 0",
        )
        
        assert result.status == ComplianceTestStatus.FAILED
        assert result.error_message is not None
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = ComplianceTestResult(
            test_id="test-001",
            test_name="Test",
            category=ComplianceTestCategory.SPARQL_BASIC,
            status=ComplianceTestStatus.PASSED,
            duration_ms=10.0,
        )
        
        d = result.to_dict()
        assert d["testId"] == "test-001"
        assert d["status"] == "passed"
        assert d["category"] == "sparql_basic"


# ============================================================================
# ComplianceReport Tests
# ============================================================================

class TestComplianceReport:
    """Tests for ComplianceReport dataclass."""
    
    def test_create_report(self):
        """Test creating a compliance report."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
        )
        
        assert report.suite_name == "Test Suite"
        assert report.version == "1.0"
        assert report.total == 0
    
    def test_report_statistics(self):
        """Test report statistics."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test 1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t2", "Test 2", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t3", "Test 3", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.FAILED, 10),
                ComplianceTestResult("t4", "Test 4", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.SKIPPED, 0),
                ComplianceTestResult("t5", "Test 5", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.ERROR, 5),
            ],
        )
        
        assert report.total == 5
        assert report.passed == 2
        assert report.failed == 1
        assert report.skipped == 1
        assert report.errors == 1
    
    def test_report_pass_rate(self):
        """Test pass rate calculation."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test 1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t2", "Test 2", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t3", "Test 3", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.FAILED, 10),
                ComplianceTestResult("t4", "Test 4", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
            ],
        )
        
        assert report.pass_rate == 75.0
    
    def test_empty_report_pass_rate(self):
        """Test pass rate for empty report."""
        report = ComplianceReport(
            suite_name="Empty",
            version="1.0",
            timestamp=datetime.now(),
        )
        
        assert report.pass_rate == 0.0
    
    def test_report_by_category(self):
        """Test grouping results by category."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test 1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t2", "Test 2", ComplianceTestCategory.SPARQL_FILTER, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t3", "Test 3", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
            ],
        )
        
        by_cat = report.by_category()
        assert len(by_cat[ComplianceTestCategory.SPARQL_BASIC]) == 2
        assert len(by_cat[ComplianceTestCategory.SPARQL_FILTER]) == 1
    
    def test_report_to_dict(self):
        """Test converting report to dict."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test 1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
            ],
        )
        
        d = report.to_dict()
        assert d["suiteName"] == "Test Suite"
        assert d["summary"]["total"] == 1
        assert d["summary"]["passed"] == 1
        assert len(d["results"]) == 1
    
    def test_report_to_junit_xml(self):
        """Test JUnit XML output."""
        report = ComplianceReport(
            suite_name="Test Suite",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test 1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 100),
                ComplianceTestResult("t2", "Test 2", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.FAILED, 50,
                          error_message="Expected 2, got 0"),
            ],
        )
        
        xml = report.to_junit_xml()
        assert '<?xml version="1.0"' in xml
        assert '<testsuite name="Test Suite"' in xml
        assert 'tests="2"' in xml
        assert 'failures="1"' in xml
        assert '<testcase name="Test 1"' in xml
        assert '<failure message=' in xml


# ============================================================================
# AuditFinding Tests
# ============================================================================

class TestAuditFinding:
    """Tests for AuditFinding dataclass."""
    
    def test_create_finding(self):
        """Test creating an audit finding."""
        finding = AuditFinding(
            id="SEC-001",
            title="API Key Entropy",
            severity=Severity.HIGH,
            category="Authentication",
            description="Keys should have sufficient entropy",
            recommendation="Use 32 bytes of random data",
            passed=True,
        )
        
        assert finding.id == "SEC-001"
        assert finding.severity == Severity.HIGH
        assert finding.passed is True
    
    def test_finding_with_evidence(self):
        """Test finding with evidence."""
        finding = AuditFinding(
            id="SEC-002",
            title="Missing Rate Limit",
            severity=Severity.MEDIUM,
            category="Authorization",
            description="Rate limiting not enforced",
            recommendation="Add rate limiting",
            evidence="GET /api/query returned 200 after 1000 requests",
            passed=False,
        )
        
        assert finding.evidence is not None
        assert finding.passed is False
    
    def test_finding_to_dict(self):
        """Test converting finding to dict."""
        finding = AuditFinding(
            id="SEC-001",
            title="Test Finding",
            severity=Severity.CRITICAL,
            category="Security",
            description="Description",
            recommendation="Fix it",
            passed=False,
        )
        
        d = finding.to_dict()
        assert d["id"] == "SEC-001"
        assert d["severity"] == "critical"
        assert d["passed"] is False


# ============================================================================
# SecurityAuditReport Tests
# ============================================================================

class TestSecurityAuditReport:
    """Tests for SecurityAuditReport dataclass."""
    
    def test_create_report(self):
        """Test creating security audit report."""
        report = SecurityAuditReport(timestamp=datetime.now())
        
        assert report.timestamp is not None
        assert report.passed_count == 0
        assert report.failed_count == 0
    
    def test_report_counts(self):
        """Test report counting."""
        report = SecurityAuditReport(
            timestamp=datetime.now(),
            findings=[
                AuditFinding("S1", "T1", Severity.HIGH, "C", "D", "R", passed=True),
                AuditFinding("S2", "T2", Severity.CRITICAL, "C", "D", "R", passed=False),
                AuditFinding("S3", "T3", Severity.MEDIUM, "C", "D", "R", passed=True),
                AuditFinding("S4", "T4", Severity.HIGH, "C", "D", "R", passed=False),
            ],
        )
        
        assert report.passed_count == 2
        assert report.failed_count == 2
        assert report.critical_count == 1
        assert report.high_count == 1
    
    def test_report_to_dict(self):
        """Test converting report to dict."""
        report = SecurityAuditReport(
            timestamp=datetime.now(),
            findings=[
                AuditFinding("S1", "T1", Severity.HIGH, "C", "D", "R", passed=True),
            ],
        )
        
        d = report.to_dict()
        assert "summary" in d
        assert d["summary"]["total"] == 1
        assert len(d["findings"]) == 1


# ============================================================================
# SecurityAuditor Tests
# ============================================================================

class TestSecurityAuditor:
    """Tests for SecurityAuditor class."""
    
    def test_create_auditor(self):
        """Test creating auditor."""
        auditor = SecurityAuditor()
        
        assert auditor is not None
        assert len(auditor.checks) > 0
    
    def test_run_audit(self):
        """Test running audit."""
        auditor = SecurityAuditor()
        report = auditor.run_audit()
        
        assert report is not None
        assert len(report.findings) > 0
    
    def test_audit_has_expected_checks(self):
        """Test audit includes expected security checks."""
        auditor = SecurityAuditor()
        report = auditor.run_audit()
        
        # Should have checks for key security areas
        finding_ids = {f.id for f in report.findings}
        
        assert "SEC-001" in finding_ids  # API Key Entropy
        assert "SEC-002" in finding_ids  # API Key Storage
        assert "SEC-003" in finding_ids  # Token Expiration
        assert "SEC-007" in finding_ids  # Tenant Isolation


# ============================================================================
# CertificationManager Tests
# ============================================================================

class TestCertificationManager:
    """Tests for CertificationManager class."""
    
    def test_create_manager_without_store(self):
        """Test creating manager without store."""
        manager = CertificationManager()
        
        assert manager.store is None
    
    def test_security_audit_without_store(self):
        """Test running security audit without store."""
        manager = CertificationManager()
        report = manager.run_security_audit()
        
        assert report is not None
        assert len(report.findings) > 0
    
    def test_sparql_compliance_requires_store(self):
        """Test that SPARQL compliance requires a store."""
        manager = CertificationManager()
        
        with pytest.raises(ValueError, match="Store required"):
            manager.run_sparql_compliance()
    
    def test_rdfstar_compliance_requires_store(self):
        """Test that RDF-Star compliance requires a store."""
        manager = CertificationManager()
        
        with pytest.raises(ValueError, match="Store required"):
            manager.run_rdfstar_compliance()
    
    def test_generate_badge_passing(self):
        """Test generating a passing badge."""
        manager = CertificationManager()
        report = ComplianceReport(
            suite_name="Test",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "T1", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
                ComplianceTestResult("t2", "T2", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10),
            ],
        )
        
        svg = manager.generate_badge(report)
        assert "<svg" in svg
        assert "passing" in svg
        assert "#4c1" in svg  # Green color
    
    def test_generate_badge_partial(self):
        """Test generating a partial badge."""
        manager = CertificationManager()
        results = [ComplianceTestResult(f"t{i}", f"T{i}", ComplianceTestCategory.SPARQL_BASIC, 
                             ComplianceTestStatus.PASSED if i < 9 else ComplianceTestStatus.FAILED, 10)
                   for i in range(10)]
        report = ComplianceReport(
            suite_name="Test",
            version="1.0",
            timestamp=datetime.now(),
            results=results,
        )
        
        svg = manager.generate_badge(report)
        assert "partial" in svg
        assert "#dfb317" in svg  # Yellow color
    
    def test_generate_badge_failing(self):
        """Test generating a failing badge."""
        manager = CertificationManager()
        results = [ComplianceTestResult(f"t{i}", f"T{i}", ComplianceTestCategory.SPARQL_BASIC, 
                             ComplianceTestStatus.PASSED if i < 5 else ComplianceTestStatus.FAILED, 10)
                   for i in range(10)]
        report = ComplianceReport(
            suite_name="Test",
            version="1.0",
            timestamp=datetime.now(),
            results=results,
        )
        
        svg = manager.generate_badge(report)
        assert "failing" in svg
        assert "#e05d44" in svg  # Red color


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_run_security_audit(self):
        """Test run_security_audit function."""
        report = run_security_audit()
        
        assert report is not None
        assert isinstance(report, SecurityAuditReport)
        assert len(report.findings) > 0


# ============================================================================
# SPARQL11TestSuite Tests (with mock store)
# ============================================================================

class MockStore:
    """Mock store for testing."""
    
    def __init__(self):
        self.triples = []
        self._turtle_data = ""
    
    def load_turtle(self, data: str) -> None:
        """Load Turtle data."""
        self._turtle_data = data
        # Simple parsing for test data
        if ":alice" in data and ":name" in data:
            self.triples.append(("alice", "name", "Alice"))
        if ":bob" in data and ":name" in data:
            self.triples.append(("bob", "name", "Bob"))
    
    def query(self, sparql: str) -> dict:
        """Execute SPARQL query."""
        # Return mock results based on query type
        if sparql.strip().upper().startswith("ASK"):
            # Check for true/false conditions
            if '"Alice"' in sparql:
                return {"boolean": True}
            elif '"Bob"' in sparql and "alice" not in self._turtle_data.lower():
                return {"boolean": False}
            return {"boolean": False}
        
        if sparql.strip().upper().startswith("SELECT"):
            # Return mock bindings
            bindings = []
            
            # Count expected results based on test data
            if ":alice" in self._turtle_data and ":bob" in self._turtle_data:
                bindings = [{"name": "Alice"}, {"name": "Bob"}]
            elif ":alice" in self._turtle_data:
                bindings = [{"name": "Alice"}]
            elif self._turtle_data.strip():
                bindings = [{"result": "value"}]
            
            return {"results": {"bindings": bindings}}
        
        if sparql.strip().upper().startswith(("INSERT", "DELETE")):
            return {"success": True}
        
        if sparql.strip().upper().startswith("CONSTRUCT"):
            return {"results": {"bindings": [{"s": "x", "p": "y", "o": "z"}]}}
        
        return {"results": {"bindings": []}}


class TestSPARQL11TestSuite:
    """Tests for SPARQL11TestSuite class."""
    
    def test_create_suite(self):
        """Test creating test suite."""
        store = MockStore()
        suite = SPARQL11TestSuite(store)
        
        assert suite.store is store
        assert len(suite.test_cases) > 0
    
    def test_suite_has_test_categories(self):
        """Test suite covers multiple categories."""
        store = MockStore()
        suite = SPARQL11TestSuite(store)
        
        categories = {tc.category for tc in suite.test_cases}
        
        assert ComplianceTestCategory.SPARQL_BASIC in categories
        assert ComplianceTestCategory.SPARQL_FILTER in categories
        assert ComplianceTestCategory.SPARQL_OPTIONAL in categories
    
    def test_run_category(self):
        """Test running a specific category."""
        store = MockStore()
        suite = SPARQL11TestSuite(store)
        
        report = suite.run_category(ComplianceTestCategory.SPARQL_ASK)
        
        assert report is not None
        assert all(r.category == ComplianceTestCategory.SPARQL_ASK for r in report.results)


# ============================================================================
# RDFStarTestSuite Tests
# ============================================================================

class MockRDFStarStore:
    """Mock store for RDF-Star testing."""
    
    def __init__(self):
        self._data = ""
    
    def load_turtle(self, data: str) -> None:
        """Load Turtle data."""
        self._data = data
    
    def query(self, sparql: str) -> dict:
        """Execute SPARQL query."""
        # Count quoted triples in data
        qt_count = self._data.count("<<")
        
        if "SELECT" in sparql.upper():
            # Return bindings based on quoted triple count
            bindings = [{"qt": f"qt{i}"} for i in range(max(1, qt_count // 2))]
            
            # Adjust for specific test expectations
            if "certainty" in sparql and "> 0.7" in sparql:
                bindings = [{"friend": "bob"}]
            elif "source" in sparql and "GROUP BY" in sparql:
                bindings = [{"source": "a", "count": 2}, {"source": "b", "count": 1}]
            
            return {"results": {"bindings": bindings}}
        
        return {"results": {"bindings": []}}


class TestRDFStarTestSuite:
    """Tests for RDFStarTestSuite class."""
    
    def test_create_suite(self):
        """Test creating RDF-Star test suite."""
        store = MockRDFStarStore()
        suite = RDFStarTestSuite(store)
        
        assert suite.store is store
        assert len(suite.test_cases) > 0
    
    def test_suite_has_rdfstar_categories(self):
        """Test suite covers RDF-Star categories."""
        store = MockRDFStarStore()
        suite = RDFStarTestSuite(store)
        
        categories = {tc.category for tc in suite.test_cases}
        
        assert ComplianceTestCategory.RDFSTAR_SYNTAX in categories
        assert ComplianceTestCategory.RDFSTAR_SEMANTICS in categories
        assert ComplianceTestCategory.RDFSTAR_SPARQL in categories
    
    def test_run_all(self):
        """Test running all RDF-Star tests."""
        store = MockRDFStarStore()
        suite = RDFStarTestSuite(store)
        
        report = suite.run_all()
        
        assert report is not None
        assert report.suite_name == "RDF-Star Working Group"


# ============================================================================
# Integration Tests
# ============================================================================

class TestCertificationIntegration:
    """Integration tests for certification module."""
    
    def test_full_security_audit_workflow(self):
        """Test complete security audit workflow."""
        # Run audit
        report = run_security_audit()
        
        # Verify report structure
        assert report.timestamp is not None
        assert len(report.findings) >= 5
        
        # Convert to dict
        d = report.to_dict()
        assert "summary" in d
        assert "findings" in d
    
    def test_compliance_report_json_serialization(self):
        """Test that compliance reports can be JSON serialized."""
        import json
        
        report = ComplianceReport(
            suite_name="Test",
            version="1.0",
            timestamp=datetime.now(),
            results=[
                ComplianceTestResult("t1", "Test", ComplianceTestCategory.SPARQL_BASIC, ComplianceTestStatus.PASSED, 10.5),
            ],
        )
        
        d = report.to_dict()
        json_str = json.dumps(d)
        
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["suiteName"] == "Test"
    
    def test_audit_finding_severity_levels(self):
        """Test all severity levels work."""
        for severity in Severity:
            finding = AuditFinding(
                id=f"TEST-{severity.value}",
                title=f"Test {severity.value}",
                severity=severity,
                category="Test",
                description="Test",
                recommendation="Test",
            )
            
            assert finding.severity == severity
            assert finding.to_dict()["severity"] == severity.value
