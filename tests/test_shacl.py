"""Tests for SHACL validation module."""

import pytest
from rdf_starbase.storage.shacl import (
    Severity,
    NodeKind,
    ConstraintType,
    ValidationResult,
    ValidationReport,
    PropertyConstraint,
    PropertyShape,
    NodeShape,
    ShapesGraph,
    SHACLValidator,
    SHACLManager,
    SH,
    XSD,
    RDF,
)


# ============================================================================
# ValidationResult Tests
# ============================================================================

class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_create_result(self):
        """Test creating a validation result."""
        result = ValidationResult(
            focus_node="http://example.org/node1",
            result_path="http://example.org/name",
            value="test",
            source_shape="http://example.org/shape1",
            source_constraint=f"{SH}MinCountConstraintComponent",
            message="Missing required value",
        )
        
        assert result.focus_node == "http://example.org/node1"
        assert result.severity == Severity.VIOLATION
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(
            focus_node="http://example.org/node1",
            result_path="http://example.org/prop",
            value="test",
            source_shape="http://example.org/shape1",
            source_constraint=f"{SH}DatatypeConstraintComponent",
            message="Wrong datatype",
            severity=Severity.WARNING,
        )
        
        d = result.to_dict()
        assert d["focusNode"] == "http://example.org/node1"
        assert d["resultPath"] == "http://example.org/prop"
        assert d["value"] == "test"
        assert d["resultSeverity"] == f"{SH}Warning"


# ============================================================================
# ValidationReport Tests
# ============================================================================

class TestValidationReport:
    """Tests for ValidationReport."""
    
    def test_empty_report_conforms(self):
        """Test empty report conforms."""
        report = ValidationReport(conforms=True)
        assert report.conforms is True
        assert len(report.results) == 0
    
    def test_add_violation_sets_nonconforming(self):
        """Test that adding a violation sets conforms=False."""
        report = ValidationReport(conforms=True)
        
        report.add_result(ValidationResult(
            focus_node="http://example.org/node1",
            result_path=None,
            value=None,
            source_shape="http://example.org/shape",
            source_constraint=f"{SH}MinCountConstraintComponent",
            message="Error",
            severity=Severity.VIOLATION,
        ))
        
        assert report.conforms is False
    
    def test_add_warning_keeps_conforming(self):
        """Test that adding a warning keeps conforms=True."""
        report = ValidationReport(conforms=True)
        
        report.add_result(ValidationResult(
            focus_node="http://example.org/node1",
            result_path=None,
            value=None,
            source_shape="http://example.org/shape",
            source_constraint=f"{SH}DatatypeConstraintComponent",
            message="Warning",
            severity=Severity.WARNING,
        ))
        
        assert report.conforms is True
    
    def test_violations_filter(self):
        """Test filtering violations."""
        report = ValidationReport(conforms=True)
        
        for i, sev in enumerate([Severity.VIOLATION, Severity.WARNING, Severity.VIOLATION]):
            report.add_result(ValidationResult(
                focus_node=f"http://example.org/node{i}",
                result_path=None,
                value=None,
                source_shape="http://example.org/shape",
                source_constraint=f"{SH}MinCountConstraintComponent",
                message=f"Message {i}",
                severity=sev,
            ))
        
        assert len(report.violations()) == 2
        assert len(report.warnings()) == 1
    
    def test_to_dict(self):
        """Test report to dictionary."""
        report = ValidationReport(conforms=True)
        report.add_result(ValidationResult(
            focus_node="http://example.org/node1",
            result_path=None,
            value=None,
            source_shape="http://example.org/shape",
            source_constraint=f"{SH}MinCountConstraintComponent",
            message="Error",
            severity=Severity.VIOLATION,
        ))
        
        d = report.to_dict()
        assert d["conforms"] is False
        assert d["violationCount"] == 1
        assert d["warningCount"] == 0
        assert len(d["results"]) == 1
    
    def test_to_rdf(self):
        """Test report to RDF Turtle."""
        report = ValidationReport(conforms=True)
        
        turtle = report.to_rdf()
        assert "@prefix sh:" in turtle
        assert "sh:conforms true" in turtle


# ============================================================================
# ShapesGraph Tests
# ============================================================================

class TestShapesGraph:
    """Tests for ShapesGraph parsing."""
    
    def test_load_simple_shape(self):
        """Test loading a simple node shape."""
        turtle = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:minCount 1 ;
                sh:datatype xsd:string
            ] .
        """
        
        graph = ShapesGraph()
        graph.load_from_turtle(turtle)
        
        assert len(graph.node_shapes) == 1
        shape = list(graph.node_shapes.values())[0]
        assert "http://example.org/PersonShape" in shape.shape_iri
    
    def test_parse_target_class(self):
        """Test parsing target class."""
        triples = [
            ("http://example.org/Shape1", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape1", f"{SH}targetClass", "http://example.org/Person"),
        ]
        
        graph = ShapesGraph()
        graph.load_from_triples(triples)
        
        assert len(graph.node_shapes) == 1
        shape = graph.node_shapes["http://example.org/Shape1"]
        assert "http://example.org/Person" in shape.target_class
    
    def test_parse_property_constraints(self):
        """Test parsing property shape constraints."""
        triples = [
            ("http://example.org/Shape1", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape1", f"{SH}property", "_:prop1"),
            ("_:prop1", f"{SH}path", "http://example.org/name"),
            ("_:prop1", f"{SH}minCount", '"1"'),
            ("_:prop1", f"{SH}maxCount", '"1"'),
        ]
        
        graph = ShapesGraph()
        graph.load_from_triples(triples)
        
        shape = graph.node_shapes["http://example.org/Shape1"]
        assert len(shape.property_shapes) == 1
        
        prop = shape.property_shapes[0]
        assert prop.path == "http://example.org/name"
        
        constraint_types = [c.constraint_type for c in prop.constraints]
        assert ConstraintType.MIN_COUNT in constraint_types
        assert ConstraintType.MAX_COUNT in constraint_types
    
    def test_parse_multiple_shapes(self):
        """Test parsing multiple shapes."""
        triples = [
            ("http://example.org/Shape1", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape1", f"{SH}targetClass", "http://example.org/Person"),
            ("http://example.org/Shape2", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape2", f"{SH}targetClass", "http://example.org/Organization"),
        ]
        
        graph = ShapesGraph()
        graph.load_from_triples(triples)
        
        assert len(graph.node_shapes) == 2
    
    def test_parse_closed_shape(self):
        """Test parsing closed shape."""
        triples = [
            ("http://example.org/Shape1", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape1", f"{SH}closed", '"true"'),
        ]
        
        graph = ShapesGraph()
        graph.load_from_triples(triples)
        
        shape = graph.node_shapes["http://example.org/Shape1"]
        assert shape.closed is True
    
    def test_parse_deactivated_shape(self):
        """Test parsing deactivated shape."""
        triples = [
            ("http://example.org/Shape1", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape1", f"{SH}deactivated", '"true"'),
        ]
        
        graph = ShapesGraph()
        graph.load_from_triples(triples)
        
        shape = graph.node_shapes["http://example.org/Shape1"]
        assert shape.deactivated is True


# ============================================================================
# SHACLValidator Tests
# ============================================================================

class TestSHACLValidator:
    """Tests for SHACLValidator."""
    
    @pytest.fixture
    def person_shapes(self):
        """Create shapes for Person validation."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/PersonShape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/PersonShape", f"{SH}targetClass", "http://example.org/Person"),
            ("http://example.org/PersonShape", f"{SH}property", "_:nameProp"),
            ("_:nameProp", f"{SH}path", "http://example.org/name"),
            ("_:nameProp", f"{SH}minCount", '"1"'),
            ("_:nameProp", f"{SH}datatype", f"{XSD}string"),
        ])
        return graph
    
    def test_validate_conforming_data(self, person_shapes):
        """Test validating conforming data."""
        validator = SHACLValidator(person_shapes)
        
        data = [
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person", None),
            ("http://example.org/alice", "http://example.org/name", '"Alice"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is True
    
    def test_validate_missing_required_property(self, person_shapes):
        """Test validation catches missing required property."""
        validator = SHACLValidator(person_shapes)
        
        data = [
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person", None),
            # Missing name property
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
        assert len(report.violations()) == 1
        assert "at least 1" in report.violations()[0].message
    
    def test_validate_wrong_datatype(self, person_shapes):
        """Test validation catches wrong datatype."""
        validator = SHACLValidator(person_shapes)
        
        data = [
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person", None),
            ("http://example.org/alice", "http://example.org/name", f'"42"^^<{XSD}integer>', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
        assert any("datatype" in v.message.lower() for v in report.violations())
    
    def test_validate_min_count(self):
        """Test minCount constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/value"),
            ("_:prop", f"{SH}minCount", '"2"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Only one value - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/value", '"one"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_max_count(self):
        """Test maxCount constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/value"),
            ("_:prop", f"{SH}maxCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Two values - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/value", '"one"', None),
            ("http://example.org/x", "http://example.org/value", '"two"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_node_kind_iri(self):
        """Test nodeKind IRI constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/ref"),
            ("_:prop", f"{SH}nodeKind", f"{SH}IRI"),
        ])
        
        validator = SHACLValidator(graph)
        
        # Literal value - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/ref", '"not-an-iri"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_min_length(self):
        """Test minLength constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/name"),
            ("_:prop", f"{SH}minLength", '"5"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Too short - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/name", '"ab"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_max_length(self):
        """Test maxLength constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/code"),
            ("_:prop", f"{SH}maxLength", '"3"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Too long - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/code", '"ABCDEF"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_pattern(self):
        """Test pattern constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/email"),
            ("_:prop", f"{SH}pattern", '"^[a-z]+@[a-z]+\\.[a-z]+$"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Invalid email - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/email", '"not-an-email"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_min_inclusive(self):
        """Test minInclusive constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/age"),
            ("_:prop", f"{SH}minInclusive", '"18"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Below minimum - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/age", '"15"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_max_inclusive(self):
        """Test maxInclusive constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/score"),
            ("_:prop", f"{SH}maxInclusive", '"100"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Above maximum - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/score", '"150"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_closed_shape(self):
        """Test closed shape constraint."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}closed", '"true"'),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/name"),
        ])
        
        validator = SHACLValidator(graph)
        
        # Extra property - should fail
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
            ("http://example.org/x", "http://example.org/name", '"Test"', None),
            ("http://example.org/x", "http://example.org/extra", '"Value"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
        assert "not allowed" in report.violations()[0].message
    
    def test_validate_target_node(self):
        """Test targetNode targeting."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetNode", "http://example.org/specificNode"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/required"),
            ("_:prop", f"{SH}minCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Specific node missing required - should fail
        data = [
            ("http://example.org/specificNode", "http://example.org/other", '"value"', None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_validate_target_subjects_of(self):
        """Test targetSubjectsOf targeting."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetSubjectsOf", "http://example.org/hasPart"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/name"),
            ("_:prop", f"{SH}minCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Subject of hasPart missing name - should fail
        data = [
            ("http://example.org/x", "http://example.org/hasPart", "http://example.org/y", None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
    
    def test_deactivated_shape_not_validated(self):
        """Test that deactivated shapes are not validated."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}deactivated", '"true"'),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/required"),
            ("_:prop", f"{SH}minCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Would fail if shape was active
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
        ]
        
        report = validator.validate(data)
        assert report.conforms is True  # Shape is deactivated
    
    def test_validate_multiple_focus_nodes(self):
        """Test validating multiple focus nodes."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Person"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/name"),
            ("_:prop", f"{SH}minCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        data = [
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person", None),
            ("http://example.org/alice", "http://example.org/name", '"Alice"', None),
            ("http://example.org/bob", f"{RDF}type", "http://example.org/Person", None),
            # Bob is missing name
        ]
        
        report = validator.validate(data)
        assert report.conforms is False
        assert len(report.violations()) == 1
        assert "bob" in report.violations()[0].focus_node


# ============================================================================
# SHACLManager Tests
# ============================================================================

class TestSHACLManager:
    """Tests for SHACLManager high-level API."""
    
    def test_load_shapes_from_turtle(self):
        """Test loading shapes from Turtle."""
        manager = SHACLManager(None)  # No store needed for basic tests
        
        turtle = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:Shape1 a sh:NodeShape ;
            sh:targetClass ex:Person .
        
        ex:Shape2 a sh:NodeShape ;
            sh:targetClass ex:Organization .
        """
        
        count = manager.load_shapes(turtle)
        assert count == 2
        assert manager.shapes_loaded is True
    
    def test_validate_triples(self):
        """Test validating triples directly."""
        manager = SHACLManager(None)
        
        turtle = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:minCount 1
            ] .
        """
        
        manager.load_shapes(turtle)
        
        # Valid triples
        report = manager.validate_triples([
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person"),
            ("http://example.org/alice", "http://example.org/name", '"Alice"'),
        ])
        
        assert report.conforms is True
    
    def test_get_shapes_summary(self):
        """Test getting shapes summary."""
        manager = SHACLManager(None)
        
        turtle = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:name "Person Shape" ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:minCount 1
            ] .
        """
        
        manager.load_shapes(turtle)
        
        summary = manager.get_shapes_summary()
        assert summary["loaded"] is True
        assert summary["nodeShapeCount"] == 1
        assert len(summary["shapes"]) == 1
    
    def test_no_shapes_loaded(self):
        """Test validation with no shapes loaded."""
        manager = SHACLManager(None)
        
        report = manager.validate_triples([
            ("http://example.org/x", "http://example.org/p", '"value"'),
        ])
        
        assert report.conforms is True  # No shapes = always valid


# ============================================================================
# Integration Tests
# ============================================================================

class TestSHACLIntegration:
    """Integration tests for SHACL validation."""
    
    def test_complex_shape_validation(self):
        """Test complex shape with multiple constraints."""
        graph = ShapesGraph()
        graph.load_from_triples([
            # Person shape
            ("http://example.org/PersonShape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/PersonShape", f"{SH}targetClass", "http://example.org/Person"),
            
            # Name property
            ("http://example.org/PersonShape", f"{SH}property", "_:name"),
            ("_:name", f"{SH}path", "http://example.org/name"),
            ("_:name", f"{SH}minCount", '"1"'),
            ("_:name", f"{SH}maxCount", '"1"'),
            ("_:name", f"{SH}datatype", f"{XSD}string"),
            ("_:name", f"{SH}minLength", '"2"'),
            
            # Age property
            ("http://example.org/PersonShape", f"{SH}property", "_:age"),
            ("_:age", f"{SH}path", "http://example.org/age"),
            ("_:age", f"{SH}datatype", f"{XSD}integer"),
            ("_:age", f"{SH}minInclusive", '"0"'),
            ("_:age", f"{SH}maxInclusive", '"150"'),
        ])
        
        validator = SHACLValidator(graph)
        
        # Valid person
        valid_data = [
            ("http://example.org/alice", f"{RDF}type", "http://example.org/Person", None),
            ("http://example.org/alice", "http://example.org/name", '"Alice"', None),
            ("http://example.org/alice", "http://example.org/age", f'"30"^^<{XSD}integer>', None),
        ]
        
        report = validator.validate(valid_data)
        assert report.conforms is True
    
    def test_report_rdf_serialization(self):
        """Test that validation report serializes to valid RDF."""
        graph = ShapesGraph()
        graph.load_from_triples([
            ("http://example.org/Shape", f"{RDF}type", f"{SH}NodeShape"),
            ("http://example.org/Shape", f"{SH}targetClass", "http://example.org/Thing"),
            ("http://example.org/Shape", f"{SH}property", "_:prop"),
            ("_:prop", f"{SH}path", "http://example.org/required"),
            ("_:prop", f"{SH}minCount", '"1"'),
        ])
        
        validator = SHACLValidator(graph)
        
        data = [
            ("http://example.org/x", f"{RDF}type", "http://example.org/Thing", None),
        ]
        
        report = validator.validate(data)
        
        turtle = report.to_rdf()
        assert "@prefix sh:" in turtle
        assert "sh:ValidationReport" in turtle
        assert "sh:conforms false" in turtle
        assert "sh:ValidationResult" in turtle
