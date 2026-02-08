"""
Scalable RDF and RDF-Star benchmark data generator.

Generates datasets at various scales for fair benchmarking:
- RDF plane: Standard triples without annotations
- RDF-Star plane: Triples with statement-level metadata (provenance, confidence, temporal)

Based on LUBM-like patterns but with RDF-Star extensions.
"""
import random
import string
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse

# Configuration
UNIVERSITIES_PER_SCALE = {
    "tiny": 1,        # ~100K triples
    "small": 10,      # ~1M triples  
    "medium": 50,     # ~5M triples
    "large": 100,     # ~10M triples
    "xlarge": 500,    # ~50M triples
}

# LUBM-like schema
PREFIXES = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix bench: <http://benchmark.example.org/> .
@prefix univ: <http://university.example.org/> .
@prefix prov: <http://www.w3.org/ns/prov#> .

"""

# Classes
CLASSES = [
    "University", "Department", "Professor", "AssociateProfessor", 
    "AssistantProfessor", "Lecturer", "Student", "GraduateStudent",
    "UndergraduateStudent", "ResearchGroup", "Publication", "Course"
]

# Predicates
PREDICATES = {
    "type": "rdf:type",
    "name": "bench:name",
    "memberOf": "bench:memberOf",
    "worksFor": "bench:worksFor",
    "teacherOf": "bench:teacherOf",
    "takesCourse": "bench:takesCourse",
    "advisor": "bench:advisor",
    "publicationAuthor": "bench:publicationAuthor",
    "subOrganizationOf": "bench:subOrganizationOf",
    "headOf": "bench:headOf",
    "degreeFrom": "bench:degreeFrom",
    "doctoralDegreeFrom": "bench:doctoralDegreeFrom",
    "mastersDegreeFrom": "bench:mastersDegreeFrom",
    "undergraduateDegreeFrom": "bench:undergraduateDegreeFrom",
    "researchInterest": "bench:researchInterest",
    "emailAddress": "bench:emailAddress",
    "telephone": "bench:telephone",
    "age": "bench:age",
}

# Research areas for variety
RESEARCH_AREAS = [
    "MachineLearning", "Databases", "Networks", "Security", "Graphics",
    "AI", "Theory", "Systems", "HCI", "Robotics", "NLP", "Vision",
    "DistributedSystems", "ProgrammingLanguages", "SoftwareEngineering"
]

# Sources for provenance
SOURCES = [
    "http://registrar.example.org/",
    "http://hr.example.org/",
    "http://research.example.org/",
    "http://publications.example.org/",
    "http://courses.example.org/",
]


def generate_name():
    """Generate a random name."""
    first = ''.join(random.choices(string.ascii_uppercase, k=1)) + \
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
    last = ''.join(random.choices(string.ascii_uppercase, k=1)) + \
           ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    return f"{first} {last}"


def escape_literal(s: str) -> str:
    """Escape a string for Turtle literal."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


class BenchmarkGenerator:
    """Generates RDF and RDF-Star benchmark data."""
    
    def __init__(self, scale: str = "small", rdf_star: bool = False, seed: int = 42):
        self.scale = scale
        self.rdf_star = rdf_star
        self.num_universities = UNIVERSITIES_PER_SCALE.get(scale, 10)
        self.triple_count = 0
        random.seed(seed)
        
        # Per-university counts (LUBM-like ratios)
        self.departments_per_univ = 15
        self.profs_per_dept = 10
        self.students_per_dept = 200
        self.courses_per_dept = 20
        self.pubs_per_prof = 5
        
    def _format_triple(self, s: str, p: str, o: str, 
                       confidence: Optional[float] = None,
                       source: Optional[str] = None,
                       timestamp: Optional[str] = None) -> str:
        """Format a triple, optionally with RDF-Star annotations."""
        self.triple_count += 1
        
        if self.rdf_star and (confidence or source or timestamp):
            # RDF-Star quoted triple with annotations
            base = f"<< {s} {p} {o} >>"
            annotations = []
            if confidence is not None:
                annotations.append(f'    bench:confidence "{confidence:.2f}"^^xsd:decimal')
                self.triple_count += 1
            if source:
                annotations.append(f'    prov:wasAttributedTo <{source}>')
                self.triple_count += 1
            if timestamp:
                annotations.append(f'    prov:generatedAtTime "{timestamp}"^^xsd:dateTime')
                self.triple_count += 1
            
            return base + " .\n" + base + "\n" + " ;\n".join(annotations) + " .\n"
        else:
            return f"{s} {p} {o} .\n"
    
    def _random_provenance(self):
        """Generate random provenance metadata."""
        if not self.rdf_star:
            return None, None, None
        
        confidence = round(random.uniform(0.7, 1.0), 2)
        source = random.choice(SOURCES)
        
        # Random timestamp in last 5 years
        days_ago = random.randint(0, 365 * 5)
        ts = datetime.now() - timedelta(days=days_ago)
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return confidence, source, timestamp
    
    def generate_university(self, univ_id: int) -> str:
        """Generate all data for one university."""
        output = []
        univ_uri = f"univ:University{univ_id}"
        
        # University itself
        conf, src, ts = self._random_provenance()
        output.append(self._format_triple(univ_uri, "rdf:type", "bench:University", conf, src, ts))
        output.append(self._format_triple(univ_uri, "bench:name", f'"University {univ_id}"'))
        
        # Departments
        for dept_id in range(self.departments_per_univ):
            dept_uri = f"univ:Dept{univ_id}_{dept_id}"
            conf, src, ts = self._random_provenance()
            output.append(self._format_triple(dept_uri, "rdf:type", "bench:Department", conf, src, ts))
            output.append(self._format_triple(dept_uri, "bench:subOrganizationOf", univ_uri))
            output.append(self._format_triple(dept_uri, "bench:name", f'"Department {dept_id}"'))
            
            # Professors
            for prof_id in range(self.profs_per_dept):
                prof_type = random.choice(["Professor", "AssociateProfessor", "AssistantProfessor"])
                prof_uri = f"univ:Prof{univ_id}_{dept_id}_{prof_id}"
                
                conf, src, ts = self._random_provenance()
                output.append(self._format_triple(prof_uri, "rdf:type", f"bench:{prof_type}", conf, src, ts))
                output.append(self._format_triple(prof_uri, "bench:name", f'"{escape_literal(generate_name())}"'))
                output.append(self._format_triple(prof_uri, "bench:worksFor", dept_uri))
                output.append(self._format_triple(prof_uri, "bench:emailAddress", 
                    f'"prof{prof_id}@dept{dept_id}.univ{univ_id}.edu"'))
                
                # Research interests
                for _ in range(random.randint(1, 3)):
                    area = random.choice(RESEARCH_AREAS)
                    conf, src, ts = self._random_provenance()
                    output.append(self._format_triple(prof_uri, "bench:researchInterest", 
                        f"bench:{area}", conf, src, ts))
                
                # Publications
                for pub_id in range(random.randint(1, self.pubs_per_prof)):
                    pub_uri = f"univ:Pub{univ_id}_{dept_id}_{prof_id}_{pub_id}"
                    conf, src, ts = self._random_provenance()
                    output.append(self._format_triple(pub_uri, "rdf:type", "bench:Publication", conf, src, ts))
                    output.append(self._format_triple(pub_uri, "bench:publicationAuthor", prof_uri))
                
                # Head of department (first prof)
                if prof_id == 0:
                    output.append(self._format_triple(prof_uri, "bench:headOf", dept_uri))
            
            # Courses
            for course_id in range(self.courses_per_dept):
                course_uri = f"univ:Course{univ_id}_{dept_id}_{course_id}"
                output.append(self._format_triple(course_uri, "rdf:type", "bench:Course"))
                output.append(self._format_triple(course_uri, "bench:name", f'"Course {course_id}"'))
                
                # Assign teacher
                teacher_id = random.randint(0, self.profs_per_dept - 1)
                teacher_uri = f"univ:Prof{univ_id}_{dept_id}_{teacher_id}"
                output.append(self._format_triple(teacher_uri, "bench:teacherOf", course_uri))
            
            # Students
            for student_id in range(self.students_per_dept):
                is_grad = random.random() < 0.3
                student_type = "GraduateStudent" if is_grad else "UndergraduateStudent"
                student_uri = f"univ:Student{univ_id}_{dept_id}_{student_id}"
                
                conf, src, ts = self._random_provenance()
                output.append(self._format_triple(student_uri, "rdf:type", f"bench:{student_type}", conf, src, ts))
                output.append(self._format_triple(student_uri, "bench:name", f'"{escape_literal(generate_name())}"'))
                output.append(self._format_triple(student_uri, "bench:memberOf", dept_uri))
                
                # Age
                age = random.randint(18, 35) if is_grad else random.randint(18, 24)
                output.append(self._format_triple(student_uri, "bench:age", f'"{age}"^^xsd:integer'))
                
                # Courses taken
                for _ in range(random.randint(2, 5)):
                    course_id = random.randint(0, self.courses_per_dept - 1)
                    course_uri = f"univ:Course{univ_id}_{dept_id}_{course_id}"
                    output.append(self._format_triple(student_uri, "bench:takesCourse", course_uri))
                
                # Advisor for grad students
                if is_grad:
                    advisor_id = random.randint(0, self.profs_per_dept - 1)
                    advisor_uri = f"univ:Prof{univ_id}_{dept_id}_{advisor_id}"
                    conf, src, ts = self._random_provenance()
                    output.append(self._format_triple(student_uri, "bench:advisor", advisor_uri, conf, src, ts))
        
        return "".join(output)
    
    def generate(self, output_path: str):
        """Generate the full dataset."""
        path = Path(output_path)
        is_gzipped = path.suffix.lower() == ".gz"
        
        opener = gzip.open if is_gzipped else open
        mode = 'wt' if is_gzipped else 'w'
        
        print(f"Generating {self.scale} dataset ({self.num_universities} universities)...")
        print(f"RDF-Star mode: {self.rdf_star}")
        
        with opener(path, mode, encoding='utf-8') as f:
            f.write(PREFIXES)
            
            for univ_id in range(self.num_universities):
                if univ_id % 10 == 0:
                    print(f"  University {univ_id}/{self.num_universities}...")
                f.write(self.generate_university(univ_id))
        
        print(f"Generated {self.triple_count:,} triples to {output_path}")
        return self.triple_count


def generate_benchmark_queries(rdf_star: bool = False) -> dict:
    """Generate benchmark queries for RDF and RDF-Star planes."""
    
    queries = {
        "rdf": {
            "Q1_count": "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }",
            
            "Q2_type_scan": """
                SELECT ?x WHERE { 
                    ?x rdf:type bench:GraduateStudent 
                }
            """,
            
            "Q3_pattern_join": """
                SELECT ?student ?advisor ?dept WHERE {
                    ?student rdf:type bench:GraduateStudent .
                    ?student bench:advisor ?advisor .
                    ?advisor bench:worksFor ?dept .
                }
            """,
            
            "Q4_filter": """
                SELECT ?student ?age WHERE {
                    ?student rdf:type bench:GraduateStudent .
                    ?student bench:age ?age .
                    FILTER(?age > 25)
                }
            """,
            
            "Q5_aggregate": """
                SELECT ?dept (COUNT(?student) as ?count) WHERE {
                    ?student bench:memberOf ?dept .
                    ?student rdf:type bench:GraduateStudent .
                }
                GROUP BY ?dept
            """,
            
            "Q6_optional": """
                SELECT ?prof ?pub WHERE {
                    ?prof rdf:type bench:Professor .
                    OPTIONAL { ?pub bench:publicationAuthor ?prof }
                }
                LIMIT 10000
            """,
            
            "Q7_multi_hop": """
                SELECT ?student ?univ WHERE {
                    ?student rdf:type bench:GraduateStudent .
                    ?student bench:memberOf ?dept .
                    ?dept bench:subOrganizationOf ?univ .
                }
            """,
        },
    }
    
    if rdf_star:
        queries["rdf_star"] = {
            "QS1_confidence_filter": """
                SELECT ?s ?p ?o WHERE {
                    << ?s ?p ?o >> bench:confidence ?conf .
                    FILTER(?conf >= 0.9)
                }
            """,
            
            "QS2_provenance_source": """
                SELECT ?s ?p ?o ?source WHERE {
                    << ?s ?p ?o >> prov:wasAttributedTo ?source .
                    FILTER(?source = <http://research.example.org/>)
                }
            """,
            
            "QS3_temporal_range": """
                SELECT ?s ?p ?o ?ts WHERE {
                    << ?s ?p ?o >> prov:generatedAtTime ?ts .
                    FILTER(?ts > "2024-01-01T00:00:00Z"^^xsd:dateTime)
                }
            """,
            
            "QS4_annotated_pattern": """
                SELECT ?student ?advisor ?conf WHERE {
                    << ?student bench:advisor ?advisor >> bench:confidence ?conf .
                    ?student rdf:type bench:GraduateStudent .
                }
            """,
            
            "QS5_provenance_aggregate": """
                SELECT ?source (COUNT(*) as ?count) WHERE {
                    << ?s ?p ?o >> prov:wasAttributedTo ?source .
                }
                GROUP BY ?source
            """,
            
            "QS6_high_confidence_join": """
                SELECT ?prof ?area WHERE {
                    << ?prof bench:researchInterest ?area >> bench:confidence ?conf .
                    ?prof rdf:type bench:Professor .
                    FILTER(?conf >= 0.95)
                }
            """,
        }
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate RDF/RDF-Star benchmark data")
    parser.add_argument("--scale", choices=list(UNIVERSITIES_PER_SCALE.keys()), 
                        default="small", help="Dataset scale")
    parser.add_argument("--rdf-star", action="store_true", 
                        help="Generate RDF-Star data with provenance")
    parser.add_argument("--output", "-o", type=str, 
                        help="Output file path (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        suffix = "_star" if args.rdf_star else ""
        output_path = f"data/sample/benchmark_{args.scale}{suffix}.ttl"
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    gen = BenchmarkGenerator(scale=args.scale, rdf_star=args.rdf_star, seed=args.seed)
    count = gen.generate(output_path)
    
    # Print expected triples per scale
    print(f"\nScale reference:")
    for scale, univs in UNIVERSITIES_PER_SCALE.items():
        expected = univs * 15 * (10 * 8 + 200 * 5 + 20 * 3)  # rough estimate
        print(f"  {scale}: ~{expected:,} base triples ({univs} universities)")
    
    # Print query info
    print(f"\nGenerated benchmark queries available via generate_benchmark_queries()")
    

if __name__ == "__main__":
    main()
