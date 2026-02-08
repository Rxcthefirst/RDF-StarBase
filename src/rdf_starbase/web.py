"""
RDF-StarBase Web API

DEPRECATED: This module has been moved to api.web.
This file provides backward compatibility - import from api.web instead.
"""

import warnings

warnings.warn(
    "rdf_starbase.web is deprecated. Import from api.web instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from api.web import (
    create_app,
    create_production_app,
    create_security_router,
    create_sql_router,
    dataframe_to_records,
    get_static_dir,
    app,
    # Pydantic models
    ProvenanceInput,
    TripleInput,
    BatchTripleInput,
    SPARQLQuery,
    SourceInput,
    SQLQuery,
)

__all__ = [
    "create_app",
    "create_production_app",
    "create_security_router",
    "create_sql_router",
    "dataframe_to_records",
    "get_static_dir",
    "app",
    "ProvenanceInput",
    "TripleInput",
    "BatchTripleInput",
    "SPARQLQuery",
    "SourceInput",
    "SQLQuery",
]
