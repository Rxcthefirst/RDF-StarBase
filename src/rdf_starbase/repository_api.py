"""
Repository Management API Router.

DEPRECATED: This module has been moved to api.repository_api.
This file provides backward compatibility - import from api.repository_api instead.
"""

import warnings

warnings.warn(
    "rdf_starbase.repository_api is deprecated. Import from api.repository_api instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from api.repository_api import (
    create_repository_router,
    dataframe_to_records,
    CreateRepositoryRequest,
    UpdateRepositoryRequest,
    RenameRepositoryRequest,
    SPARQLQueryRequest,
    SQLQueryRequest,
    SQLAggregateRequest,
    RepositoryResponse,
)

__all__ = [
    "create_repository_router",
    "dataframe_to_records",
    "CreateRepositoryRequest",
    "UpdateRepositoryRequest",
    "RenameRepositoryRequest",
    "SPARQLQueryRequest",
    "SQLQueryRequest",
    "SQLAggregateRequest",
    "RepositoryResponse",
]
