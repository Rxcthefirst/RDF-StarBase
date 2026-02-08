"""
AI Grounding API

DEPRECATED: This module has been moved to api.ai_grounding.
This file provides backward compatibility - import from api.ai_grounding instead.
"""

import warnings

warnings.warn(
    "rdf_starbase.ai_grounding is deprecated. Import from api.ai_grounding instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from api.ai_grounding import (
    create_ai_router,
    integrate_ai_router,
    dataframe_to_grounded_facts,
    # Pydantic Models
    ConfidenceLevel,
    FactWithProvenance,
    Citation,
    GroundedFact,
    AIQueryRequest,
    AIQueryResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    EntityContextResponse,
    MaterializeRequest,
    MaterializeResponse,
)

__all__ = [
    "create_ai_router",
    "integrate_ai_router",
    "dataframe_to_grounded_facts",
    "ConfidenceLevel",
    "FactWithProvenance",
    "Citation",
    "GroundedFact",
    "AIQueryRequest",
    "AIQueryResponse",
    "ClaimVerificationRequest",
    "ClaimVerificationResponse",
    "EntityContextResponse",
    "MaterializeRequest",
    "MaterializeResponse",
]
