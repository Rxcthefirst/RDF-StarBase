# =============================================================================
# RDF-StarBase Docker Image
# =============================================================================
# Multi-stage build: Node.js for frontend, Python for backend
# Final image serves both frontend and API from a single container
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Frontend
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm ci --silent

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RDFSTARBASE_REPOSITORY_PATH=/data/repositories \
    RDFSTARBASE_HOST=0.0.0.0 \
    RDFSTARBASE_PORT=8000 \
    RDFSTARBASE_SERVE_STATIC=true

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python package files and install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with all dependencies (including web and query)
RUN pip install --no-cache-dir -e ".[dev,web,query]"

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create data directory
RUN mkdir -p /data/repositories

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entry point - run the FastAPI server
CMD ["python", "-m", "uvicorn", "rdf_starbase.web:app", "--host", "0.0.0.0", "--port", "8000"]
