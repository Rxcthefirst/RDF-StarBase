#!/bin/bash
# Start RDF-StarBase API server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Set Python path to include src directory
export PYTHONPATH="$PROJECT_ROOT/src"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start the server
echo "Starting RDF-StarBase API server..."
echo "API Documentation: http://localhost:8000/docs"
echo "ReDoc: http://localhost:8000/redoc"
echo "Web Interface: http://localhost:8000/app/"
echo ""

python3 -m uvicorn api.web:app --reload --host 0.0.0.0 --port 8000
