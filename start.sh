#!/bin/bash

echo "Starting Personal Chat AI Backend..."
echo "Environment: ${PYTHON_ENV:-development}"
echo "Database URL: ${DATABASE_URL:-not configured}"

# Start the FastAPI server with auto-reload
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload