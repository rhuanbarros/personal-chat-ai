from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from agents.gemini_agent import GeminiAgent

# Load environment variables
load_dotenv(dotenv_path=".env.local", override=True)

app = FastAPI(
    title="Personal Chat AI Backend",
    description="FastAPI backend for Personal Chat application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware( # type: ignore
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_agent = GeminiAgent()

class InvokeRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.post("/invoke_gemini")
async def invoke_gemini_agent(request: InvokeRequest) -> Dict[str, str]:
    response_content = gemini_agent.invoke_completion(request.messages)
    return {"response": response_content}

@app.get("/")
async def root():
    return {"message": "Personal Chat AI Backend is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("PYTHON_ENV", "development"),
        "database_url": os.getenv("DATABASE_URL", "not configured")
    }

@app.get("/api/test")
async def test_endpoint():
    return {
        "message": "Test endpoint working!",
        "data": {
            "backend": "FastAPI",
            "version": "1.0.0",
            "status": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # type: ignore 