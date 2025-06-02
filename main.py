from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from agents.gemini_agent_basic import GeminiAgentBasic

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

class InvokeRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    model_name: Optional[str] = "gemini-2.0-flash"
    model_provider: Optional[str] = "google"

@app.post("/invoke_gemini")
async def invoke_gemini_agent(request: InvokeRequest) -> Dict[str, str]:
    # Create agent with the requested parameters (excluding model_provider)
    gemini_agent = GeminiAgentBasic(
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p
    )
    response_content = gemini_agent.invoke_completion(request.messages)
    return {"response": response_content}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("PYTHON_ENV", "development"),
    }

