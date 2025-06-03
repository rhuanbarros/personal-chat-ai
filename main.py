from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from agents.gemini_agent_basic import GeminiAgentBasic
from agents.model_basic import ModelBasic
from utils.logging_utils import log_model_response, log_request_details

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


@app.post("/invoke_basic_model")
async def invoke_basic_model(request: InvokeRequest) -> Dict[str, str]:
    # Log request details
    log_request_details(request)
    
    model_agent = ModelBasic(
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
        model_provider=request.model_provider
    )
    
    # Use the new method that returns both content and metadata
    response_content, response_metadata = model_agent.invoke_completion_with_metadata(request.messages)

    # Log response details with metadata
    log_model_response(response_content, response_metadata)

    return {"response": response_content}

@app.post("/invoke_agent")
async def invoke_agent(request: InvokeRequest) -> Dict[str, str]:
    # Log request details
    log_request_details(request)
    
    gemini_agent = GeminiAgentBasic(
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p
    )
    
    # Use the new method that returns both content and metadata
    response_content, response_metadata = gemini_agent.invoke_completion_with_metadata(request.messages)
    
    # Log response details with metadata
    log_model_response(response_content, response_metadata)
    
    return {"response": response_content}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("PYTHON_ENV", "development"),
    }

