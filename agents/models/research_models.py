from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ResearchInput(BaseModel):
    context: str = Field(description="User's context information (may contain private data)")
    objective: str = Field(description="What the user wants to accomplish or search for")

class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    relevance_score: float = Field(ge=0, le=1, description="Relevance score from 0 to 1")
    summary: str
    source_domain: str

class ResearchOutput(BaseModel):
    original_objective: str
    anonymized_queries: List[str]
    selected_documents: List[SearchResult]
    total_documents_found: int
    research_timestamp: datetime 