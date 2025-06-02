import pytest
import os
from dotenv import load_dotenv
from agents.gemini_agent_with_tools import GeminiAgentWithTools

def test_invoke_with_tools():
    """Test that invoke_with_tools method works and returns valid text."""
    # Load environment variables from .env.local
    load_dotenv(".env.local")
    
    # Instantiate the agent
    agent = GeminiAgentWithTools()
    
    # Test with a simple query
    user_input = "What is the capital of France?"
    response = agent.invoke_with_tools(user_input)
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "No response generated."
    assert response != "An error occurred while processing your request with tools."
    
    print(f"Response: {response}") 