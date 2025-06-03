import pytest
import os
from unittest.mock import patch
from dotenv import load_dotenv
from agents.model_basic import ModelBasic

def is_valid_api_key(api_key, provider):
    """Check if API key appears to be valid (basic validation)."""
    if not api_key:
        return False
    
    # Basic format validation
    if provider == "google":
        return len(api_key) > 10 and not api_key.startswith("test") and api_key != "adasdsad"
    elif provider == "openai":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider == "anthropic":
        return len(api_key) > 10 and not api_key.startswith("test") and api_key != "adasdsad"
    
    return False

@pytest.fixture
def disable_langfuse():
    """Fixture to disable Langfuse during tests to avoid error messages."""
    with patch('agents.model_basic.get_langfuse_callbacks', return_value=[]):
        yield

def test_google_provider(disable_langfuse):
    """Test that Google provider works and returns valid text."""
    # Load environment variables from .env.local
    load_dotenv(".env.local")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not is_valid_api_key(api_key, "google"):
        pytest.skip("Valid GEMINI_API_KEY not found in environment")
    
    # Instantiate the agent with Google provider
    agent = ModelBasic(model_provider="google")
    
    # Test with a simple query
    messages = [{"role": "user", "content": "Say hello in one word"}]
    response = agent.invoke_completion(messages)
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "No messages provided."
    assert response != "An error occurred while processing your request."
    
    print(f"Google response: {response}")

def test_openai_provider(disable_langfuse):
    """Test that OpenAI provider works and returns valid text."""
    # Load environment variables from .env.local
    load_dotenv(".env.local")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not is_valid_api_key(api_key, "openai"):
        pytest.skip("Valid OPENAI_API_KEY not found in environment")
    
    # Instantiate the agent with OpenAI provider
    agent = ModelBasic(model_provider="openai")
    
    # Test with a simple query
    messages = [{"role": "user", "content": "Say hello in one word"}]
    response = agent.invoke_completion(messages)
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "No messages provided."
    assert response != "An error occurred while processing your request."
    
    print(f"OpenAI response: {response}")

def test_anthropic_provider(disable_langfuse):
    """Test that Anthropic provider works and returns valid text."""
    # Load environment variables from .env.local
    load_dotenv(".env.local")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not is_valid_api_key(api_key, "anthropic"):
        pytest.skip("Valid ANTHROPIC_API_KEY not found in environment")
    
    # Instantiate the agent with Anthropic provider
    agent = ModelBasic(model_provider="anthropic")
    
    # Test with a simple query
    messages = [{"role": "user", "content": "Say hello in one word"}]
    response = agent.invoke_completion(messages)
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert response != "No messages provided."
    assert response != "An error occurred while processing your request."
    
    print(f"Anthropic response: {response}")

def test_unsupported_provider():
    """Test that unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model provider: invalid"):
        ModelBasic(model_provider="invalid") 