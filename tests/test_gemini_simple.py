import pytest
import requests

# Use the actual running server
BASE_URL = "http://localhost:8000"

def test_gemini_endpoint_basic():
    # Send a simple message to the actual running API
    response = requests.post(f"{BASE_URL}/invoke_gemini", json={
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    })
    
    # Verify the response
    assert response.status_code == 200
    assert "response" in response.json()
    assert len(response.json()["response"]) > 0  # Just check that there's some text response 