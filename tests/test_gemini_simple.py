import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_gemini_endpoint_basic():
    # Send a simple message
    response = client.post("/invoke_gemini", json={
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    })
    
    # Verify the response
    assert response.status_code == 200
    assert "response" in response.json()
    assert len(response.json()["response"]) > 0  # Just check that there's some text response 