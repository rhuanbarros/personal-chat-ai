"""
Logging utilities for model requests and responses.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel
from typing import List


class InvokeRequest(BaseModel):
    """Request model for invoke endpoints - redefined here to avoid circular imports"""
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    model_name: Optional[str] = "gemini-2.0-flash"
    model_provider: Optional[str] = "google"


def log_model_response(response_content: str, response_metadata: Optional[Dict] = None) -> None:
    """
    Log the model response and reasoning tokens if available.
    
    Args:
        response_content: The response content from the model
        response_metadata: Optional metadata from the model response
    """
    print("\n=== Model Response ===")
    print(f"Response Content: {response_content}")
    
    if response_metadata:
        print(f"Response Metadata: {response_metadata}")
        
        # Check for reasoning/thinking tokens in usage metadata using LangChain standard format
        usage_metadata = response_metadata.get('usage_metadata', {})
        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)
            
            # Get reasoning tokens from output_token_details (LangChain standard format)
            output_token_details = usage_metadata.get('output_token_details', {})
            reasoning_tokens = output_token_details.get('reasoning', 0) if output_token_details else 0
            
            # Also check for legacy 'thoughts_token_count' for backward compatibility
            legacy_thoughts_tokens = usage_metadata.get('thoughts_token_count', 0)
            
            # Use reasoning tokens if available, otherwise fall back to legacy format
            actual_reasoning_tokens = reasoning_tokens if reasoning_tokens > 0 else legacy_thoughts_tokens
            
            print(f"Token Usage:")
            print(f"  Input tokens: {input_tokens}")
            print(f"  Output tokens: {output_tokens}")
            print(f"  Total tokens: {total_tokens}")
            
            if actual_reasoning_tokens > 0:
                print(f"  🧠 Reasoning tokens: {actual_reasoning_tokens}")
                reasoning_ratio = (actual_reasoning_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                print(f"  📈 Reasoning ratio: {reasoning_ratio:.1f}% of total tokens")
                print(f"  💡 Model used reasoning process!")
            else:
                print(f"  No reasoning tokens detected")
    
    print("===================\n")


def log_request_details(request: InvokeRequest) -> None:
    """
    Log the request details.
    
    Args:
        request: The invoke request object
    """
    print("\n=== Request Details ===")
    print(f"Model Provider: {request.model_provider}")
    print(f"Model Name: {request.model_name}")
    print(f"Temperature: {request.temperature}")
    print(f"Top P: {request.top_p}")
    print("\nMessages:")
    for i, msg in enumerate(request.messages, 1):
        print(f"  Message {i}:")
        print(f"    Role: {msg.get('role', 'unknown')}")
        print(f"    Content: {msg.get('content', '')}")
    print("===================\n")


def log_reasoning_tokens_info(usage_metadata: Dict[str, Any]) -> None:
    """
    Extract and log reasoning tokens information from usage metadata.
    
    Args:
        usage_metadata: Usage metadata dictionary from model response
    """
    if not usage_metadata:
        print("No usage metadata available")
        return
    
    input_tokens = usage_metadata.get('input_tokens', 0)
    output_tokens = usage_metadata.get('output_tokens', 0)
    total_tokens = usage_metadata.get('total_tokens', 0)
    
    # Get reasoning tokens from output_token_details (LangChain standard format)
    output_token_details = usage_metadata.get('output_token_details', {})
    reasoning_tokens = output_token_details.get('reasoning', 0) if output_token_details else 0
    
    # Also check for legacy 'thoughts_token_count' for backward compatibility
    legacy_thoughts_tokens = usage_metadata.get('thoughts_token_count', 0)
    
    # Use reasoning tokens if available, otherwise fall back to legacy format
    actual_reasoning_tokens = reasoning_tokens if reasoning_tokens > 0 else legacy_thoughts_tokens
    
    print("🔍 Token Analysis:")
    print(f"  📥 Input: {input_tokens} tokens")
    print(f"  📤 Output: {output_tokens} tokens")
    print(f"  📊 Total: {total_tokens} tokens")
    
    if actual_reasoning_tokens > 0:
        print(f"  🧠 Reasoning: {actual_reasoning_tokens} tokens")
        reasoning_ratio = (actual_reasoning_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  📈 Reasoning ratio: {reasoning_ratio:.1f}% of total tokens")
        print(f"  💡 Model engaged in reasoning process!")
    else:
        print(f"  🚫 No reasoning tokens used") 