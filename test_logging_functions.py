#!/usr/bin/env python3
"""
Test script to demonstrate the logging functions and reasoning token detection.
"""

from utils.logging_utils import log_model_response, log_request_details, log_reasoning_tokens_info, InvokeRequest

def test_logging_functions():
    """Test the logging functions with mock data."""
    
    print("🧪 Testing Logging Functions")
    print("=" * 50)
    
    # Test request logging
    sample_request = InvokeRequest(
        messages=[
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": "42"},
            {"role": "user", "content": "Please explain that answer."}
        ],
        temperature=0.7,
        top_p=0.9,
        model_name="gemini-2.0-flash",
        model_provider="google"
    )
    
    log_request_details(sample_request)
    
    # Test response logging without reasoning tokens
    print("📝 Testing response without reasoning tokens:")
    sample_response_no_reasoning = "Life, the universe, and everything has many interpretations, but 42 was Douglas Adams' humorous answer in 'The Hitchhiker's Guide to the Galaxy'."
    sample_metadata_no_reasoning = {
        'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
        'finish_reason': 'STOP',
        'model_name': 'gemini-2.0-flash',
        'safety_ratings': [],
        'usage_metadata': {
            'input_tokens': 25,
            'output_tokens': 30,
            'total_tokens': 55,
            'output_token_details': {
                'reasoning': 0
            }
        }
    }
    
    log_model_response(sample_response_no_reasoning, sample_metadata_no_reasoning)
    
    # Test response logging with reasoning tokens (LangChain standard format)
    print("🧠 Testing response WITH reasoning tokens (LangChain standard format):")
    sample_response_with_reasoning = "After thinking about this question deeply, I believe the meaning of life involves finding purpose, creating connections, and contributing positively to the world around us."
    sample_metadata_with_reasoning = {
        'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
        'finish_reason': 'STOP',
        'model_name': 'gemini-2.0-flash-thinking',
        'safety_ratings': [],
        'usage_metadata': {
            'input_tokens': 25,
            'output_tokens': 35,
            'total_tokens': 185,
            'output_token_details': {
                'reasoning': 125  # This indicates reasoning was used (LangChain standard)
            }
        }
    }
    
    log_model_response(sample_response_with_reasoning, sample_metadata_with_reasoning)
    
    # Test reasoning token analysis
    print("🔍 Testing detailed reasoning token analysis:")
    log_reasoning_tokens_info(sample_metadata_with_reasoning['usage_metadata'])
    
    # Test legacy format for backward compatibility
    print("🔄 Testing legacy format (thoughts_token_count) for backward compatibility:")
    sample_metadata_legacy = {
        'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
        'finish_reason': 'STOP',
        'model_name': 'gemini-2.0-flash-thinking',
        'safety_ratings': [],
        'usage_metadata': {
            'input_tokens': 25,
            'output_tokens': 35,
            'total_tokens': 185,
            'thoughts_token_count': 125  # Legacy format
        }
    }
    
    log_model_response("Legacy format test response.", sample_metadata_legacy)
    
    # Test real-world example from your output
    print("🌍 Testing real-world example (based on your output):")
    real_world_metadata = {
        'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
        'finish_reason': 'STOP',
        'model_name': 'models/gemini-2.5-flash-preview-05-20',
        'safety_ratings': [],
        'usage_metadata': {
            'input_tokens': 62,
            'output_tokens': 1780,
            'total_tokens': 3451,
            'input_token_details': {'cache_read': 0},
            'output_token_details': {'reasoning': 1609}
        }
    }
    
    log_model_response("Real world test with actual reasoning tokens.", real_world_metadata)
    
    print("\n✅ All logging function tests completed!")

if __name__ == "__main__":
    test_logging_functions() 