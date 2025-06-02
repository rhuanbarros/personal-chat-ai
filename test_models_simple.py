"""
Very simple test for just the Pydantic models
"""
import pytest
from datetime import datetime

def test_research_models():
    """Test that we can import and use the research models"""
    try:
        from agents.models.research_models import ResearchInput, ResearchOutput, SearchResult
        
        # Test ResearchInput
        research_input = ResearchInput(
            context="I'm working on a Python project",
            objective="Find best practices for testing"
        )
        assert research_input.context == "I'm working on a Python project"
        assert research_input.objective == "Find best practices for testing"
        print("✅ ResearchInput model works!")
        
        # Test SearchResult
        search_result = SearchResult(
            title="Python Testing Guide",
            url="https://example.com/testing",
            content="This is a guide about Python testing...",
            relevance_score=0.8,
            summary="A comprehensive guide to testing",
            source_domain="example.com"
        )
        assert search_result.relevance_score == 0.8
        assert 0 <= search_result.relevance_score <= 1
        print("✅ SearchResult model works!")
        
        # Test ResearchOutput
        research_output = ResearchOutput(
            original_objective="Find testing best practices",
            anonymized_queries=["python testing"],
            selected_documents=[search_result],
            total_documents_found=1,
            research_timestamp=datetime.now()
        )
        assert len(research_output.selected_documents) == 1
        assert research_output.total_documents_found == 1
        print("✅ ResearchOutput model works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_validation():
    """Test model validation"""
    try:
        from agents.models.research_models import SearchResult
        
        # This should raise a validation error
        with pytest.raises(ValueError):
            SearchResult(
                title="Test",
                url="https://example.com",
                content="Test content",
                relevance_score=1.5,  # Invalid: > 1
                summary="Test summary",
                source_domain="example.com"
            )
        print("✅ Validation works correctly!")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ResearchAgent models...")
    if test_research_models() and test_validation():
        print("🎉 All basic tests passed!")
    else:
        print("❌ Some tests failed") 