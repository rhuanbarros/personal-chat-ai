"""
Simple test file for ResearchAgent basic functionality
"""
import pytest
from datetime import datetime
from agents.models.research_models import ResearchInput, ResearchOutput, SearchResult

def test_research_input_model():
    """Test ResearchInput Pydantic model"""
    research_input = ResearchInput(
        context="I'm working on a Python project",
        objective="Find best practices for testing"
    )
    
    assert research_input.context == "I'm working on a Python project"
    assert research_input.objective == "Find best practices for testing"

def test_search_result_model():
    """Test SearchResult Pydantic model"""
    search_result = SearchResult(
        title="Python Testing Guide",
        url="https://example.com/testing",
        content="This is a guide about Python testing...",
        relevance_score=0.8,
        summary="A comprehensive guide to testing",
        source_domain="example.com"
    )
    
    assert search_result.title == "Python Testing Guide"
    assert search_result.relevance_score == 0.8
    assert 0 <= search_result.relevance_score <= 1

def test_search_result_invalid_score():
    """Test that SearchResult validates relevance score bounds"""
    with pytest.raises(ValueError):
        SearchResult(
            title="Test",
            url="https://example.com",
            content="Test content",
            relevance_score=1.5,  # Invalid: > 1
            summary="Test summary",
            source_domain="example.com"
        )
    
    with pytest.raises(ValueError):
        SearchResult(
            title="Test",
            url="https://example.com", 
            content="Test content",
            relevance_score=-0.1,  # Invalid: < 0
            summary="Test summary",
            source_domain="example.com"
        )

def test_research_output_model():
    """Test ResearchOutput Pydantic model"""
    search_results = [
        SearchResult(
            title="Python Testing",
            url="https://example.com/test1",
            content="Testing content 1",
            relevance_score=0.9,
            summary="Summary 1",
            source_domain="example.com"
        ),
        SearchResult(
            title="Unit Testing",
            url="https://example.com/test2", 
            content="Testing content 2",
            relevance_score=0.7,
            summary="Summary 2",
            source_domain="example.com"
        )
    ]
    
    research_output = ResearchOutput(
        original_objective="Find testing best practices",
        anonymized_queries=["python testing best practices", "unit testing guide"],
        selected_documents=search_results,
        total_documents_found=5,
        research_timestamp=datetime.now()
    )
    
    assert research_output.original_objective == "Find testing best practices"
    assert len(research_output.selected_documents) == 2
    assert research_output.total_documents_found == 5
    assert len(research_output.anonymized_queries) == 2

def test_regex_anonymization():
    """Test basic regex anonymization patterns"""
    import re
    
    def simple_anonymize(text):
        """Simple anonymization function for testing"""
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # URLs
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        return text
    
    test_text = """
    Contact John at john.doe@company.com or call 555-123-4567.
    Visit https://secret.example.com for more info.
    """
    
    anonymized = simple_anonymize(test_text)
    
    assert "[EMAIL]" in anonymized
    assert "[PHONE]" in anonymized  
    assert "[URL]" in anonymized
    assert "john.doe@company.com" not in anonymized
    assert "555-123-4567" not in anonymized
    assert "https://secret.example.com" not in anonymized

def test_domain_extraction():
    """Test URL domain extraction logic"""
    from urllib.parse import urlparse
    
    def extract_domain(url):
        """Simple domain extraction for testing"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"
    
    test_cases = [
        ("https://www.example.com/path", "www.example.com"),
        ("http://subdomain.example.org/page", "subdomain.example.org"),
        ("https://example.net", "example.net"),
        ("invalid-url", "unknown"),
        ("", "unknown")
    ]
    
    for url, expected_domain in test_cases:
        assert extract_domain(url) == expected_domain

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 