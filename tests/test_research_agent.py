import pytest
import asyncio
import os
from datetime import datetime
from agents.research_agent import ResearchAgent
from agents.models.research_models import ResearchInput, ResearchOutput, SearchResult

class TestResearchAgent:
    """Test cases for the ResearchAgent"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create agent with test parameters
        self.agent = ResearchAgent(
            model_name="gemini-2.0-flash",
            temperature=0.1,  # Low temperature for more consistent results
            num_queries=2,
            tavily_max_results=3,  # Fewer results for faster testing
            relevance_threshold=0.3  # Lower threshold for testing
        )

    def test_agent_initialization(self):
        """Test that the agent initializes correctly"""
        assert self.agent is not None
        assert self.agent.num_queries == 2
        assert self.agent.relevance_threshold == 0.3
        assert self.agent.llm is not None
        assert self.agent.tavily_search_tool is not None
        assert self.agent.workflow is not None

    def test_context_anonymization(self):
        """Test the context anonymization functionality"""
        sensitive_context = """
        My name is John Doe and my email is john.doe@company.com. 
        I work at Acme Corp and my API key is sk-1234567890abcdef1234567890abcdef12345678.
        My phone number is 555-123-4567 and I visited https://example.com/secret-page.
        My IP address is 192.168.1.100.
        """
        
        anonymized = self.agent.anonymize_context(sensitive_context)
        
        # Check that sensitive information has been anonymized
        assert "john.doe@company.com" not in anonymized.lower()
        assert "sk-1234567890abcdef1234567890abcdef12345678" not in anonymized
        assert "555-123-4567" not in anonymized
        assert "192.168.1.100" not in anonymized
        assert "https://example.com/secret-page" not in anonymized

    def test_regex_anonymization_fallback(self):
        """Test the regex-based anonymization fallback"""
        sensitive_text = """
        Contact: user@example.com
        API Key: sk-abcd1234567890abcdef1234567890abcdef123456
        Phone: 123-456-7890
        IP: 10.0.0.1
        URL: https://secret.example.com/api
        """
        
        anonymized = self.agent._regex_anonymize(sensitive_text)
        
        assert "[EMAIL]" in anonymized
        assert "[API_KEY]" in anonymized
        assert "[PHONE]" in anonymized
        assert "[IP_ADDRESS]" in anonymized
        assert "[URL]" in anonymized

    def test_query_generation(self):
        """Test search query generation"""
        anonymized_context = "I need to find information about [COMPANY_NAME] for a business analysis."
        objective = "Research latest AI trends and market analysis"
        
        queries = self.agent.generate_search_queries(anonymized_context, objective, 3)
        
        assert len(queries) <= 3
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q.strip()) > 0 for q in queries)

    def test_domain_extraction(self):
        """Test URL domain extraction"""
        test_cases = [
            ("https://www.example.com/path", "www.example.com"),
            ("http://subdomain.example.org/page?param=value", "subdomain.example.org"),
            ("https://example.net", "example.net"),
            ("invalid-url", "unknown"),
            ("", "unknown")
        ]
        
        for url, expected_domain in test_cases:
            assert self.agent._extract_domain(url) == expected_domain

    def test_fallback_document_analysis(self):
        """Test the fallback document analysis method"""
        mock_documents = [
            {
                "title": "AI Trends in 2024",
                "url": "https://example.com/ai-trends",
                "content": "Artificial intelligence is rapidly evolving with machine learning advances..."
            },
            {
                "title": "Cooking Recipes",
                "url": "https://cooking.com/recipes",
                "content": "Here are some delicious recipes for dinner..."
            }
        ]
        
        objective = "AI trends and machine learning"
        results = self.agent._fallback_document_analysis(mock_documents, objective)
        
        # Should filter out irrelevant documents
        assert len(results) <= len(mock_documents)
        
        # Check that results are SearchResult objects
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.relevance_score >= 0
            assert result.relevance_score <= 1
            assert result.source_domain in ["example.com", "cooking.com"]

    @pytest.mark.asyncio
    async def test_research_input_validation(self):
        """Test that research method handles input validation"""
        # Test with valid input
        research_input = ResearchInput(
            context="I'm working on a [COMPANY_NAME] project",
            objective="Find recent developments in renewable energy"
        )
        
        result = await self.agent.research(research_input)
        
        assert isinstance(result, ResearchOutput)
        assert result.original_objective == research_input.objective
        assert isinstance(result.research_timestamp, datetime)
        assert isinstance(result.total_documents_found, int)
        assert isinstance(result.selected_documents, list)
        assert isinstance(result.anonymized_queries, list)

    @pytest.mark.asyncio
    async def test_research_with_empty_context(self):
        """Test research with minimal context"""
        research_input = ResearchInput(
            context="",
            objective="Python programming best practices"
        )
        
        result = await self.agent.research(research_input)
        
        assert isinstance(result, ResearchOutput)
        assert result.original_objective == research_input.objective

    def test_research_models_validation(self):
        """Test that Pydantic models validate correctly"""
        # Valid ResearchInput
        valid_input = ResearchInput(
            context="Some context here",
            objective="Find information about X"
        )
        assert valid_input.context == "Some context here"
        assert valid_input.objective == "Find information about X"
        
        # Valid SearchResult
        valid_result = SearchResult(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            relevance_score=0.8,
            summary="Test summary",
            source_domain="example.com"
        )
        assert valid_result.relevance_score == 0.8
        assert 0 <= valid_result.relevance_score <= 1
        
        # Invalid relevance score should raise validation error
        with pytest.raises(ValueError):
            SearchResult(
                title="Test",
                url="https://example.com",
                content="Test",
                relevance_score=1.5,  # Invalid: > 1
                summary="Test",
                source_domain="example.com"
            )

# Integration test that requires API keys
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="API keys not available"
)
class TestResearchAgentIntegration:
    """Integration tests that require actual API calls"""

    @pytest.mark.asyncio
    async def test_full_research_workflow(self):
        """Test the complete research workflow with real API calls"""
        agent = ResearchAgent(
            num_queries=1,  # Limit to 1 query for faster testing
            tavily_max_results=2,  # Limit results
            relevance_threshold=0.3
        )
        
        research_input = ResearchInput(
            context="I'm a software developer working on web applications",
            objective="Latest trends in web development frameworks 2024"
        )
        
        result = await agent.research(research_input)
        
        # Verify the structure of the result
        assert isinstance(result, ResearchOutput)
        assert len(result.anonymized_queries) > 0
        assert result.total_documents_found >= 0
        assert all(isinstance(doc, SearchResult) for doc in result.selected_documents)
        
        # Check that documents have valid scores
        for doc in result.selected_documents:
            assert 0 <= doc.relevance_score <= 1
            assert len(doc.title) > 0
            assert len(doc.url) > 0
            assert len(doc.summary) > 0 