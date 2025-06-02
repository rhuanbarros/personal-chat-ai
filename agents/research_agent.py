from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from .models.research_models import ResearchInput, SearchResult, ResearchOutput
from .prompts.research_prompts import (
    ANONYMIZATION_PROMPT,
    QUERY_GENERATION_PROMPT, 
    DOCUMENT_ANALYSIS_PROMPT,
    WEB_SEARCH_SYSTEM_PROMPT
)

class ResearchState(TypedDict):
    """State object for LangGraph workflow"""
    research_input: Optional[ResearchInput]
    anonymized_context: str
    search_queries: List[str]
    raw_search_results: List[Dict[str, Any]]
    analyzed_documents: List[SearchResult]
    final_output: Optional[ResearchOutput]
    error: Optional[str]

class ResearchAgent:
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.7, 
                 top_p: float = 1.0, num_queries: int = 2, tavily_max_results: int = 5,
                 tavily_topic: str = "general", relevance_threshold: float = 0.5):
        """
        Initializes the Research Agent with configurable parameters.

        Args:
            model_name (str): The name of the Gemini model to use.
            temperature (float): The temperature for the model's output.
            top_p (float): The top_p (nucleus sampling) parameter.
            num_queries (int): Number of search queries to generate.
            tavily_max_results (int): Maximum number of search results per query.
            tavily_topic (str): The topic for Tavily search ("general" or "news").
            relevance_threshold (float): Minimum relevance score to include documents.
        """
        self.num_queries = num_queries
        self.relevance_threshold = relevance_threshold
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize Tavily Search Tool
        self.tavily_search_tool = TavilySearch(
            max_results=tavily_max_results,
            topic=tavily_topic,
            api_key=os.getenv("TAVILY_API_KEY")
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow for the research process."""
        
        def anonymize_node(state: ResearchState) -> ResearchState:
            """Node to anonymize the user context."""
            try:
                if not state.get("research_input"):
                    state["error"] = "No research input provided"
                    return state
                
                anonymized_context = self.anonymize_context(state["research_input"].context)
                state["anonymized_context"] = anonymized_context
                return state
            except Exception as e:
                state["error"] = f"Error in anonymization: {str(e)}"
                return state

        def query_generation_node(state: ResearchState) -> ResearchState:
            """Node to generate search queries."""
            try:
                if state.get("error"):
                    return state
                
                if not state.get("research_input"):
                    state["error"] = "No research input provided"
                    return state
                
                queries = self.generate_search_queries(
                    state["anonymized_context"],
                    state["research_input"].objective,
                    self.num_queries
                )
                state["search_queries"] = queries
                return state
            except Exception as e:
                state["error"] = f"Error in query generation: {str(e)}"
                return state

        def search_node(state: ResearchState) -> ResearchState:
            """Node to execute web searches."""
            try:
                if state.get("error"):
                    return state
                
                all_results = []
                for query in state["search_queries"]:
                    try:
                        results = self.tavily_search_tool.invoke({"query": query})
                        if isinstance(results, list):
                            all_results.extend(results)
                        elif isinstance(results, dict) and "results" in results:
                            all_results.extend(results["results"])
                    except Exception as e:
                        print(f"Error searching for query '{query}': {e}")
                        continue
                
                state["raw_search_results"] = all_results
                return state
            except Exception as e:
                state["error"] = f"Error in web search: {str(e)}"
                return state

        def analysis_node(state: ResearchState) -> ResearchState:
            """Node to analyze and filter search results."""
            try:
                if state.get("error"):
                    return state
                
                if not state.get("research_input"):
                    state["error"] = "No research input provided"
                    return state
                
                analyzed_docs = self.analyze_and_filter_documents(
                    state["raw_search_results"],
                    state["research_input"].objective
                )
                state["analyzed_documents"] = analyzed_docs
                return state
            except Exception as e:
                state["error"] = f"Error in document analysis: {str(e)}"
                return state

        def output_node(state: ResearchState) -> ResearchState:
            """Node to format the final output."""
            try:
                if state.get("error"):
                    return state
                
                if not state.get("research_input"):
                    state["error"] = "No research input provided"
                    return state
                
                output = ResearchOutput(
                    original_objective=state["research_input"].objective,
                    anonymized_queries=state["search_queries"],
                    selected_documents=state["analyzed_documents"],
                    total_documents_found=len(state["raw_search_results"]),
                    research_timestamp=datetime.now()
                )
                state["final_output"] = output
                return state
            except Exception as e:
                state["error"] = f"Error creating output: {str(e)}"
                return state

        # Build the graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("anonymize", anonymize_node)
        workflow.add_node("query_generation", query_generation_node)
        workflow.add_node("search", search_node)
        workflow.add_node("analysis", analysis_node)
        workflow.add_node("output", output_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "anonymize")
        workflow.add_edge("anonymize", "query_generation")
        workflow.add_edge("query_generation", "search")
        workflow.add_edge("search", "analysis")
        workflow.add_edge("analysis", "output")
        workflow.add_edge("output", END)
        
        return workflow.compile()

    def anonymize_context(self, context: str) -> str:
        """
        Remove or anonymize private information from context.
        
        Args:
            context (str): The original context containing potential private information.
            
        Returns:
            str: Anonymized context with sensitive information replaced.
        """
        try:
            # Use LLM for intelligent anonymization
            prompt = ANONYMIZATION_PROMPT.format(context=context)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            anonymized = response.content.strip()
            
            # Fallback regex-based anonymization for extra safety
            anonymized = self._regex_anonymize(anonymized)
            
            return anonymized
        except Exception as e:
            print(f"Error in LLM anonymization, using regex fallback: {e}")
            return self._regex_anonymize(context)

    def _regex_anonymize(self, text: str) -> str:
        """Fallback regex-based anonymization."""
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # API keys (common patterns)
        text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[API_KEY]', text)
        text = re.sub(r'sk-[A-Za-z0-9]{48}', '[API_KEY]', text)
        
        # IP addresses
        text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP_ADDRESS]', text)
        
        # Phone numbers (US format)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # URLs with sensitive info
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        
        return text

    def generate_search_queries(self, anonymized_context: str, objective: str, num_queries: int) -> List[str]:
        """
        Generate N search queries based on anonymized context and objective.
        
        Args:
            anonymized_context (str): The anonymized context.
            objective (str): The user's objective.
            num_queries (int): Number of queries to generate.
            
        Returns:
            List[str]: List of search queries.
        """
        try:
            prompt = QUERY_GENERATION_PROMPT.format(
                anonymized_context=anonymized_context,
                objective=objective,
                num_queries=num_queries
            )
            
            response = self.llm.invoke([
                SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # Parse queries from response
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            
            # Ensure we have the requested number of queries
            if len(queries) < num_queries:
                # Add a fallback query if we don't have enough
                queries.append(objective)
            
            return queries[:num_queries]
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            # Fallback to using the objective as a single query
            return [objective]

    def analyze_and_filter_documents(self, documents: List[Dict], objective: str) -> List[SearchResult]:
        """
        Use LLM to analyze, filter, and score documents.
        
        Args:
            documents (List[Dict]): Raw search results from Tavily.
            objective (str): The research objective.
            
        Returns:
            List[SearchResult]: Filtered and scored documents.
        """
        if not documents:
            return []
        
        try:
            # Format documents for analysis
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_doc = {
                    "index": i,
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "content": doc.get("content", "")[:1000],  # Limit content length
                }
                formatted_docs.append(formatted_doc)
            
            prompt = DOCUMENT_ANALYSIS_PROMPT.format(
                objective=objective,
                search_results=json.dumps(formatted_docs, indent=2)
            )
            
            response = self.llm.invoke([
                SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # Parse JSON response
            try:
                analysis_results = json.loads(response.content.strip())
            except json.JSONDecodeError:
                # Fallback: keep all documents with default scoring
                return self._fallback_document_analysis(documents, objective)
            
            # Convert to SearchResult objects
            search_results = []
            for result in analysis_results:
                if not isinstance(result, dict):
                    continue
                
                doc_index = result.get("index", 0)
                if doc_index >= len(documents):
                    continue
                
                original_doc = documents[doc_index]
                relevance_score = float(result.get("relevance_score", 0.0))
                
                if relevance_score >= self.relevance_threshold:
                    search_result = SearchResult(
                        title=original_doc.get("title", ""),
                        url=original_doc.get("url", ""),
                        content=original_doc.get("content", ""),
                        relevance_score=relevance_score,
                        summary=result.get("summary", ""),
                        source_domain=self._extract_domain(original_doc.get("url", ""))
                    )
                    search_results.append(search_result)
            
            # Sort by relevance score (highest first)
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return search_results
            
        except Exception as e:
            print(f"Error in document analysis: {e}")
            return self._fallback_document_analysis(documents, objective)

    def _fallback_document_analysis(self, documents: List[Dict], objective: str) -> List[SearchResult]:
        """Fallback document analysis when LLM analysis fails."""
        search_results = []
        for doc in documents:
            # Simple relevance scoring based on keyword matching
            title = doc.get("title", "").lower()
            content = doc.get("content", "").lower()
            objective_words = objective.lower().split()
            
            score = 0.0
            for word in objective_words:
                if word in title:
                    score += 0.3
                if word in content:
                    score += 0.1
            
            score = min(score, 1.0)  # Cap at 1.0
            
            if score >= self.relevance_threshold:
                search_result = SearchResult(
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    content=doc.get("content", ""),
                    relevance_score=score,
                    summary=content[:200] + "..." if len(content) > 200 else content,
                    source_domain=self._extract_domain(doc.get("url", ""))
                )
                search_results.append(search_result)
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    async def research(self, research_input: ResearchInput) -> ResearchOutput:
        """
        Main method that orchestrates the entire research process.
        
        Args:
            research_input (ResearchInput): The research request.
            
        Returns:
            ResearchOutput: The structured research results.
        """
        try:
            # Initialize state
            initial_state: ResearchState = {
                "research_input": research_input,
                "anonymized_context": "",
                "search_queries": [],
                "raw_search_results": [],
                "analyzed_documents": [],
                "final_output": None,
                "error": None
            }
            
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            if final_state.get("error"):
                # Return error as empty result
                return ResearchOutput(
                    original_objective=research_input.objective,
                    anonymized_queries=[],
                    selected_documents=[],
                    total_documents_found=0,
                    research_timestamp=datetime.now()
                )
            
            return final_state.get("final_output") or ResearchOutput(
                original_objective=research_input.objective,
                anonymized_queries=[],
                selected_documents=[],
                total_documents_found=0,
                research_timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in research process: {e}")
            # Return empty result on error
            return ResearchOutput(
                original_objective=research_input.objective,
                anonymized_queries=[],
                selected_documents=[],
                total_documents_found=0,
                research_timestamp=datetime.now()
            ) 