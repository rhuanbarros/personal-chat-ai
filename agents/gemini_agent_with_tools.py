from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import os
from typing import List, Union, Dict, Any, Optional

class GeminiAgentWithTools:
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.7, top_p: float = 1.0, 
                 tavily_max_results: int = 5, tavily_topic: str = "general"):
        """
        Initializes the Gemini Agent with web search capabilities.

        Args:
            model_name (str): The name of the Gemini model to use.
            temperature (float): The temperature for the model's output.
            top_p (float): The top_p (nucleus sampling) parameter for the model's output.
            tavily_max_results (int): Maximum number of search results to return from Tavily.
            tavily_topic (str): The topic for Tavily search ("general" or "news").
        """
        # Initialize the LLM
        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
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
        
        # Create the agent with tools
        self.agent = create_react_agent(self.llm, [self.tavily_search_tool])

    def invoke_completion(self, messages: List[Union[BaseMessage, Dict[str, Any]]]) -> str:
        """
        Invokes the AI model with a list of messages and returns the response.
        This method now uses the basic LLM without tools.

        Args:
            messages (List[Union[BaseMessage, Dict[str, Any]]]): A list of messages, 
                where each message can be an instance of a Langchain BaseMessage 
                (HumanMessage, AIMessage, SystemMessage) or a dictionary 
                with "role" and "content" keys.

        Returns:
            str: The AI's response content.
        """
        if not messages:
            return "No messages provided."

        langchain_messages: List[BaseMessage] = self._convert_to_langchain_messages(messages)

        try:
            response = self.llm.invoke(langchain_messages)
            return response.content  # type: ignore
        except Exception as e:
            # Basic error handling
            print(f"Error invoking Gemini model: {e}")
            return "An error occurred while processing your request."

    def invoke_with_tools_stream(self, user_input: str) -> str:
        """
        Invokes the agent with tools (including web search) and returns the final response.

        Args:
            user_input (str): The user's input/query.

        Returns:
            str: The agent's final response after potentially using tools.
        """
        try:
            # Stream the agent's execution and get the final result
            final_response = ""
            for step in self.agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="values",
            ):
                if step["messages"]:
                    last_message = step["messages"][-1]
                    if isinstance(last_message, AIMessage) and not hasattr(last_message, 'tool_calls'):
                        final_response = last_message.content
            
            return final_response or "No response generated."
        except Exception as e:
            print(f"Error invoking agent with tools: {e}")
            return "An error occurred while processing your request with tools."

    def invoke_with_tools(self, user_input: str) -> str:
        """
        Invokes the agent with tools (including web search) synchronously without streaming.

        Args:
            user_input (str): The user's input/query.

        Returns:
            str: The agent's final response after potentially using tools.
        """
        try:
            # Invoke the agent directly without streaming
            result = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
            
            # Extract the final AI message content
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content or "No response generated."
            
            return "No response generated."
        except Exception as e:
            print(f"Error invoking agent with tools: {e}")
            return "An error occurred while processing your request with tools."

    def search_web(self, query: str, include_domains: Optional[List[str]] = None, 
                   exclude_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Directly performs a web search using Tavily.

        Args:
            query (str): The search query.
            include_domains (Optional[List[str]]): List of domains to include in search.
            exclude_domains (Optional[List[str]]): List of domains to exclude from search.

        Returns:
            Dict[str, Any]: The search results from Tavily.
        """
        try:
            search_params = {"query": query}
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            result = self.tavily_search_tool.invoke(search_params)
            return result
        except Exception as e:
            print(f"Error performing web search: {e}")
            return {"error": "An error occurred while performing the web search."}

    def _convert_to_langchain_messages(self, messages: List[Union[BaseMessage, Dict[str, Any]]]) -> List[BaseMessage]:
        """
        Helper method to convert various message formats to LangChain BaseMessage objects.

        Args:
            messages: List of messages in various formats.

        Returns:
            List[BaseMessage]: Converted LangChain messages.
        """
        langchain_messages: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                elif role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                else:
                    # Fallback for unknown roles in dicts, treat as human message
                    langchain_messages.append(HumanMessage(content=str(msg)))
            elif isinstance(msg, BaseMessage):   # type: ignore 
                langchain_messages.append(msg) # Already a Langchain message object
            else:
                # Fallback for other types (e.g. simple strings), treat as human message
                langchain_messages.append(HumanMessage(content=str(msg)))
        
        return langchain_messages 