from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from typing import List, Union, Dict, Any, Tuple
from .langfuse_config import get_langfuse_callbacks

class ModelBasic:
    def __init__(self, model_provider: str, model_name: str = None, temperature: float = 0.7, top_p: float = 1.0, **kwargs):
        """
        Initializes the Model Agent with different providers.

        Args:
            model_provider (str): The provider to use ("google", "openai", "anthropic").
            model_name (str): The name of the model to use. If None, uses provider defaults.
            temperature (float): The temperature for the model's output.
            top_p (float): The top_p (nucleus sampling) parameter for the model's output.
            **kwargs: Additional provider-specific arguments.
        """
        self.model_provider = model_provider.lower()
        
        if self.model_provider == "google":
            default_model = model_name or "gemini-2.0-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=default_model,
                temperature=temperature,
                top_p=top_p,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                **kwargs
            )
        elif self.model_provider == "openai":
            default_model = model_name or "gpt-3.5-turbo"
            self.llm = ChatOpenAI(
                model=default_model,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs
            )
        elif self.model_provider == "anthropic":
            default_model = model_name or "claude-3-haiku-20240307"
            self.llm = ChatAnthropic(
                model=default_model,
                temperature=temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}. Supported providers: 'google', 'openai', 'anthropic'")

    def invoke_completion_with_metadata(self, messages: List[Union[BaseMessage, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
        """
        Invokes the AI model with a list of messages and returns both response content and metadata.

        Args:
            messages (List[Union[BaseMessage, Dict[str, Any]]]): A list of messages, 
                where each message can be an instance of a Langchain BaseMessage 
                (HumanMessage, AIMessage, SystemMessage) or a dictionary 
                with "role" and "content" keys.

        Returns:
            Tuple[str, Dict[str, Any]]: The AI's response content and metadata containing usage info.
        """
        if not messages:
            return "No messages provided.", {}

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

        try:
            # Get Langfuse callbacks for observability (with error handling)
            callbacks = []
            try:
                callbacks = get_langfuse_callbacks()
            except Exception as e:
                print(f"Warning: Failed to get Langfuse callbacks: {e}")
                callbacks = []
            
            # Create config with callbacks if available
            config = {"callbacks": callbacks} if callbacks else {}
            
            response = self.llm.invoke(langchain_messages, config=config)
            
            # Extract metadata from response
            metadata = {}
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata or {}
            
            # Add usage metadata if available
            if hasattr(response, 'usage_metadata'):
                metadata['usage_metadata'] = response.usage_metadata or {}
            
            return response.content or "", metadata  # type: ignore
        except Exception as e:
            # Basic error handling
            print(f"Error invoking {self.model_provider} model: {e}")
            return "An error occurred while processing your request.", {}

    def invoke_completion(self, messages: List[Union[BaseMessage, Dict[str, Any]]]) -> str:
        """
        Invokes the AI model with a list of messages and returns the response.

        Args:
            messages (List[Union[BaseMessage, Dict[str, Any]]]): A list of messages, 
                where each message can be an instance of a Langchain BaseMessage 
                (HumanMessage, AIMessage, SystemMessage) or a dictionary 
                with "role" and "content" keys.

        Returns:
            str: The AI's response content.
        """
        content, _ = self.invoke_completion_with_metadata(messages)
        return content 