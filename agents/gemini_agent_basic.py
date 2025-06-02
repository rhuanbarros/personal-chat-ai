from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from typing import List, Union, Dict, Any

class GeminiAgentBasic:
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.7, top_p: float = 1.0):
        """
        Initializes the Gemini Agent.

        Args:
            model_name (str): The name of the Gemini model to use.
            temperature (float): The temperature for the model's output.
            top_p (float): The top_p (nucleus sampling) parameter for the model's output.
        """
        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            google_api_key=os.getenv("GEMINI_API_KEY") 
        )

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
        if not messages:
            return "No messages provided."

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
            response = self.llm.invoke(langchain_messages)
            return response.content  # type: ignore
        except Exception as e:
            # Basic error handling
            print(f"Error invoking Gemini model: {e}")
            return "An error occurred while processing your request." 