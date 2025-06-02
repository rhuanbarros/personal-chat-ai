"""
Langfuse configuration module for the agents package.
Provides a centralized way to configure and use Langfuse for LLM observability.
"""

import os
from typing import Optional
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv

class LangfuseConfig:
    """Singleton class to manage Langfuse configuration."""
    
    _instance: Optional['LangfuseConfig'] = None
    _handler: Optional[CallbackHandler] = None
    
    def __new__(cls) -> 'LangfuseConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Langfuse configuration if not already done."""
        if self._handler is None:
            # Load environment variables first
            load_dotenv(dotenv_path=".env.local", override=True)
            self._initialize_handler()
    
    def _initialize_handler(self) -> None:
        """Initialize the Langfuse handler with environment variables."""
        try:
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_API_KEY")  # This is actually the public key in your .env.local
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not secret_key or not public_key:
                print("Warning: Langfuse credentials not found in environment variables.")
                print("Please check LANGFUSE_SECRET_KEY and LANGFUSE_API_KEY in .env.local")
                self._handler = None
                return
            
            print(f"Initializing Langfuse with:")
            print(f"  Host: {host}")
            print(f"  Public Key: {public_key[:10]}...")
            print(f"  Secret Key: {secret_key[:10]}...")
            
            self._handler = CallbackHandler(
                secret_key=secret_key,
                public_key=public_key,
                host=host,
                debug=True  # Enable debug mode
            )
            print(f"Langfuse handler initialized successfully with host: {host}")
            
            # Test the connection
            try:
                # Try to flush any pending traces to test connectivity
                self._handler.langfuse.flush()
                print("Langfuse connection test successful")
            except Exception as e:
                print(f"Warning: Langfuse connection test failed: {e}")
            
        except Exception as e:
            print(f"Error initializing Langfuse handler: {e}")
            self._handler = None
    
    def get_handler(self) -> Optional[CallbackHandler]:
        """Get the Langfuse callback handler."""
        return self._handler
    
    def is_enabled(self) -> bool:
        """Check if Langfuse is properly configured and enabled."""
        return self._handler is not None

# Create a global instance
langfuse_config = LangfuseConfig()

def get_langfuse_handler() -> Optional[CallbackHandler]:
    """Convenience function to get the Langfuse handler."""
    return langfuse_config.get_handler()

def get_langfuse_callbacks() -> list:
    """Get a list containing the Langfuse callback handler, or empty list if not configured."""
    handler = get_langfuse_handler()
    if handler:
        print("Langfuse callback handler added to LLM call")
        return [handler]
    else:
        print("No Langfuse handler available, skipping tracing")
        return [] 