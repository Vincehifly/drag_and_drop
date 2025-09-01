"""LLM client initialization and configuration."""

import os
from langchain_openai import AzureChatOpenAI



def create_azure_openai_llm(temperature: float = 0.1, json_mode: bool = False) -> AzureChatOpenAI:
    """
    Create AzureChatOpenAI instance with error handling and optional JSON mode.
    
    Args:
        temperature: Controls randomness in responses (0.0 to 1.0)
        json_mode: Enable JSON response format for structured output
        
    Returns:
        Configured AzureChatOpenAI instance
        
    Raises:
        ValueError: If required environment variables are missing or client creation fails
    """
    try:
        base_config = {
            "model": "gpt-4o",
            "azure_deployment": os.environ["DEPLOYMENT_NAME"],
            "api_key": os.environ["AZUREOPENAIAPIKEY"],
            "api_version": os.environ["AZUREOPENAIAPIVERSION"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "temperature": temperature
        }
        
        if json_mode:
            base_config["model_kwargs"] = {"response_format": {"type": "json_object"}}
            
        return AzureChatOpenAI(**base_config)
    except KeyError as e:
        missing_env = str(e).strip("'")
        raise ValueError(f"Missing required environment variable: {missing_env}. Please set this environment variable.")
    except Exception as e:
        raise ValueError(f"Failed to create Azure OpenAI LLM: {e}")

def create_json_mode_llm(temperature: float = 0.1) -> AzureChatOpenAI:
    """
    Create AzureChatOpenAI instance specifically for JSON responses.
    
    Args:
        temperature: Controls randomness in responses
        
    Returns:
        AzureChatOpenAI instance configured for JSON output
    """
    return create_azure_openai_llm(temperature=temperature, json_mode=True)

def create_conversation_llm(temperature: float = 0.3) -> AzureChatOpenAI:
    """
    Create AzureChatOpenAI instance for conversational responses.
    
    Args:
        temperature: Controls randomness in responses (higher for more creative responses)
        
    Returns:
        AzureChatOpenAI instance for natural conversation
    """
    return create_azure_openai_llm(temperature=temperature, json_mode=False)
