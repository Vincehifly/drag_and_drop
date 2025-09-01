"""
Google Form Agent Package

A modular conversational agent with tool integration,
featuring human-in-the-loop capabilities and JSON mode LLM support.

Main Components:
- models: State definitions and data types  
- llm_client: LLM initialization with JSON mode support
- validation: Field validation and data completeness checking
- tools: Tool implementations and registry
- nodes: LangGraph node implementations
- graph_builder: Graph construction and compilation
- prompts: Prompt templates and building functions
"""

from .models import (
    ConversationState,
    NodeAction,
    make_initial_conversation_state,
)

from .llm_client import (
    create_azure_openai_llm,
    create_json_mode_llm,
    create_conversation_llm,
)

from .validation import (
    validate_field,
    get_missing_fields,
    is_data_complete,
)

from .tools import (
    execute_tool,
    AVAILABLE_TOOLS,
    format_search_results,
)

__version__ = "2.0.0"
__author__ = "Assistant"

__all__ = [
    # Types and models
    "ConversationState",
    "NodeAction",
    "make_initial_conversation_state",
    
    # LLM clients
    "create_azure_openai_llm",
    "create_json_mode_llm", 
    "create_conversation_llm",
    
    # Validation
    "validate_field",
    "get_missing_fields",
    "is_data_complete",
    
    # Tools
    "execute_tool",
    "AVAILABLE_TOOLS",
    "format_search_results",
]
