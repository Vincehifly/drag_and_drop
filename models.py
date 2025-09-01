"""Data models and state definitions for the LangGraph agent."""

from typing import Dict, Any, TypedDict, List, Literal, Annotated
from operator import add

# --- State Definitions --- #

class ConversationState(TypedDict):
    """Enhanced state for human-in-the-loop conversational agents."""
    messages: List[Dict[str, Any]]
    user_input: str
    extracted_data: Dict[str, Any]
    query_spec: Dict[str, Any]
    tool_result: Dict[str, Any]
    last_tool_summary: str
    last_tool_context: Dict[str, Any]
    next_action: str
    chosen_tool: str
    tool_category: str
    error_message: str
    conversation_active: bool
    decision_justification: str
    decision_context: Dict[str, Any]  # NEW: Capture decision context
    session_id: str
    action_history: List[Dict[str, Any]]
    clarification_message: str
    validation_errors: List[str]

# Type definitions
NodeAction = Literal["chat", "input_extractor", "decision_router", "tool_use", "end"]
