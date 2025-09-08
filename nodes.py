"""LangGraph node implementations with clean, minimal logging."""

import json
from typing import Dict, Any, List
from langgraph.types import interrupt

from models import ConversationState
from tools import execute_tool, AVAILABLE_TOOLS
try:
    from tools import format_search_results as _format_search_results
except Exception:
    _format_search_results = None
from validation import get_missing_fields, is_data_complete
from llm_client import create_json_mode_llm, create_conversation_llm
from prompts import build_tool_answer_prompt, build_input_translation_prompt, build_output_translation_prompt
from utilities import *

# Global variables for prompt functions (injected at runtime)
BASE_SYSTEM_PROMPT = None
TOOL_DESCRIPTIONS = None
build_input_extraction_prompt = None
build_decision_prompt = None
build_final_response_prompt = None
build_tool_use_prompt = None
build_tool_input_prompt = None

# Utility functions moved to utilities.py

def are_required_fields_complete(data: Dict[str, Any], fields: list) -> bool:
    """Check whether all required fields in `fields` are present and truthy in `data`."""
    for f in fields:
        if bool(f.get("required")):
            name = f.get("name")
            if not name:
                continue
            val = data.get(name)
            if val is None or (isinstance(val, str) and not val.strip()):
                return False
    return True

def summarize_outcome(old_state: Dict[str, Any], new_state: Dict[str, Any], node_name: str) -> str:
    """Summarize the key outcome of a node execution for action memory."""
    
    # Check for data extraction
    old_extracted = old_state.get("extracted_data", {})
    new_extracted = new_state.get("extracted_data", {})
    if new_extracted != old_extracted:
        new_fields = [k for k in new_extracted if k not in old_extracted or old_extracted[k] != new_extracted[k]]
        if new_fields:
            return f"extracted {', '.join(new_fields)}: {', '.join([str(new_extracted[f]) for f in new_fields])}"
    
    # Check for message changes (chat responses)
    old_msgs = len(old_state.get("messages", []))
    new_msgs = len(new_state.get("messages", []))
    if new_msgs > old_msgs and node_name == "CHAT_NODE":
        return "provided conversational response"
    elif new_msgs > old_msgs and "tool" in node_name.lower():
        return f"tool executed: {new_state.get('tool_result', {}).get('type', 'unknown')}"
    
    # Check for user input collection
    if new_state.get("user_input") and not old_state.get("user_input"):
        return f"received user input: '{new_state.get('user_input', '')[:30]}...'"
    
    # Check for next action decisions
    if new_state.get("next_action") != old_state.get("next_action"):
        decision = new_state.get("decision_justification", "")
        return f"decided '{new_state.get('next_action')}' - {decision[:50]}..."
    
    # Default
    return "state updated"

# Utility functions moved to utilities.py

# Utility functions moved to utilities.py

def set_prompt_functions(
    base_system_prompt,
    tool_descriptions,
    input_extraction_fn,
    decision_fn,
    conversational_response_fn,
    tool_use_fn
):
    """Set prompt functions for the nodes."""
    global BASE_SYSTEM_PROMPT, TOOL_DESCRIPTIONS
    global build_input_extraction_prompt, build_decision_prompt
    global build_conversational_response_prompt, build_tool_use_prompt
    
    BASE_SYSTEM_PROMPT = base_system_prompt
    TOOL_DESCRIPTIONS = tool_descriptions
    build_input_extraction_prompt = input_extraction_fn
    build_decision_prompt = decision_fn
    build_conversational_response_prompt = conversational_response_fn
    build_tool_use_prompt = tool_use_fn
    # tool input prompt is optional
    try:
        from prompts import build_tool_input_prompt as _tip
        global build_tool_input_prompt
        build_tool_input_prompt = _tip
    except Exception:
        build_tool_input_prompt = None

def chat_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """
    Chat node that ONLY generates responses. No interrupting.
    """
    old_state = dict(state)
    messages = list(state.get("messages", []))
    extracted_data = state.get("extracted_data", {})
    user_input = state.get("user_input", "")
    decision_justification = state.get("decision_justification", "")
    required_fields = config.get("required_fields", [])
    # Use only 'agent_prompt' (no mixed keys)
    system_prompt = config.get("agent_prompt", "You are a helpful assistant.")

    # Get last tool context for grounding
    last_ctx = state.get("last_tool_context", {}) or {}

    # Build the prompt using the function from prompts.py
    chat_prompt = build_conversational_response_prompt(
        system_prompt=system_prompt,
        user_message=user_input,
        required_inputs=required_fields,
        conversation_history=messages,
        collected_data=extracted_data,
        decision_justification=decision_justification,
        tool_context=last_ctx
    )

    # Call the LLM to generate the response
    agent_message = llm.invoke(chat_prompt).content if hasattr(llm, "invoke") else llm(chat_prompt)

    # Add agent response to messages
    messages.append({"role": "assistant", "content": agent_message})

    new_state = {
        **state,  # Keep all existing state
        "messages": messages,
        "user_input": "",  # Clear input after processing
        "conversation_active": True
    }

    log_node_execution("CHAT_NODE", old_state, new_state, verbose)
    return new_state


def wait_user_input_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """
    Node that waits for user input using LangGraph's interrupt mechanism.
    This node calls interrupt() to pause execution and wait for human input.
    """
    old_state = dict(state)
    
    # If user_input already present in state (pre-filled by tests), skip interrupt
    existing_input = state.get("user_input", "")
    if existing_input and isinstance(existing_input, str) and existing_input.strip():
        user_input = existing_input
        messages = list(state.get("messages", []))
        # Ensure the current user_input is reflected in messages
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_input:
            messages.append({"role": "user", "content": user_input})
    else:
        # Use LangGraph's interrupt() to pause and wait for user input
        user_input = interrupt("Please provide your input:")
        # Add user message to conversation
        messages = list(state.get("messages", []))
        if user_input and user_input.strip():
            messages.append({"role": "user", "content": user_input})
    
    new_state = {
        **state,  # Keep all existing state
        "messages": messages,
        "user_input": user_input  # Keep the user input for processing
    }
    
    log_node_execution("WAIT_USER_INPUT", old_state, new_state, verbose)
    return new_state

def input_extractor_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Extract structured data from conversation history using JSON mode LLM."""
    old_state = dict(state)
    
    required_fields = config.get("required_fields", [])
    if not required_fields:
        new_state = {**state, "next_action": "decision_router"}
        log_node_execution("INPUT_EXTRACTOR", old_state, new_state, verbose)
        return new_state
    
    messages = state.get("messages", [])
    if not messages:
        new_state = {**state, "next_action": "decision_router"}
        log_node_execution("INPUT_EXTRACTOR", old_state, new_state, verbose)
        return new_state
    
    # Create extraction prompt
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    field_descriptions = "\n".join([f"- {field['name']}: {field.get('description', field.get('prompt', ''))}" for field in required_fields])
    
    prompt = f"""Extract the following information from the conversation:

Required fields:
{field_descriptions}

Conversation:
{conversation_text}

Return only a JSON object with the extracted values. If a field is not found, omit it from the JSON.
"""

    try:
        json_llm = create_json_mode_llm()
        response = json_llm.invoke(prompt)
        extracted_data = json.loads(response.content)
        # Ensure all extracted values are strings and strip whitespace
        extracted_data = {k: v.strip() if isinstance(v, str) else v for k, v in extracted_data.items()}
    except Exception:
        extracted_data = state.get("extracted_data", {})
    
    new_state = {
        **state,
        "extracted_data": extracted_data,
        "next_action": "decision_router"
    }
    
    log_node_execution("INPUT_EXTRACTOR", old_state, new_state, verbose)
    return new_state


"""
The previous tool_input_node has been removed from the graph. Keep a no-op placeholder for backward compatibility if referenced.
"""
def tool_input_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    old_state = dict(state)
    # Directly route to execution; assume prior nodes prepared inputs.
    chosen = state.get("chosen_tool") or state.get("next_action") or config.get("tool_type")
    new_state = {**state, "next_action": "tool_execution", "chosen_tool": chosen}
    log_node_execution("TOOL_INPUT_NOOP", old_state, new_state, verbose)
    return new_state


def query_planner_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Plan a retrieval query using per-tool input_schema when available.

    Produces a normalized query_spec and minimal extracted_data entries (e.g., query),
    then routes to tool_execution. If ambiguous, asks clarification and routes to chat.
    """
    old_state = dict(state)

    extracted_data = dict(state.get("extracted_data", {}))
    messages = state.get("messages", [])

    # Resolve selected tool and schema
    selected_tool, tool_cfg = get_selected_tool_config(config, state)
    tool_type = selected_tool or config.get("tool_type", "unknown")
    normalized_schema_fields = normalize_input_schema((tool_cfg or {}).get("input_schema"))

    # If we already have sufficient query fields, skip to execution
    try:
        if normalized_schema_fields and are_required_fields_complete(extracted_data, normalized_schema_fields):
            new_state = {**state, "next_action": "tool_execution", "chosen_tool": tool_type, "query_spec": {k: extracted_data.get(k) for k in [f.get("name") for f in normalized_schema_fields if f.get("name")]}}
            log_node_execution("QUERY_PLANNER", old_state, new_state, verbose)
            return new_state
    except Exception:
        pass

    # Build extraction prompt focused on retrieval query fields
    system_prompt = config.get("agent_prompt", "You are a helpful assistant.")
    user_msg = state.get("user_input", "")

    # Prepare a tool_input_spec-like structure for prompt building
    if normalized_schema_fields:
        tool_input_spec = [{"name": f.get("name"), "description": f.get("description") or ""} for f in normalized_schema_fields if f.get("name")]
    else:
        tool_input_spec = [{"name": "query", "prompt": "What should I search for?"}]

    if build_tool_input_prompt:
        try:
            prompt = build_tool_input_prompt(
                system_prompt=system_prompt,
                tool_type=tool_type,
                tool_input_spec=tool_input_spec,
                messages=messages,
                user_message=user_msg
            )
        except Exception:
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt = f"""SYSTEM PROMPT: {system_prompt}

Extract retrieval query fields from the conversation.

Conversation:
{conversation_text}

Return only a JSON object with the extracted values. If a field is not found, omit it.
"""
    else:
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt = f"""SYSTEM PROMPT: {system_prompt}

Extract retrieval query fields from the conversation.

Conversation:
{conversation_text}

Return only a JSON object with the extracted values. If a field is not found, omit it.
"""

    try:
        json_llm = create_json_mode_llm()
        response = json_llm.invoke(prompt)
        planned = json.loads(response.content)
        planned = {k: v.strip() if isinstance(v, str) else v for k, v in planned.items()}
    except Exception as e:
        planned = {}
        clarification = f"Query planning failed: {str(e)}"

    # Validate planned spec against schema fields if available
    from validation import validate_against_schema
    if not normalized_schema_fields:
        # No input_schema configured - this should not happen in a properly configured agent
        return {
            **state,
            "messages": messages + [{"role": "assistant", "content": f"Error: No input schema configured for tool '{tool_type}'. Please check the agent configuration."}],
            "next_action": "chat",
            "validation_errors": ["Missing input schema configuration"]
        }
    
    schema_fields = normalized_schema_fields
    validated, val_errors = validate_against_schema(planned, schema_fields)

    if val_errors or not validated:
        # Ask for clarification
        messages2 = list(messages)
        clarify_msg = ", ".join(val_errors) if val_errors else tool_input_spec[0].get("prompt", "Could you clarify your request?")
        messages2.append({"role": "assistant", "content": f"{clarify_msg}"})
        new_state = {**state, "messages": messages2, "next_action": "chat", "clarification_message": ", ".join(val_errors) if val_errors else ""}
        log_node_execution("QUERY_PLANNER", old_state, new_state, verbose)
        return new_state

    # Merge minimal fields into extracted_data and store query_spec
    merged = dict(extracted_data)
    for k, v in validated.items():
        if k not in merged or not merged.get(k):
            merged[k] = v

    new_state = {**state, "extracted_data": merged, "query_spec": validated, "next_action": "tool_execution", "chosen_tool": tool_type}
    log_node_execution("QUERY_PLANNER", old_state, new_state, verbose)
    return new_state


def structured_extractor_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Extract structured inputs for ALL tools (input or retrieval) using per-tool input_schema.

    Adapts the prompt based on tool type:
    - Input tools: Extract data to be sent to external services
    - Retrieval tools: Extract query parameters for information retrieval
    
    Routes to validate_inputs by default.
    """
    old_state = dict(state)

    extracted_data = dict(state.get("extracted_data", {}))
    messages = state.get("messages", [])

    selected_tool, tool_cfg = get_selected_tool_config(config, state)
    tool_type = selected_tool or config.get("tool_type", "unknown")
    tool_category = get_tool_category(config, tool_type)
    normalized_schema_fields = normalize_input_schema((tool_cfg or {}).get("input_schema"))

    # If schema-required fields look complete, proceed to validation
    try:
        if normalized_schema_fields and are_required_fields_complete(extracted_data, normalized_schema_fields):
            new_state = {**state, "next_action": "validate_inputs", "chosen_tool": tool_type, "tool_category": tool_category}
            log_node_execution("STRUCTURED_EXTRACTOR", old_state, new_state, verbose)
            return new_state
    except Exception:
        pass

    # Prepare prompt spec based on tool category
    if normalized_schema_fields:
        tool_input_spec = [{"name": f.get("name"), "description": f.get("description") or ""} for f in normalized_schema_fields if f.get("name")]
    else:
        if tool_category == "retrieval":
            tool_input_spec = [{"name": "query", "prompt": "What should I search for?"}]
        else:
            tool_input_spec = [{"name": "tool_input", "prompt": "Please provide the input for this operation."}]

    system_prompt = config.get("agent_prompt", "You are a helpful assistant.")
    user_msg = state.get("user_input", "")

    # Build appropriate prompt based on tool category
    if tool_category == "retrieval":
        # For retrieval tools, focus on query extraction
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt = f"""SYSTEM PROMPT: {system_prompt}

TOOL: {tool_type} - This is a retrieval tool that searches for information.

You are preparing input for the `{tool_type}` tool. Extract the requested tool input from the conversation and return only a JSON object containing those fields.

Tool input fields:
{chr(10).join([f"- {f.get('name')}: {f.get('description') or 'Field description'}" for f in tool_input_spec])}

Conversation:
{conversation_text}

USER MESSAGE: "{user_msg}"

Return only a JSON object with the extracted values for these fields. If a field is not found, omit it.
"""
    else:
        # For input tools, focus on data collection
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt = f"""SYSTEM PROMPT: {system_prompt}

TOOL: {tool_type} - This is an input tool that saves or sends data.

You are preparing input for the `{tool_type}` tool. Extract the requested tool input from the conversation and return only a JSON object containing those fields.

Tool input fields:
{chr(10).join([f"- {f.get('name')}: {f.get('description') or 'Field description'}" for f in tool_input_spec])}

Conversation:
{conversation_text}

USER MESSAGE: "{user_msg}"

Return only a JSON object with the extracted values for these fields. If a field is not found, omit it.
"""

    try:
        json_llm = create_json_mode_llm()
        response = json_llm.invoke(prompt)
        extracted = json.loads(response.content)
        extracted = {k: v.strip() if isinstance(v, str) else v for k, v in extracted.items()}
    except Exception as e:
        extracted = {}
        clarification = f"Extraction failed: {str(e)}"

    # Merge with existing extracted data
    merged = dict(extracted_data)
    for k, v in (extracted or {}).items():
        if k not in merged or not merged.get(k):
            merged[k] = v

    # Ask for clarification if nothing was extracted and no messages yet
    if not extracted and not messages:
        messages2 = list(messages)
        if tool_category == "retrieval":
            messages2.append({"role": "assistant", "content": "What would you like me to search for?"})
        else:
            messages2.append({"role": "assistant", "content": tool_input_spec[0].get("prompt", "Could you share the necessary details?")})
        new_state = {**state, "messages": messages2, "next_action": "chat"}
        log_node_execution("STRUCTURED_EXTRACTOR", old_state, new_state, verbose)
        return new_state

    new_state = {**state, "extracted_data": merged, "next_action": "validate_inputs", "chosen_tool": tool_type, "tool_category": tool_category}
    log_node_execution("STRUCTURED_EXTRACTOR", old_state, new_state, verbose)
    return new_state


def validate_inputs_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Validate extracted_data against per-tool input_schema.

    On errors, append a human-friendly message and route to chat; else proceed to tool_execution.
    """
    old_state = dict(state)

    extracted_data = dict(state.get("extracted_data", {}))
    messages = state.get("messages", [])

    selected_tool, tool_cfg = get_selected_tool_config(config, state)
    tool_type = selected_tool or config.get("tool_type", "unknown")
    normalized_schema_fields = normalize_input_schema((tool_cfg or {}).get("input_schema"))

    from validation import validate_against_schema
    schema_fields = normalized_schema_fields
    validated, val_errors = validate_against_schema(extracted_data, schema_fields)

    if val_errors:
        # Compose a brief assistant message listing the issues
        msg = "; ".join(val_errors)
        messages2 = list(messages)
        messages2.append({"role": "assistant", "content": f"I need a bit more information or corrections: {msg}"})
        new_state = {**state, "messages": messages2, "validation_errors": val_errors, "next_action": "chat"}
        log_node_execution("VALIDATE_INPUTS", old_state, new_state, verbose)
        return new_state

    # Use validated data (normalized) and proceed
    merged = dict(extracted_data)
    merged.update(validated)
    new_state = {**state, "extracted_data": merged, "validation_errors": [], "next_action": "tool_execution", "chosen_tool": tool_type}
    log_node_execution("VALIDATE_INPUTS", old_state, new_state, verbose)
    return new_state


def exit_evaluator_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Evaluate configured exit conditions and decide whether to end the conversation.

    Supported condition types:
    - prompt: expression is string regex or array of phrases; matches user messages (case-insensitive)
    - logical: "is_data_complete" or "all_inputs_collected" based on tool schema required fields
    - tool_event: {tool: name, status: "success"|"error"}
    - max_turns: integer threshold on count of user messages
    """
    old_state = dict(state)

    conditions = config.get("exit_conditions", []) or []
    mode = (config.get("exit_condition_mode") or "or").lower()

    # Use utility functions from utilities.py

    results = [bool(evaluate(c, state, state.get("messages", []))) for c in conditions]
    met = any(results) if mode == "or" else all(results) if results else False

    if met:
        new_state = {**state, "conversation_active": False, "next_action": "end"}
    else:
        new_state = {**state, "conversation_active": True}

    log_node_execution("EXIT_EVALUATOR", old_state, new_state, verbose)
    return new_state





def tool_answer_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Tool answer node: produce a concise user-facing answer for ALL tools and append to messages.

    Handles both input tools (data saving) and retrieval tools (information search).
    Always routes to wait_user_input after generating the response.
    """
    old_state = dict(state)

    messages = list(state.get("messages", []))
    tool_result = state.get("tool_result", {}) or {}
    query_spec = state.get("query_spec", {}) or {}
    extracted_data = state.get("extracted_data", {}) or {}
    system_prompt = config.get("agent_prompt", "You are a helpful assistant.")
    user_msg = state.get("user_input", "")
    tool_category = state.get("tool_category", "input")
    chosen_tool = state.get("chosen_tool", "unknown")

    # Note: tool_result formatting is now handled in the prompt function

    # Create a simple summary for context based on tool category
    last_tool_summary = ""
    if isinstance(tool_result, dict):
        if tool_result.get("success") is True:
            if tool_category == "retrieval":
                last_tool_summary = f"Search completed successfully: {tool_result.get('message', 'Information found')}"
            else:
                last_tool_summary = f"Data saved successfully: {tool_result.get('message', 'Operation completed')}"
        else:
            last_tool_summary = f"Tool execution failed: {tool_result.get('error', 'Unknown error')}"

    # Produce an LLM-based informative answer (assistant message)
    try:
        # Build the prompt using the function from prompts.py
        answer_prompt = build_tool_answer_prompt(
            system_prompt=system_prompt,
            tool_type=chosen_tool,
            tool_result=tool_result,
            user_query=user_msg,
            extracted_data=extracted_data,
            tool_category=tool_category,
            tool_outcome="success" if tool_result.get("success") else "failure",
            tool_summary=last_tool_summary,
            decision_context=state.get("decision_context", {})  # NEW: Pass decision context
        )

        llm_answer = llm.invoke(answer_prompt).content if hasattr(llm, "invoke") else llm(answer_prompt)
        llm_answer = (llm_answer or "").strip()
    except Exception:
        llm_answer = last_tool_summary

    # Create minimal tool context for downstream nodes
    tool_ctx = {
        "tool": chosen_tool,
        "category": tool_category,
        "status": "success" if tool_result.get("success") else "failure",
        "summary": last_tool_summary,
        "data": extracted_data
    }

    messages.append({"role": "assistant", "content": llm_answer})

    new_state = {
        **state,
        "messages": messages,
        "user_input": "",
        "last_tool_summary": last_tool_summary,
        "last_tool_context": tool_ctx,
        "next_action": "wait_user_input"  # Always route to wait_user_input
    }

    log_node_execution("TOOL_ANSWER", old_state, new_state, verbose)
    return new_state

def get_tool_requirements(config: dict, tool_name: str) -> List[str]:
    """Get required fields for a specific tool."""
    if not tool_name:
        return []
    
    try:
        for tool in config.get("tools", []):
            if tool.get("name") == tool_name:
                schema = tool.get("input_schema", {})
                if isinstance(schema, dict) and schema.get("properties"):
                    return schema.get("required", [])
                elif isinstance(schema, list):
                    return [f.get("name") for f in schema if f.get("required")]
        return []
    except Exception:
        return []

# Utility function moved to utilities.py

def decision_router_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Route to next action using LLM and build_decision_prompt (multi-tool aware)."""
    old_state = dict(state)
    extracted_data = state.get("extracted_data", {})
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")
    system_prompt = config.get("agent_prompt", "You are a helpful assistant.")
    # Build enhanced config with action history for prompt
    enhanced_config = dict(config)
    # Reduce tools into a minimal prompt-friendly catalog
    try:
        tools_min = []
        for t in (config.get("tools") or []):
            if not t.get("enabled", True) or not t.get("name"):
                continue
            tools_min.append({
                "name": t.get("name"),
                "type": t.get("type"),
                "impl": t.get("impl"),
                "description": t.get("description", "")
            })
        enhanced_config["tools"] = tools_min
    except Exception:
        pass
    enhanced_config["action_history"] = state.get("action_history", [])
    # Derive required input names (if any) from input tool schemas for prompt display
    required_input_names = []
    try:
        for t in (config.get("tools") or []):
            if (t.get("type") or "").lower() != "input":
                continue
            fields = normalize_input_schema((t.get("input_schema") or {}))
            for f in fields:
                if f.get("required") and f.get("name"):
                    required_input_names.append(f.get("name"))
        # de-duplicate while preserving order
        seen = set()
        required_input_names = [n for n in required_input_names if not (n in seen or seen.add(n))]
    except Exception:
        required_input_names = []
    enhanced_config["required_input_names"] = required_input_names
    # Include runtime query_spec for prompt rendering (TOOL RESULT block)
    enhanced_config["query_spec"] = state.get("query_spec", {}) or {}
    # Also include the most recent tool result and the synthesized summary/context so the decision
    # router can consider tool outcomes when deciding next steps
    enhanced_config["tool_result"] = state.get("tool_result", {})
    enhanced_config["last_tool_summary"] = state.get("last_tool_summary", "")
    enhanced_config["last_tool_context"] = state.get("last_tool_context", {})
    
    # Build the configuration-driven decision prompt with action memory
    decision_prompt = build_decision_prompt(
        system_prompt=system_prompt,
        config=enhanced_config,
        user_message=user_input,
        messages=messages,
        extracted_data=extracted_data
    )
    # Call the LLM to get the decision
    llm_response = llm.invoke(decision_prompt).content if hasattr(llm, "invoke") else llm(decision_prompt)

    # No verbose debug prints here; rely on concise per-node logging

    # Parse the LLM response for DECISION: [action] - [reason]
    import re
    match = re.search(r"DECISION:\s*([\w\-_]+)\s*-\s*(.*)", llm_response, re.IGNORECASE)
    if match:
        next_action = match.group(1).strip()
        justification = match.group(2).strip()
    else:
        # Fallback: default to chat if parsing fails
        next_action = "chat"
        justification = llm_response.strip()

    # Normalize decision: if it matches a configured tool, set chosen_tool & category
    chosen_tool = None
    tool_category = None
    for t in (config.get("tools") or []):
        name = t.get("name")
        if name and name.lower() == next_action.lower():
            chosen_tool = name
            tool_category = t.get("type")
            break

    # Loop guard: if we are repeating the same tool decision too many times, defer to chat
    action_history = state.get("action_history", []) or []
    recent = action_history[-3:] if len(action_history) >= 3 else action_history
    repeated_tool = False
    if chosen_tool and recent:
        decisions = [a.get("outcome", "") for a in recent if a.get("node") == "DECISION_ROUTER"]
        # naive check: the decision outcome contains the tool name
        if len(decisions) >= 2 and all(chosen_tool in (d or "") for d in decisions):
            repeated_tool = True
    if repeated_tool:
        chosen_tool = None
        tool_category = None
        next_action = "chat"

    # Build decision context for downstream nodes
    decision_context = {
        "user_intent": user_input,
        "available_data": extracted_data,
        "tool_requirements": get_tool_requirements(config, chosen_tool) if chosen_tool else [],
        "reasoning": justification,
        "alternatives_considered": get_alternative_actions(config, extracted_data)
    }
    
    new_state = {
        **state,
        "next_action": chosen_tool or next_action,
        "decision_justification": justification,
        "chosen_tool": chosen_tool or state.get("chosen_tool"),
        "tool_category": tool_category or state.get("tool_category"),
        "decision_context": decision_context  # NEW: Capture decision context
    }

    log_node_execution("DECISION_ROUTER", old_state, new_state, verbose)
    return new_state

def tool_execution_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Execute the configured tool (e.g., save to Google Sheets)."""
    old_state = dict(state)
    try:
        extracted_data = state.get("extracted_data", {})
        # Ensure all values are strings to avoid .strip errors in tools
        extracted_data = {k: str(v) if not isinstance(v, str) else v for k, v in extracted_data.items()}
        # Determine the tool to execute: prefer preserved chosen_tool. If the next_action
        # is the control token 'tool_execution', fall back to the configured tool name
        # (config['tool_type']) rather than using the control token itself.
        # Resolve tool name to execute
        chosen = state.get("chosen_tool")
        next_action = state.get("next_action")
        tool_type = None
        if chosen:
            tool_type = chosen
        # If next_action is a tool name (not the control token), use it
        if not tool_type and next_action and next_action != "tool_execution":
            tool_type = next_action
        # Fallback: try to infer from query_spec (prefer retrieval tools)
        if not tool_type:
            qspec = state.get("query_spec") or {}
            if qspec:
                for t in config.get("tools", []) or []:
                    if t.get("type") == "retrieval":
                        tool_type = t.get("name")
                        break
        # Final fallback to legacy config.tool_type or generic
        if not tool_type:
            tool_type = config.get("tool_type") or "generic"
        # If chosen tool not set, try to infer from query_spec or tools cfg
        if not chosen:
            # prefer explicit query_spec hint
            qspec = state.get("query_spec") or {}
            if qspec:
                # try to find a retrieval tool in config.tools
                for t in config.get("tools", []) or []:
                    if t.get("type") == "retrieval":
                        tool_type = t.get("name")
                        break
        # Debug: show resolved tool_type and runtime config
        try:
            runtime_cfg = build_runtime_tool_config(config, state)
        except Exception as e:
            runtime_cfg = {}



        # If the resolved tool_type is not a registered implementation, try to infer from runtime_cfg or config
        if tool_type not in AVAILABLE_TOOLS:
            # First, check if this is a tool name that maps to a different implementation
            selected_tool, tool_cfg = get_selected_tool_config(config, state)
            if tool_cfg and tool_cfg.get("impl") in AVAILABLE_TOOLS:
                # Use the implementation instead of the tool name
                tool_type = tool_cfg.get("impl")
            else:
                # prefer a key in runtime_cfg that matches a known tool
                inferred = None
                try:
                    for k in runtime_cfg.keys():
                        if k in AVAILABLE_TOOLS:
                            inferred = k
                            break
                except Exception:
                    inferred = None
                # fallback: check config.tools list
                if not inferred:
                    try:
                        for t in config.get("tools", []) or []:
                            if t.get("type") == "retrieval" or t.get("type") == "input":
                                name = t.get("name")
                                impl = t.get("impl")
                                # Prefer the implementation if it's available
                                if impl and impl in AVAILABLE_TOOLS:
                                    inferred = impl
                                    break
                                elif name in AVAILABLE_TOOLS:
                                    inferred = name
                                    break
                    except Exception:
                        inferred = None
                if inferred:
                    tool_type = inferred

        tool_result = execute_tool(tool_type, extracted_data, runtime_cfg, verbose)
        import json
        messages = list(state.get("messages", []))
        # Do NOT append a visible assistant message for tool results; keep UI clean and grounded

        # Log concise tool summary if available
        try:
            summary = tool_result.get("summary") if isinstance(tool_result, dict) else None
            if summary:
                session = state.get("session_id", "no-session")
                s = str(summary).replace("\n", " ")
                if len(s) > 300:
                    s = s[:297] + "..."
                print(f"TOOL_SUMMARY | session={session} | summary={s}")
        except Exception:
            pass

        new_state = {
            **state,
            "tool_result": tool_result,
            "messages": messages
        }
        # Debug: show the tool_result that will be stored in state
    except Exception as e:
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": f"Tool error: {str(e)}"})
        new_state = {
            **state,
            "tool_result": {"error": str(e)},
            "messages": messages
        }
    
    log_node_execution("TOOL_EXECUTION", old_state, new_state, verbose)
    return new_state


def final_response_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """Generate final response to user."""
    old_state = dict(state)
    
    tool_result = state.get("tool_result", {})
    messages = list(state.get("messages", []))
    
    if tool_result.get("error"):
        final_message = f"I encountered an error: {tool_result['error']}. Please try again."
    else:
        final_message = "Perfect! I've successfully saved your project preferences. Is there anything else you'd like to update?"
    
    messages.append({"role": "assistant", "content": final_message})
    
    new_state = {
        **state,
        "messages": messages,
        "conversation_active": False,
        "next_action": "end"
    }
    
    log_node_execution("FINAL_RESPONSE", old_state, new_state, verbose)
    return new_state


def end_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """End the conversation."""
    old_state = dict(state)
    
    new_state = {
        **state,
        "conversation_active": False,
        "next_action": "end"
    }
    
    log_node_execution("END", old_state, new_state, verbose)
    return new_state


def input_translation_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """
    Translate user input from source language to target language (English) for processing.
    Only executes if translation is enabled in config.
    """
    old_state = state.copy()
    
    # Check if translation is enabled
    translation_config = config.get("translation", {})
    translation_enabled = translation_config.get("enabled", False)
    
    if not translation_enabled:
        if verbose:
            print("Translation disabled, skipping input translation")
        # Set translation fields to indicate no translation
        new_state = {
            **old_state,
            "translation_enabled": False,
            "original_language": "English",
            "translated_input": old_state.get("user_input", ""),
            "target_language": "English"
        }
        log_node_execution("INPUT_TRANSLATION", old_state, new_state, verbose)
        return new_state
    
    # Get translation settings
    source_language = translation_config.get("source_language", "Hungarian")
    target_language = translation_config.get("target_language", "English")
    user_input = old_state.get("user_input", "")
    
    if verbose:
        print(f"Translating input from {source_language} to {target_language}")
    
    # Create translation prompt using the structured template
    translation_prompt = build_input_translation_prompt(
        system_prompt=BASE_SYSTEM_PROMPT or "You are a helpful translation assistant.",
        user_message=user_input,
        source_language=source_language,
        target_language=target_language
    )
    
    try:
        # Perform translation
        translation_result = llm.invoke(translation_prompt).content.strip()
        
        if verbose:
            print(f"Translation result: '{translation_result}'")
        
        # Update state with translation results
        new_state = {
            **old_state,
            "translation_enabled": True,
            "original_language": source_language,
            "translated_input": translation_result,
            "target_language": target_language,
            "user_input": translation_result  # Update user_input with translated version for processing
        }
        
        log_node_execution("INPUT_TRANSLATION", old_state, new_state, verbose)
        return new_state
        
    except Exception as e:
        if verbose:
            print(f"Translation failed: {e}")
        
        # Fallback: use original input without translation
        new_state = {
            **old_state,
            "translation_enabled": False,
            "original_language": source_language,
            "translated_input": user_input,
            "target_language": target_language,
            "error_message": f"Translation failed: {e}"
        }
        
        log_node_execution("INPUT_TRANSLATION", old_state, new_state, verbose)
        return new_state


def output_translation_node(state: ConversationState, config: dict, llm, verbose: bool = True) -> Dict[str, Any]:
    """
    Translate agent response from target language (English) back to response language.
    Only executes if translation is enabled in config.
    """
    old_state = state.copy()
    
    # Check if translation is enabled
    translation_config = config.get("translation", {})
    translation_enabled = translation_config.get("enabled", False)
    
    if not translation_enabled:
        if verbose:
            print("Translation disabled, skipping output translation")
        log_node_execution("OUTPUT_TRANSLATION", old_state, old_state, verbose)
        return old_state
    
    # Get translation settings
    target_language = translation_config.get("target_language", "English")
    response_language = translation_config.get("response_language", "Hungarian")
    
    # Get the last message from the agent (should be the response to translate)
    messages = old_state.get("messages", [])
    if not messages:
        if verbose:
            print("No messages to translate")
        log_node_execution("OUTPUT_TRANSLATION", old_state, old_state, verbose)
        return old_state
    
    # Get the last agent message
    last_message = messages[-1]
    if last_message.get("role") != "assistant":
        if verbose:
            print("Last message is not from assistant, skipping translation")
        log_node_execution("OUTPUT_TRANSLATION", old_state, old_state, verbose)
        return old_state
    
    agent_response = last_message.get("content", "")
    
    if verbose:
        print(f"Translating response from {target_language} to {response_language}")
    
    # Create translation prompt using the structured template
    translation_prompt = build_output_translation_prompt(
        system_prompt=BASE_SYSTEM_PROMPT or "You are a helpful translation assistant.",
        assistant_message=agent_response,
        source_language=target_language,
        target_language=response_language
    )
    
    try:
        # Perform translation
        translation_result = llm.invoke(translation_prompt).content.strip()
        
        if verbose:
            print(f"Response translation result: '{translation_result}'")
        
        # Update the last message with translated content
        updated_messages = messages.copy()
        updated_messages[-1] = {
            **last_message,
            "content": translation_result
        }
        
        new_state = {
            **old_state,
            "messages": updated_messages
        }
        
        log_node_execution("OUTPUT_TRANSLATION", old_state, new_state, verbose)
        return new_state
        
    except Exception as e:
        if verbose:
            print(f"Response translation failed: {e}")
        
        # Fallback: keep original response
        new_state = {
            **old_state,
            "error_message": f"Response translation failed: {e}"
        }
        
        log_node_execution("OUTPUT_TRANSLATION", old_state, new_state, verbose)
        return new_state