"""Utility functions for the LangGraph agent system."""

from typing import Dict, Any, List
import json

# Tool Configuration Utilities

def get_selected_tool_config(config: Dict[str, Any], state: Dict[str, Any]):
    """Resolve the active tool name and its tool config entry from config.tools."""
    # Prefer an explicitly preserved chosen tool, then next_action, then legacy config key
    preferred = state.get("chosen_tool") or state.get("next_action") or config.get("tool_type")
    tools = config.get("tools") or []
    if preferred and isinstance(tools, list):
        for tool in tools:
            if tool.get("name") == preferred and tool.get("enabled", True):
                return preferred, tool
    # Fallback: if no match, return first enabled tool if available
    for tool in tools:
        if tool.get("enabled", True):
            return tool.get("name"), tool
    # Final fallback to legacy
    return preferred, None

def get_tool_category(config: Dict[str, Any], tool_name: str) -> str:
    """Get the category (input/retrieval) for a tool."""
    tools = config.get("tools") or []
    for t in tools:
        if t.get("name") == tool_name:
            return t.get("type") or "input"
    return "input"

def build_runtime_tool_config(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a runtime config object for tool implementations using legacy keys expected by tools.py.

    - sheets_tool expects: config['sheet'] with spreadsheet_title/worksheet_name and optional top-level 'credentials_path'
      and uses validate functions reading config['required_fields'] → we adapt from input_schema.required.
    - web_search_tool expects: config['web_search'] dict with search options.
    """
    runtime_cfg: Dict[str, Any] = {}
    tool_name, tool_cfg = get_selected_tool_config(config, state)
    tool_cfg = tool_cfg or {}
    # carry over agent-level basics
    if config.get("credentials_path"):
        runtime_cfg["credentials_path"] = config.get("credentials_path")
    # map per-tool config
    per_tool_conf = tool_cfg.get("config", {}) or {}
    if tool_cfg.get("impl") == "sheets":
        runtime_cfg["sheet"] = {
            "spreadsheet_title": per_tool_conf.get("spreadsheet_title"),
            "worksheet_name": per_tool_conf.get("worksheet_name")
        }
        # adapt required_fields from input_schema
        fields = normalize_input_schema(tool_cfg.get("input_schema"))
        # convert to legacy shape: list of {name, required, format}
        legacy_required = []
        for f in fields:
            legacy_required.append({
                "name": f.get("name"),
                "required": bool(f.get("required")),
                "validate": "email" if (f.get("format") == "email") else None
            })
        runtime_cfg["required_fields"] = legacy_required
    elif tool_name == "web_search":
        runtime_cfg["web_search"] = per_tool_conf
    elif tool_cfg.get("impl") == "api_retrieval":
        # api_retrieval tool expects config under "api_retrieval" key
        runtime_cfg["api_retrieval"] = per_tool_conf
        # Also add the tool name for context
        runtime_cfg["tool_name"] = tool_name
    else:
        # pass through generic config under its name, and copy required_fields when available
        runtime_cfg[tool_name] = per_tool_conf
        fields = normalize_input_schema(tool_cfg.get("input_schema"))
        legacy_required = [{"name": f.get("name"), "required": bool(f.get("required"))} for f in fields]
        runtime_cfg["required_fields"] = legacy_required
    return runtime_cfg

def normalize_input_schema(input_schema: Any) -> list:
    """Normalize a per-tool input schema to a simple field list for extraction/validation.

    Accepts either a JSON Schema-like object {type: 'object', properties: {...}, required: [...]} or
    a list of field descriptors [{name, type, required, format}], returning a list with at least name/type/required.
    """
    if not input_schema:
        return []
    
    # Handle list format: [{name, type, required, format}]
    if isinstance(input_schema, list):
        return input_schema
    
    # Handle JSON Schema format: {type: 'object', properties: {...}, required: [...]}
    if isinstance(input_schema, dict):
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        if properties:
            fields = []
            for field_name, field_spec in properties.items():
                if isinstance(field_spec, dict):
                    field_info = {
                        "name": field_name,
                        "type": field_spec.get("type", "string"),
                        "required": field_name in required,
                        "format": field_spec.get("format"),
                        "description": field_spec.get("description", "")
                    }
                    fields.append(field_info)
            return fields
    
    return []

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

def get_alternative_actions(config: dict, extracted_data: dict) -> List[str]:
    """Get alternative actions that could have been considered."""
    alternatives = ["chat", "end"]
    
    # Add available tools as alternatives
    try:
        for tool in config.get("tools", []):
            if tool.get("enabled", True):
                alternatives.append(tool.get("name", ""))
    except Exception:
        pass
    
    return [alt for alt in alternatives if alt]

# State Management Utilities

def are_required_fields_complete(data: Dict[str, Any], fields: list) -> bool:
    """Check if all required fields are present in the data."""
    if not fields:
        return True
    
    for field in fields:
        field_name = field.get("name") if isinstance(field, dict) else field
        if field_name and field_name not in data:
            return False
        # Check if the field has a value (not empty string, None, etc.)
        if field_name in data and not data[field_name]:
            return False
    
    return True

def summarize_outcome(old_state: Dict[str, Any], new_state: Dict[str, Any], node_name: str) -> str:
    """Summarize the outcome of a node execution."""
    try:
        if node_name == "STRUCTURED_EXTRACTOR":
            old_data = old_state.get("extracted_data", {})
            new_data = new_state.get("extracted_data", {})
            if new_data and len(new_data) > len(old_data):
                return f"Extracted {len(new_data)} field(s): {', '.join(new_data.keys())}"
            return "No new data extracted"
        
        elif node_name == "VALIDATE_INPUTS":
            errors = new_state.get("validation_errors", [])
            if errors:
                return f"Validation failed: {', '.join(errors)}"
            return "Validation passed"
        
        elif node_name == "TOOL_EXECUTION":
            result = new_state.get("tool_result", {})
            if result.get("success"):
                return f"Tool executed successfully: {result.get('message', 'Operation completed')}"
            else:
                return f"Tool execution failed: {result.get('error', 'Unknown error')}"
        
        elif node_name == "CHAT_NODE":
            messages = new_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant":
                    content = last_msg.get("content", "")
                    return f"Generated response: {content[:50]}{'...' if len(content) > 50 else ''}"
            return "No response generated"
        
        else:
            return f"Node {node_name} completed"
    
    except Exception:
        return f"Node {node_name} completed (summary failed)"

def track_action(node_name: str, old_state: Dict[str, Any], new_state: Dict[str, Any]):
    """Track action for decision memory (always, regardless of verbose setting)."""
    # Get existing action history
    action_history = list(old_state.get("action_history", []))
    
    # Create action record
    action_record = {
        "step": len(action_history) + 1,
        "node": node_name,
        "transition": f"{list(old_state.keys())[0] if old_state else 'start'} → {node_name}",
        "outcome": summarize_outcome(old_state, new_state, node_name),
        "timestamp": len(new_state.get("messages", []))  # Use message count as simple timestamp
    }
    
    # Add to history (keep only last 5 actions to prevent prompt bloat)
    action_history.append(action_record)
    if len(action_history) > 5:
        action_history = action_history[-5:]
    
    # Update state with action history
    new_state["action_history"] = action_history

def log_node_execution(node_name: str, old_state: Dict[str, Any], new_state: Dict[str, Any], verbose: bool = True):
    """Log node execution with state changes and track actions for decision memory."""
    # Track action for decision memory (always, regardless of verbose setting)
    track_action(node_name, old_state, new_state)
    
    # If verbose is disabled, do not print anything (action_history still tracked)
    if not verbose:
        return
    # Node-specific concise logging per user instruction
    import json

    def _truncate(val, limit=200):
        try:
            if isinstance(val, (dict, list)):
                s = json.dumps(val, ensure_ascii=False)
            else:
                s = str(val)
        except Exception:
            s = str(val)
        if len(s) > limit:
            return s[:limit-3] + "..."
        return s

    node_label = node_name.upper()
    session = new_state.get("session_id", "no-session")

    if node_label == "DECISION_ROUTER":
        justification = new_state.get("decision_justification", "")
        print(f"NODE:DECISION_ROUTER | session={session} | decision_justification={_truncate(justification,300)}")
        return

    if node_label in ("RESULT_SYNTH_INPUT", "RESULT_SYNTH_RETRIEVAL"):
        summary = new_state.get("last_tool_summary", "")
        context = new_state.get("last_tool_context", {})
        top_answer = ""
        try:
            if isinstance(context, dict):
                top_answer = context.get("top_answer", "")
        except Exception:
            top_answer = ""
        print(f"NODE:{node_label} | session={session} | tool_summary={_truncate(summary)} | top_answer={_truncate(top_answer)} | tool_context={_truncate(context)}")
        return

    if node_label == "CHAT_NODE":
        msgs = new_state.get("messages", []) or []
        assistant_msg = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                assistant_msg = m.get("content", "")
                break
        print(f"NODE:{node_label} | session={session} | assistant_response={_truncate(assistant_msg)}")
        return

    if node_label == "TOOL_EXECUTION":
        tr = new_state.get("tool_result", {}) or {}
        success = tr.get("success")
        summary = tr.get("summary") or tr.get("message") or ""
        print(f"NODE:{node_label} | session={session} | success={success} | summary={_truncate(summary)}")
        return

    if node_label == "QUERY_PLANNER":
        qs = new_state.get("query_spec", {}) or {}
        print(f"NODE:{node_label} | session={session} | query_spec={_truncate(qs)}")
        return

    # Fallback: minimal change log for other nodes
    changed_keys = []
    for key, new_val in new_state.items():
        if key == "action_history":
            continue
        old_val = old_state.get(key, None)
        if old_val != new_val:
            changed_keys.append(key)
    if changed_keys:
        print(f"NODE:{node_label} | session={session} | changed_keys={','.join(changed_keys)}")
    else:
        print(f"NODE:{node_label} | session={session} | (no change)")

# Exit Condition Utilities

def count_user_messages(msgs: list) -> int:
    """Count user messages in the conversation."""
    return len([m for m in msgs if m.get("role") == "user"])

def prompt_matches(expr, msgs: list) -> bool:
    """Check if any user message matches the exit expression."""
    if not expr or not msgs:
        return False
    
    # Get the last user message
    user_messages = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    if not user_messages:
        return False
    
    last_user_message = user_messages[-1].lower()
    
    # Handle regex patterns
    if expr.startswith("/") and expr.endswith("/"):
        try:
            import re
            pattern = expr[1:-1]
            return bool(re.search(pattern, last_user_message, re.IGNORECASE))
        except Exception:
            return False
    
    # Handle pipe-separated phrases
    phrases = [p.strip().lower() for p in expr.split("|")]
    return any(phrase in last_user_message for phrase in phrases)

def logical_matches(expr, st: Dict[str, Any]) -> bool:
    """Check if logical exit condition matches."""
    if not expr:
        return False
    
    # Handle named predicates
    if expr == "is_data_complete":
        extracted_data = st.get("extracted_data", {})
        # For logical conditions, we need to get the tool requirements from the config
        # This is a simplified version - in practice, you might need to pass config here
        return len(extracted_data) > 0  # Simplified check
    
    return False

def tool_event_matches(expr, st: Dict[str, Any]) -> bool:
    """Check if tool event exit condition matches."""
    if not expr or not isinstance(expr, dict):
        return False
    
    tool_result = st.get("tool_result", {})
    if not tool_result:
        return False
    
    # Check tool name match
    expected_tool = expr.get("tool")
    if expected_tool and tool_result.get("type") != expected_tool:
        return False
    
    # Check status match
    expected_status = expr.get("status")
    if expected_status and tool_result.get("success") != (expected_status == "success"):
        return False
    
    return True

def max_turns_matches(expr, msgs: list) -> bool:
    """Check if max turns exit condition matches."""
    if not isinstance(expr, (int, str)):
        return False
    
    try:
        max_turns = int(expr)
        return len(msgs) >= max_turns
    except (ValueError, TypeError):
        return False

def evaluate(cond: Dict[str, Any], state: Dict[str, Any], messages: list) -> bool:
    """Evaluate an exit condition."""
    if not cond or "type" not in cond:
        return False
    
    cond_type = cond.get("type")
    expr = cond.get("expression")
    
    if cond_type == "prompt":
        return prompt_matches(expr, messages)
    
    elif cond_type == "logical":
        return logical_matches(expr, state)
    
    elif cond_type == "tool_event":
        return tool_event_matches(expr, state)
    
    elif cond_type == "max_turns":
        return max_turns_matches(expr, messages)
    
    return False
