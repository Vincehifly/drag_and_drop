# CONTEXT for GitHub Copilot:
# - Write minimal, clean code without excessive prints or debug output
# - Focus on core functionality, avoid verbose instructions
# - Use modern Python patterns and best practices
# - For LangGraph: use interrupt()/Command(resume=...) pattern for human-in-the-loop
# - Prefer concise, working solutions over detailed explanations in code

from typing import Dict, List, Any
import json

# --- Base System Prompt --- #

BASE_SYSTEM_PROMPT = """You are a helpful, professional assistant that collects structured information and uses authorized tools when needed.

Core rules:
1. Safety & privacy: Never expose credentials, API keys, or raw PII in prompts or logs. Mask PII when summarizing (e.g., j***@example.com).
2. Tool usage: Use tools only when required to fulfill user intent or when explicitly instructed. Prefer conversational clarification before calling write tools.
3. JSON output policy: When asked to extract structured fields, return JSON ONLY (no surrounding text). Omit fields you cannot confidently extract.

Tone: professional, concise, and helpful. Always follow per-agent `agent_prompt` for domain-specific behavior.
"""

# --- Tool Descriptions --- #

TOOL_DESCRIPTIONS = {
    "web_search": "searches the internet for information, answers questions, finds current data",
    "sheets": "saves information to a database, stores contact details, records user data",
    "email": "sends emails, notifications, or messages to specified recipients"
}

# --- Essential Template Builders --- #

def build_input_extraction_prompt(
    system_prompt: str,
    tool_type: str,
    required_fields: List[Dict],
    user_message: str,
    collected: Dict
) -> str:
    """Build comprehensive input extraction prompt with all injection types."""
    
    field_descriptions = []
    for field in required_fields:
        field_key = field.get('key', field.get('name', ''))
        field_descriptions.append(f"- {field_key}: {field.get('prompt', field.get('description', 'No description'))}")
    
    return f"""
SYSTEM PROMPT INJECTION: {system_prompt}

TOOL PROMPT INJECTION: You are working with a {tool_type} tool that will use the collected information.

REQUIRED INPUT PROMPT INJECTION: You must extract the following required fields from user messages:
{chr(10).join(field_descriptions)}

TASK: Extract field values from the user's message and return them as JSON.

USER MESSAGE: "{user_message}"

CURRENTLY COLLECTED: {collected}

INSTRUCTIONS:
1. Only extract values that are clearly mentioned in the user's message
2. Don't extract values that are already collected unless the user is updating them
3. Don't make assumptions or fill in missing information
4. Return ONLY the new/updated field values as valid JSON
5. If no extractable values found, return empty JSON object {{}}

EXAMPLES:
User: "Hi, my name is John Smith and my email is john@example.com"
Output: {{"name": "John Smith", "email": "john@example.com"}}

User: "I work at Microsoft"
Output: {{"company": "Microsoft"}}

User: "Actually, my email is john.smith@gmail.com"
Output: {{"email": "john.smith@gmail.com"}}

User: "How are you doing today?"
Output: {{}}

Now extract from the user message above:
"""

def build_decision_prompt(
    system_prompt: str,
    config: dict,
    user_message: str,
    messages: list = None,
    extracted_data: dict = None
) -> str:
    """
    Build a configuration-driven decision router prompt that lists configured tools (if any).
    """
    # Prefer explicit tools list (new schema); fall back to legacy single-tool fields
    tools_cfg = config.get("tools") or []
    tool_display = ""
    configured_tool = None
    required_input_context = ""

    if tools_cfg and isinstance(tools_cfg, list):
        lines = []
        for tool in tools_cfg:
            if not tool.get("enabled", True):
                continue
            tool_name = tool.get("name", "unnamed")
            tool_type = tool.get("type", "unknown")
            tool_desc = tool.get("description", "")
            lines.append(f"- {tool_name}: {tool_desc}")
        
        if lines:
            tool_display = "CONFIGURED TOOLS:\n" + "\n".join(lines)
            
            # Build required input context for input tools
            input_tools = [t for t in tools_cfg if t.get("type") == "input" and t.get("enabled", True)]
            if input_tools:
                input_contexts = []
                for tool in input_tools:
                    tool_name = tool.get("name")
                    schema = tool.get("input_schema", {})
                    if schema and isinstance(schema, dict) and schema.get("properties"):
                        props = schema["properties"]
                        required = schema.get("required", [])
                        field_lines = []
                        for field_name, field_spec in props.items():
                            required_marker = " (required)" if field_name in required else ""
                            field_lines.append(f"- {field_name}: {field_spec.get('description', 'No description')}{required_marker}")
                        
                        if field_lines:
                            input_contexts.append(f"REQUIRED INPUTS FOR {tool_name}:\n" + "\n".join(field_lines))
                
                if input_contexts:
                    required_input_context = "\n\n".join(input_contexts)

    # Build conversation context
    conversation_context = ""
    if messages and len(messages) > 0:
        # Get last few messages for context
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        conversation_lines = []
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_lines.append(f"{role}: {content}")
        
        if conversation_lines:
            conversation_context = "CONVERSATION:\n" + "\n".join(conversation_lines)

    # Note: extracted_data is not available during decision routing
    # Data collection happens after decision is made

    # Build tool result context
    tool_result_block = ""
    if config.get("tool_result"):
        tool_result = config["tool_result"]
        if isinstance(tool_result, dict) and tool_result.get("type"):
            tool_result_block = f"""
TOOL RESULT:
- Tool: {tool_result.get('type')}
- Status: {tool_result.get('success', 'unknown')}
- Data: {tool_result.get('data', {})}
- Message: {tool_result.get('message', '')}
- Error: {tool_result.get('error', '') if not tool_result.get('success') else 'None'}
"""

    # Build action context
    action_context = ""
    action_history = config.get("action_history", [])
    if action_history and len(action_history) > 0:
        # Get last few actions for context
        recent_actions = action_history[-3:] if len(action_history) > 3 else action_history
        action_lines = []
        for action in recent_actions:
            step = action.get("step", 0)
            from_node = action.get("from_node", "unknown")
            to_node = action.get("to_node", "unknown")
            outcome = action.get("outcome", "")
            action_lines.append(f"Step {step}: {from_node} → {to_node} - {outcome}")
        
        if action_lines:
            action_context = "RECENT ACTIONS:\n" + "\n".join(action_lines)

    # Build options block
    options_block = ""
    if tools_cfg and isinstance(tools_cfg, list):
        tool_names = [t.get("name") for t in tools_cfg if t.get("enabled", True)]
        if tool_names:
            options_block = f"- {', '.join(tool_names)} if you have all required inputs and want to use the tool"

    # Build the complete prompt
    prompt = f"""SYSTEM PROMPT: {system_prompt}

{tool_display}

{required_input_context}

{conversation_context}

{tool_result_block}

{action_context}

TASK: Decide the next action based on the current state, conversation flow, and recent actions.

CURRENT USER MESSAGE: "{user_message}"

DECISION OPTIONS:
{options_block}
- "chat" if you need to collect more information or provide a conversational response
- "end" if the conversation should be concluded

INSTRUCTIONS:
- Analyze the conversation flow and user intent from the history
- Consider recent actions to avoid repetitive patterns or loops
- Use a configured tool ONLY when all required inputs are available and tool usage is appropriate
- Choose "chat" for collecting missing inputs or providing conversational responses
- Choose "end" when the task is complete or user wants to finish
- If you see repeated patterns in recent actions, adjust your strategy accordingly
- Explain your reasoning briefly, referencing recent actions if relevant

Return your decision as: DECISION: [one of: {', '.join([t.get('name') for t in tools_cfg if t.get('enabled', True)])}, chat, end] - [brief reason]
"""

    return prompt

def build_conversational_response_prompt(
    system_prompt: str,
    user_message: str,
    required_inputs: list = None,
    conversation_history: list = None,
    collected_data: dict = None,
    decision_justification: str = "",
    tool_context: dict = None
) -> str:
    """
    Build a prompt for general conversational responses with comprehensive context.
    """
    # Build required inputs context
    required_input_context = ""
    if required_inputs:
        field_keys = [f.get('key', f.get('name', '')) for f in required_inputs]
        required_input_context = f"Required inputs: {', '.join(field_keys)}"
    
    # Build conversation history context
    conversation_context = ""
    if conversation_history:
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        conversation_lines = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            conversation_lines.append(f"{role}: {content}")
        conversation_context = f"CONVERSATION HISTORY:\n{chr(10).join(conversation_lines)}"
    
    # Build collected data context
    collected_context = ""
    if collected_data:
        collected_context = f"COLLECTED DATA: {collected_data}"
    
    # Build decision context
    decision_context = ""
    if decision_justification:
        decision_context = f"DECISION CONTEXT: {decision_justification}"
    
    # Build tool context
    tool_context_block = ""
    if tool_context:
        top_answer = tool_context.get("top_answer", "").strip()
        facts = tool_context.get("facts", [])
        if isinstance(facts, list) and facts:
            facts_block = "\n".join([f"- {str(f)}" for f in facts[:3]])
        else:
            facts_block = "(none)"
        
        tool_context_block = f"""TOOL CONTEXT:
- Top answer: {top_answer}
- Facts:
{facts_block}"""
    
    prompt = f"""SYSTEM PROMPT: {system_prompt}

{required_input_context}

{conversation_context}

{collected_context}

{decision_context}

{tool_context_block}

TASK: Provide a helpful response that guides the user toward tool usage when appropriate, or offers conversational assistance when no tool is needed.

USER MESSAGE: "{user_message}"

INSTRUCTIONS:
1. Assess if the user's request requires tool usage
2. If yes, guide them toward providing required inputs
3. If no, provide helpful conversational responses
4. Always be clear about what's needed for tools
5. Suggest next steps that move toward completion
"""

    return prompt

def build_tool_use_prompt(
    system_prompt: str,
    tool_name: str,
    tool_description: str,
    user_message: str,
    tool_input: dict = None
) -> str:
    """
    Build a prompt for tool usage confirmation and execution.
    """
    tool_input_text = ""
    if tool_input:
        tool_input_text = f"\nTOOL INPUT: {tool_input}"

    prompt = f"""SYSTEM PROMPT: {system_prompt}

TOOL: {tool_name} - {tool_description}

USER MESSAGE: "{user_message}"

{tool_input_text}

TASK: Confirm that you will use the {tool_name} tool to fulfill the user's request.

INSTRUCTIONS:
1. Acknowledge the user's request
2. Confirm which tool you will use
3. Explain what the tool will do
4. Ask for any missing information if needed

Provide a clear, helpful response.
"""

    return prompt


def build_tool_answer_prompt(
    system_prompt: str,
    tool_type: str,
    tool_result: dict,
    user_query: str = "",
    extracted_data: dict = None,
    tool_category: str = "input",
    tool_outcome: str = "success",
    tool_summary: str = "",
    decision_context: dict = None
) -> str:
    """
    Build a prompt for generating tool answer responses based on tool category.
    """
    # Build tool result context
    tool_result_context = ""
    if tool_result:
        try:
            tool_result_context = f"TOOL RESULT:\n{json.dumps(tool_result, indent=2, default=str)}"
        except Exception:
            tool_result_context = f"TOOL RESULT:\n{str(tool_result)}"
    
    # Build decision context section
    decision_context_block = ""
    if decision_context:
        decision_context_block = f"""DECISION CONTEXT:
- User Intent: {decision_context.get('user_intent', 'Not specified')}
- Available Data: {decision_context.get('available_data', {})}
- Tool Requirements: {decision_context.get('tool_requirements', [])}
- Reasoning: {decision_context.get('reasoning', 'Not specified')}
- Alternatives Considered: {decision_context.get('alternatives_considered', [])}

"""
    
    # Build category-specific instructions
    if tool_category == "retrieval":
        task_description = "Based on the search results above, provide a helpful answer to the user's query."
        instructions = "Be concise but informative, and reference the key findings from the results."
    else:
        task_description = "Confirm to the user that their data has been processed."
        instructions = "If successful, provide a brief confirmation. If there was an error, explain what went wrong and suggest how to proceed."
    
    prompt = f"""SYSTEM PROMPT: {system_prompt}

{decision_context_block}

{tool_result_context}

USER TASK/QUERY: {user_query}

TOOL TYPE: {tool_type}
TOOL OUTCOME: {tool_outcome}
TOOL SUMMARY: {tool_summary}

ASSISTANT RESPONSE:
{task_description}
{instructions}
"""

    return prompt
