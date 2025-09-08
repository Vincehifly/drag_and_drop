"""LangGraph graph construction and compilation."""
# CONTEXT for GitHub Copilot:
# - Write minimal, clean code without excessive prints or debug output
# - Focus on core functionality, avoid verbose instructions
# - Use modern Python patterns and best practices
# - For LangGraph: use interrupt()/Command(resume=...) pattern for human-in-the-loop
# - Prefer concise, working solutions over detailed explanations in code

from langgraph.graph import StateGraph

from models import ConversationState
from nodes import (
    chat_node, wait_user_input_node, decision_router_node,
    tool_execution_node, end_node, set_prompt_functions,
    structured_extractor_node, validate_inputs_node, exit_evaluator_node,
    tool_answer_node, input_translation_node, output_translation_node
)

def create_conversation_graph(config: dict, llm, prompt_functions: dict, verbose: bool = True):
    """
    Create and compile the LangGraph conversation graph.
    
    Args:
        config: Configuration dictionary
        llm: LLM instance
        prompt_functions: Dictionary of prompt building functions
        verbose: Enable verbose logging
    
    Returns:
        Compiled graph ready for execution
    """
    if verbose:
        print("Creating conversation graph...")
    
    # Set prompt functions for nodes
    set_prompt_functions(
        prompt_functions["base_system_prompt"],
        prompt_functions["tool_descriptions"],
        prompt_functions["build_input_extraction_prompt"],
        prompt_functions["build_decision_prompt"],
        prompt_functions["build_conversational_response_prompt"],
        prompt_functions["build_tool_use_prompt"]
    )
    
    # Create graph with state schema
    graph = StateGraph(ConversationState)
    
    # Add nodes with config and LLM binding - SIMPLIFIED UNIFIED ARCHITECTURE
    graph.add_node("decision_router", lambda state: decision_router_node(state, config, llm, verbose))
    graph.add_node("chat", lambda state: chat_node(state, config, llm, verbose))
    graph.add_node("wait_user_input", lambda state: wait_user_input_node(state, config, llm, verbose))
    graph.add_node("structured_extractor", lambda state: structured_extractor_node(state, config, llm, verbose))
    graph.add_node("validate_inputs", lambda state: validate_inputs_node(state, config, llm, verbose))
    graph.add_node("exit_evaluator", lambda state: exit_evaluator_node(state, config, llm, verbose))
    graph.add_node("tool_answer", lambda state: tool_answer_node(state, config, llm, verbose))
    graph.add_node("tool_execution", lambda state: tool_execution_node(state, config, llm, verbose))
    graph.add_node("end", lambda state: end_node(state, config, llm, verbose))
    
    # Add translation nodes
    graph.add_node("input_translation", lambda state: input_translation_node(state, config, llm, verbose))
    graph.add_node("output_translation", lambda state: output_translation_node(state, config, llm, verbose))
    
    # Set entry point - start by waiting for user input
    graph.set_entry_point("wait_user_input")
    
    # Discover tools from config
    tools_cfg = config.get("tools", []) or []
    tool_names = []
    try:
        tool_names = [t.get("name") for t in tools_cfg if t.get("name") and t.get("enabled", True)]
    except Exception:
        tool_names = []
    
    # Check if translation is enabled
    translation_enabled = config.get("translation", {}).get("enabled", False)
    
    # SIMPLIFIED UNIFIED ARCHITECTURE WITH TRANSLATION:
    # With translation: wait_user_input → input_translation → decision_router → [chat/tools] → output_translation → wait_user_input
    # Without translation: wait_user_input → decision_router → [chat/tools] → wait_user_input
    
    # Decision router: routes to chat, tools, or end
    def decision_router_fn(state):
        """Route from decision_router to next step based on decision."""
        next_action = state.get("next_action", "chat")
        if verbose:
            print(f"Decision routing to: {next_action}")
        
        if next_action in tool_names:
            return "structured_extractor"  # All tools go to structured_extractor
        elif next_action == "end":
            return "end"
        else:
            return "chat"  # Continue conversation
    
    # Decision router conditional edges
    graph.add_conditional_edges(
        "decision_router",
        decision_router_fn,
        {
            "structured_extractor": "structured_extractor",
            "chat": "chat",
            "end": "end"
        }
    )
    
    # Chat conversation flow: route based on translation setting
    if translation_enabled:
        # With translation: chat -> output_translation -> wait_user_input
        graph.add_edge("chat", "output_translation")
    else:
        # Without translation: chat -> wait_user_input
        graph.add_edge("chat", "wait_user_input")
    
    # Route user input based on translation setting
    if translation_enabled:
        # With translation: wait_user_input -> input_translation -> decision_router
        graph.add_edge("wait_user_input", "input_translation")
        graph.add_edge("input_translation", "decision_router")
    else:
        # Without translation: wait_user_input -> decision_router
        graph.add_edge("wait_user_input", "decision_router")
    
    # Unified tool execution flow: structured_extractor → validate_inputs → tool_execution → tool_answer → wait_user_input
    graph.add_edge("structured_extractor", "validate_inputs")
    
    # Conditional routing from validate_inputs: chat if errors, tool_execution if valid
    def validate_router(state):
        validation_errors = state.get("validation_errors", [])
        return "chat" if validation_errors else "tool_execution"
    
    graph.add_conditional_edges(
        "validate_inputs",
        validate_router,
        {
            "chat": "chat",
            "tool_execution": "tool_execution"
        }
    )
    
    # After tool execution, always go to tool_answer
    graph.add_edge("tool_execution", "tool_answer")
    
    # After tool_answer, route based on translation setting
    if translation_enabled:
        # With translation: tool_answer -> output_translation -> wait_user_input
        graph.add_edge("tool_answer", "output_translation")
        graph.add_edge("output_translation", "wait_user_input")
    else:
        # Without translation: tool_answer -> wait_user_input
        graph.add_edge("tool_answer", "wait_user_input")
    
    # Exit evaluation routing
    def exit_router(state):
        return "end" if (state.get("next_action") == "end" or state.get("conversation_active") is False) else "decision_router"
    
    graph.add_conditional_edges(
        "exit_evaluator",
        exit_router,
        {
            "end": "end",
            "decision_router": "decision_router"
        }
    )
    
    # End node terminates the graph
    graph.add_edge("end", "__end__")
    
    # Compile graph without checkpointing for testing environment
    compiled_graph = graph.compile()
    
    if verbose:
        print("Graph compiled successfully with simplified unified architecture!")
        if translation_enabled:
            print("Translation ENABLED - Flow: wait_user_input → input_translation → decision_router → [chat/tools] → output_translation → wait_user_input")
        else:
            print("Translation DISABLED - Flow: wait_user_input → decision_router → [chat/tools] → wait_user_input")
        print("All tools follow the same path: structured_extractor → validate_inputs → tool_execution → tool_answer")
        
    return compiled_graph

def visualize_graph(graph, save_path: str = None, verbose: bool = True):
    """
    Generate a visual representation of the graph.
    
    Args:
        graph: Compiled LangGraph
        save_path: Optional path to save the visualization
        verbose: Enable verbose logging
    """
    try:
        # Generate mermaid diagram
        mermaid = graph.get_graph().draw_mermaid()
        
        if verbose:
            print("Graph visualization (Mermaid):")
            print(mermaid)
        
        # Save to file if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(mermaid)
            if verbose:
                print(f"Graph visualization saved to: {save_path}")
        
        return mermaid
        
    except Exception as e:
        if verbose:
            print(f"Could not generate graph visualization: {e}")
        return None

def get_graph_state(compiled_graph, config):
    """Get current graph state for debugging."""
    try:
        return compiled_graph.get_state(config)
    except Exception as e:
        print(f"Could not get graph state: {e}")
        return None
