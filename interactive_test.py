#!/usr/bin/env python3
"""
Interactive test script for the simplified unified graph architecture.
Lets you pick from predefined agent configurations and chat interactively.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from graph_builder import create_conversation_graph
from llm_client import create_azure_openai_llm
from prompts import (
    BASE_SYSTEM_PROMPT,
    TOOL_DESCRIPTIONS,
    build_input_extraction_prompt,
    build_decision_prompt,
    build_conversational_response_prompt,
    build_tool_use_prompt
)

def load_agent_configs():
    """Load available agent configurations from agents.json."""
    try:
        with open("agents.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ agents.json not found!")
        return {}
    except json.JSONDecodeError:
        print("❌ Invalid JSON in agents.json!")
        return {}

def display_agent_menu(agents):
    """Display the menu of available agents."""
    print("\n🤖 Available Agent Configurations:")
    print("=" * 50)
    
    for i, (agent_id, agent_config) in enumerate(agents.items(), 1):
        name = agent_config.get("name", agent_id)
        tools = agent_config.get("tools", [])
        tool_names = [tool.get("name", "unnamed") for tool in tools]
        
        print(f"{i}. {name}")
        print(f"   Tools: {', '.join(tool_names)}")
        print(f"   ID: {agent_id}")
        print()
    
    print("0. Exit")
    print("=" * 50)

def select_agent(agents):
    """Let user select an agent configuration."""
    while True:
        try:
            choice = input("Select an agent (0-{}): ".format(len(agents)))
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(agents):
                agent_id = list(agents.keys())[choice_num - 1]
                return agent_id, agents[agent_id]
            else:
                print("❌ Invalid choice. Please select 0-{}".format(len(agents)))
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None

def create_prompt_functions():
    """Create the prompt functions dictionary."""
    return {
        "base_system_prompt": BASE_SYSTEM_PROMPT,
        "tool_descriptions": TOOL_DESCRIPTIONS,
        "build_input_extraction_prompt": build_input_extraction_prompt,
        "build_decision_prompt": build_decision_prompt,
        "build_conversational_response_prompt": build_conversational_response_prompt,
        "build_tool_use_prompt": build_tool_use_prompt
    }

async def chat_with_agent(agent_config, verbose=True):
    """Chat interactively with the selected agent."""
    print(f"\n🚀 Starting chat with: {agent_config['name']}")
    print("=" * 60)
    print("💡 Type 'exit', 'quit', or 'goodbye' to end the conversation")
    print("💡 Debug logs will show the simplified unified architecture in action")
    print("=" * 60)
    
    # Create LLM client
    try:
        llm = create_azure_openai_llm()
        print("✅ Created Azure OpenAI LLM client")
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        print("💡 Make sure your .env file has the correct Azure OpenAI credentials")
        return
    
    # Create prompt functions
    prompt_functions = create_prompt_functions()
    
    # Create and compile the graph
    try:
        print("🔧 Creating conversation graph...")
        graph = create_conversation_graph(agent_config, llm, prompt_functions, verbose=verbose)
        print("✅ Graph created and compiled successfully")
        print(f"📊 Graph has {len(graph.nodes)} nodes")
        print("🔗 Simplified unified architecture: structured_extractor → validate_inputs → tool_execution → tool_answer → wait_user_input")
    except Exception as e:
        print(f"❌ Failed to create graph: {e}")
        return
    
    # Initialize conversation state
    conversation_state = {
        "messages": [],
        "user_input": "",
        "conversation_active": True,
        "extracted_data": {},
        "query_spec": {},
        "tool_result": {},
        "last_tool_summary": "",
        "last_tool_context": {},
        "next_action": "",
        "chosen_tool": "",
        "tool_category": "",
        "decision_justification": "",
        "error_message": "",
        "validation_errors": [],
        "clarification_message": "",
        "session_id": f"session_{agent_config['name']}_{os.getpid()}",
        "action_history": []
    }
    
    turn_count = 0
    max_turns = 25
    
    while conversation_state["conversation_active"] and turn_count < max_turns:
        turn_count += 1
        print(f"\n🔄 Turn {turn_count}")
        print("-" * 30)
        
        # Get user input
        try:
            user_input = input("👤 You: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() in ["exit", "quit", "goodbye"]:
            print("👋 Goodbye! Thanks for testing the simplified unified graph!")
            break
        
        if not user_input:
            print("⚠️  Please enter a message")
            continue
        
        # Update conversation state
        conversation_state["user_input"] = user_input
        conversation_state["messages"].append({"role": "user", "content": user_input})
        
        try:
            # Run the graph
            print("🤖 Agent is thinking...")
            print("🔍 Following simplified unified path: decision_router → [tool/chat] → ...")
            
            result = await graph.ainvoke(conversation_state, config={"configurable": {"thread_id": conversation_state["session_id"]}})
            
            # Update conversation state with result
            conversation_state.update(result)
            
            # Display agent response
            if conversation_state.get("messages"):
                last_message = conversation_state["messages"][-1]
                if last_message.get("role") == "assistant":
                    print(f"🤖 Agent: {last_message.get('content', 'No response')}")
                else:
                    print("🤖 Agent: (Processing response...)")
            
            # Display debug information
            print("\n🔍 Debug Information:")
            print(f"   Next Action: {conversation_state.get('next_action', 'None')}")
            print(f"   Chosen Tool: {conversation_state.get('chosen_tool', 'None')}")
            print(f"   Tool Category: {conversation_state.get('tool_category', 'None')}")
            
            if conversation_state.get("decision_justification"):
                print(f"   Decision Reason: {conversation_state['decision_justification']}")
            
            # Display tool results or errors
            if conversation_state.get("tool_result"):
                tool_result = conversation_state["tool_result"]
                if tool_result.get("success"):
                    print(f"   ✅ Tool executed successfully: {tool_result.get('message', '')}")
                else:
                    print(f"   ❌ Tool execution failed: {tool_result.get('error', '')}")
            
            if conversation_state.get("error_message"):
                print(f"   ⚠️  Error: {conversation_state['error_message']}")
            
            if conversation_state.get("validation_errors"):
                print(f"   🔍 Validation errors: {conversation_state['validation_errors']}")
            
            # Display action history for debugging
            if conversation_state.get("action_history"):
                recent_actions = conversation_state["action_history"][-3:]
                print(f"   📋 Recent actions: {len(recent_actions)} actions")
                for action in recent_actions:
                    print(f"      - {action.get('from_node', '?')} → {action.get('to_node', '?')}")
            
        except Exception as e:
            print(f"❌ Error during graph execution: {e}")
            conversation_state["error_message"] = str(e)
        
        # Check exit conditions
        if turn_count >= max_turns:
            print("⚠️  Maximum turns reached. Ending conversation.")
            break
        
        if not conversation_state.get("conversation_active"):
            print("🔚 Conversation ended by agent.")
            break
    
    print("\n🎉 Chat session completed!")
    print(f"📊 Total turns: {turn_count}")
    print(f"💬 Total messages: {len(conversation_state.get('messages', []))}")
    
    # Show final state summary
    print("\n📋 Final State Summary:")
    print(f"   Next Action: {conversation_state.get('next_action', 'None')}")
    print(f"   Chosen Tool: {conversation_state.get('chosen_tool', 'None')}")
    print(f"   Tool Result: {'Success' if conversation_state.get('tool_result', {}).get('success') else 'None/Failed'}")
    print(f"   Conversation Active: {conversation_state.get('conversation_active', False)}")

def main():
    """Main function to run the interactive test."""
    print("🧪 Interactive Testing Suite for Simplified Unified Graph")
    print("=" * 70)
    print("🔍 This will test the new simplified unified architecture where")
    print("   all tools follow the same path: structured_extractor → validate_inputs → tool_execution → tool_answer")
    print("=" * 70)
    
    # Check if Azure OpenAI credentials are available
    if not os.getenv("AZUREOPENAIAPIKEY"):
        print("⚠️  Azure OpenAI API key not found in .env file")
        print("💡 Please set the following environment variables in your .env file:")
        print("   - AZUREOPENAIAPIKEY")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        print("   - AZUREOPENAIAPIVERSION")
        print("\n📋 You can still test the graph structure without full execution.")
        
        # Load agents for structure testing
        agents = load_agent_configs()
        if agents:
            print(f"\n✅ Loaded {len(agents)} agent configurations for structure testing")
            display_agent_menu(agents)
            selection = select_agent(agents)
            if selection:
                agent_id, agent_config = selection
                print(f"\n🔍 Testing graph structure for: {agent_config['name']}")
                # Test graph structure without execution
                test_graph_structure(agent_config)
        return
    
    # Load agent configurations
    agents = load_agent_configs()
    if not agents:
        print("❌ No agent configurations found in agents.json")
        return
    
    print(f"✅ Loaded {len(agents)} agent configurations")
    
    while True:
        # Display menu
        display_agent_menu(agents)
        
        # Get user selection
        selection = select_agent(agents)
        if not selection:
            break
        
        agent_id, agent_config = selection
        
        # Start chat with selected agent
        try:
            asyncio.run(chat_with_agent(agent_config, verbose=True))
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during chat: {e}")
        
        # Ask if user wants to try another agent
        print("\n" + "=" * 50)
        try:
            another = input("Try another agent? (y/n): ").lower().strip()
            if another not in ["y", "yes"]:
                break
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print("\n🎉 Testing completed! Thanks for testing the simplified unified graph!")

def test_graph_structure(agent_config):
    """Test the graph structure without running it."""
    print("🔍 Testing Graph Structure")
    print("=" * 40)
    
    try:
        # Create LLM client (mock)
        class MockLLM:
            def __init__(self):
                self.name = "mock_llm"
        
        llm = MockLLM()
        
        # Create prompt functions
        prompt_functions = {
            "base_system_prompt": "Test prompt",
            "tool_descriptions": {"test": "test tool"},
            "build_input_extraction_prompt": lambda *args: "test",
            "build_decision_prompt": lambda *args: "test",
            "build_conversational_response_prompt": lambda *args: "test",
            "build_tool_use_prompt": lambda *args: "test"
        }
        
        # Create graph
        graph = create_conversation_graph(agent_config, llm, prompt_functions, verbose=False)
        
        # Check graph structure
        print(f"✅ Graph created successfully")
        print(f"📊 Graph has {len(graph.nodes)} nodes")
        
        # List nodes
        print("\n📋 Graph Nodes:")
        for node in graph.nodes:
            print(f"   - {node}")
        
        # Describe architecture instead of listing edges
        print("\n🔗 Simplified Unified Architecture:")
        print("   - Entry point: wait_user_input")
        print("   - Decision router handles routing to tools/chat/end")
        print("   - All tools follow unified path: structured_extractor → validate_inputs → tool_execution → tool_answer")
        print("   - Tool answer always routes to wait_user_input")
        
        # Verify required nodes are present
        required_nodes = {"decision_router", "structured_extractor", "validate_inputs", "tool_execution", "tool_answer", "wait_user_input", "chat", "end"}
        missing_nodes = required_nodes - set(graph.nodes)
        if missing_nodes:
            print(f"   ⚠️  Missing nodes: {missing_nodes}")
        else:
            print("   ✅ All required nodes present")
        
        print("\n✅ Graph structure test completed successfully!")
        
    except Exception as e:
        print(f"❌ Graph structure test failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
