#!/usr/bin/env python3
"""
Data Science RAG Agent
A specialized agent that answers data science questions using RAG with a cheat sheet knowledge base.
"""

import asyncio
import os
from langchain_core.messages import SystemMessage
# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from rag_setup import setup_rag
from tools import AVAILABLE_TOOLS

# Use your existing LLM client with environment variables from .env file
from llm_client import create_conversation_llm
llm = create_conversation_llm(temperature=0.1)
print("✅ Using Azure OpenAI LLM")

# Initialize RAG system
print("🔄 Initializing RAG system...")
rag_system = setup_rag()
if not rag_system:
    print("⚠️ RAG system not available - continuing without RAG capabilities")

# Check Tavily availability
tavily_available = os.getenv("tavily_api_key") is not None
if tavily_available:
    print("✅ Tavily web search available")
else:
    print("⚠️ Tavily web search not available - set tavily_api_key environment variable")


@tool
def rag_search_tool(query: str) -> str:
    """Search the cheat sheet knowledge base for information. Use this when the user asks questions that might be answered by the cheat sheet content.
    
    Args:
        query: The question or search query to look up in the knowledge base
    """
    if not rag_system:
        return "RAG system is not available. Please check the setup."
    
    print(f"🔍 [RAG TOOL] Searching for: {query}")
    
    try:
        results = rag_system.search(query, n_results=3)
        
        if not results:
            return "No relevant information found in the knowledge base for your query."
        
        response = "Based on the cheat sheet, here's what I found:\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"**Source {i}:**\n{result['text']}\n\n"
        
        print(f"✅ [RAG TOOL] Found {len(results)} relevant results")
        return response
        
    except Exception as e:
        print(f"❌ [RAG TOOL] Error: {e}")
        return f"Error searching the knowledge base: {e}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information. Use this when you need up-to-date information not available in the knowledge base.
    
    Args:
        query: The search query to look up on the web
    """
    if not tavily_available:
        return "Web search is not available. Please set the tavily_api_key environment variable."
    
    print(f"🌐 [WEB SEARCH] Searching for: {query}")
    
    try:
        # Use the web search tool from tools.py
        extracted_data = {"search_query": query}
        config = {"max_results": 5}
        
        result = AVAILABLE_TOOLS["web_search"](extracted_data, config, verbose=False)
        
        if not result.get("success"):
            return f"Web search failed: {result.get('message', 'Unknown error')}"
        
        search_data = result.get("data", {})
        results = search_data.get("results", [])
        
        if not results:
            return "No relevant information found on the web for your query."
        
        response = "Here's what I found on the web:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "No description available")
            
            response += f"**Result {i}: {title}**\n"
            response += f"URL: {url}\n"
            response += f"Description: {snippet}\n\n"
        
        print(f"✅ [WEB SEARCH] Found {len(results)} results")
        return response
        
    except Exception as e:
        print(f"❌ [WEB SEARCH] Error: {e}")
        return f"Error searching the web: {e}"






def create_agent(use_custom_prompt=True):
    """Create the ReAct agent with RAG and web search tools for data science questions."""
    tools = [rag_search_tool, web_search_tool]
    
    if use_custom_prompt:
        # Custom system message as string (gets converted to SystemMessage automatically)
        custom_prompt = """You are a specialized data science AI assistant with access to multiple information sources.

YOUR GOAL:
Answer data science related questions by using ONLY information from your search tools. Do not supplement with your training data.

STRICT SOURCE REQUIREMENTS:
- You MUST base your responses ONLY on information found through your search tools
- You MUST NOT add information from your training data beyond what's in the search results
- You MUST clearly indicate when information comes from the knowledge base vs. web search
- If search results are insufficient, acknowledge this limitation explicitly
- You MUST NOT provide detailed explanations beyond what's found in the sources

YOUR CAPABILITIES:
1. KNOWLEDGE BASE SEARCH: Search a comprehensive cheat sheet for established data science concepts
2. WEB SEARCH: Search the web for current information, recent developments, or specific details

TOOL SELECTION STRATEGY:
- Use rag_search_tool(query) for:
  * Fundamental data science concepts (machine learning, statistics, algorithms)
  * Programming techniques (pandas, numpy, matplotlib)
  * Established methodologies and best practices
  * General data science education topics

- Use web_search_tool(query) for:
  * Current events or recent developments in data science
  * Latest software versions, updates, or releases
  * Specific company or product information
  * News about data science trends or breakthroughs
  * Information not likely to be in a cheat sheet

CONVERSATION FLOW:
- When users ask data science questions, first try the knowledge base
- If the knowledge base doesn't have sufficient information, use web search
- You can use both tools in sequence for comprehensive answers
- Synthesize information from multiple sources when helpful
- ALWAYS stay within the bounds of what you found in the search results

TOOL USAGE:
- rag_search_tool(query): Search the cheat sheet knowledge base
- web_search_tool(query): Search the web for current information
- Choose the most appropriate tool based on the question type

RESPONSE STYLE:
- Be friendly, professional, and educational
- Provide answers based STRICTLY on search results
- Quote or paraphrase the exact content from sources when possible
- If sources have limited information, acknowledge this limitation
- Always cite your sources (knowledge base vs. web search)
- Do not expand beyond what's provided in the search results

Remember: You are a source-based assistant. Use ONLY information from your search tools. Do not supplement with training data."""
        
        # Create a system message with clear instructions
        system_message = SystemMessage(content=custom_prompt)
        
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message,  # Use SystemMessage class
            checkpointer=MemorySaver()
        )
    else:
        # Default LangGraph agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=MemorySaver()
        )
    
    return agent


async def interactive_chat():
    """Interactive chat interface for testing multi-tool data science agent."""
    print("=== Multi-Tool Data Science Agent ===")
    print("This agent answers data science questions using:")
    print("1. Knowledge base (cheat sheet) - for established concepts")
    print("2. Web search - for current information and recent developments")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 60)
    
    agent = create_agent()
    config = {"configurable": {"thread_id": "multi-tool-data-science-demo"}}
    
    print("Try asking data science questions like:")
    print("KNOWLEDGE BASE QUESTIONS:")
    print("- 'What is machine learning?'")
    print("- 'How do I use pandas?'")
    print("- 'Explain neural networks'")
    print("- 'What are the different types of regression?'")
    print()
    print("WEB SEARCH QUESTIONS:")
    print("- 'What are the latest trends in AI in 2024?'")
    print("- 'What's new in pandas 2.0?'")
    print("- 'Recent breakthroughs in machine learning'")
    print("- 'Latest data science job market trends'")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif not user_input:
                continue
            
            print("\nAgent:")
            # ReAct agent expects messages in proper format
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
            print(f"Response: {response['messages'][-1].content}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_demo():
    """Run a quick demo of the multi-tool data science agent."""
    print("=== Multi-Tool Data Science Agent Demo ===")
    
    agent = create_agent()
    config = {"configurable": {"thread_id": "multi-tool-data-science-demo"}}
    
    # Demo 1: Knowledge base question
    print("Demo 1: Knowledge base question")
    print("Asking: 'What is machine learning?'")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What is machine learning?"}]
    }, config)
    print(f"Agent: {response['messages'][-1].content}")
    print()
    
    # Demo 2: Web search question
    print("Demo 2: Web search question")
    print("Asking: 'What are the latest trends in AI in 2024?'")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What are the latest trends in AI in 2024?"}]
    }, config)
    print(f"Agent: {response['messages'][-1].content}")
    print()
    
    # Demo 3: Mixed question (might use both tools)
    print("Demo 3: Comprehensive question")
    print("Asking: 'Explain machine learning and what are the latest developments?'")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Explain machine learning and what are the latest developments?"}]
    }, config)
    print(f"Agent: {response['messages'][-1].content}")
    print()
    
    print("Demo completed!")


def main():
    """Main function - choose between demo or interactive chat."""
    print("Choose mode:")
    print("1. Run demo examples (shows both knowledge base and web search)")
    print("2. Interactive chat (try both types of questions)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        asyncio.run(interactive_chat())
    else:
        print("Invalid choice. Running interactive chat...")
        asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()