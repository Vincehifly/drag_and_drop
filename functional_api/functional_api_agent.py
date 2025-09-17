#!/usr/bin/env python3
"""
LangGraph Functional API Agent Template

A minimal, runnable template that uses your existing tools and llm_client.

- Functional API (@entrypoint, @task)
- Uses tools.AVAILABLE_TOOLS
- LLM-based decision making
- Streaming via StreamWriter
- Persistence via MemorySaver
"""

import asyncio
import os
from typing import Any, Dict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import StreamWriter

from tools import AVAILABLE_TOOLS, execute_tool

# Use your existing LLM client with environment variables from .env file
from llm_client import create_conversation_llm
llm = create_conversation_llm(temperature=0.2)
print("✅ Using Azure OpenAI LLM")

# Import the React agent from the other file
from langgraph_react_agent import create_agent
react_agent = create_agent()
print("✅ LangGraph React agent loaded")

# Per-thread persistence config
CONFIG = {"configurable": {"thread_id": "functional-template-thread"}}

# Translation settings
ENABLE_TRANSLATION = True  # Set to True to enable language detection and translation
DETECTED_LANGUAGE = "en"  # Store the detected language of the original message


@task()
def detect_and_translate_to_english(user_input: str) -> Dict[str, str]:
    """Detect the language of the input and translate to English if needed."""
    global DETECTED_LANGUAGE
    
    # First, detect the language
    detect_prompt = f"""Detect the language of the following text and respond with only the language code (e.g., 'en', 'hu', 'es', 'fr', 'de', etc.):

Text: {user_input}

Language code:"""
    
    detected_lang = llm.invoke(detect_prompt).content.strip().lower()
    DETECTED_LANGUAGE = detected_lang
    
    # If already English, return as is
    if detected_lang in ['en', 'english']:
        return {
            "original_language": "en",
            "translated_input": user_input,
            "needs_translation": False
        }
    
    # Translate to English
    translate_prompt = f"""Translate the following text to English. Only return the English translation, nothing else.

Original text ({detected_lang}): {user_input}

English translation:"""
    
    translated_text = llm.invoke(translate_prompt).content.strip()
    
    return {
        "original_language": detected_lang,
        "translated_input": translated_text,
        "needs_translation": True
    }


@task()
def react_agent_task(user_input: str) -> Dict[str, Any]:
    """Use the LangGraph React agent to handle the user input with RAG and web search capabilities."""
    print(f"🤖 [REACT AGENT] Processing: {user_input}")
    
    # Get the current thread_id from the context (this should be passed automatically)
    # For now, use a unique thread_id based on the input to ensure isolation
    import hashlib
    thread_id = f"react-{hashlib.md5(user_input.encode()).hexdigest()[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    response = react_agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
    
    # Extract the response content
    message_content = response['messages'][-1].content
    
    # Check if any tools were used by looking at the response structure
    # The React agent will have used tools internally if needed
    tool_used = None
    if "🔍 [RAG TOOL]" in str(response) or "🌐 [WEB SEARCH]" in str(response):
        if "🔍 [RAG TOOL]" in str(response):
            tool_used = "rag_search"
        if "🌐 [WEB SEARCH]" in str(response):
            tool_used = "web_search"
    
    return {
        "message": message_content,
        "tool": tool_used,
        "data": {"response": message_content}
    }


@task()
def translate_response_to_original_language(english_response: str, original_language: str) -> str:
    """Translate the English response back to the original language."""
    global DETECTED_LANGUAGE
    
    # If original language was English, return as is
    if original_language in ['en', 'english']:
        return english_response
    
    # Translate back to original language
    translate_prompt = f"""Translate the following English text to {original_language}. Only return the translation, nothing else.

English text: {english_response}

{original_language} translation:"""
    
    translated_response = llm.invoke(translate_prompt).content.strip()
    return translated_response


# Entrypoint: detect language -> translate to English -> react agent -> translate back
@entrypoint(checkpointer=MemorySaver())
def agent(user_input: str, writer: StreamWriter) -> Dict[str, Any]:
    # Step 1: Detect language and translate to English if needed
    if ENABLE_TRANSLATION:
        translation_result = detect_and_translate_to_english(user_input).result()
        processing_input = translation_result["translated_input"]
        original_language = translation_result["original_language"]
    else:
        processing_input = user_input
        original_language = "en"
    
    # Step 2: Use React agent with RAG and web search capabilities
    result = react_agent_task(processing_input).result()
    
    # Step 3: Translate response back to original language if needed
    if ENABLE_TRANSLATION and original_language != "en":
        final_response = translate_response_to_original_language(
            result["message"], 
            original_language
        ).result()
        result["message"] = final_response
    
    return result




async def run_streaming_example(prompt: str):
    """Run streaming example"""
    print(f"User: {prompt}")
    print("Agent reasoning:")
    
    async for chunk in agent.astream(prompt, CONFIG):
        print(">", chunk)


async def interactive_chat():
    """Interactive chat interface"""
    global ENABLE_TRANSLATION
    print("=== LangGraph Functional API Agent - Interactive Chat ===")
    print("This agent uses:")
    print("1. Language Detection and Translation (any language to English)")
    print("2. React Agent with RAG and Web Search capabilities")
    print("3. Response Translation back to original language")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'stream' to enable streaming mode")
    print("Type 'translate' to toggle language detection and translation")
    print(f"Translation mode: {'ON' if ENABLE_TRANSLATION else 'OFF'}")
    print("=" * 60)
    
    streaming_mode = False
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'stream':
                streaming_mode = not streaming_mode
                print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue
            elif user_input.lower() == 'translate':
                ENABLE_TRANSLATION = not ENABLE_TRANSLATION
                print(f"Translation mode: {'ON' if ENABLE_TRANSLATION else 'OFF'}")
                continue
            elif not user_input:
                continue
            
            if streaming_mode:
                print("\nAgent:")
                async for chunk in agent.astream(user_input, CONFIG):
                    print(">", chunk)
            else:
                print("\nAgent:")
                response = agent.invoke(user_input, CONFIG)
                print(f"Response: {response['message']}")
                if response.get('tool'):
                    print(f"Tool used: {response['tool']}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function - choose between demo or interactive chat"""
    print("Choose mode:")
    print("1. Run demo examples")
    print("2. Interactive chat")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n=== LangGraph Functional API Agent Demo ===\n")
        
        # Example 1: Simple invoke
        print("1. Simple invoke example:")
        out = agent.invoke("Search for the latest news about LangGraph", CONFIG)
        print("Response:", out)
        print()
        
        # Example 2: Streaming (async)
        print("2. Streaming example:")
        asyncio.run(run_streaming_example("What is LangGraph functional API?"))
        print()
        
        # Example 3: Chat example
        print("3. Chat example:")
        chat_out = agent.invoke("Hello, how are you?", CONFIG)
        print("Response:", chat_out)
        print()
        
        print("Demo completed!")
        
    elif choice == "2":
        asyncio.run(interactive_chat())
    else:
        print("Invalid choice. Running interactive chat...")
        asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()
