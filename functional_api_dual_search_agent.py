#!/usr/bin/env python3
"""
Functional API Dual-Source Agent
- Runs RAG and Web Search in parallel using await/gather
- Uses existing tools: rag_setup.RAGSetup and tools.web_search_tool
- Provides a simple entrypoint and interactive demo
"""

import asyncio
from typing import Any, Dict, List, Optional

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import StreamWriter

from rag_setup import setup_rag, RAGSetup
from tools import web_search_tool

# Optional LLM for synthesis
from llm_client import create_conversation_llm
llm = create_conversation_llm(temperature=0.2)

# Persistent RAG instance (lazy)
_RAG_INSTANCE: Optional[RAGSetup] = None


def get_rag() -> Optional[RAGSetup]:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = setup_rag()
    return _RAG_INSTANCE


@task()
def rag_search_task(query: str, top_k: int = 5) -> Dict[str, Any]:
    print(f"🔍 RAG: Searching for '{query}'")
    rag = get_rag()
    if rag is None:
        print("❌ RAG: Setup failed")
        return {
            "success": False,
            "type": "rag",
            "message": "RAG not available (setup failed)",
            "data": {"query": query, "results": []},
        }
    results = rag.search(query, n_results=top_k) or []
    print(f"✅ RAG: Found {len(results)} results")
    return {
        "success": True,
        "type": "rag",
        "message": f"RAG returned {len(results)} results",
        "data": {"query": query, "results": results, "result_count": len(results)},
    }


@task()
def web_search_task(query: str, max_results: int = 5) -> Dict[str, Any]:
    print(f"🌐 WEB: Searching for '{query}'")
    cfg = {"max_results": max_results}
    result = web_search_tool({"query": query}, cfg, verbose=False)
    print(f"✅ WEB: Found {result.get('data', {}).get('result_count', 0)} results")
    return result


def _synthesize_answer(user_query: str, rag_data: Dict[str, Any], web_data: Dict[str, Any]) -> str:
    print(f"🔄 SYNTHESIS: RAG success={rag_data.get('success')}, Web success={web_data.get('success')}")
    
    rag_bits: List[str] = []
    try:
        rag_results = rag_data.get("data", {}).get("results") or []
        print(f"🔄 SYNTHESIS: RAG has {len(rag_results)} results")
        for r in rag_results[:5]:
            snippet = r.get("text") or ""
            rag_bits.append(snippet[:350])
    except Exception as e:
        print(f"❌ SYNTHESIS: RAG processing error: {e}")

    web_bits: List[str] = []
    try:
        web_results = web_data.get("data", {}).get("results") or []
        print(f"🔄 SYNTHESIS: Web has {len(web_results)} results")
        for r in web_results[:5]:
            piece = (r.get("title") or "") + " - " + (r.get("snippet") or "")
            web_bits.append(piece[:350])
    except Exception as e:
        print(f"❌ SYNTHESIS: Web processing error: {e}")

    prompt = (
        "You are a helpful assistant. Use the following RAG and Web Search context to answer the user.\n"
        "Cite sources inline with short markers like [RAG] or [WEB:n].\n\n"
        f"User question: {user_query}\n\n"
        f"RAG context (max 5 chunks):\n- " + "\n- ".join(rag_bits) + "\n\n"
        f"Web context (max 5 items):\n- " + "\n- ".join(web_bits) + "\n\n"
        "Answer concisely."
    )
    return llm.invoke(prompt).content.strip()


async def _emit(writer: StreamWriter, text: str) -> None:
    """Emit a streaming chunk supporting both callable and object-style writers."""
    try:
        # Object-style: writer.write(...)
        write_method = getattr(writer, "write", None)
        if callable(write_method):
            maybe_coro = write_method(text)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
            return
        # Callable-style: writer(...)
        if callable(writer):
            maybe_coro = writer(text)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
            return
    except Exception:
        # Swallow streaming errors to avoid failing the whole run
        return


@entrypoint(checkpointer=MemorySaver())
async def dual_search_agent(user_input: str, writer: StreamWriter) -> Dict[str, Any]:
    # Stream initial status
    await _emit(writer, "🔍 Starting parallel search...\n")
    
    # Run tasks concurrently
    await _emit(writer, "📚 Searching RAG database...\n")
    await _emit(writer, "🌐 Searching web...\n")
    
    rag_future = rag_search_task(user_input)
    web_future = web_search_task(user_input)
    
    # Wait for both to complete
    rag_res, web_res = await asyncio.gather(rag_future, web_future)
    
    # Stream results
    await _emit(writer, f"✅ RAG: {rag_res.get('message', 'No results')}\n")
    await _emit(writer, f"✅ Web: {web_res.get('message', 'No results')}\n")
    
    # Synthesize final response
    await _emit(writer, "🔄 Synthesizing answer...\n")
    answer = _synthesize_answer(user_input, rag_res, web_res)
    
    # Stream final answer
    await _emit(writer, "📝 Final Answer:\n")
    await _emit(writer, f"{answer}\n")

    return {
        "message": answer,
        "sources": {
            "rag": rag_res.get("data", {}).get("results", []),
            "web": web_res.get("data", {}).get("results", []),
        },
        "tool_summaries": {
            "rag": rag_res.get("message"),
            "web": web_res.get("message"),
        },
    }


def create_dual_search_agent():
    return dual_search_agent


async def main():
    print("=== Dual Search Agent (Functional API) ===")
    print("Runs RAG + Web search in parallel and synthesizes an answer.")
    print("Type 'stream' to toggle streaming mode, 'quit' to exit.\n")
    
    streaming_mode = True
    config = {"configurable": {"thread_id": "dual-agent-thread"}}
    
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            elif q.lower() == "stream":
                streaming_mode = not streaming_mode
                print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue
            
            print("\nAssistant:")
            
            if streaming_mode:
                # Use streaming mode
                async for chunk in dual_search_agent.astream(q, config):
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            else:
                # Use regular invoke
                out = await dual_search_agent.ainvoke(q, config)
                print(out.get("message"))
            
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
