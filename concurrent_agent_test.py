#!/usr/bin/env python3
"""
Concurrent LangGraph Agent Testing

This script tests how multiple instances/sessions of the functional API agent
can be run simultaneously. It explores:

1. Multiple sessions with different thread IDs
2. Concurrent execution of different queries
3. Resource sharing and isolation
4. Performance characteristics
5. Error handling in concurrent scenarios
"""

import asyncio
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
import json
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Import your functional API agent
from functional_api_agent import agent


class ConcurrentAgentTester:
    """Test runner for concurrent agent execution scenarios"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def log_result(self, test_name: str, session_id: str, query: str, 
                  result: Any, duration: float, error: str = None):
        """Log test results"""
        self.results.append({
            "test_name": test_name,
            "session_id": session_id,
            "query": query,
            "result": str(result)[:200] if result else None,  # Truncate for readability
            "duration": duration,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def print_results_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*80)
        print("CONCURRENT AGENT TEST RESULTS SUMMARY")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            status = "✅ SUCCESS" if not result["error"] else "❌ ERROR"
            print(f"\n{i}. {result['test_name']} - {result['session_id']}")
            print(f"   Status: {status}")
            print(f"   Query: {result['query']}")
            print(f"   Duration: {result['duration']:.2f}s")
            if result["error"]:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Result: {result['result']}...")
        
        # Performance summary
        successful_tests = [r for r in self.results if not r["error"]]
        if successful_tests:
            avg_duration = sum(r["duration"] for r in successful_tests) / len(successful_tests)
            min_duration = min(r["duration"] for r in successful_tests)
            max_duration = max(r["duration"] for r in successful_tests)
            
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"Total tests: {len(self.results)}")
            print(f"Successful: {len(successful_tests)}")
            print(f"Failed: {len(self.results) - len(successful_tests)}")
            print(f"Average duration: {avg_duration:.2f}s")
            print(f"Min duration: {min_duration:.2f}s")
            print(f"Max duration: {max_duration:.2f}s")
    
    def test_sequential_different_sessions(self):
        """Test 1: Sequential execution with different session IDs"""
        print("\n🔄 TEST 1: Sequential execution with different session IDs")
        
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "Tell me about data science"
        ]
        
        for i, query in enumerate(queries):
            session_id = f"session_{i+1}"
            config = {"configurable": {"thread_id": session_id}}
            
            start_time = time.time()
            try:
                result = agent.invoke(query, config)
                duration = time.time() - start_time
                self.log_result("Sequential Different Sessions", session_id, query, result, duration)
                print(f"  ✅ {session_id}: {duration:.2f}s - {query}")
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Sequential Different Sessions", session_id, query, None, duration, str(e))
                print(f"  ❌ {session_id}: {duration:.2f}s - Error: {e}")
    
    def test_sequential_same_session(self):
        """Test 2: Sequential execution with same session ID (conversation memory)"""
        print("\n🔄 TEST 2: Sequential execution with same session ID")
        
        config = {"configurable": {"thread_id": "same_session"}}
        
        queries = [
            "My name is Alice",
            "What's my name?",
            "I like Python programming",
            "What do I like?"
        ]
        
        for i, query in enumerate(queries):
            start_time = time.time()
            try:
                result = agent.invoke(query, config)
                duration = time.time() - start_time
                self.log_result("Sequential Same Session", "same_session", query, result, duration)
                print(f"  ✅ Query {i+1}: {duration:.2f}s - {query}")
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Sequential Same Session", "same_session", query, None, duration, str(e))
                print(f"  ❌ Query {i+1}: {duration:.2f}s - Error: {e}")
    
    def test_concurrent_different_sessions(self):
        """Test 3: Concurrent execution with different session IDs"""
        print("\n🔄 TEST 3: Concurrent execution with different session IDs")
        
        async def run_agent_async(session_id: str, query: str):
            """Async wrapper for agent execution"""
            config = {"configurable": {"thread_id": session_id}}
            start_time = time.time()
            try:
                # Run agent in thread pool since it's not natively async
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.invoke(query, config))
                duration = time.time() - start_time
                self.log_result("Concurrent Different Sessions", session_id, query, result, duration)
                print(f"  ✅ {session_id}: {duration:.2f}s - {query}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Concurrent Different Sessions", session_id, query, None, duration, str(e))
                print(f"  ❌ {session_id}: {duration:.2f}s - Error: {e}")
                return None
        
        async def run_concurrent_test():
            queries = [
                ("session_a", "What is AI?"),
                ("session_b", "What is ML?"),
                ("session_c", "What is DL?"),
                ("session_d", "What is NLP?")
            ]
            
            tasks = [run_agent_async(session_id, query) for session_id, query in queries]
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            print(f"  🕒 Total concurrent execution time: {total_time:.2f}s")
            return results
        
        asyncio.run(run_concurrent_test())
    
    def test_concurrent_same_session(self):
        """Test 4: Concurrent execution with same session ID (potential conflicts)"""
        print("\n🔄 TEST 4: Concurrent execution with same session ID")
        
        def run_agent_sync(query: str, query_id: int):
            """Synchronous wrapper for threading"""
            config = {"configurable": {"thread_id": "shared_session"}}
            start_time = time.time()
            try:
                result = agent.invoke(query, config)
                duration = time.time() - start_time
                self.log_result("Concurrent Same Session", "shared_session", query, result, duration)
                print(f"  ✅ Query {query_id}: {duration:.2f}s - {query}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Concurrent Same Session", "shared_session", query, None, duration, str(e))
                print(f"  ❌ Query {query_id}: {duration:.2f}s - Error: {e}")
                return None
        
        queries = [
            "Tell me about Python",
            "What is JavaScript?",
            "Explain Java programming",
            "What is C++ used for?"
        ]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_agent_sync, query, i+1) for i, query in enumerate(queries)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        print(f"  🕒 Total concurrent execution time: {total_time:.2f}s")
    
    def test_stress_many_sessions(self, num_sessions: int = 10):
        """Test 5: Stress test with many concurrent sessions"""
        print(f"\n🔄 TEST 5: Stress test with {num_sessions} concurrent sessions")
        
        async def run_stress_session(session_id: int):
            """Single stress test session"""
            config = {"configurable": {"thread_id": f"stress_session_{session_id}"}}
            query = f"What is session {session_id} about?"
            
            start_time = time.time()
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.invoke(query, config))
                duration = time.time() - start_time
                self.log_result("Stress Test", f"stress_session_{session_id}", query, result, duration)
                return session_id, True, duration
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Stress Test", f"stress_session_{session_id}", query, None, duration, str(e))
                return session_id, False, duration
        
        async def run_stress_test():
            tasks = [run_stress_session(i) for i in range(num_sessions)]
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = sum(1 for _, success, _ in results if success)
            failed = num_sessions - successful
            avg_duration = sum(duration for _, _, duration in results) / num_sessions
            
            print(f"  🕒 Total stress test time: {total_time:.2f}s")
            print(f"  ✅ Successful sessions: {successful}/{num_sessions}")
            print(f"  ❌ Failed sessions: {failed}/{num_sessions}")
            print(f"  ⏱️ Average session duration: {avg_duration:.2f}s")
        
        asyncio.run(run_stress_test())
    
    def test_streaming_concurrent(self):
        """Test 6: Concurrent streaming execution"""
        print("\n🔄 TEST 6: Concurrent streaming execution")
        
        async def run_streaming_session(session_id: str, query: str):
            """Single streaming session"""
            config = {"configurable": {"thread_id": session_id}}
            start_time = time.time()
            chunks = []
            
            try:
                async for chunk in agent.astream(query, config):
                    chunks.append(str(chunk))
                
                duration = time.time() - start_time
                result = f"Streamed {len(chunks)} chunks"
                self.log_result("Concurrent Streaming", session_id, query, result, duration)
                print(f"  ✅ {session_id}: {duration:.2f}s - {len(chunks)} chunks")
                return chunks
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Concurrent Streaming", session_id, query, None, duration, str(e))
                print(f"  ❌ {session_id}: {duration:.2f}s - Error: {e}")
                return None
        
        async def run_concurrent_streaming():
            queries = [
                ("stream_a", "Explain machine learning in detail"),
                ("stream_b", "What are neural networks and how do they work?"),
                ("stream_c", "Tell me about deep learning applications")
            ]
            
            tasks = [run_streaming_session(session_id, query) for session_id, query in queries]
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            print(f"  🕒 Total concurrent streaming time: {total_time:.2f}s")
        
        asyncio.run(run_concurrent_streaming())
    
    def test_resource_isolation(self):
        """Test 7: Resource isolation between sessions"""
        print("\n🔄 TEST 7: Resource isolation between sessions")
        
        # Session A: Set some context
        config_a = {"configurable": {"thread_id": "isolation_test_a"}}
        config_b = {"configurable": {"thread_id": "isolation_test_b"}}
        
        queries_a = [
            "My favorite color is blue",
            "Remember that I work at TechCorp"
        ]
        
        queries_b = [
            "My favorite color is red", 
            "Remember that I work at DataInc"
        ]
        
        # Set context in both sessions
        print("  Setting context in session A...")
        for query in queries_a:
            start_time = time.time()
            try:
                result = agent.invoke(query, config_a)
                duration = time.time() - start_time
                self.log_result("Resource Isolation A", "isolation_test_a", query, result, duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Resource Isolation A", "isolation_test_a", query, None, duration, str(e))
        
        print("  Setting context in session B...")
        for query in queries_b:
            start_time = time.time()
            try:
                result = agent.invoke(query, config_b)
                duration = time.time() - start_time
                self.log_result("Resource Isolation B", "isolation_test_b", query, result, duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Resource Isolation B", "isolation_test_b", query, None, duration, str(e))
        
        # Test isolation
        isolation_queries = [
            "What is my favorite color?",
            "Where do I work?"
        ]
        
        print("  Testing isolation...")
        for query in isolation_queries:
            # Test session A
            start_time = time.time()
            try:
                result_a = agent.invoke(query, config_a)
                duration = time.time() - start_time
                self.log_result("Isolation Test A", "isolation_test_a", query, result_a, duration)
                print(f"    Session A - {query}: {str(result_a.get('message', ''))[:50]}...")
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Isolation Test A", "isolation_test_a", query, None, duration, str(e))
            
            # Test session B
            start_time = time.time()
            try:
                result_b = agent.invoke(query, config_b)
                duration = time.time() - start_time
                self.log_result("Isolation Test B", "isolation_test_b", query, result_b, duration)
                print(f"    Session B - {query}: {str(result_b.get('message', ''))[:50]}...")
            except Exception as e:
                duration = time.time() - start_time
                self.log_result("Isolation Test B", "isolation_test_b", query, None, duration, str(e))


def main():
    """Main test runner"""
    print("🚀 STARTING CONCURRENT LANGGRAPH AGENT TESTS")
    print("="*80)
    
    tester = ConcurrentAgentTester()
    
    try:
        # Run all tests
        tester.test_sequential_different_sessions()
        tester.test_sequential_same_session()
        tester.test_concurrent_different_sessions()
        tester.test_concurrent_same_session()
        tester.test_stress_many_sessions(5)  # Start with 5 sessions
        tester.test_streaming_concurrent()
        tester.test_resource_isolation()
        
        # Print comprehensive results
        tester.print_results_summary()
        
        # Save results to file
        with open("concurrent_test_results.json", "w") as f:
            json.dump(tester.results, f, indent=2)
        print(f"\n💾 Results saved to concurrent_test_results.json")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
