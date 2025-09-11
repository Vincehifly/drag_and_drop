#!/usr/bin/env python3
"""
Multiple Workflow Concurrency Test

This test specifically examines whether multiple LangGraph workflows can run 
simultaneously without interference, simulating an API-based architecture 
where multiple users make concurrent requests.

Focus: Workflow isolation, resource contention, and concurrent execution safety.
"""

import asyncio
import time
import threading
from typing import Dict, List, Tuple
from datetime import datetime
import json

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from functional_api_agent import agent


class WorkflowConcurrencyTest:
    """Test multiple workflow execution concurrency"""
    
    def __init__(self):
        self.results = []
        self.active_workflows = {}
        self.lock = threading.Lock()
    
    def log_workflow_event(self, workflow_id: str, event: str, data: Dict = None):
        """Thread-safe logging of workflow events"""
        with self.lock:
            self.results.append({
                "workflow_id": workflow_id,
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.get_ident(),
                "data": data or {}
            })
    
    async def execute_single_workflow(self, workflow_id: str, query: str, expected_duration: float = None):
        """
        Execute a single workflow and monitor its behavior
        
        Args:
            workflow_id: Unique identifier for this workflow instance
            query: The query to process
            expected_duration: Expected execution time (for monitoring)
        """
        # Create unique session for this workflow
        session_id = f"workflow_{workflow_id}_{int(time.time()*1000)}"
        config = {"configurable": {"thread_id": session_id}}
        
        self.log_workflow_event(workflow_id, "workflow_started", {
            "query": query,
            "session_id": session_id,
            "expected_duration": expected_duration
        })
        
        start_time = time.time()
        
        try:
            # Execute the workflow
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: agent.invoke(query, config))
            
            duration = time.time() - start_time
            
            self.log_workflow_event(workflow_id, "workflow_completed", {
                "duration": duration,
                "success": True,
                "response_length": len(result.get('message', '')),
                "tool_used": result.get('tool'),
                "deviation": abs(duration - expected_duration) if expected_duration else None
            })
            
            return {
                "workflow_id": workflow_id,
                "success": True,
                "duration": duration,
                "result": result
            }
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.log_workflow_event(workflow_id, "workflow_failed", {
                "duration": duration,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            return {
                "workflow_id": workflow_id,
                "success": False,
                "duration": duration,
                "error": str(e)
            }
    
    async def test_concurrent_workflows(self):
        """
        Test multiple workflows running concurrently
        
        This simulates an API server receiving multiple requests simultaneously
        """
        print("🔄 TESTING CONCURRENT WORKFLOW EXECUTION")
        print("="*60)
        
        # Define different workflow scenarios (simulating different API endpoints/users)
        workflows = [
            # Fast queries (simulate simple API calls)
            ("fast_1", "Hello", 1.0),
            ("fast_2", "What is 2+2?", 1.0),
            ("fast_3", "Hi there", 1.0),
            
            # Medium complexity (simulate typical API usage)
            ("medium_1", "What is machine learning?", 3.0),
            ("medium_2", "Explain Python programming", 3.0),
            
            # Complex queries (simulate heavy API usage)
            ("complex_1", "Compare machine learning and deep learning approaches in detail", 5.0),
            ("complex_2", "What are the latest developments in artificial intelligence?", 5.0),
        ]
        
        print(f"Launching {len(workflows)} concurrent workflows...")
        
        # Record overall test start
        test_start = time.time()
        self.log_workflow_event("test_master", "concurrent_test_started", {
            "workflow_count": len(workflows),
            "workflow_types": ["fast", "medium", "complex"]
        })
        
        # Launch all workflows concurrently (simulating simultaneous API requests)
        tasks = [
            self.execute_single_workflow(wf_id, query, expected_duration)
            for wf_id, query, expected_duration in workflows
        ]
        
        # Wait for all workflows to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        test_duration = time.time() - test_start
        
        self.log_workflow_event("test_master", "concurrent_test_completed", {
            "total_duration": test_duration,
            "results_count": len(results)
        })
        
        # Analyze results
        successful_workflows = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_workflows = [r for r in results if isinstance(r, dict) and not r.get('success')]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"\n📊 CONCURRENT EXECUTION RESULTS:")
        print(f"Total execution time: {test_duration:.2f}s")
        print(f"Successful workflows: {len(successful_workflows)}/{len(workflows)}")
        print(f"Failed workflows: {len(failed_workflows)}")
        print(f"Exceptions: {len(exceptions)}")
        
        if successful_workflows:
            durations = [r['duration'] for r in successful_workflows]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            print(f"Average workflow duration: {avg_duration:.2f}s")
            print(f"Fastest workflow: {min_duration:.2f}s")
            print(f"Slowest workflow: {max_duration:.2f}s")
            
            # Calculate theoretical sequential time
            theoretical_sequential = sum(durations)
            concurrent_efficiency = theoretical_sequential / test_duration if test_duration > 0 else 0
            
            print(f"Concurrent efficiency: {concurrent_efficiency:.2f}x speedup")
        
        return {
            "total_duration": test_duration,
            "successful": len(successful_workflows),
            "failed": len(failed_workflows),
            "efficiency": concurrent_efficiency if successful_workflows else 0
        }
    
    async def test_workflow_isolation(self):
        """
        Test that workflows don't interfere with each other's state
        """
        print("\n🔒 TESTING WORKFLOW ISOLATION")
        print("="*60)
        
        # Create workflows that modify session state
        isolation_workflows = [
            ("isolate_A", "My name is Alice and I work at TechCorp"),
            ("isolate_B", "My name is Bob and I work at DataInc"), 
            ("isolate_C", "My name is Charlie and I study at University")
        ]
        
        print("Setting up isolated workflow states...")
        
        # Set up states concurrently
        setup_tasks = [
            self.execute_single_workflow(wf_id, setup_query)
            for wf_id, setup_query in isolation_workflows
        ]
        
        setup_results = await asyncio.gather(*setup_tasks)
        
        # Test isolation by querying each workflow's memory
        print("Testing workflow memory isolation...")
        
        test_tasks = [
            self.execute_single_workflow(f"{wf_id}_test", "What is my name and where do I work?")
            for wf_id, _ in isolation_workflows
        ]
        
        test_results = await asyncio.gather(*test_tasks)
        
        # Analyze isolation
        print(f"\n🔍 ISOLATION ANALYSIS:")
        
        isolation_success = True
        for i, (wf_id, original_info) in enumerate(isolation_workflows):
            if i < len(test_results) and test_results[i]['success']:
                response = test_results[i]['result'].get('message', '').lower()
                
                # Check if the workflow remembers its own info
                expected_name = original_info.split()[3].lower()  # Extract name
                remembers_own = expected_name in response
                
                # Check for contamination from other workflows
                other_names = [info.split()[3].lower() for j, (_, info) in enumerate(isolation_workflows) if j != i]
                contaminated = any(name in response for name in other_names)
                
                status = "✅" if remembers_own and not contaminated else "❌"
                print(f"  {status} {wf_id}: Remembers own info: {remembers_own}, Contaminated: {contaminated}")
                
                if not remembers_own or contaminated:
                    isolation_success = False
        
        print(f"\nWorkflow isolation: {'✅ PASSED' if isolation_success else '❌ FAILED'}")
        
        return {"isolation_success": isolation_success}
    
    async def test_resource_contention(self):
        """
        Test behavior under resource contention (many simultaneous workflows)
        """
        print("\n⚡ TESTING RESOURCE CONTENTION")
        print("="*60)
        
        # Create many lightweight workflows simultaneously
        contention_count = 15
        lightweight_query = "What is AI?"
        
        print(f"Launching {contention_count} simultaneous workflows...")
        
        start_time = time.time()
        
        contention_tasks = [
            self.execute_single_workflow(f"contention_{i}", lightweight_query, 2.0)
            for i in range(contention_count)
        ]
        
        results = await asyncio.gather(*contention_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = contention_count - successful
        
        print(f"\n⚡ RESOURCE CONTENTION RESULTS:")
        print(f"Simultaneous workflows: {contention_count}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Successful: {successful}/{contention_count}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
            durations = [r['duration'] for r in successful_results]
            avg_duration = sum(durations) / len(durations)
            
            print(f"Average workflow duration under contention: {avg_duration:.2f}s")
            
            # Calculate contention overhead
            baseline_duration = 2.0  # Expected duration for this query type
            overhead = (avg_duration - baseline_duration) / baseline_duration * 100
            print(f"Contention overhead: {overhead:.1f}%")
        
        return {
            "simultaneous_workflows": contention_count,
            "successful": successful,
            "total_time": total_time,
            "success_rate": successful / contention_count
        }
    
    def generate_detailed_report(self):
        """Generate a detailed report of all workflow events"""
        print("\n📋 DETAILED WORKFLOW EVENT LOG")
        print("="*60)
        
        # Group events by workflow
        workflow_events = {}
        for event in self.results:
            wf_id = event['workflow_id']
            if wf_id not in workflow_events:
                workflow_events[wf_id] = []
            workflow_events[wf_id].append(event)
        
        # Print summary for each workflow
        for workflow_id, events in workflow_events.items():
            if workflow_id == "test_master":
                continue
                
            print(f"\n🔄 Workflow: {workflow_id}")
            
            start_event = next((e for e in events if e['event'] == 'workflow_started'), None)
            end_event = next((e for e in events if e['event'] in ['workflow_completed', 'workflow_failed']), None)
            
            if start_event and end_event:
                print(f"  Query: {start_event['data'].get('query', 'N/A')}")
                print(f"  Thread ID: {start_event['thread_id']}")
                print(f"  Duration: {end_event['data'].get('duration', 'N/A'):.2f}s")
                print(f"  Status: {'✅ Success' if end_event['event'] == 'workflow_completed' else '❌ Failed'}")
                
                if end_event['event'] == 'workflow_completed':
                    tool_used = end_event['data'].get('tool_used')
                    if tool_used:
                        print(f"  Tool used: {tool_used}")


async def main():
    """Run the complete workflow concurrency test"""
    print("🧪 MULTIPLE WORKFLOW CONCURRENCY TEST")
    print("="*80)
    print("This test examines whether multiple LangGraph workflows can run")
    print("simultaneously in an API-based architecture without interference.")
    print("="*80)
    
    test_runner = WorkflowConcurrencyTest()
    
    try:
        # Test 1: Basic concurrent execution
        concurrent_results = await test_runner.test_concurrent_workflows()
        
        # Test 2: Workflow isolation
        isolation_results = await test_runner.test_workflow_isolation()
        
        # Test 3: Resource contention
        contention_results = await test_runner.test_resource_contention()
        
        # Generate detailed report
        test_runner.generate_detailed_report()
        
        # Final summary
        print("\n" + "="*80)
        print("🎯 FINAL TEST SUMMARY")
        print("="*80)
        
        print(f"\n✅ CONCURRENT EXECUTION:")
        print(f"  • Efficiency: {concurrent_results['efficiency']:.2f}x speedup")
        print(f"  • Success rate: {concurrent_results['successful']}/{concurrent_results['successful'] + concurrent_results['failed']}")
        
        print(f"\n✅ WORKFLOW ISOLATION:")
        print(f"  • Isolation working: {'✅ Yes' if isolation_results['isolation_success'] else '❌ No'}")
        
        print(f"\n✅ RESOURCE CONTENTION:")
        print(f"  • Success rate under load: {contention_results['success_rate']*100:.1f}%")
        print(f"  • Handled {contention_results['simultaneous_workflows']} simultaneous workflows")
        
        print(f"\n🚀 API ARCHITECTURE READINESS:")
        overall_success = (
            concurrent_results['efficiency'] > 1.0 and
            concurrent_results['successful'] > 0 and
            contention_results['success_rate'] > 0.8
        )
        
        print(f"  • Multiple workflows can run concurrently: {'✅ YES' if overall_success else '❌ NO'}")
        print(f"  • Suitable for API-based architecture: {'✅ YES' if overall_success else '❌ NO'}")
        print(f"  • Recommended max concurrent workflows: {int(contention_results['simultaneous_workflows'] * contention_results['success_rate'])}")
        
        # Save results to file
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "concurrent_execution": concurrent_results,
            "workflow_isolation": isolation_results,
            "resource_contention": contention_results,
            "overall_success": overall_success,
            "detailed_events": test_runner.results
        }
        
        with open("workflow_concurrency_test_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: workflow_concurrency_test_results.json")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
