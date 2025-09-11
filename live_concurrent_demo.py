#!/usr/bin/env python3
"""
Live Concurrent Agent Demo

This script provides a live demonstration of concurrent LangGraph agent sessions.
Run this to see how multiple users can interact with the agent simultaneously.
"""

import asyncio
import time
import random
from typing import List, Tuple

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from functional_api_agent import agent


class LiveConcurrentDemo:
    """Live demonstration of concurrent agent capabilities"""
    
    def __init__(self):
        self.active_sessions = {}
        
    async def simulate_user_session(self, user_id: str, queries: List[str], delay_range: Tuple[float, float] = (1.0, 3.0)):
        """Simulate a user session with multiple queries"""
        session_id = f"demo_{user_id}"
        config = {"configurable": {"thread_id": session_id}}
        
        print(f"👤 {user_id} joined the session")
        self.active_sessions[user_id] = {"queries": 0, "start_time": time.time()}
        
        for i, query in enumerate(queries):
            try:
                # Random delay between queries (simulating user thinking time)
                if i > 0:
                    delay = random.uniform(*delay_range)
                    await asyncio.sleep(delay)
                
                print(f"👤 {user_id}: {query}")
                
                # Execute query
                start_time = time.time()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.invoke(query, config))
                duration = time.time() - start_time
                
                # Show response
                response = result.get('message', 'No response')
                print(f"🤖 → {user_id}: {response[:80]}... ({duration:.1f}s)")
                
                self.active_sessions[user_id]["queries"] += 1
                
            except Exception as e:
                print(f"❌ {user_id}: Error - {e}")
        
        session_duration = time.time() - self.active_sessions[user_id]["start_time"]
        print(f"👋 {user_id} finished session ({session_duration:.1f}s total)")
    
    async def run_demo_scenario(self, scenario_name: str):
        """Run a specific demo scenario"""
        print(f"\n🎬 SCENARIO: {scenario_name}")
        print("="*60)
        
        if scenario_name == "data_science_team":
            # Simulate a data science team asking different questions
            users = [
                ("Alice", [
                    "What is machine learning?",
                    "How do neural networks work?", 
                    "What's the difference between supervised and unsupervised learning?"
                ]),
                ("Bob", [
                    "What is Python used for in data science?",
                    "Explain data preprocessing techniques",
                    "What are the best practices for model evaluation?"
                ]),
                ("Charlie", [
                    "What is deep learning?",
                    "Tell me about natural language processing",
                    "How does computer vision work?"
                ])
            ]
            
        elif scenario_name == "rapid_questions":
            # Simulate rapid-fire questions from different users
            users = [
                ("User1", ["What is AI?", "Define ML", "Explain DL"]),
                ("User2", ["Hello", "What's 2+2?", "Tell me about Python"]),
                ("User3", ["Hi there", "What is data science?", "Goodbye"]),
                ("User4", ["Quick question", "What is statistics?", "Thanks"])
            ]
            
        elif scenario_name == "mixed_complexity":
            # Mix of simple and complex queries
            users = [
                ("Beginner", [
                    "Hi, I'm new to AI",
                    "What should I learn first?",
                    "Any good resources?"
                ]),
                ("Expert", [
                    "Compare transformer architectures with CNNs",
                    "Explain gradient descent optimization techniques", 
                    "What are the latest developments in reinforcement learning?"
                ]),
                ("Student", [
                    "I have an exam tomorrow",
                    "Can you help me understand linear regression?",
                    "What about logistic regression?"
                ])
            ]
        else:
            print("Unknown scenario")
            return
        
        # Run all user sessions concurrently
        start_time = time.time()
        tasks = [self.simulate_user_session(user_id, queries) for user_id, queries in users]
        await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Summary
        total_queries = sum(len(queries) for _, queries in users)
        print(f"\n📊 SCENARIO COMPLETE")
        print(f"Users: {len(users)}")
        print(f"Total queries: {total_queries}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per query: {total_time/total_queries:.1f}s")
    
    async def interactive_demo(self):
        """Interactive demo where you can add users and queries"""
        print("\n🎮 INTERACTIVE CONCURRENT DEMO")
        print("="*60)
        print("Commands:")
        print("  add <user_id> <query> - Add a query for a user")
        print("  run - Execute all queued queries concurrently")
        print("  clear - Clear all queued queries")
        print("  quit - Exit")
        
        queued_queries = {}
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                    
                elif command.lower() == 'run':
                    if queued_queries:
                        print(f"\n🚀 Running {len(queued_queries)} concurrent user sessions...")
                        tasks = [
                            self.simulate_user_session(user_id, queries, (0.5, 1.5))
                            for user_id, queries in queued_queries.items()
                        ]
                        await asyncio.gather(*tasks, return_exceptions=True)
                        queued_queries.clear()
                    else:
                        print("No queries queued. Use 'add <user> <query>' first.")
                        
                elif command.lower() == 'clear':
                    queued_queries.clear()
                    print("Cleared all queued queries.")
                    
                elif command.startswith('add '):
                    parts = command[4:].split(' ', 1)
                    if len(parts) == 2:
                        user_id, query = parts
                        if user_id not in queued_queries:
                            queued_queries[user_id] = []
                        queued_queries[user_id].append(query)
                        print(f"Added query for {user_id}: {query}")
                    else:
                        print("Usage: add <user_id> <query>")
                        
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main demo function"""
    print("🎭 LIVE CONCURRENT LANGGRAPH AGENT DEMO")
    print("="*80)
    
    demo = LiveConcurrentDemo()
    
    print("\nChoose a demo:")
    print("1. Data Science Team Simulation")
    print("2. Rapid Questions Simulation") 
    print("3. Mixed Complexity Simulation")
    print("4. Interactive Demo")
    print("5. All Scenarios")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            await demo.run_demo_scenario("data_science_team")
            
        elif choice == "2":
            await demo.run_demo_scenario("rapid_questions")
            
        elif choice == "3":
            await demo.run_demo_scenario("mixed_complexity")
            
        elif choice == "4":
            await demo.interactive_demo()
            
        elif choice == "5":
            # Run all scenarios
            scenarios = ["data_science_team", "rapid_questions", "mixed_complexity"]
            for scenario in scenarios:
                await demo.run_demo_scenario(scenario)
                print("\n" + "⏳ Waiting 3 seconds before next scenario...\n")
                await asyncio.sleep(3)
                
        else:
            print("Invalid choice. Running data science team demo...")
            await demo.run_demo_scenario("data_science_team")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
    
    print("\n🎭 Demo completed! Thank you for testing concurrent agent sessions.")


if __name__ == "__main__":
    asyncio.run(main())
