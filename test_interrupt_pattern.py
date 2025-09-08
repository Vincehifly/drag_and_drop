#!/usr/bin/env python3
"""
Test Interrupt Pattern with Functional API
Demonstrates proper time travel using interrupt() and Command(resume=...)
"""

from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

@task()
def step_1(input_query):
    """Append bar."""
    print(f"Step 1: Processing '{input_query}'")
    return f"{input_query} bar"

@task()
def human_feedback(input_query):
    """Append user input."""
    print(f"Human feedback needed for: '{input_query}'")
    feedback = interrupt(f"Please provide feedback: {input_query}")
    print(f"Received feedback: {feedback}")
    return f"{input_query} {feedback}"

@task()
def step_3(input_query):
    """Append qux."""
    print(f"Step 3: Processing '{input_query}'")
    return f"{input_query} qux"

@entrypoint(checkpointer=MemorySaver())
def graph(input_query):
    """Main workflow with interrupts."""
    print(f"\n🚀 Starting workflow with: '{input_query}'")
    print("=" * 50)
    
    result_1 = step_1(input_query).result()
    print(f"✅ Step 1 completed: {result_1}")
    
    result_2 = human_feedback(result_1).result()
    print(f"✅ Human feedback completed: {result_2}")
    
    result_3 = step_3(result_2).result()
    print(f"✅ Step 3 completed: {result_3}")
    
    return result_3

def test_interrupt_pattern():
    """Test the interrupt pattern."""
    print("🧪 TESTING INTERRUPT PATTERN")
    print("=" * 50)
    print("This demonstrates time travel with Functional API using interrupts.")
    print()
    
    config = {"configurable": {"thread_id": "test-thread"}}
    
    print("🔄 FIRST RUN (will pause at human_feedback):")
    print("-" * 40)
    
    # First run - will pause at interrupt
    try:
        for event in graph.stream("foo", config):
            print("Event:", event)
            print()
            
            # Check if this is an interrupt event
            if isinstance(event, dict) and "__interrupt__" in event:
                interrupt_obj = event["__interrupt__"][0]
                print(f"🔄 INTERRUPT: {interrupt_obj.value}")
                
                # Get user input for the interrupt
                user_feedback = input("Your feedback: ").strip()
                
                # Resume with the user's feedback
                print("🔄 Resuming with your feedback...")
                for resume_event in graph.stream(Command(resume=user_feedback), config):
                    print("Resume event:", resume_event)
                    print()
                
                print("✅ Workflow completed!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    test_interrupt_pattern()

if __name__ == "__main__":
    main()
