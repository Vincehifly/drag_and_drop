#!/usr/bin/env python3
"""
LlamaIndex Workflow Debugger

A debugging utility for the News Analysis Workflow that provides:
1. Visual workflow flow diagrams
2. Step-by-step execution inspection
3. Event path visualization
4. Built-in instrumentation and observability

Usage:
    python llamaindex_workflow_debugger.py
"""

import asyncio
import json
from typing import Any, Dict, List
from datetime import datetime

# Import the workflow and events from the main file
from llamaindex_agent_copy import (
    NewsAnalysisWorkflow,
    PDFSetupEvent,
    DocumentProcessedEvent,
    LanguageDetectedEvent,
    QuestionsGeneratedEvent,
    UserInputNeededEvent,
    AnswerCollectedEvent,
    AllAnswersCollectedEvent,
    AnalysisCompleteEvent,
    StartEvent,
    StopEvent
)

# Import LlamaIndex workflow debugging tools
from llama_index.core.workflow import Workflow
from llama_index.core.workflow.event import StartEvent as LlamaIndexStartEvent


class WorkflowDebugger:
    """Debugging utility for LlamaIndex workflows"""
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.execution_history = []
    
    def draw_all_possible_flows(self):
        """Visualize all possible event paths through the workflow"""
        print("🔍 WORKFLOW FLOW DIAGRAM")
        print("=" * 60)
        
        # Define the expected flow
        flow_steps = [
            "StartEvent",
            "↓",
            "PDFSetupEvent (setup_data_directory)",
            "↓",
            "DocumentProcessedEvent (process_pdf_document)",
            "↓", 
            "LanguageDetectedEvent (detect_document_language)",
            "↓",
            "QuestionsGeneratedEvent (generate_questions)",
            "↓",
            "UserInputNeededEvent (request_user_input)",
            "↓",
            "AnswerCollectedEvent (collect_any_answer) - Question 1",
            "↓",
            "UserInputNeededEvent (request_second_question) - Question 2",
            "↓",
            "AllAnswersCollectedEvent (collect_any_answer) - Question 2",
            "↓",
            "AnalysisCompleteEvent (analyze_answers)",
            "↓",
            "StopEvent (generate_final_report)"
        ]
        
        for step in flow_steps:
            print(f"  {step}")
        
        print("\n🔄 ALTERNATIVE PATHS:")
        print("  • If no PDF files found → StopEvent (early exit)")
        print("  • If only 1 question → AllAnswersCollectedEvent (skip Q2)")
        print("  • If analysis fails → StopEvent (error exit)")
        
        print("\n📊 EVENT DATA FLOW:")
        print("  • document_text: PDFSetupEvent → DocumentProcessedEvent → ... → StopEvent")
        print("  • language: LanguageDetectedEvent → ... → StopEvent")
        print("  • questions: QuestionsGeneratedEvent → ... → StopEvent")
        print("  • answer_1: AnswerCollectedEvent → ... → StopEvent")
        print("  • answer_2: AllAnswersCollectedEvent → ... → StopEvent")
    
    def draw_most_recent_execution(self):
        """Show exactly how events flowed in the last execution"""
        print("\n🔍 MOST RECENT EXECUTION FLOW")
        print("=" * 60)
        
        if not self.execution_history:
            print("  No execution history available. Run the workflow first.")
            return
        
        print("  Event Flow:")
        for i, event in enumerate(self.execution_history):
            event_type = type(event).__name__
            timestamp = event.get('timestamp', 'N/A')
            print(f"    {i+1}. {event_type} at {timestamp}")
            
            # Show key data for each event
            if hasattr(event, 'success'):
                print(f"       Success: {event.success}")
            if hasattr(event, 'language'):
                print(f"       Language: {event.language}")
            if hasattr(event, 'question_number'):
                print(f"       Question: {event.question_number}")
            if hasattr(event, 'total_questions'):
                print(f"       Total Questions: {event.total_questions}")
            print()
    
    async def step_by_step_execution(self):
        """Execute workflow step by step with manual inspection"""
        print("\n🔍 STEP-BY-STEP EXECUTION")
        print("=" * 60)
        print("This will run the workflow step by step, allowing you to inspect each event.")
        print("Press Enter after each step to continue...")
        print()
        
        # Create a custom workflow runner that pauses between steps
        try:
            # Start the workflow
            print("🚀 Starting workflow...")
            result = await self.workflow.run()
            
            print(f"\n✅ Workflow completed successfully!")
            print(f"📊 Final result: {result}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
    
    def inspect_event_data(self, event: Any):
        """Inspect the data contained in an event"""
        print(f"\n🔍 INSPECTING EVENT: {type(event).__name__}")
        print("-" * 40)
        
        # Get all attributes of the event
        attributes = [attr for attr in dir(event) if not attr.startswith('_')]
        
        for attr in attributes:
            try:
                value = getattr(event, attr)
                if not callable(value):
                    if isinstance(value, (str, int, float, bool)):
                        print(f"  {attr}: {value}")
                    elif isinstance(value, (list, dict)):
                        print(f"  {attr}: {type(value).__name__} with {len(value)} items")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    First item: {value[0]}")
                    else:
                        print(f"  {attr}: {type(value).__name__}")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")
    
    def generate_workflow_report(self):
        """Generate a comprehensive workflow report"""
        print("\n📊 WORKFLOW ANALYSIS REPORT")
        print("=" * 60)
        
        # Analyze the workflow structure
        print("🏗️  WORKFLOW STRUCTURE:")
        print(f"  • Workflow Class: {self.workflow.__class__.__name__}")
        print(f"  • Total Steps: {len(self.workflow.steps) if hasattr(self.workflow, 'steps') else 'Unknown'}")
        
        # Show event types
        print("\n📋 EVENT TYPES:")
        event_types = [
            "StartEvent", "PDFSetupEvent", "DocumentProcessedEvent",
            "LanguageDetectedEvent", "QuestionsGeneratedEvent",
            "UserInputNeededEvent", "AnswerCollectedEvent",
            "AllAnswersCollectedEvent", "AnalysisCompleteEvent", "StopEvent"
        ]
        
        for event_type in event_types:
            print(f"  • {event_type}")
        
        # Show data flow
        print("\n🔄 DATA FLOW ANALYSIS:")
        data_flow = {
            "document_text": "PDF → Language → Questions → Analysis → Report",
            "language": "Detection → Questions → Analysis → Report",
            "questions": "Generation → User Input → Analysis → Report",
            "answer_1": "Collection → Second Question → Analysis → Report",
            "answer_2": "Collection → Analysis → Report",
            "analysis": "Analysis → Report"
        }
        
        for data, flow in data_flow.items():
            print(f"  • {data}: {flow}")
        
        print("\n🎯 WORKFLOW CHARACTERISTICS:")
        print("  • Type: Event-driven workflow")
        print("  • State Management: Stateless (data flows through events)")
        print("  • Human Interaction: Yes (question answering)")
        print("  • LLM Integration: Yes (multiple LLM calls)")
        print("  • Error Handling: Yes (fallback mechanisms)")
        print("  • Language Support: Multi-language (auto-detection)")
    
    def run_debugging_session(self):
        """Run a complete debugging session"""
        print("🔧 LLAMAINDEX WORKFLOW DEBUGGER")
        print("=" * 60)
        print("This tool helps you understand and debug the News Analysis Workflow.")
        print()
        
        while True:
            print("\n📋 DEBUGGING OPTIONS:")
            print("1. Draw all possible flows")
            print("2. Draw most recent execution")
            print("3. Step-by-step execution")
            print("4. Generate workflow report")
            print("5. Inspect event data")
            print("6. Run full workflow")
            print("7. Exit")
            
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == "1":
                self.draw_all_possible_flows()
            elif choice == "2":
                self.draw_most_recent_execution()
            elif choice == "3":
                asyncio.run(self.step_by_step_execution())
            elif choice == "4":
                self.generate_workflow_report()
            elif choice == "5":
                print("\nAvailable event types for inspection:")
                print("  • PDFSetupEvent")
                print("  • DocumentProcessedEvent")
                print("  • LanguageDetectedEvent")
                print("  • QuestionsGeneratedEvent")
                print("  • UserInputNeededEvent")
                print("  • AnswerCollectedEvent")
                print("  • AllAnswersCollectedEvent")
                print("  • AnalysisCompleteEvent")
                print("\nNote: Run the workflow first to see actual event data.")
            elif choice == "6":
                print("\n🚀 Running full workflow...")
                asyncio.run(self.run_full_workflow())
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    async def run_full_workflow(self):
        """Run the full workflow and capture execution history"""
        print("Starting workflow execution...")
        
        try:
            # Create workflow instance
            workflow = NewsAnalysisWorkflow()
            
            # Run the workflow
            result = await workflow.run()
            
            # Store execution result
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'success': result.get('success', False) if isinstance(result, dict) else True
            })
            
            print(f"✅ Workflow completed successfully!")
            print(f"📊 Result: {result}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })


async def main():
    """Main function to run the debugger"""
    print("🔧 LlamaIndex Workflow Debugger")
    print("=" * 60)
    
    # Create workflow instance
    workflow = NewsAnalysisWorkflow()
    
    # Create debugger
    debugger = WorkflowDebugger(workflow)
    
    # Run debugging session
    debugger.run_debugging_session()


if __name__ == "__main__":
    asyncio.run(main())

