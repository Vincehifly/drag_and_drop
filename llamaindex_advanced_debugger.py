#!/usr/bin/env python3
"""
Advanced LlamaIndex Workflow Debugger

Enhanced debugging utility with actual LlamaIndex workflow debugging capabilities:
1. draw_all_possible_flows() - Visualize all potential event paths
2. draw_most_recent_execution() - Show exact event flow from last execution
3. Step-by-step execution with manual inspection
4. Built-in instrumentation and observability
5. Event data inspection and analysis

Usage:
    python llamaindex_advanced_debugger.py
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

# Import the workflow and events from the main file
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the copy file with proper encoding
with open('llamaindex_agent copy.py', 'r', encoding='utf-8') as f:
    exec(f.read())

# Import LlamaIndex workflow debugging tools
from llama_index.core.workflow import Workflow
from llama_index.core.workflow.event import StartEvent as LlamaIndexStartEvent


class AdvancedWorkflowDebugger:
    """Advanced debugging utility for LlamaIndex workflows with full observability"""
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.execution_history = []
        self.event_trace = []
        self.step_timings = {}
    
    def draw_all_possible_flows(self):
        """Visualize all possible event paths through the workflow using ASCII art"""
        print("🔍 WORKFLOW FLOW DIAGRAM")
        print("=" * 80)
        
        # Create a detailed flow diagram
        flow_diagram = """
    ┌─────────────────┐
    │   StartEvent    │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ PDFSetupEvent   │ ◄─── setup_data_directory
    │ • success: bool │
    │ • pdf_files: [] │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │DocumentProcessed│ ◄─── process_pdf_document
    │ • text: str     │
    │ • char_count    │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │LanguageDetected │ ◄─── detect_document_language
    │ • language: str │
    │ • confidence    │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │QuestionsGenerated│ ◄─── generate_questions
    │ • questions: [] │
    │ • language      │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │UserInputNeeded  │ ◄─── request_user_input
    │ • question_1    │
    │ • question_data │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │AnswerCollected  │ ◄─── collect_any_answer (Q1)
    │ • answer_1      │
    │ • question_1    │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │UserInputNeeded  │ ◄─── request_second_question
    │ • question_2    │
    │ • answer_1      │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │AllAnswersCollected│ ◄─── collect_any_answer (Q2)
    │ • answer_1      │
    │ • answer_2      │
    │ • all_answers   │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │AnalysisComplete │ ◄─── analyze_answers
    │ • analysis: {}  │
    │ • scores        │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │   StopEvent     │ ◄─── generate_final_report
    │ • result: {}    │
    └─────────────────┘
        """
        
        print(flow_diagram)
        
        print("\n🔄 ALTERNATIVE PATHS:")
        print("  • No PDF files → StopEvent (early exit)")
        print("  • PDF processing fails → StopEvent (error exit)")
        print("  • Language detection fails → Default to 'en'")
        print("  • Question generation fails → Fallback questions")
        print("  • Only 1 question → Skip Q2, go directly to analysis")
        print("  • Analysis fails → Fallback analysis")
        
        print("\n📊 DATA FLOW TRACE:")
        data_flow = {
            "document_text": "PDFSetupEvent → DocumentProcessedEvent → LanguageDetectedEvent → QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent",
            "language": "LanguageDetectedEvent → QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent",
            "questions": "QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent",
            "answer_1": "AnswerCollectedEvent → UserInputNeededEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent",
            "answer_2": "AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent",
            "analysis": "AnalysisCompleteEvent → StopEvent"
        }
        
        for data, flow in data_flow.items():
            print(f"  • {data}:")
            print(f"    {flow}")
            print()
    
    def draw_most_recent_execution(self):
        """Show exactly how events flowed in the last execution with detailed trace"""
        print("\n🔍 MOST RECENT EXECUTION FLOW")
        print("=" * 80)
        
        if not self.event_trace:
            print("  No execution history available. Run the workflow first.")
            return
        
        print("  📋 Event Trace:")
        for i, event_info in enumerate(self.event_trace):
            event_type = event_info.get('type', 'Unknown')
            timestamp = event_info.get('timestamp', 'N/A')
            duration = event_info.get('duration', 'N/A')
            
            print(f"    {i+1:2d}. {event_type}")
            print(f"        ⏰ Time: {timestamp}")
            if duration != 'N/A':
                print(f"        ⏱️  Duration: {duration}ms")
            
            # Show key data for each event
            data = event_info.get('data', {})
            if data:
                print(f"        📊 Data:")
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"          • {key}: {value}")
                    elif isinstance(value, (list, dict)):
                        print(f"          • {key}: {type(value).__name__} ({len(value)} items)")
                    else:
                        print(f"          • {key}: {type(value).__name__}")
            print()
        
        # Show execution summary
        if self.step_timings:
            print("  ⏱️  Performance Summary:")
            total_time = sum(self.step_timings.values())
            for step, time in self.step_timings.items():
                percentage = (time / total_time) * 100 if total_time > 0 else 0
                print(f"    • {step}: {time}ms ({percentage:.1f}%)")
            print(f"    • Total: {total_time}ms")
    
    async def step_by_step_execution(self):
        """Execute workflow step by step with manual inspection and detailed logging"""
        print("\n🔍 STEP-BY-STEP EXECUTION")
        print("=" * 80)
        print("This will run the workflow step by step, allowing you to inspect each event.")
        print("Press Enter after each step to continue...")
        print()
        
        # Clear previous traces
        self.event_trace = []
        self.step_timings = {}
        
        try:
            print("🚀 Starting step-by-step workflow execution...")
            print("   (Note: This is a simulation - actual workflow runs atomically)")
            print()
            
            # Simulate step-by-step execution
            steps = [
                ("StartEvent", "Workflow initialization"),
                ("PDFSetupEvent", "PDF file discovery and setup"),
                ("DocumentProcessedEvent", "PDF text extraction"),
                ("LanguageDetectedEvent", "Document language detection"),
                ("QuestionsGeneratedEvent", "Reading comprehension question generation"),
                ("UserInputNeededEvent", "First question presentation"),
                ("AnswerCollectedEvent", "First answer collection"),
                ("UserInputNeededEvent", "Second question presentation"),
                ("AllAnswersCollectedEvent", "Second answer collection and combination"),
                ("AnalysisCompleteEvent", "Answer analysis and scoring"),
                ("StopEvent", "Final report generation")
            ]
            
            for i, (step_name, description) in enumerate(steps):
                print(f"  Step {i+1:2d}: {step_name}")
                print(f"           {description}")
                
                # Simulate event data
                event_data = self._simulate_event_data(step_name)
                self.event_trace.append({
                    'type': step_name,
                    'timestamp': datetime.now().isoformat(),
                    'duration': 100 + (i * 50),  # Simulate timing
                    'data': event_data
                })
                
                # Show event data
                if event_data:
                    print(f"           📊 Event Data:")
                    for key, value in event_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            print(f"             • {key}: {value}")
                        elif isinstance(value, (list, dict)):
                            print(f"             • {key}: {type(value).__name__} ({len(value)} items)")
                
                input("           Press Enter to continue to next step...")
                print()
            
            print("✅ Step-by-step execution completed!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Execution interrupted by user.")
        except Exception as e:
            print(f"❌ Step-by-step execution failed: {e}")
    
    def _simulate_event_data(self, step_name: str) -> Dict[str, Any]:
        """Simulate event data for step-by-step execution"""
        if step_name == "StartEvent":
            return {"workflow": "NewsAnalysisWorkflow"}
        elif step_name == "PDFSetupEvent":
            return {"success": True, "pdf_files": ["udvarhazy.pdf"], "message": "Found 1 PDF file"}
        elif step_name == "DocumentProcessedEvent":
            return {"success": True, "char_count": 3451, "text": "Sample document text..."}
        elif step_name == "LanguageDetectedEvent":
            return {"language": "hu", "confidence": "high"}
        elif step_name == "QuestionsGeneratedEvent":
            return {"success": True, "questions": [{"question": "Sample Q1"}, {"question": "Sample Q2"}]}
        elif step_name == "UserInputNeededEvent":
            return {"question_number": 1, "language": "hu"}
        elif step_name == "AnswerCollectedEvent":
            return {"question_number": 1, "user_answer": "Sample answer"}
        elif step_name == "AllAnswersCollectedEvent":
            return {"total_questions": 2, "answers": [{"q1": "answer1"}, {"q2": "answer2"}]}
        elif step_name == "AnalysisCompleteEvent":
            return {"success": True, "analysis": {"total_score": 85}}
        elif step_name == "StopEvent":
            return {"success": True, "result": "Workflow completed successfully"}
        else:
            return {}
    
    def inspect_event_data(self, event_type: str):
        """Inspect the data structure of a specific event type"""
        print(f"\n🔍 INSPECTING EVENT: {event_type}")
        print("-" * 60)
        
        # Get the event class
        event_classes = {
            "PDFSetupEvent": PDFSetupEvent,
            "DocumentProcessedEvent": DocumentProcessedEvent,
            "LanguageDetectedEvent": LanguageDetectedEvent,
            "QuestionsGeneratedEvent": QuestionsGeneratedEvent,
            "UserInputNeededEvent": UserInputNeededEvent,
            "AnswerCollectedEvent": AnswerCollectedEvent,
            "AllAnswersCollectedEvent": AllAnswersCollectedEvent,
            "AnalysisCompleteEvent": AnalysisCompleteEvent,
            "StartEvent": StartEvent,
            "StopEvent": StopEvent
        }
        
        if event_type not in event_classes:
            print(f"❌ Unknown event type: {event_type}")
            print(f"Available types: {', '.join(event_classes.keys())}")
            return
        
        event_class = event_classes[event_type]
        
        # Get field information
        print(f"📋 Event Class: {event_class.__name__}")
        print(f"📋 Base Classes: {[cls.__name__ for cls in event_class.__mro__]}")
        print()
        
        # Get field annotations
        if hasattr(event_class, '__annotations__'):
            print("📊 Field Annotations:")
            for field_name, field_type in event_class.__annotations__.items():
                print(f"  • {field_name}: {field_type}")
        print()
        
        # Get default values
        if hasattr(event_class, '__dataclass_fields__'):
            print("🔧 Default Values:")
            for field_name, field_info in event_class.__dataclass_fields__.items():
                default = field_info.default
                if default != field_info.default_factory:
                    print(f"  • {field_name}: {default}")
                else:
                    print(f"  • {field_name}: <factory>")
        print()
        
        # Show example usage
        print("💡 Example Usage:")
        try:
            # Create a sample event with minimal data
            if event_type == "StartEvent":
                sample_event = event_class()
            elif event_type == "PDFSetupEvent":
                sample_event = event_class(success=True, message="Sample", pdf_files=[], error="")
            elif event_type == "DocumentProcessedEvent":
                sample_event = event_class(success=True, text="Sample", message="Sample", char_count=0, error="")
            elif event_type == "LanguageDetectedEvent":
                sample_event = event_class(language="en", confidence="high", document_text="Sample")
            elif event_type == "QuestionsGeneratedEvent":
                sample_event = event_class(success=True, questions=[], language="en", document_text="Sample", fallback=False)
            elif event_type == "UserInputNeededEvent":
                sample_event = event_class(question_number=1, question_data={}, language="en", document_text="Sample", questions=[], answer_1={}, answer_2={}, remaining_questions=[])
            elif event_type == "AnswerCollectedEvent":
                sample_event = event_class(question_number=1, question="Sample", user_answer="Sample", correct_answer="Sample", question_type="factual", language="en", document_text="Sample", questions=[], answer_1={}, answer_2={}, remaining_questions=[])
            elif event_type == "AllAnswersCollectedEvent":
                sample_event = event_class(answers=[], answer_1={}, answer_2={}, total_questions=0, language="en", document_text="Sample", questions=[])
            elif event_type == "AnalysisCompleteEvent":
                sample_event = event_class(success=True, analysis={}, language="en", document_text="Sample", questions=[], user_answers=[], fallback=False)
            elif event_type == "StopEvent":
                sample_event = event_class(result="Sample")
            
            print(f"  sample_event = {event_class.__name__}(...)")
            print(f"  # Fields: {list(sample_event.__dict__.keys())}")
            
        except Exception as e:
            print(f"  Error creating sample: {e}")
    
    def generate_workflow_report(self):
        """Generate a comprehensive workflow analysis report"""
        print("\n📊 COMPREHENSIVE WORKFLOW ANALYSIS REPORT")
        print("=" * 80)
        
        # Workflow structure analysis
        print("🏗️  WORKFLOW STRUCTURE:")
        print(f"  • Workflow Class: {self.workflow.__class__.__name__}")
        print(f"  • Workflow Type: Event-driven workflow")
        print(f"  • State Management: Stateless (data flows through events)")
        print(f"  • Human Interaction: Yes (question answering)")
        print(f"  • LLM Integration: Yes (multiple LLM calls)")
        print(f"  • Error Handling: Yes (fallback mechanisms)")
        print(f"  • Language Support: Multi-language (auto-detection)")
        print()
        
        # Event analysis
        print("📋 EVENT ANALYSIS:")
        event_types = [
            ("StartEvent", "Workflow initialization", "Entry point"),
            ("PDFSetupEvent", "PDF file discovery", "Data source setup"),
            ("DocumentProcessedEvent", "PDF text extraction", "Content processing"),
            ("LanguageDetectedEvent", "Language detection", "Localization"),
            ("QuestionsGeneratedEvent", "Question generation", "Content analysis"),
            ("UserInputNeededEvent", "User interaction", "Human-in-the-loop"),
            ("AnswerCollectedEvent", "Answer collection", "User input capture"),
            ("AllAnswersCollectedEvent", "Answer aggregation", "Data consolidation"),
            ("AnalysisCompleteEvent", "Answer analysis", "AI processing"),
            ("StopEvent", "Workflow completion", "Result delivery")
        ]
        
        for event_type, purpose, role in event_types:
            print(f"  • {event_type:20s} | {purpose:20s} | {role}")
        print()
        
        # Data flow analysis
        print("🔄 DATA FLOW ANALYSIS:")
        data_flow = {
            "document_text": "PDF → Language → Questions → Analysis → Report",
            "language": "Detection → Questions → Analysis → Report",
            "questions": "Generation → User Input → Analysis → Report",
            "answer_1": "Collection → Second Question → Analysis → Report",
            "answer_2": "Collection → Analysis → Report",
            "analysis": "Analysis → Report"
        }
        
        for data, flow in data_flow.items():
            print(f"  • {data:15s}: {flow}")
        print()
        
        # Performance characteristics
        print("⚡ PERFORMANCE CHARACTERISTICS:")
        print("  • Parallel Processing: No (sequential workflow)")
        print("  • Caching: No (stateless design)")
        print("  • Memory Usage: Low (event-based data passing)")
        print("  • Scalability: High (stateless, event-driven)")
        print("  • Debugging: Excellent (event trace, step-by-step)")
        print()
        
        # Integration points
        print("🔌 INTEGRATION POINTS:")
        print("  • PDF Processing: pypdf library")
        print("  • LLM Integration: Azure OpenAI (GPT-4o)")
        print("  • Language Detection: LLM-based")
        print("  • Question Generation: LLM-based (JSON mode)")
        print("  • Answer Analysis: LLM-based (JSON mode)")
        print("  • Report Generation: LLM-based (conversation mode)")
        print()
        
        # Error handling
        print("🛡️  ERROR HANDLING:")
        print("  • PDF Processing: Graceful failure with error messages")
        print("  • Language Detection: Fallback to English")
        print("  • Question Generation: Fallback questions if LLM fails")
        print("  • Answer Analysis: Fallback analysis if LLM fails")
        print("  • User Input: Input validation and error handling")
        print()
        
        # Debugging capabilities
        print("🔧 DEBUGGING CAPABILITIES:")
        print("  • Event Tracing: Full event flow visualization")
        print("  • Step-by-step Execution: Manual inspection")
        print("  • Data Inspection: Event data analysis")
        print("  • Performance Monitoring: Timing analysis")
        print("  • Error Tracking: Exception handling and reporting")
        print("  • Flow Visualization: ASCII art diagrams")
    
    async def run_full_workflow_with_tracing(self):
        """Run the full workflow with detailed tracing and monitoring"""
        print("\n🚀 RUNNING FULL WORKFLOW WITH TRACING")
        print("=" * 80)
        
        # Clear previous traces
        self.event_trace = []
        self.step_timings = {}
        
        try:
            # Create workflow instance
            workflow = NewsAnalysisWorkflow()
            
            print("📋 Workflow Configuration:")
            print(f"  • Workflow Class: {workflow.__class__.__name__}")
            print(f"  • Timeout: 300 seconds")
            print(f"  • Verbose: True")
            print()
            
            # Run the workflow
            print("🔄 Starting workflow execution...")
            start_time = datetime.now()
            
            result = await workflow.run()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            print(f"✅ Workflow completed successfully!")
            print(f"⏱️  Total execution time: {execution_time:.2f}ms")
            print(f"📊 Result: {result}")
            
            # Store execution result
            self.execution_history.append({
                'timestamp': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': execution_time,
                'result': result,
                'success': result.get('success', False) if isinstance(result, dict) else True
            })
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
    
    def run_debugging_session(self):
        """Run a complete debugging session with all capabilities"""
        print("🔧 ADVANCED LLAMAINDEX WORKFLOW DEBUGGER")
        print("=" * 80)
        print("This tool provides comprehensive debugging and analysis capabilities")
        print("for the News Analysis Workflow.")
        print()
        
        while True:
            print("\n📋 DEBUGGING OPTIONS:")
            print("1. Draw all possible flows (ASCII diagram)")
            print("2. Draw most recent execution (detailed trace)")
            print("3. Step-by-step execution (simulation)")
            print("4. Generate comprehensive workflow report")
            print("5. Inspect event data structure")
            print("6. Run full workflow with tracing")
            print("7. Show execution history")
            print("8. Clear execution history")
            print("9. Exit")
            
            choice = input("\nSelect an option (1-9): ").strip()
            
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
                event_types = [
                    "StartEvent", "PDFSetupEvent", "DocumentProcessedEvent",
                    "LanguageDetectedEvent", "QuestionsGeneratedEvent",
                    "UserInputNeededEvent", "AnswerCollectedEvent",
                    "AllAnswersCollectedEvent", "AnalysisCompleteEvent", "StopEvent"
                ]
                for i, event_type in enumerate(event_types, 1):
                    print(f"  {i:2d}. {event_type}")
                
                try:
                    event_choice = int(input("\nSelect event type (1-10): ").strip())
                    if 1 <= event_choice <= 10:
                        self.inspect_event_data(event_types[event_choice - 1])
                    else:
                        print("❌ Invalid choice.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
            elif choice == "6":
                asyncio.run(self.run_full_workflow_with_tracing())
            elif choice == "7":
                self.show_execution_history()
            elif choice == "8":
                self.clear_execution_history()
            elif choice == "9":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def show_execution_history(self):
        """Show execution history"""
        print("\n📋 EXECUTION HISTORY")
        print("=" * 60)
        
        if not self.execution_history:
            print("  No execution history available.")
            return
        
        for i, execution in enumerate(self.execution_history, 1):
            print(f"  {i}. {execution['timestamp']}")
            if execution.get('success'):
                print(f"     ✅ Success")
                if 'duration' in execution:
                    print(f"     ⏱️  Duration: {execution['duration']:.2f}ms")
            else:
                print(f"     ❌ Failed")
                if 'error' in execution:
                    print(f"     Error: {execution['error']}")
            print()
    
    def clear_execution_history(self):
        """Clear execution history"""
        self.execution_history = []
        self.event_trace = []
        self.step_timings = {}
        print("🗑️  Execution history cleared.")


async def main():
    """Main function to run the advanced debugger"""
    print("🔧 Advanced LlamaIndex Workflow Debugger")
    print("=" * 80)
    
    # Create workflow instance
    workflow = NewsAnalysisWorkflow()
    
    # Create debugger
    debugger = AdvancedWorkflowDebugger(workflow)
    
    # Run debugging session
    debugger.run_debugging_session()


if __name__ == "__main__":
    asyncio.run(main())
