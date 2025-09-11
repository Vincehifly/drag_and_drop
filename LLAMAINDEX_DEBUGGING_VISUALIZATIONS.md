# LlamaIndex Workflow Debugging Visualizations

This document demonstrates the powerful debugging capabilities available in LlamaIndex workflows, specifically `draw_all_possible_flows()` and `draw_most_recent_execution()`.

## 1. draw_all_possible_flows()

This method visualizes all potential event paths through a workflow, showing the complete flow structure and possible execution paths.

### Example Output:

```
🔍 WORKFLOW FLOW DIAGRAM
================================================================================

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

🔄 ALTERNATIVE PATHS:
  • No PDF files → StopEvent (early exit)
  • PDF processing fails → StopEvent (error exit)
  • Language detection fails → Default to 'en'
  • Question generation fails → Fallback questions
  • Only 1 question → Skip Q2, go directly to analysis
  • Analysis fails → Fallback analysis

📊 DATA FLOW TRACE:
  • document_text: PDFSetupEvent → DocumentProcessedEvent → LanguageDetectedEvent → QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent
  • language: LanguageDetectedEvent → QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent
  • questions: QuestionsGeneratedEvent → UserInputNeededEvent → AnswerCollectedEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent
  • answer_1: AnswerCollectedEvent → UserInputNeededEvent → AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent
  • answer_2: AllAnswersCollectedEvent → AnalysisCompleteEvent → StopEvent
  • analysis: AnalysisCompleteEvent → StopEvent
```

### What it shows:
- **Complete workflow structure** with ASCII art diagrams
- **Event relationships** and data flow between steps
- **Alternative execution paths** and error handling routes
- **Data propagation** through the entire workflow
- **Step dependencies** and execution order

## 2. draw_most_recent_execution()

This method shows exactly how events flowed in the last execution, providing detailed trace information with timing and data.

### Example Output:

```
🔍 MOST RECENT EXECUTION FLOW
================================================================================

  📋 Event Trace:
     1. StartEvent
        ⏰ Time: 2025-09-10T12:43:19.123456
        ⏱️  Duration: 5ms
        📊 Data:
          • workflow: NewsAnalysisWorkflow

     2. PDFSetupEvent
        ⏰ Time: 2025-09-10T12:43:19.128456
        ⏱️  Duration: 15ms
        📊 Data:
          • success: True
          • pdf_files: ['udvarhazy.pdf']
          • message: Found 1 PDF file(s) ready for analysis

     3. DocumentProcessedEvent
        ⏰ Time: 2025-09-10T12:43:19.143456
        ⏱️  Duration: 1200ms
        📊 Data:
          • success: True
          • char_count: 3451
          • text: Sample document text...

     4. LanguageDetectedEvent
        ⏰ Time: 2025-09-10T12:43:20.343456
        ⏱️  Duration: 800ms
        📊 Data:
          • language: hu
          • confidence: high
          • document_text: Sample document text...

     5. QuestionsGeneratedEvent
        ⏰ Time: 2025-09-10T12:43:21.143456
        ⏱️  Duration: 2000ms
        📊 Data:
          • success: True
          • questions: List (2 items)
          • language: hu
          • document_text: Sample document text...

     6. UserInputNeededEvent
        ⏰ Time: 2025-09-10T12:43:21.148456
        ⏱️  Duration: 5ms
        📊 Data:
          • question_number: 1
          • language: hu
          • document_text: Sample document text...
          • questions: List (2 items)

     7. AnswerCollectedEvent
        ⏰ Time: 2025-09-10T12:43:51.148456
        ⏱️  Duration: 30000ms (user input)
        📊 Data:
          • question_number: 1
          • user_answer: úttörő
          • correct_answer: Úttörő szerepet töltött be...
          • question_type: main_idea
          • language: hu
          • answer_1: Dict (5 items)

     8. UserInputNeededEvent
        ⏰ Time: 2025-09-10T12:43:51.153456
        ⏱️  Duration: 5ms
        📊 Data:
          • question_number: 2
          • language: hu
          • answer_1: Dict (5 items)
          • questions: List (2 items)

     9. AllAnswersCollectedEvent
        ⏰ Time: 2025-09-10T12:44:16.153456
        ⏱️  Duration: 25000ms (user input)
        📊 Data:
          • total_questions: 2
          • answers: List (2 items)
          • answer_1: Dict (5 items)
          • answer_2: Dict (5 items)
          • language: hu

    10. AnalysisCompleteEvent
        ⏰ Time: 2025-09-10T12:44:19.153456
        ⏱️  Duration: 3000ms
        📊 Data:
          • success: True
          • analysis: Dict (2 items)
          • language: hu
          • questions: List (2 items)
          • user_answers: List (2 items)

    11. StopEvent
        ⏰ Time: 2025-09-10T12:44:19.653456
        ⏱️  Duration: 500ms
        📊 Data:
          • success: True
          • result: Dict (7 items)

  ⏱️  Performance Summary:
    • StartEvent: 5ms (0.0%)
    • PDFSetupEvent: 15ms (0.0%)
    • DocumentProcessedEvent: 1200ms (1.6%)
    • LanguageDetectedEvent: 800ms (1.1%)
    • QuestionsGeneratedEvent: 2000ms (2.7%)
    • UserInputNeededEvent (Q1): 5ms (0.0%)
    • AnswerCollectedEvent: 30000ms (40.5%)
    • UserInputNeededEvent (Q2): 5ms (0.0%)
    • AllAnswersCollectedEvent: 25000ms (33.8%)
    • AnalysisCompleteEvent: 3000ms (4.1%)
    • StopEvent: 500ms (0.7%)
    • Total: 74035ms
```

### What it shows:
- **Exact event sequence** that occurred during execution
- **Timing information** for each step (including user input time)
- **Event data** that was passed between steps
- **Performance breakdown** showing which steps took the most time
- **User interaction points** (where the workflow paused for input)
- **Error locations** (if any failures occurred)

## 3. Additional Debugging Features

### Step-by-Step Execution
```
🔍 STEP-BY-STEP EXECUTION
================================================================================

  Step  1: StartEvent
           Workflow initialization
           📊 Event Data:
             • workflow: NewsAnalysisWorkflow
           Press Enter to continue to next step...

  Step  2: PDFSetupEvent
           PDF file discovery and setup
           📊 Event Data:
             • success: True
             • pdf_files: ['udvarhazy.pdf']
           Press Enter to continue to next step...

  Step  3: DocumentProcessedEvent
           PDF text extraction
           📊 Event Data:
             • success: True
             • char_count: 3451
           Press Enter to continue to next step...
```

### Event Data Inspection
```
🔍 INSPECTING EVENT: AnswerCollectedEvent
------------------------------------------------------------
📋 Event Class: AnswerCollectedEvent
📋 Base Classes: ['AnswerCollectedEvent', 'Event', 'object']

📊 Field Annotations:
  • question_number: int
  • question: str
  • user_answer: str
  • correct_answer: str
  • question_type: str
  • language: str
  • document_text: str
  • questions: List[Dict[str, Any]]
  • answer_1: Dict[str, Any]
  • answer_2: Dict[str, Any]
  • remaining_questions: List[Dict[str, Any]]

🔧 Default Values:
  • question_number: <required>
  • question: <required>
  • user_answer: <required>
  • correct_answer: <required>
  • question_type: <required>
  • language: <required>
  • document_text: <required>
  • questions: []
  • answer_1: {}
  • answer_2: {}
  • remaining_questions: []

💡 Example Usage:
  sample_event = AnswerCollectedEvent(...)
  # Fields: ['question_number', 'question', 'user_answer', 'correct_answer', 'question_type', 'language', 'document_text', 'questions', 'answer_1', 'answer_2', 'remaining_questions']
```

## 4. Usage in Code

```python
from llama_index.core.workflow import Workflow

# Create workflow instance
workflow = NewsAnalysisWorkflow()

# Run the workflow
result = await workflow.run()

# Visualize all possible flows
workflow.draw_all_possible_flows()

# Show the most recent execution
workflow.draw_most_recent_execution()

# Step-by-step execution (if supported)
workflow.step_by_step_execution()
```

## 5. Benefits

### For Development:
- **Debug complex workflows** by seeing exact execution paths
- **Identify performance bottlenecks** through timing analysis
- **Understand data flow** between workflow steps
- **Test alternative paths** and error handling

### For Production:
- **Monitor workflow performance** in real-time
- **Troubleshoot failures** by examining execution traces
- **Optimize workflow efficiency** based on timing data
- **Audit workflow execution** for compliance and debugging

### For Learning:
- **Understand workflow patterns** through visualization
- **Learn event-driven architecture** concepts
- **See data propagation** in action
- **Debug step-by-step** for educational purposes

These debugging capabilities make LlamaIndex workflows highly observable and debuggable, providing developers with powerful tools to understand, optimize, and troubleshoot complex event-driven workflows.

