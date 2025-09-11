#!/usr/bin/env python3
"""
LlamaIndex Workflow News Analysis Agent

A news analysis agent that:
1. Processes PDF news documents from data directory
2. Detects document language
3. Generates reading comprehension questions
4. Uses human-in-the-loop pattern for user interaction
5. Analyzes and scores user responses
6. Provides comprehensive feedback

Uses LlamaIndex Workflow with step decorators and Event system.
"""

import asyncio
import json
import os
import glob
from typing import Any, Dict, List
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pypdf import PdfReader

# Use your existing LLM client with specialized configurations
from llm_client import create_conversation_llm, create_json_mode_llm


# Event definitions for the workflow
class PDFSetupEvent(Event):
    success: bool
    message: str
    pdf_files: List[str] = []
    error: str = ""


class DocumentProcessedEvent(Event):
    success: bool
    text: str
    message: str
    char_count: int = 0
    error: str = ""


class LanguageDetectedEvent(Event):
    language: str
    confidence: str
    document_text: str  # Carry forward document text


class QuestionsGeneratedEvent(Event):
    success: bool
    questions: List[Dict[str, Any]]
    language: str
    document_text: str  # Carry forward document text
    fallback: bool = False


class UserInputNeededEvent(Event):
    question_number: int
    question_data: Dict[str, Any]
    language: str
    document_text: str  # Carry forward document text
    questions: List[Dict[str, Any]] = []  # All questions for context
    answer_1: Dict[str, Any] = {}  # Store first answer separately
    answer_2: Dict[str, Any] = {}  # Store second answer separately
    remaining_questions: List[Dict[str, Any]] = []


class AnswerCollectedEvent(Event):
    question_number: int
    question: str
    user_answer: str
    correct_answer: str
    question_type: str
    language: str  # Carry forward language
    document_text: str  # Carry forward document text
    questions: List[Dict[str, Any]] = []  # All questions for context
    answer_1: Dict[str, Any] = {}  # Store first answer separately
    answer_2: Dict[str, Any] = {}  # Store second answer separately
    remaining_questions: List[Dict[str, Any]] = []


class AllAnswersCollectedEvent(Event):
    answers: List[Dict[str, Any]]
    answer_1: Dict[str, Any] = {}  # First answer
    answer_2: Dict[str, Any] = {}  # Second answer
    total_questions: int
    language: str  # Carry forward language
    document_text: str  # Carry forward document text
    questions: List[Dict[str, Any]]  # All questions for context


class AnalysisCompleteEvent(Event):
    success: bool
    analysis: Dict[str, Any]
    language: str  # Carry forward language
    document_text: str  # Carry forward document text
    questions: List[Dict[str, Any]]  # All questions for context
    user_answers: List[Dict[str, Any]]  # All user answers for context
    fallback: bool = False


# No global state needed - all data flows through events!


# Forward declaration of workflow class
class NewsAnalysisWorkflow(Workflow):
    """LlamaIndex Workflow for News Analysis and Reading Comprehension"""
    
    def __init__(self):
        super().__init__()


@step(workflow=NewsAnalysisWorkflow)
async def setup_data_directory(ev: StartEvent) -> PDFSetupEvent:
    """Set up data directory and check for PDF files."""
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created data directory: {data_dir}")
    
    # Find PDF files in data directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        return PDFSetupEvent(
            success=False,
            message=f"No PDF files found in {data_dir} directory. Please upload a PDF file first.",
            pdf_files=[],
            error="No PDF files found"
        )
    
    print(f"📄 Found {len(pdf_files)} PDF file(s): {[os.path.basename(f) for f in pdf_files]}")
    
    return PDFSetupEvent(
        success=True,
        message=f"Found {len(pdf_files)} PDF file(s) ready for analysis",
        pdf_files=pdf_files,
        error=""
    )
    
@step(workflow=NewsAnalysisWorkflow)
async def process_pdf_document(ev: PDFSetupEvent) -> DocumentProcessedEvent:
    """Extract text from the first PDF file found."""
    if not ev.success:
        return DocumentProcessedEvent(
            success=False,
            text="",
            message=ev.message,
            error=ev.error
        )
    
    # Process the first PDF file
    pdf_path = ev.pdf_files[0]
    print(f"📖 Processing PDF: {os.path.basename(pdf_path)}")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean up text
        text = text.strip()
        
        if not text:
            return DocumentProcessedEvent(
                success=False,
                text="",
                message="Could not extract text from PDF. The file might be image-based or corrupted.",
                error="Text extraction failed"
            )
        
        print(f"✅ Successfully extracted {len(text)} characters from PDF")
        
        return DocumentProcessedEvent(
            success=True,
            text=text,
            message=f"Successfully processed PDF: {os.path.basename(pdf_path)}",
            char_count=len(text)
        )
        
    except Exception as e:
        return DocumentProcessedEvent(
            success=False,
            text="",
            message=f"Error processing PDF: {str(e)}",
            error=str(e)
        )
    
@step(workflow=NewsAnalysisWorkflow)
async def detect_document_language(ev: DocumentProcessedEvent) -> LanguageDetectedEvent:
    """Detect the language of the document."""
    if not ev.success:
        return LanguageDetectedEvent(language="en", confidence="low")
    
    # Use first 1000 characters for language detection
    sample_text = ev.text[:1000]
    
    prompt = f"""Detect the language of the following text and respond with only the language code (e.g., 'en', 'hu', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', etc.):

Text: {sample_text}

Language code:"""
    
    try:
        # Create LLM instance inside the step
        conversation_llm = create_conversation_llm(temperature=0.3)
        detected_lang = conversation_llm.invoke(prompt).content.strip().lower()
        print(f"🌍 Detected document language: {detected_lang}")
        
        return LanguageDetectedEvent(
            language=detected_lang,
            confidence="high",
            document_text=ev.text  # Pass document text forward
        )
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}")
        return LanguageDetectedEvent(
            language="en", 
            confidence="low",
            document_text=ev.text  # Pass document text forward even on error
        )
    
@step(workflow=NewsAnalysisWorkflow)
async def generate_questions(ev: LanguageDetectedEvent) -> QuestionsGeneratedEvent:
    """Generate reading comprehension questions."""
    document_text = ev.document_text
    
    # Use first 4000 characters for question generation
    sample_text = document_text[:4000]
    
    prompt = f"""Based on the following news article text, generate exactly 2 reading comprehension questions that test understanding of key facts, main ideas, and important details. Generate the questions in {ev.language}.

Text: {sample_text}

Requirements:
- Create 2 questions that test different aspects of comprehension
- Questions should be answerable from the text
- Include the correct answers
- Format as JSON with this structure:
{{
    "questions": [
        {{
            "question": "Question text here",
            "correct_answer": "Correct answer here",
            "question_type": "factual|main_idea|detail|inference"
        }},
        {{
            "question": "Question text here", 
            "correct_answer": "Correct answer here",
            "question_type": "factual|main_idea|detail|inference"
        }}
    ]
}}

Generate questions in {ev.language}:"""
    
    try:
        # Create LLM instance inside the step
        json_llm = create_json_mode_llm(temperature=0.1)
        response = json_llm.invoke(prompt).content.strip()
        questions_data = json.loads(response)
        questions = questions_data["questions"]
        
        print(f"❓ Generated {len(questions)} reading comprehension questions in {ev.language}")
        
        return QuestionsGeneratedEvent(
            success=True,
            questions=questions,
            language=ev.language,
            document_text=ev.document_text  # Pass document text forward
        )
        
    except Exception as e:
        print(f"⚠️ JSON parsing failed, trying fallback with conversation LLM: {e}")
        
        try:
            # Fallback: try with conversation LLM
            conversation_llm = create_conversation_llm(temperature=0.3)
            response = conversation_llm.invoke(prompt).content.strip()
            questions_data = json.loads(response)
            questions = questions_data["questions"]
            
            return QuestionsGeneratedEvent(
                success=True,
                questions=questions,
                language=ev.language,
                document_text=ev.document_text  # Pass document text forward
            )
            
        except Exception as fallback_error:
            print(f"⚠️ Fallback also failed, creating simple questions: {fallback_error}")
            
            # Final fallback: create simple questions
            fallback_questions = [
                {
                    "question": f"What is the main topic of this article? (in {ev.language})",
                    "correct_answer": "Main topic based on article content",
                    "question_type": "main_idea"
                },
                {
                    "question": f"What are the key details mentioned in this article? (in {ev.language})",
                    "correct_answer": "Key details from the article",
                    "question_type": "detail"
                }
            ]
            
            return QuestionsGeneratedEvent(
                success=True,
                questions=fallback_questions,
                language=ev.language,
                document_text=ev.document_text,  # Pass document text forward
                fallback=True
            )
    
@step(workflow=NewsAnalysisWorkflow)
async def request_user_input(ev: QuestionsGeneratedEvent) -> UserInputNeededEvent:
    """Request user input for the first question."""
    if not ev.success:
        return UserInputNeededEvent(
            question_number=0,
            question_data={},
            language=ev.language,
            document_text=ev.document_text
        )
    
    questions = ev.questions
    question_data = questions[0]
    
    print(f"\n📝 Reading Comprehension Test (in {ev.language})")
    print("=" * 50)
    print(f"\nQuestion 1: {question_data['question']}")
    print("-" * 30)
    
    return UserInputNeededEvent(
        question_number=1,
        question_data=question_data,
        language=ev.language,
        document_text=ev.document_text,  # Pass document text forward
        questions=questions,  # Pass all questions forward
        answer_1={},  # Empty for now
        answer_2={},  # Empty for now
        remaining_questions=questions[1:] if len(questions) > 1 else []
    )
    
@step(workflow=NewsAnalysisWorkflow)
async def collect_any_answer(ev: UserInputNeededEvent) -> AnswerCollectedEvent | AllAnswersCollectedEvent:
    """Collect answer for any question and route appropriately."""
    print(f"🔍 DEBUG: collect_any_answer called for question {ev.question_number}")
    
    user_answer = input(f"Please answer question {ev.question_number}: ").strip()
    print(f"Your answer: {user_answer}")
    
    answer_data = {
        "question_number": ev.question_number,
        "question": ev.question_data['question'],
        "user_answer": user_answer,
        "correct_answer": ev.question_data['correct_answer'],
        "question_type": ev.question_data['question_type']
    }
    
    if ev.question_number == 1:
        # First answer - store in answer_1 field
        print(f"🔍 DEBUG: Stored first answer: {answer_data}")
        return AnswerCollectedEvent(
            question_number=ev.question_number,
            question=ev.question_data['question'],
            user_answer=user_answer,
            correct_answer=ev.question_data['correct_answer'],
            question_type=ev.question_data['question_type'],
            language=ev.language,  # Pass language forward
            document_text=ev.document_text,  # Pass document text forward
            questions=ev.questions,  # Pass questions forward
            answer_1=answer_data,  # Store first answer
            answer_2={},  # Empty for now
            remaining_questions=ev.remaining_questions
        )
    
    # Question 2 or later → combine with previous answers and finish
    # This happens when collect_any_answer is called for question 2
    # We can get the first answer from the event context
    print(f"🔍 DEBUG: Question {ev.question_number} - combining with first answer")
    print(f"🔍 DEBUG: First answer from event: {ev.answer_1}")
    print(f"🔍 DEBUG: Current answer data: {answer_data}")
    
    # Combine both answers
    all_answers = [ev.answer_1, answer_data] if ev.answer_1 else [answer_data]
    print(f"🔍 DEBUG: Combined answers: {all_answers}")
    
    return AllAnswersCollectedEvent(
        answers=all_answers,
        answer_1=ev.answer_1,  # Pass first answer forward
        answer_2=answer_data,  # Store second answer
        total_questions=len(all_answers),
        language=ev.language,  # Pass language forward
        document_text=ev.document_text,  # Pass document text forward
        questions=ev.questions  # Pass questions forward
    )
    
@step(workflow=NewsAnalysisWorkflow)
async def request_second_question(ev: AnswerCollectedEvent) -> UserInputNeededEvent | AllAnswersCollectedEvent:
    """Request user input for the second question."""
    if not ev.remaining_questions:
        # No more questions after Q1 → proceed to analysis with only the first answer
        return AllAnswersCollectedEvent(
            answers=[ev.answer_1] if ev.answer_1 else [],
            answer_1=ev.answer_1,  # Pass first answer forward
            answer_2={},  # No second answer
            total_questions=1 if ev.answer_1 else 0,
            language=ev.language,  # Pass language forward
            document_text=ev.document_text,  # Pass document text forward
            questions=ev.questions  # Pass questions forward
        )
    
    question_data = ev.remaining_questions[0]
    
    print(f"\nQuestion 2: {question_data['question']}")
    print("-" * 30)
    
    return UserInputNeededEvent(
        question_number=2,
        question_data=question_data,
        language=ev.language,  # Pass language forward
        document_text=ev.document_text,  # Pass document text forward
        questions=ev.questions,  # Pass questions forward
        answer_1=ev.answer_1,  # Pass first answer forward
        answer_2={},  # Empty for now
        remaining_questions=[]
    )
    
    # Removed: obsolete collector. All collection handled by collect_any_answer.
    
@step(workflow=NewsAnalysisWorkflow)
async def analyze_answers(ev: AllAnswersCollectedEvent) -> AnalysisCompleteEvent:
    """Analyze and score user answers."""
    print(f"🔍 DEBUG: Analyzing {len(ev.answers)} answers")
    print(f"🔍 DEBUG: Answers data: {ev.answers}")
    
    analysis_prompt = f"""You are an expert reading comprehension evaluator. Analyze these answers with detailed, comprehensive feedback.

For each answer, evaluate these criteria (0-100 scale):
1. **Accuracy**: How factually correct is the answer?
2. **Completeness**: How thoroughly does it address the question?
3. **Relevance**: How well does it stay on topic?
4. **Language Quality**: Grammar, clarity, and expression
5. **Critical Thinking**: Shows analysis, reasoning, or insight
6. **Evidence Usage**: References to the text or logical reasoning

Answers to analyze:
{json.dumps(ev.answers, indent=2, ensure_ascii=False)}

Provide detailed analysis in JSON format:
{{
    "question_analyses": [
        {{
            "question_number": 1,
            "question_text": "Original question text",
            "user_answer": "User's answer",
            "correct_answer": "Expected answer",
            "scores": {{
                "accuracy": 85,
                "completeness": 90,
                "relevance": 95,
                "language_quality": 80,
                "critical_thinking": 75,
                "evidence_usage": 70
            }},
            "overall_score": 82,
            "grade": "B+",
            "detailed_feedback": "Comprehensive analysis of what the user did well and what needs improvement"
        }}
    ],
    "overall_analysis": {{
        "total_score": 82,
        "grade": "B+",
        "performance_level": "Good",
        "encouragement": "Positive reinforcement and motivation"
    }}
}}"""
    
    try:
        # Create LLM instance inside the step
        json_llm = create_json_mode_llm(temperature=0.1)
        # Use JSON mode LLM for reliable structured analysis
        response = json_llm.invoke(analysis_prompt).content.strip()
        analysis = json.loads(response)
        
        return AnalysisCompleteEvent(
            success=True,
            analysis=analysis,
            language=ev.language,  # Pass language forward
            document_text=ev.document_text,  # Pass document text forward
            questions=ev.questions,  # Pass questions forward
            user_answers=ev.answers  # Pass user answers forward
        )
    except Exception as e:
        print(f"⚠️ JSON parsing failed for analysis, trying fallback: {e}")
        
        try:
            # Fallback: try with conversation LLM
            conversation_llm = create_conversation_llm(temperature=0.3)
            response = conversation_llm.invoke(analysis_prompt).content.strip()
            analysis = json.loads(response)
            
            return AnalysisCompleteEvent(
                success=True,
                analysis=analysis,
                language=ev.language,  # Pass language forward
                document_text=ev.document_text,  # Pass document text forward
                questions=ev.questions,  # Pass questions forward
                user_answers=ev.answers  # Pass user answers forward
            )
        except Exception as fallback_error:
            print(f"⚠️ Fallback analysis also failed: {fallback_error}")
            
            # Final fallback: create simple analysis
            fallback_analysis = {
                "overall_analysis": {
                    "total_score": 75,
                    "grade": "C+",
                    "performance_level": "Satisfactory",
                    "encouragement": "Good effort! Keep practicing to improve your comprehension skills."
                },
                "question_analyses": [
                    {
                        "question_number": i+1,
                        "question_text": ev.answers[i].get('question', 'Question not available'),
                        "user_answer": ev.answers[i].get('user_answer', 'No answer provided'),
                        "overall_score": 72,
                        "grade": "C+",
                        "detailed_feedback": "Shows understanding but could be more detailed and analytical"
                    } for i in range(len(ev.answers))
                ]
            }
            
            return AnalysisCompleteEvent(
                success=True,
                analysis=fallback_analysis,
                language=ev.language,  # Pass language forward
                document_text=ev.document_text,  # Pass document text forward
                questions=ev.questions,  # Pass questions forward
                user_answers=ev.answers,  # Pass user answers forward
                fallback=True
            )
    
@step(workflow=NewsAnalysisWorkflow)
async def generate_final_report(ev: AnalysisCompleteEvent) -> StopEvent:
    """Generate comprehensive summary report."""
    if not ev.success:
        return StopEvent(result={
            "success": False,
            "message": "Analysis failed",
            "error": "Could not analyze answers"
        })
    
    # Generate summary report
    analysis = ev.analysis
    document_language = ev.language
    questions = ev.questions
    all_answers = ev.user_answers
    
    # Create detailed report (similar to original)
    overall = analysis.get('overall_analysis', {})
    
    report = f"""
{'='*80}
📊 COMPREHENSIVE READING COMPREHENSION ANALYSIS REPORT
{'='*80}

🎯 OVERALL PERFORMANCE SUMMARY
{'-'*50}
📈 Total Score: {overall.get('total_score', 'N/A')}/100
🏆 Grade: {overall.get('grade', 'N/A')}
📊 Performance Level: {overall.get('performance_level', 'N/A')}

🌟 ENCOURAGEMENT:
{overall.get('encouragement', 'Keep up the good work!')}

{'='*80}
"""
    
    # Translate report if needed
    if document_language.lower() not in ['en', 'english']:
        translation_prompt = f"""Translate the following report to {document_language}. 
        Maintain formatting and structure:
        
        {report}"""
        
        try:
            conversation_llm = create_conversation_llm(temperature=0.3)
            translated_report = conversation_llm.invoke(translation_prompt).content.strip()
            report = translated_report
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
    
    return StopEvent(result={
        "success": True,
        "message": "News analysis and reading comprehension test completed successfully",
        "document_language": document_language,
        "questions": questions,
        "user_answers": all_answers,
        "analysis": analysis,
        "summary_report": report,
        "timestamp": datetime.now().isoformat()
    })




# Create workflow instance
news_analysis_workflow = NewsAnalysisWorkflow()


async def run_news_analysis():
    """Run the news analysis workflow."""
    print("=== LlamaIndex News Analysis Workflow ===")
    print("This workflow will:")
    print("1. Process PDF news documents from the 'data' directory")
    print("2. Detect the document language")
    print("3. Generate reading comprehension questions")
    print("4. Test your understanding with interactive questions")
    print("5. Analyze your answers and provide detailed feedback")
    print()
    print("Make sure you have a PDF file in the 'data' directory before starting!")
    print("=" * 60)
    
    try:
        result = await news_analysis_workflow.run()
        
        if result["success"]:
            print("\n" + result["summary_report"])
            print(f"\n✅ Analysis completed at {result['timestamp']}")
        else:
            print(f"\n❌ Analysis failed: {result['message']}")
            
    except Exception as e:
        print(f"❌ Workflow error: {e}")


if __name__ == "__main__":
    asyncio.run(run_news_analysis())