#!/usr/bin/env python3
"""
LangGraph Functional API News Analysis Agent

A news analysis agent that:
1. Processes PDF news documents from data directory
2. Detects document language
3. Generates reading comprehension questions
4. Uses interrupts to ask questions and collect answers
5. Analyzes and scores user responses
6. Provides comprehensive feedback

- Functional API (@entrypoint, @task)
- PDF processing with PyPDF
- Language detection and question generation
- Interrupt functionality for user interaction
- Answer analysis and scoring
"""

import asyncio
import json
import os
import glob
from typing import Any, Dict, List
from datetime import datetime

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
from langgraph.types import StreamWriter, Command, interrupt
from pypdf import PdfReader

# Use your existing LLM client with environment variables from .env file
from llm_client import create_conversation_llm
llm = create_conversation_llm(temperature=0.3)
print("✅ Using Azure OpenAI LLM")

# Per-thread persistence config
CONFIG = {"configurable": {"thread_id": "news-analysis-thread"}}

# Global variables for document analysis
DOCUMENT_LANGUAGE = "en"
DOCUMENT_TEXT = ""
QUESTIONS = []
USER_ANSWERS = []


@task()
def setup_data_directory() -> Dict[str, Any]:
    """Set up data directory and check for PDF files."""
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created data directory: {data_dir}")
    
    # Find PDF files in data directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        return {
            "success": False,
            "message": f"No PDF files found in {data_dir} directory. Please upload a PDF file first.",
            "pdf_files": []
        }
    
    print(f"📄 Found {len(pdf_files)} PDF file(s): {[os.path.basename(f) for f in pdf_files]}")
    
    return {
        "success": True,
        "message": f"Found {len(pdf_files)} PDF file(s) ready for analysis",
        "pdf_files": pdf_files
    }


@task()
def process_pdf_document(pdf_files: List[str]) -> Dict[str, Any]:
    """Extract text from the first PDF file found."""
    global DOCUMENT_TEXT
    
    if not pdf_files:
        return {
            "success": False,
            "message": "No PDF files available for processing",
            "text": ""
        }
    
    # Process the first PDF file
    pdf_path = pdf_files[0]
    print(f"📖 Processing PDF: {os.path.basename(pdf_path)}")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean up text
        text = text.strip()
        
        if not text:
            return {
                "success": False,
                "message": "Could not extract text from PDF. The file might be image-based or corrupted.",
                "text": ""
            }
        
        DOCUMENT_TEXT = text
        print(f"✅ Successfully extracted {len(text)} characters from PDF")
        
        return {
            "success": True,
            "message": f"Successfully processed PDF: {os.path.basename(pdf_path)}",
            "text": text,
            "char_count": len(text)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "text": ""
        }


@task()
def detect_document_language(document_text: str) -> Dict[str, str]:
    """Detect the language of the document."""
    global DOCUMENT_LANGUAGE
    
    # Use first 1000 characters for language detection
    sample_text = document_text[:1000]
    
    prompt = f"""Detect the language of the following text and respond with only the language code (e.g., 'en', 'hu', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', etc.):

Text: {sample_text}

Language code:"""
    
    detected_lang = llm.invoke(prompt).content.strip().lower()
    DOCUMENT_LANGUAGE = detected_lang
    
    print(f"🌍 Detected document language: {detected_lang}")
    
    return {
        "language": detected_lang,
        "confidence": "high"  # Could be enhanced with confidence scoring
    }


@task()
def generate_reading_comprehension_questions(document_text: str, language: str) -> Dict[str, Any]:
    """Generate 2 reading comprehension questions based on the document."""
    global QUESTIONS
    
    # Use first 4000 characters for question generation
    sample_text = document_text[:4000]
    
    prompt = f"""Based on the following news article text, generate exactly 2 reading comprehension questions that test understanding of key facts, main ideas, and important details. Generate the questions in {language}.

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

Generate questions in {language}:"""
    
    response = llm.invoke(prompt).content.strip()
    
    try:
        # Try to parse JSON response
        import json
        questions_data = json.loads(response)
        QUESTIONS = questions_data["questions"]
        
        print(f"❓ Generated {len(QUESTIONS)} reading comprehension questions in {language}")
        
        return {
            "success": True,
            "questions": QUESTIONS,
            "language": language
        }
        
    except Exception as e:
        # Fallback: create simple questions if JSON parsing fails
        print(f"⚠️ JSON parsing failed, creating fallback questions: {e}")
        
        fallback_questions = [
            {
                "question": f"What is the main topic of this article? (in {language})",
                "correct_answer": "Main topic based on article content",
                "question_type": "main_idea"
            },
            {
                "question": f"What are the key details mentioned in this article? (in {language})",
                "correct_answer": "Key details from the article",
                "question_type": "detail"
            }
        ]
        
        QUESTIONS = fallback_questions
        
        return {
            "success": True,
            "questions": fallback_questions,
            "language": language,
            "fallback": True
        }


@task()
def ask_question_1(questions: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
    """Ask first question with interrupt for user input."""
    question_data = questions[0]
    
    print(f"\n📝 Reading Comprehension Test (in {language})")
    print("=" * 50)
    print(f"\nQuestion 1: {question_data['question']}")
    print("-" * 30)
    
    # Use interrupt to pause for user input
    user_answer = interrupt(f"Please answer question 1 in {language}: {question_data['question']}")
    
    answer_data = {
        "question_number": 1,
        "question": question_data['question'],
        "user_answer": user_answer,
        "correct_answer": question_data['correct_answer'],
        "question_type": question_data['question_type']
    }
    
    print(f"Your answer: {user_answer}")
    
    return {
        "success": True,
        "answer_1": answer_data,
        "remaining_questions": questions[1:]
    }

@task()
def ask_question_2(question_1_result: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Ask second question with interrupt for user input."""
    questions = question_1_result["remaining_questions"]
    question_data = questions[0]
    
    print(f"\nQuestion 2: {question_data['question']}")
    print("-" * 30)
    
    # Use interrupt to pause for user input
    user_answer = interrupt(f"Please answer question 2 in {language}: {question_data['question']}")
    
    answer_data = {
        "question_number": 2,
        "question": question_data['question'],
        "user_answer": user_answer,
        "correct_answer": question_data['correct_answer'],
        "question_type": question_data['question_type']
    }
    
    print(f"Your answer: {user_answer}")
    print()
    
    # Combine both answers
    all_answers = [question_1_result["answer_1"], answer_data]
    
    return {
        "success": True,
        "answers": all_answers,
        "total_questions": 2
    }


@task()
def analyze_user_answers(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and score user answers with comprehensive evaluation."""
    
    analysis_prompt = f"""You are an expert reading comprehension evaluator. Analyze these answers with detailed, comprehensive feedback.

For each answer, evaluate these criteria (0-100 scale):
1. **Accuracy**: How factually correct is the answer?
2. **Completeness**: How thoroughly does it address the question?
3. **Relevance**: How well does it stay on topic?
4. **Language Quality**: Grammar, clarity, and expression
5. **Critical Thinking**: Shows analysis, reasoning, or insight
6. **Evidence Usage**: References to the text or logical reasoning

Answers to analyze:
{json.dumps(answers, indent=2, ensure_ascii=False)}

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
            "detailed_feedback": "Comprehensive analysis of what the user did well and what needs improvement",
            "strengths": ["Specific strengths identified"],
            "weaknesses": ["Specific areas for improvement"],
            "suggestions": ["Concrete suggestions for better answers"],
            "key_points_missed": ["Important points the user didn't mention"],
            "excellent_points": ["Particularly good aspects of the answer"]
        }}
    ],
    "overall_analysis": {{
        "total_score": 82,
        "grade": "B+",
        "performance_level": "Good",
        "score_breakdown": {{
            "average_accuracy": 85,
            "average_completeness": 90,
            "average_relevance": 95,
            "average_language": 80,
            "average_critical_thinking": 75,
            "average_evidence_usage": 70
        }},
        "strengths_summary": "Overall strengths across all answers",
        "improvement_areas": "Main areas needing improvement",
        "recommendations": "Specific recommendations for better performance",
        "encouragement": "Positive reinforcement and motivation"
    }}
}}"""
    
    response = llm.invoke(analysis_prompt).content.strip()
    
    try:
        analysis = json.loads(response)
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        # Fallback analysis
        print(f"⚠️ JSON parsing failed for analysis: {e}")
        
        fallback_analysis = {
            "overall_analysis": {
                "total_score": 75,
                "grade": "C+",
                "performance_level": "Satisfactory",
                "score_breakdown": {
                    "average_accuracy": 80,
                    "average_completeness": 70,
                    "average_relevance": 85,
                    "average_language": 75,
                    "average_critical_thinking": 65,
                    "average_evidence_usage": 60
                },
                "strengths_summary": "Shows basic understanding of the material",
                "improvement_areas": "Detail retention and critical analysis",
                "recommendations": "Focus on specific details and deeper analysis",
                "encouragement": "Good effort! Keep practicing to improve your comprehension skills."
            },
            "question_analyses": [
                {
                    "question_number": i+1,
                    "question_text": answers[i].get('question', 'Question not available'),
                    "user_answer": answers[i].get('user_answer', 'No answer provided'),
                    "correct_answer": answers[i].get('correct_answer', 'Answer not available'),
                    "scores": {
                        "accuracy": 80,
                        "completeness": 70,
                        "relevance": 85,
                        "language_quality": 75,
                        "critical_thinking": 65,
                        "evidence_usage": 60
                    },
                    "overall_score": 72,
                    "grade": "C+",
                    "detailed_feedback": "Shows understanding but could be more detailed and analytical",
                    "strengths": ["Engaged with the content", "Attempted to answer"],
                    "weaknesses": ["Lacks specific details", "Limited analysis"],
                    "suggestions": ["Include more specific examples", "Explain your reasoning"],
                    "key_points_missed": ["Some important details from the text"],
                    "excellent_points": ["Shows effort and engagement"]
                } for i in range(len(answers))
            ]
        }
        
        return {
            "success": True,
            "analysis": fallback_analysis,
            "fallback": True
        }


@task()
def translate_report(report: str, target_language: str) -> str:
    """Translate the report to the target language."""
    
    if target_language.lower() in ['en', 'english']:
        return report  # No translation needed for English
    
    translation_prompt = f"""Translate the following comprehensive reading comprehension analysis report to {target_language}. 
    
    Maintain all formatting, emojis, and structure. Only translate the text content while preserving:
    - All emojis and symbols
    - Section headers and formatting
    - Score numbers and percentages
    - Bullet points and lists
    - The overall structure and layout
    
    Report to translate:
    {report}
    
    Provide only the translated report without any additional commentary."""
    
    try:
        response = llm.invoke(translation_prompt).content.strip()
        return response
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return report  # Return original if translation fails


@task()
def generate_summary_report(analysis: Dict[str, Any], language: str) -> str:
    """Generate a comprehensive summary report of the reading comprehension analysis."""
    
    # Extract key data for easier formatting
    overall = analysis.get('overall_analysis', {})
    questions = analysis.get('question_analyses', [])
    
    # Create a detailed visual report
    report = f"""
{'='*80}
📊 COMPREHENSIVE READING COMPREHENSION ANALYSIS REPORT
{'='*80}

🎯 OVERALL PERFORMANCE SUMMARY
{'-'*50}
📈 Total Score: {overall.get('total_score', 'N/A')}/100
🏆 Grade: {overall.get('grade', 'N/A')}
📊 Performance Level: {overall.get('performance_level', 'N/A')}

📋 DETAILED SCORE BREAKDOWN
{'-'*50}"""
    
    # Add score breakdown if available
    score_breakdown = overall.get('score_breakdown', {})
    if score_breakdown:
        report += f"""
• Accuracy: {score_breakdown.get('average_accuracy', 'N/A')}/100
• Completeness: {score_breakdown.get('average_completeness', 'N/A')}/100
• Relevance: {score_breakdown.get('average_relevance', 'N/A')}/100
• Language Quality: {score_breakdown.get('average_language', 'N/A')}/100
• Critical Thinking: {score_breakdown.get('average_critical_thinking', 'N/A')}/100
• Evidence Usage: {score_breakdown.get('average_evidence_usage', 'N/A')}/100"""
    
    # Add question-by-question analysis
    if questions:
        report += f"""

📝 QUESTION-BY-QUESTION ANALYSIS
{'-'*50}"""
        
        for i, q in enumerate(questions, 1):
            report += f"""

🔍 QUESTION {i}
{'-'*20}
❓ Question: {q.get('question_text', 'N/A')}
💭 Your Answer: {q.get('user_answer', 'N/A')}
✅ Expected Answer: {q.get('correct_answer', 'N/A')}

📊 SCORES:
• Overall: {q.get('overall_score', 'N/A')}/100 ({q.get('grade', 'N/A')})
• Accuracy: {q.get('scores', {}).get('accuracy', 'N/A')}/100
• Completeness: {q.get('scores', {}).get('completeness', 'N/A')}/100
• Relevance: {q.get('scores', {}).get('relevance', 'N/A')}/100
• Language: {q.get('scores', {}).get('language_quality', 'N/A')}/100
• Critical Thinking: {q.get('scores', {}).get('critical_thinking', 'N/A')}/100
• Evidence Usage: {q.get('scores', {}).get('evidence_usage', 'N/A')}/100

💡 FEEDBACK:
{q.get('detailed_feedback', 'No detailed feedback available')}

✅ STRENGTHS:
{chr(10).join([f"• {strength}" for strength in q.get('strengths', [])])}

⚠️ AREAS FOR IMPROVEMENT:
{chr(10).join([f"• {weakness}" for weakness in q.get('weaknesses', [])])}

💡 SUGGESTIONS:
{chr(10).join([f"• {suggestion}" for suggestion in q.get('suggestions', [])])}

🎯 KEY POINTS YOU MISSED:
{chr(10).join([f"• {point}" for point in q.get('key_points_missed', [])])}

⭐ EXCELLENT POINTS:
{chr(10).join([f"• {point}" for point in q.get('excellent_points', [])])}"""
    
    # Add overall analysis
    report += f"""

🎯 OVERALL ANALYSIS
{'-'*50}
📈 STRENGTHS SUMMARY:
{overall.get('strengths_summary', 'No strengths summary available')}

⚠️ IMPROVEMENT AREAS:
{overall.get('improvement_areas', 'No improvement areas identified')}

💡 RECOMMENDATIONS:
{overall.get('recommendations', 'No specific recommendations available')}

🌟 ENCOURAGEMENT:
{overall.get('encouragement', 'Keep up the good work!')}

{'='*80}
📚 NEXT STEPS FOR IMPROVEMENT
{'-'*50}
1. Review the detailed feedback for each question
2. Focus on the identified improvement areas
3. Practice with similar reading materials
4. Pay attention to specific details and critical analysis
5. Continue practicing to build confidence

{'='*80}
"""
    
    return report


# Main workflow: PDF processing -> language detection -> question generation -> interrupt -> answer analysis -> summary
@entrypoint(checkpointer=MemorySaver())
def news_analysis_agent(input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main news analysis workflow with reading comprehension testing."""
    
    print("📰 Starting News Analysis and Reading Comprehension Test")
    print("=" * 60)
    
    # Step 1: Setup data directory and find PDF files
    setup_result = setup_data_directory().result()
    if not setup_result["success"]:
        return {
            "success": False,
            "message": setup_result["message"],
            "error": "No PDF files found"
        }
    
    # Step 2: Process PDF document
    process_result = process_pdf_document(setup_result["pdf_files"]).result()
    if not process_result["success"]:
        return {
            "success": False,
            "message": process_result["message"],
            "error": "PDF processing failed"
        }
    
    # Step 3: Detect document language
    language_result = detect_document_language(process_result["text"]).result()
    document_language = language_result["language"]
    
    # Step 4: Generate reading comprehension questions
    questions_result = generate_reading_comprehension_questions(
        process_result["text"], 
        document_language
        ).result()
    
    if not questions_result["success"]:
        return {
            "success": False,
            "message": "Failed to generate questions",
            "error": "Question generation failed"
        }
    
    # Step 5: Ask questions using interrupt functionality
    question_1_result = ask_question_1(
        questions_result["questions"], 
        document_language
    ).result()
    
    answers_result = ask_question_2(
        question_1_result,
        document_language
    ).result()
    
    # Step 6: Analyze user answers
    analysis_result = analyze_user_answers(answers_result["answers"]).result()
    
    # Step 7: Generate summary report
    summary_report = generate_summary_report(
        analysis_result["analysis"], 
        document_language
    ).result()
    
    # Step 8: Translate report to document language
    translated_report = translate_report(
        summary_report, 
        document_language
    ).result()
    
    # Return comprehensive results
    return {
        "success": True,
        "message": "News analysis and reading comprehension test completed successfully",
        "document_language": document_language,
        "questions": questions_result["questions"],
        "user_answers": answers_result["answers"],
        "analysis": analysis_result["analysis"],
        "summary_report": summary_report,
        "translated_report": translated_report,
        "timestamp": datetime.now().isoformat()
    }


async def interactive_news_analysis():
    """Interactive interface for news analysis."""
    print("=== News Analysis and Reading Comprehension Agent ===")
    print("This agent will:")
    print("1. Process PDF news documents from the 'data' directory")
    print("2. Detect the document language")
    print("3. Generate reading comprehension questions")
    print("4. Test your understanding with interactive questions")
    print("5. Analyze your answers and provide detailed feedback")
    print()
    print("Make sure you have a PDF file in the 'data' directory before starting!")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nPress Enter to start news analysis (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif not user_input:
                print("\n🚀 Starting news analysis...")
                
                # Run the news analysis agent with streaming to handle interrupts
                config = {"configurable": {"thread_id": "news-analysis-thread"}}
                
                try:
                    # Handle multiple interrupts properly
                    current_stream = news_analysis_agent.stream({}, config)
                    
                    while True:
                        try:
                            event = next(current_stream)
                            print("Event:", event)
                            print()
                            
                            # Check if this is an interrupt event
                            if isinstance(event, dict) and "__interrupt__" in event:
                                interrupt_obj = event["__interrupt__"][0]
                                print(f"🔄 Interrupt: {interrupt_obj.value}")
                                
                                # Get user input for the interrupt
                                user_answer = input("Your answer: ").strip()
                                
                                # Resume with the user's answer and continue the stream
                                print("🔄 Resuming with your answer...")
                                current_stream = news_analysis_agent.stream(Command(resume=user_answer), config)
                                continue
                            
                        except StopIteration:
                            print("✅ News analysis completed successfully!")
                            break
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            break
                    
                except Exception as e:
                    print(f"❌ News analysis failed: {e}")
            else:
                print("Please press Enter to start or type 'quit' to exit.")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function - run interactive news analysis."""
    print("Choose mode:")
    print("1. Interactive news analysis")
    print("2. Quick test (if PDF available)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(interactive_news_analysis())
    elif choice == "2":
        print("\n🚀 Running quick news analysis test...")
        print("Note: This will use streaming mode to handle interrupts properly.")
        print()
        
        try:
            # Use streaming to handle interrupts properly
            config = {"configurable": {"thread_id": "quick-test-thread"}}
            
            # Handle multiple interrupts properly
            current_stream = news_analysis_agent.stream({}, config)
            
            while True:
                try:
                    event = next(current_stream)
                    print("Event:", event)
                    print()
                    
                    # Check if this is an interrupt event
                    if isinstance(event, dict) and "__interrupt__" in event:
                        interrupt_obj = event["__interrupt__"][0]
                        print(f"🔄 Interrupt: {interrupt_obj.value}")
                        
                        # Get user input for the interrupt
                        user_answer = input("Your answer: ").strip()
                        
                        # Resume with the user's answer and continue the stream
                        print("🔄 Resuming with your answer...")
                        current_stream = news_analysis_agent.stream(Command(resume=user_answer), config)
                        continue
                    
                except StopIteration:
                    print("✅ News analysis completed!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    else:
        print("Invalid choice. Running interactive mode...")
        asyncio.run(interactive_news_analysis())


if __name__ == "__main__":
    main()