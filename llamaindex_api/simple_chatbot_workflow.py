"""
Simple LlamaIndex Chatbot Workflow (No OpenAI required for testing)
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from llama_index.core.workflow.context import Context

class ChatMessage(Event):
    user_message: str
    session_id: str
    message: str = ""

class BotResponse(Event):
    user_message: str
    bot_response: str
    session_id: str
    message: str = ""


# Create workflow class
class SimpleChatbotWorkflow(Workflow):
    """Simple chatbot workflow with context-based conversation history"""
    
    @step
    async def start_conversation_step(self, ev: StartEvent, ctx: Context) -> ChatMessage:
        """Initialize conversation and load conversation history from context"""
        print(f"🔍 DEBUG: StartEvent input_data: {ev.input_data}")
        session_id = ev.input_data.get("session_id", "unknown")
        user_message = ev.input_data.get("user_message", "")
        
        # Load conversation history from context
        conversation_history = await ctx.get("conversation_history", [])
        print(f"📚 Loaded {len(conversation_history)} previous messages from context")
        
        print(f"🤖 Starting conversation for session: {session_id}")
        print(f"📝 User message in start_conversation: '{user_message}'")
        
        return ChatMessage(
            user_message=user_message,
            session_id=session_id,
            message="Conversation started. Processing user message..."
        )

    @step
    async def process_user_message_step(self, ev: ChatMessage, ctx: Context) -> StopEvent:
        """Process user message and generate context-aware response"""
        print(f"💬 Processing message: '{ev.user_message}'")
        print(f"🔍 DEBUG: ChatMessage - session_id: {ev.session_id}, user_message: '{ev.user_message}'")
        
        # Get conversation history from context
        conversation_history = await ctx.get("conversation_history", [])
        print(f"📚 Processing with {len(conversation_history)} previous messages in context")
        
        # Generate context-aware response
        bot_response = self.generate_contextual_response(ev.user_message, conversation_history)
        
        # Add new message to conversation history
        new_message = {
            "user_message": ev.user_message,
            "bot_response": bot_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update conversation history in context
        updated_history = conversation_history + [new_message]
        await ctx.set("conversation_history", updated_history)
        print(f"💾 Updated context with {len(updated_history)} total messages")
        
        print(f"🤖 Bot response: {bot_response}")
        
        # Simple conversation summary
        conversation_summary = f"User asked: '{ev.user_message}' and bot responded: '{bot_response}'"
        
        result = StopEvent(result={
            "session_id": ev.session_id,
            "conversation_summary": conversation_summary,
            "final_user_message": ev.user_message,
            "final_bot_response": bot_response,
            "conversation_history": updated_history,  # Include history in result
            "ended_at": datetime.now().isoformat()
        })
        
        print(f"🔍 DEBUG: Returning StopEvent with result: {result.result}")
        return result
    
    def generate_contextual_response(self, user_message: str, conversation_history: List[Dict]) -> str:
        """Generate response based on conversation history and current message"""
        user_msg = user_message.lower()
        
        # Context-aware responses based on conversation history
        if len(conversation_history) > 0:
            last_bot_response = conversation_history[-1].get("bot_response", "").lower()
            
            # Check for follow-up questions
            if "python" in user_msg and "python" in last_bot_response:
                return "Building on what I said about Python, here's more detail: Python is excellent for data science, web development, and automation!"
            elif "help" in user_msg and "help" in last_bot_response:
                return "I'm glad you're interested! Let me elaborate on how I can help you with programming and technology."
            elif "thank" in user_msg or "thanks" in user_msg:
                return "You're very welcome! I'm here whenever you need help with programming or just want to chat!"
        
        # Original response logic for new conversations
        if "hello" in user_msg or "hi" in user_msg:
            if len(conversation_history) == 0:
                return "Hello! Nice to meet you! How can I help you today?"
            else:
                return "Hello again! How can I help you today?"
        elif "how are you" in user_msg:
            return "I'm doing great, thank you for asking! How are you doing?"
        elif "python" in user_msg:
            return "Python is a great programming language! It's versatile and easy to learn."
        elif "help" in user_msg:
            return "I'm here to help! You can ask me about programming, technology, or just chat!"
        elif "bye" in user_msg or "goodbye" in user_msg:
            return "Goodbye! It was nice chatting with you. Have a great day!"
        else:
            return f"Thanks for saying '{user_message}'. That's interesting! Tell me more about it."
    

class SimpleChatbotManager:
    """Manager for simple chatbot workflows with context-based conversation history"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowCheckpointer] = {}
        self.context_snapshots: Dict[str, Dict] = {}  # Store context snapshots instead of conversation histories
    
    def get_or_create_workflow(self, session_id: str) -> WorkflowCheckpointer:
        """Get existing workflow or create new one for session"""
        if session_id not in self.workflows:
            workflow = SimpleChatbotWorkflow(timeout=60, verbose=True)
            checkpointer = WorkflowCheckpointer(workflow=workflow)
            self.workflows[session_id] = checkpointer
            self.context_snapshots[session_id] = {}  # Initialize empty context snapshot
            print(f"✅ Created new workflow for session: {session_id}")
        else:
            print(f"🔄 Using existing workflow for session: {session_id}")
        
        return self.workflows[session_id]
    
    async def send_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Send message to chatbot and get response using context snapshots"""
        checkpointer = self.get_or_create_workflow(session_id)
        
        print(f"🚀 Processing message for session: {session_id}")
        print(f"📝 User message: '{user_message}'")
        
        try:
            # Load previous context snapshot
            previous_context = self.context_snapshots.get(session_id, {})
            print(f"📚 Restoring context snapshot with {len(previous_context)} keys")
            
            # Run workflow
            print(f"🔄 Running workflow...")
            handler = checkpointer.run(input_data={"session_id": session_id, "user_message": user_message})
            result = await handler
            print(f"📊 Workflow result type: {type(result)}")
            print(f"📊 Workflow result: {result}")
            
            # Snapshot the context after completion
            context_snapshot = handler.ctx.to_dict()
            self.context_snapshots[session_id] = context_snapshot
            print(f"💾 Saved context snapshot with {len(context_snapshot)} keys")
            
            # Extract conversation history from context for backward compatibility
            conversation_history = []
            if "state" in context_snapshot and "state_data" in context_snapshot["state"]:
                state_data = context_snapshot["state"]["state_data"]
                if "_data" in state_data and "conversation_history" in state_data["_data"]:
                    conversation_history = state_data["_data"]["conversation_history"]
            
            if isinstance(result, dict) and "final_bot_response" in result:
                print(f"✅ Workflow completed with result")
                
                return {
                    "session_id": session_id,
                    "bot_response": result.get("final_bot_response", ""),
                    "conversation_summary": result.get("conversation_summary", ""),
                    "conversation_history": conversation_history,  # Include history from context
                    "status": "completed"
                }
            else:
                print(f"⚠️ Workflow result is not a dict with expected data")
                print(f"📊 Result type: {type(result)}")
                print(f"📊 Result value: {result}")
                return {
                    "session_id": session_id,
                    "bot_response": "I'm ready to chat! Please send me a message.",
                    "status": "waiting_for_input"
                }
                
        except Exception as e:
            print(f"❌ Error in conversation: {e}")
            import traceback
            traceback.print_exc()
            return {
                "session_id": session_id,
                "bot_response": f"Sorry, I encountered an error: {str(e)}",
                "status": "error"
            }
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session from context snapshot"""
        if session_id in self.context_snapshots:
            context_snapshot = self.context_snapshots[session_id]
            if "state" in context_snapshot and "state_data" in context_snapshot["state"]:
                state_data = context_snapshot["state"]["state_data"]
                if "_data" in state_data and "conversation_history" in state_data["_data"]:
                    return state_data["_data"]["conversation_history"]
        return []
    
    def list_active_sessions(self) -> List[str]:
        """List all active conversation sessions"""
        return list(self.workflows.keys())
    
    def get_session_checkpoints(self, session_id: str) -> List[Dict]:
        """Get checkpoints for a specific session"""
        if session_id in self.workflows:
            checkpointer = self.workflows[session_id]
            if session_id in checkpointer.checkpoints:
                return [
                    {
                        "id": checkpoint.id,
                        "last_completed_step": checkpoint.last_completed_step,
                        "input_event": type(checkpoint.input_event).__name__,
                        "output_event": type(checkpoint.output_event).__name__
                    }
                    for checkpoint in checkpointer.checkpoints[session_id]
                ]
        return []
