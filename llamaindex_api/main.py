"""
FastAPI Chatbot with LlamaIndex Workflow Checkpointing
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from chatbot_workflow import ChatbotManager

# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex Chatbot API",
    description="Stateless chatbot with conversation checkpointing",
    version="1.0.0"
)

# Initialize chatbot manager
chatbot_manager = ChatbotManager()

# Pydantic models for API
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    bot_response: str
    conversation_summary: Optional[str] = None
    status: str

class SessionInfo(BaseModel):
    session_id: str
    checkpoints: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LlamaIndex Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "sessions": "GET /sessions",
            "history": "GET /sessions/{session_id}/history",
            "checkpoints": "GET /sessions/{session_id}/checkpoints"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response.
    This endpoint is stateless - it uses checkpointing to maintain conversation state.
    """
    try:
        result = await chatbot_manager.send_message(
            session_id=request.session_id,
            user_message=request.message
        )
        
        return ChatResponse(
            session_id=result["session_id"],
            bot_response=result["bot_response"],
            conversation_summary=result.get("conversation_summary"),
            status=result["status"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active conversation sessions"""
    sessions = chatbot_manager.list_active_sessions()
    return {
        "active_sessions": sessions,
        "total_sessions": len(sessions)
    }

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a specific session"""
    history = chatbot_manager.get_conversation_history(session_id)
    return {
        "session_id": session_id,
        "conversation_history": history,
        "message_count": len(history)
    }

@app.get("/sessions/{session_id}/checkpoints")
async def get_session_checkpoints(session_id: str):
    """Get checkpoints for a specific session"""
    checkpoints = chatbot_manager.get_session_checkpoints(session_id)
    return {
        "session_id": session_id,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints)
    }

@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get complete session information"""
    checkpoints = chatbot_manager.get_session_checkpoints(session_id)
    history = chatbot_manager.get_conversation_history(session_id)
    
    return SessionInfo(
        session_id=session_id,
        checkpoints=checkpoints,
        conversation_history=history
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    if session_id in chatbot_manager.workflows:
        del chatbot_manager.workflows[session_id]
        del chatbot_manager.conversation_histories[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
