#!/usr/bin/env python3
"""
Multi-Session Agent Web Server

A simple FastAPI server that demonstrates how multiple users can interact
with the functional API agent concurrently, each with their own session.

Usage:
1. Install FastAPI: pip install fastapi uvicorn
2. Run: python multi_session_server.py
3. Open browser: http://localhost:8000/docs
4. Test multiple concurrent requests

Key Features:
- Each user gets a unique session ID
- Sessions maintain separate conversation history
- Concurrent requests are handled asynchronously
- RESTful API with streaming support
"""

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import StreamingResponse, HTMLResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI not available. Install with: pip install fastapi uvicorn")

import asyncio
import json
import uuid
from typing import Dict, Optional
from datetime import datetime
from functional_api_agent import agent


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    enable_streaming: bool = False


class QueryResponse(BaseModel):
    session_id: str
    query: str
    response: str
    timestamp: str
    duration: float
    tool_used: Optional[str] = None


class MultiSessionAgentServer:
    """Multi-session agent server"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Multi-Session LangGraph Agent API",
            description="Concurrent agent sessions with conversation memory",
            version="1.0.0"
        )
        self.active_sessions: Dict[str, Dict] = {}
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Root endpoint with simple HTML interface"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Multi-Session Agent</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .session { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    input, textarea { width: 100%; padding: 8px; margin: 5px 0; }
                    button { padding: 10px 20px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer; }
                    button:hover { background: #005a87; }
                    .response { background: white; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Multi-Session LangGraph Agent</h1>
                    <p>This server demonstrates concurrent agent sessions. Each session maintains separate conversation memory.</p>
                    
                    <div class="session">
                        <h3>Test Session</h3>
                        <input type="text" id="sessionId" placeholder="Session ID (optional - auto-generated if empty)">
                        <textarea id="query" placeholder="Enter your question..." rows="3"></textarea>
                        <button onclick="sendQuery()">Send Query</button>
                        <button onclick="sendStreamQuery()">Send Streaming Query</button>
                        <div id="response" class="response" style="display: none;"></div>
                    </div>
                    
                    <div class="session">
                        <h3>📊 API Endpoints</h3>
                        <ul>
                            <li><strong>POST /query</strong> - Send a query to the agent</li>
                            <li><strong>POST /stream</strong> - Stream agent response</li>
                            <li><strong>GET /sessions</strong> - List active sessions</li>
                            <li><strong>GET /sessions/{session_id}</strong> - Get session info</li>
                            <li><strong>DELETE /sessions/{session_id}</strong> - Clear session</li>
                            <li><strong>GET /docs</strong> - API documentation</li>
                        </ul>
                    </div>
                </div>
                
                <script>
                    async function sendQuery() {
                        const sessionId = document.getElementById('sessionId').value || null;
                        const query = document.getElementById('query').value;
                        const responseDiv = document.getElementById('response');
                        
                        if (!query) {
                            alert('Please enter a query');
                            return;
                        }
                        
                        responseDiv.style.display = 'block';
                        responseDiv.innerHTML = '🔄 Processing...';
                        
                        try {
                            const response = await fetch('/query', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ query, session_id: sessionId })
                            });
                            
                            const result = await response.json();
                            responseDiv.innerHTML = `
                                <strong>Session:</strong> ${result.session_id}<br>
                                <strong>Response:</strong> ${result.response}<br>
                                <strong>Duration:</strong> ${result.duration.toFixed(2)}s<br>
                                ${result.tool_used ? `<strong>Tool Used:</strong> ${result.tool_used}<br>` : ''}
                                <strong>Time:</strong> ${result.timestamp}
                            `;
                        } catch (error) {
                            responseDiv.innerHTML = `❌ Error: ${error.message}`;
                        }
                    }
                    
                    async function sendStreamQuery() {
                        const sessionId = document.getElementById('sessionId').value || null;
                        const query = document.getElementById('query').value;
                        const responseDiv = document.getElementById('response');
                        
                        if (!query) {
                            alert('Please enter a query');
                            return;
                        }
                        
                        responseDiv.style.display = 'block';
                        responseDiv.innerHTML = '🌊 Streaming...';
                        
                        try {
                            const response = await fetch('/stream', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ query, session_id: sessionId, enable_streaming: true })
                            });
                            
                            const reader = response.body.getReader();
                            responseDiv.innerHTML = `<strong>Session:</strong> ${sessionId || 'auto'}<br><strong>Streaming Response:</strong><br>`;
                            
                            while (true) {
                                const { done, value } = await reader.read();
                                if (done) break;
                                
                                const chunk = new TextDecoder().decode(value);
                                responseDiv.innerHTML += chunk + '<br>';
                            }
                        } catch (error) {
                            responseDiv.innerHTML = `❌ Streaming Error: ${error.message}`;
                        }
                    }
                </script>
            </body>
            </html>
            """
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_agent(request: QueryRequest):
            """Send a query to the agent"""
            session_id = request.session_id or str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Track session
                if session_id not in self.active_sessions:
                    self.active_sessions[session_id] = {
                        "created": datetime.now().isoformat(),
                        "queries": 0
                    }
                
                self.active_sessions[session_id]["queries"] += 1
                self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
                
                # Execute agent
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.invoke(request.query, config))
                
                duration = asyncio.get_event_loop().time() - start_time
                
                return QueryResponse(
                    session_id=session_id,
                    query=request.query,
                    response=result.get("message", str(result)),
                    timestamp=datetime.now().isoformat(),
                    duration=duration,
                    tool_used=result.get("tool")
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stream")
        async def stream_agent(request: QueryRequest):
            """Stream agent response"""
            session_id = request.session_id or str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}
            
            async def generate_stream():
                try:
                    # Track session
                    if session_id not in self.active_sessions:
                        self.active_sessions[session_id] = {
                            "created": datetime.now().isoformat(),
                            "queries": 0
                        }
                    
                    self.active_sessions[session_id]["queries"] += 1
                    self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
                    
                    yield f"data: Session: {session_id}\n\n"
                    yield f"data: Query: {request.query}\n\n"
                    yield f"data: ---\n\n"
                    
                    async for chunk in agent.astream(request.query, config):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    yield f"data: [DONE]\n\n"
                    
                except Exception as e:
                    yield f"data: ERROR: {str(e)}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        
        @self.app.get("/sessions")
        async def get_sessions():
            """Get all active sessions"""
            return self.active_sessions
        
        @self.app.get("/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get specific session info"""
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            return self.active_sessions[session_id]
        
        @self.app.delete("/sessions/{session_id}")
        async def clear_session(session_id: str):
            """Clear/delete a session"""
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                return {"message": f"Session {session_id} cleared"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
    
    def run(self, host: str = "localhost", port: int = 8000):
        """Run the server"""
        print(f"🚀 Starting Multi-Session Agent Server on http://{host}:{port}")
        print(f"📚 API docs available at: http://{host}:{port}/docs")
        print(f"🌐 Web interface available at: http://{host}:{port}/")
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    """Main function"""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI is required to run the server")
        print("Install with: pip install fastapi uvicorn")
        return
    
    server = MultiSessionAgentServer()
    server.run()


if __name__ == "__main__":
    main()
