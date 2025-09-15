# LlamaIndex Chatbot API

A stateless FastAPI chatbot using LlamaIndex workflows with checkpointing for conversation management.

## Features

- **Stateless API**: Each request is independent
- **Conversation Checkpointing**: Maintains conversation state using LlamaIndex checkpoints
- **Session Management**: Multiple concurrent conversations
- **RESTful API**: Easy to integrate with any client
- **Terminal Test Client**: Interactive testing tool
- **Two Versions**: Simple (no OpenAI) and Full (with OpenAI)

## Quick Start

### Option 1: Simple Version (No OpenAI Required)

```bash
cd llamaindex_api
pip install -r requirements.txt
python simple_main.py
```

Then test with:
```bash
python simple_test_client.py
```

### Option 2: Full Version (Requires OpenAI)

```bash
cd llamaindex_api
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
python main.py
```

Then test with:
```bash
python test_client.py
```

### Option 3: Quick Start Script

```bash
cd llamaindex_api
pip install -r requirements.txt
python quick_start.py
```

## API Endpoints

### Chat
```
POST /chat
{
  "session_id": "unique_session_id",
  "message": "Hello, how are you?"
}
```

### Session Management
```
GET /sessions                           # List all sessions
GET /sessions/{session_id}/history      # Get conversation history
GET /sessions/{session_id}/checkpoints  # Get workflow checkpoints
GET /sessions/{session_id}/info         # Get complete session info
DELETE /sessions/{session_id}           # Delete session
```

## How It Works

### Stateless Architecture

1. **Client sends message** with session_id
2. **API looks up session** in checkpoint storage
3. **Resumes workflow** from last checkpoint
4. **Processes message** through LlamaIndex workflow
5. **Saves new checkpoint** and returns response
6. **Client receives response** and can continue conversation

### Checkpointing Flow

```
StartEvent → ChatMessage → BotResponse → ChatMessage → ...
     ↓           ↓            ↓           ↓
Checkpoint  Checkpoint  Checkpoint  Checkpoint
```

Each step creates a checkpoint, allowing the conversation to be resumed from any point.

## Test Client Usage

### Interactive Mode
```bash
python test_client.py
# Choose option 1
# Type messages, use commands:
# - 'info' - Show session info
# - 'history' - Show conversation history  
# - 'checkpoints' - Show workflow checkpoints
# - 'quit' - Exit
```

### Demo Mode
```bash
python test_client.py
# Choose option 2
# Watch automated conversation with checkpointing
```

## Example Session

```python
# Start conversation
POST /chat
{
  "session_id": "user123",
  "message": "Hello!"
}

# Response
{
  "session_id": "user123",
  "bot_response": "Hello! How can I help you today?",
  "status": "waiting_for_input"
}

# Continue conversation
POST /chat
{
  "session_id": "user123", 
  "message": "What's the weather like?"
}

# Response
{
  "session_id": "user123",
  "bot_response": "I don't have access to real-time weather data...",
  "status": "waiting_for_input"
}
```

## Architecture Benefits

- **Scalable**: Multiple API instances can share checkpoint storage
- **Fault Tolerant**: Conversations survive server restarts
- **Stateless**: No server-side session storage needed
- **Resumable**: Conversations can be paused and resumed
- **Auditable**: Complete conversation history in checkpoints

## Development

### Adding New Workflow Steps

1. Define new Event classes
2. Add new @step methods to ChatbotWorkflow
3. Update step routing logic
4. Test with client

### Customizing the LLM

Modify the `ChatbotWorkflow.__init__()` method to use different LLM configurations.

### Adding Persistence

Replace in-memory checkpoint storage with database persistence for production use.
