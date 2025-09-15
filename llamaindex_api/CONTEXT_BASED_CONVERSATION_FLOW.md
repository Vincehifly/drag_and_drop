# Context-Based Conversation Flow in LlamaIndex Chatbot

## Overview

This document explains how conversation history is handled in the LlamaIndex chatbot using **Context snapshots** instead of external storage. This approach leverages LlamaIndex's built-in `Context` object to maintain conversation state across workflow executions.

## Architecture

### Key Components

1. **`SimpleChatbotWorkflow`** - The workflow class with context-aware steps
2. **`SimpleChatbotManager`** - Manages context snapshots and workflow instances
3. **`Context`** - LlamaIndex's built-in context object for state management
4. **Context Snapshots** - Serialized context state stored between workflow executions

## Data Flow

### 1. Initial Message (First Conversation)

```
User Message → API Request → Manager → New Workflow → Context (Empty) → Response
```

**Step-by-step:**

1. **User sends first message** via API
2. **Manager creates new workflow** for session
3. **Context starts empty** (`conversation_history = []`)
4. **Workflow processes message** with empty context
5. **Context is updated** with new conversation
6. **Context snapshot is saved** after workflow completion
7. **Response sent to user**

### 2. Subsequent Messages (Continued Conversation)

```
User Message → API Request → Manager → Restore Context → Workflow → Update Context → Save Snapshot → Response
```

**Step-by-step:**

1. **User sends follow-up message** via API
2. **Manager loads previous context snapshot** for session
3. **New workflow instance created** with restored context
4. **Workflow processes message** with full conversation history
5. **Context is updated** with new message
6. **Context snapshot is saved** for next time
7. **Response sent to user**

## Implementation Details

### Context Storage Structure

The context snapshot contains the conversation history in this structure:

```python
context_snapshot = {
    "state": {
        "state_data": {
            "_data": {
                "conversation_history": [
                    {
                        "user_message": "Hello!",
                        "bot_response": "Hi there!",
                        "timestamp": "2025-09-11T19:15:31.684119"
                    },
                    {
                        "user_message": "How are you?",
                        "bot_response": "I'm doing great!",
                        "timestamp": "2025-09-11T19:15:33.218687"
                    }
                ]
            }
        }
    },
    # ... other context data
}
```

### Workflow Steps

#### 1. `start_conversation_step`

```python
@step
async def start_conversation_step(self, ev: StartEvent, ctx: Context) -> ChatMessage:
    # Load conversation history from context
    conversation_history = await ctx.get("conversation_history", [])
    
    # Process with context awareness
    return ChatMessage(...)
```

#### 2. `process_user_message_step`

```python
@step
async def process_user_message_step(self, ev: ChatMessage, ctx: Context) -> StopEvent:
    # Get conversation history from context
    conversation_history = await ctx.get("conversation_history", [])
    
    # Generate context-aware response
    bot_response = self.generate_contextual_response(ev.user_message, conversation_history)
    
    # Update context with new message
    updated_history = conversation_history + [new_message]
    await ctx.set("conversation_history", updated_history)
    
    return StopEvent(result={...})
```

### Manager Operations

#### Context Snapshot Management

```python
class SimpleChatbotManager:
    def __init__(self):
        self.context_snapshots: Dict[str, Dict] = {}  # Store context snapshots
    
    async def send_message(self, session_id: str, user_message: str):
        # Load previous context snapshot
        previous_context = self.context_snapshots.get(session_id, {})
        
        # Run workflow
        handler = checkpointer.run(input_data={...})
        result = await handler
        
        # Snapshot context after completion
        context_snapshot = handler.ctx.to_dict()
        self.context_snapshots[session_id] = context_snapshot
```

## Benefits of Context-Based Approach

### ✅ **Advantages**

1. **Native LlamaIndex Integration** - Uses built-in Context object
2. **Stateless API** - Each workflow is independent
3. **Context Persistence** - Conversation history survives between API calls
4. **Context-Aware Responses** - Bot can reference previous messages
5. **Automatic Serialization** - Context handles serialization/deserialization
6. **Workflow Isolation** - Each session has independent context

### 🔄 **Data Flow Summary**

```
Message 1: Empty Context → Process → Save Snapshot
Message 2: Load Snapshot → Process → Save Snapshot  
Message 3: Load Snapshot → Process → Save Snapshot
...
```

## Context-Aware Response Generation

The bot generates responses based on conversation history:

```python
def generate_contextual_response(self, user_message: str, conversation_history: List[Dict]) -> str:
    # Check for follow-up questions
    if len(conversation_history) > 0:
        last_bot_response = conversation_history[-1].get("bot_response", "").lower()
        
        if "python" in user_msg and "python" in last_bot_response:
            return "Building on what I said about Python, here's more detail..."
    
    # Generate appropriate response
    return response
```

## API Response Structure

Each API response includes the full conversation history:

```json
{
    "session_id": "test_session_123",
    "bot_response": "Hello! Nice to meet you!",
    "conversation_summary": "User asked: 'Hello!' and bot responded: 'Hello! Nice to meet you!'",
    "conversation_history": [
        {
            "user_message": "Hello!",
            "bot_response": "Hello! Nice to meet you!",
            "timestamp": "2025-09-11T19:15:31.684119"
        }
    ],
    "status": "completed"
}
```

## Comparison with Event-Based Approach

| Aspect | Context-Based | Event-Based |
|--------|---------------|-------------|
| **Storage** | Context snapshots | Event data |
| **Complexity** | Lower | Higher |
| **LlamaIndex Integration** | Native | Manual |
| **Context Access** | `await ctx.get()` | Event properties |
| **Serialization** | Automatic | Manual |
| **State Management** | Built-in | Custom |

## Conclusion

The context-based approach provides a clean, native way to handle conversation history in LlamaIndex workflows. It leverages the framework's built-in capabilities while maintaining a stateless API architecture that can scale effectively.

The key insight is that **Context snapshots can be extracted after workflow completion** and **restored before the next workflow execution**, creating a seamless conversation experience across independent workflow instances.

