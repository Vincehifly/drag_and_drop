# Drag and Drop Agent Builder

> A configurable conversational AI agent built with LangGraph for data collection and information retrieval.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.10+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Your First Agent
```bash
python interactive_test.py
```

### 3. Choose an Agent
- **sheets_agent**: Collects and saves data to Google Sheets
- **web_search_agent**: Searches the web for information
- **api_retrieval_agent**: Fetches data from external APIs

## 📚 Core Concepts

### What This Agent Does
This agent can:
- **Collect and save data** using input tools (Google Sheets, email, etc.)
- **Retrieve information** using retrieval tools (web search, API calls, etc.)
- **Adapt behavior** based on JSON configuration files
- **Handle complex conversations** with intelligent routing and validation

### Key Architecture Principles
- **Unified Tool Flow**: All tools follow the same execution path
- **Configuration-Driven**: Behavior controlled by JSON configs
- **Schema Validation**: Inputs validated against tool schemas
- **Decision Transparency**: Clear reasoning for every action

## 🏗️ Architecture Overview

### Simplified Unified Flow
```
User Input → Decision Router → [Tool Path or Chat Path] → Wait for Next Input
```

**Tool Path**: `Structured Extractor → Validate Inputs → Tool Execution → Tool Answer`
**Chat Path**: `Chat Node → Response`

### Core Components
- **Decision Router**: Analyzes conversation and chooses next action
- **Structured Extractor**: Extracts data from user messages
- **Input Validator**: Validates data against tool schemas
- **Tool Executor**: Runs the selected tool
- **Tool Answer**: Generates user responses

## 🎯 Graph Visualization

### Agent Workflow Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AGENT WORKFLOW GRAPH STRUCTURE                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   🔄 WAIT USER  │
                    │     INPUT       │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   🧠 DECISION   │
                    │    ROUTER       │
                    └─────────┬───────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
            ┌─────────────┐ ┌─────┐ ┌─────┐
            │ 📝 STRUCTURED│ │ 💬 │ │ 🚪 │
            │  EXTRACTOR  │ │CHAT│ │ END │
            └─────┬───────┘ └─┬───┘ └─┬───┘
                  │           │       │
                  ▼           │       │
            ┌─────────────┐   │       │
            │ ✅ VALIDATE │   │       │
            │   INPUTS    │   │       │
            └─────┬───────┘   │       │
                  │           │       │
            ┌─────┼─────┐     │       │
            │     │     │     │       │
            ▼     ▼     ▼     │       │
    ┌─────────────┐ ┌─────┐   │       │
    │ ⚙️ TOOL     │ │ 💬 │   │       │
    │ EXECUTION   │ │CHAT│   │       │
    └─────┬───────┘ └─┬───┘   │       │
          │           │       │       │
          ▼           │       │       │
    ┌─────────────┐   │       │       │
    │ 📤 TOOL     │   │       │       │
    │   ANSWER    │   │       │       │
    └─────┬───────┘   │       │       │
          │           │       │       │
          └───────────┼───────┼───────┘
                      │       │
                      ▼       ▼
                    ┌─────────────┐
                    │   🔄 WAIT  │
                    │ USER INPUT │
                    └─────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    EXECUTION PATHS                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

🛠️  TOOL PATH:     Wait Input → Decision Router → Structured Extractor → Validate Inputs → Tool Execution → Tool Answer → Wait Input
💬  CHAT PATH:      Wait Input → Decision Router → Chat Node → Wait Input  
🚪  EXIT PATH:      Wait Input → Decision Router → End Node → Terminate
🔄  MAIN LOOP:      All paths return to "Wait User Input" for continuous conversation
```

### Node Color Legend
- 🟢 **Green**: Input/Output nodes (User Input, Tool Answer)
- 🔵 **Blue**: Control nodes (Decision Router)
- 🟠 **Orange**: Processing nodes (Structured Extractor)
- 🟣 **Purple**: Validation nodes (Validate Inputs)
- 🔴 **Red**: Action nodes (Tool Execution)
- 🟦 **Cyan**: Response nodes (Tool Answer)
- 🟫 **Brown**: Communication nodes (Chat)
- 🔘 **Grey**: Termination nodes (End)

## 🧩 Node Definitions

### Core Execution Nodes

#### 1. **wait_user_input_node** 🟢
- **Purpose**: Waits for user input using LangGraph's interrupt mechanism
- **Input**: None (waits for user)
- **Output**: User message added to state
- **State Changes**: `user_input`, `messages`
- **Key Features**: 
  - Human-in-the-loop interaction
  - Session management
  - Input validation

#### 2. **decision_router_node** 🔵
- **Purpose**: Analyzes conversation state and chooses next action
- **Input**: User message, conversation history, available tools
- **Output**: Next action decision with justification
- **State Changes**: `next_action`, `chosen_tool`, `tool_category`, `decision_justification`, `decision_context`
- **Key Features**: 
  - Multi-tool awareness
  - Action history tracking
  - Loop prevention
  - Decision context capture

#### 3. **structured_extractor_node** 🟠
- **Purpose**: Extracts structured data from user messages for tool usage
- **Input**: User message, tool requirements
- **Output**: Extracted data in structured format
- **State Changes**: `extracted_data`
- **Key Features**: 
  - Schema-driven extraction
  - Validation preparation
  - Context-aware parsing

#### 4. **validate_inputs_node** 🟣
- **Purpose**: Validates extracted data against tool schemas
- **Input**: Extracted data, tool schema
- **Output**: Validation result with errors if any
- **State Changes**: `validation_errors`
- **Key Features**: 
  - Schema validation
  - Error reporting
  - Missing field identification

#### 5. **tool_execution_node** 🔴
- **Purpose**: Executes the selected tool with validated data
- **Input**: Tool name, validated data, tool configuration
- **Output**: Tool execution result
- **State Changes**: `tool_result`
- **Key Features**: 
  - Dynamic tool resolution
  - Error handling
  - Result formatting

#### 6. **tool_answer_node** 🟦
- **Purpose**: Generates user-facing responses from tool results
- **Input**: Tool result, decision context, user query
- **Output**: Natural language response
- **State Changes**: `last_tool_summary`, `last_tool_context`, `messages`
- **Key Features**: 
  - Decision context integration
  - Tool category-specific responses
  - Context preservation

#### 7. **chat_node** 🟫
- **Purpose**: Generates conversational responses when no tool is needed
- **Input**: User message, conversation context, decision context
- **Output**: Helpful conversational response
- **State Changes**: `messages`
- **Key Features**: 
  - Tool-aware responses
  - Context integration
  - Natural conversation flow

#### 8. **end_node** 🔘
- **Purpose**: Gracefully ends the conversation
- **Input**: Current state
- **Output**: Final state with conversation marked as inactive
- **State Changes**: `conversation_active`, `next_action`

### Support Nodes

#### **exit_evaluator_node**
- **Purpose**: Evaluates exit conditions for the conversation
- **Input**: Current state, exit conditions
- **Output**: Exit decision
- **State Changes**: `next_action`

#### **query_planner_node**
- **Purpose**: Plans queries for retrieval tools
- **Input**: User query, tool capabilities
- **Output**: Query specification
- **State Changes**: `query_spec`

## 💬 Prompt System

### Prompt Architecture

The agent uses a sophisticated, multi-layered prompt system with context prioritization:

```
┌─────────────────────────────────────┐
│           Base System Prompt        │ ← Core behavioral guidelines
├─────────────────────────────────────┤
│        Agent-Specific Prompt        │ ← Domain-specific instructions
├─────────────────────────────────────┤
│         Tool Descriptions           │ ← Tool capabilities and usage
├─────────────────────────────────────┤
│        Context Injection            │ ← Dynamic state-based content
├─────────────────────────────────────┤
│       Decision Guidance             │ ← Action selection instructions
└─────────────────────────────────────┘
```

### Prompt Skeletons

#### **1. Decision Router Prompt**

**Purpose**: Analyze conversation state and choose next action

**Structure**:
```
SYSTEM PROMPT: {agent_prompt}

CONFIGURED TOOLS:
{tool_display}

{required_input_context}

{conversation_context}

{tool_result_block}

{action_context}

TASK: Decide the next action based on the current state, conversation flow, and recent actions.

CURRENT USER MESSAGE: "{user_message}"

DECISION OPTIONS:
{options_block}
- "chat" if you need to collect more information or provide a conversational response
- "end" if the conversation should be concluded

INSTRUCTIONS:
- Analyze the conversation flow and user intent from the history
- Consider recent actions to avoid repetitive patterns or loops
- Use a configured tool ONLY when all required inputs are available and tool usage is appropriate
- Choose "chat" for collecting missing inputs or providing conversational responses
- Choose "end" when the task is complete or user wants to finish
- If you see repeated patterns in recent actions, adjust your strategy accordingly
- Explain your reasoning briefly, referencing recent actions if relevant

Return your decision as: DECISION: [one of: {tool_names}, chat, end] - [brief reason]
```

**Context Elements**:
- **Tool Display**: List of available tools with descriptions
- **Required Input Context**: What each tool needs to function
- **Conversation Context**: Recent message history (last 5 messages)
- **Tool Result Block**: Previous tool execution results
- **Action Context**: Recent action history to prevent loops

#### **2. Conversational Response Prompt**

**Purpose**: Generate helpful conversational responses with tool awareness

**Structure**:
```
SYSTEM PROMPT: {system_prompt}

{required_input_context}

{conversation_context}

{collected_context}

{decision_context}

{tool_context_block}

TASK: Provide a helpful response that guides the user toward tool usage when appropriate, or offers conversational assistance when no tool is needed.

USER MESSAGE: "{user_message}"

INSTRUCTIONS:
1. Assess if the user's request requires tool usage
2. If yes, guide them toward providing required inputs
3. If no, provide helpful conversational responses
4. Always be clear about what's needed for tools
5. Suggest next steps that move toward completion
```

**Context Elements**:
- **Required Inputs**: Fields needed for available tools
- **Conversation History**: Recent conversation context
- **Collected Data**: Currently available information
- **Decision Context**: Why previous decisions were made
- **Tool Context**: Previous tool results and context

#### **3. Tool Answer Prompt**

**Purpose**: Generate user-facing responses from tool execution results

**Structure**:
```
SYSTEM PROMPT: {system_prompt}

{decision_context_block}

{tool_result_context}

USER TASK/QUERY: {user_query}

TOOL TYPE: {tool_type}
TOOL OUTCOME: {tool_outcome}
TOOL SUMMARY: {tool_summary}

ASSISTANT RESPONSE:
{task_description}
{instructions}
```

**Context Elements**:
- **Decision Context**: Why the tool was chosen and what the user intended
- **Tool Result**: Raw tool execution results
- **Tool Metadata**: Type, outcome, and summary
- **Category-Specific Instructions**: Different guidance for input vs. retrieval tools

#### **4. Input Extraction Prompt**

**Purpose**: Extract structured data from user messages for tool usage

**Structure**:
```
SYSTEM PROMPT INJECTION: {system_prompt}

TOOL PROMPT INJECTION: You are working with a {tool_type} tool that will use the collected information.

REQUIRED INPUT PROMPT INJECTION: You must extract the following required fields from user messages:
{field_descriptions}

TASK: Extract field values from the user's message and return them as JSON.

USER MESSAGE: "{user_message}"

CURRENTLY COLLECTED: {collected}

INSTRUCTIONS:
1. Only extract values that are clearly mentioned in the user's message
2. Don't extract values that are already collected unless the user is updating them
3. Don't make assumptions or fill in missing information
4. Return ONLY the new/updated field values as valid JSON
5. If no extractable values found, return empty JSON object {}

Return only a JSON object with the extracted values for these fields. If a field is not found, omit it.
```

**Context Elements**:
- **System Prompt**: Core behavioral guidelines
- **Tool Context**: Specific tool being prepared for
- **Field Requirements**: What data needs to be extracted
- **Current State**: Already collected information

### Context Prioritization

The prompt system uses intelligent context prioritization to manage token usage:

#### **Priority 1 (Critical - Always Include)**
- System prompt and core instructions
- Tool names and basic requirements
- Current user message

#### **Priority 2 (High Value - Include if Space Available)**
- Missing required fields
- Recent tool results
- Decision context

#### **Priority 3 (Medium Value - Include if Significant)**
- Recent actions (last 2-3)
- Collected data summary
- Conversation summary (last 2-3 messages)

#### **Priority 4 (Low Value - Skip if Over Budget)**
- Full conversation history
- Detailed action logging
- Verbose tool descriptions

### Prompt Optimization Features

#### **Smart Truncation**
- Dynamic context window management
- Token budget enforcement
- Intelligent field prioritization

#### **Context Scoring**
- Relevance-based context selection
- Tool-specific context prioritization
- User intent matching

#### **Template Reusability**
- Modular prompt components
- Consistent structure across nodes
- Easy customization and maintenance
  
## 🛠️ Usage Guide

### Basic Configuration
Create an agent configuration file:

```json
{
  "name": "contact_saver",
  "agent_prompt": "You are a helpful assistant that collects contact information.",
  "tools": [
    {
      "name": "contact_saver",
      "type": "input",
      "impl": "sheets",
      "config": {
        "spreadsheet_title": "Contacts",
        "worksheet_name": "main"
      },
      "input_schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "email"]
      }
    }
  ]
}
```

### Available Tool Types

#### Input Tools (Save Data)
- **Google Sheets** (`impl: "sheets"`): Save to spreadsheets
- **Email** (`impl: "email"`): Send emails via SMTP

#### Retrieval Tools (Fetch Data)
- **Web Search** (`impl: "web_search"`): Search the internet
- **API Retrieval** (`impl: "api_retrieval"`): Call external APIs

### Running Agents

#### Interactive Mode
```bash
python interactive_test.py
```

#### Programmatic Mode
```python
from graph_builder import create_conversation_graph
from llm_client import create_azure_openai_llm

# Create LLM client
llm = create_azure_openai_llm()

# Load configuration
with open("agent_config.json", "r") as f:
    config = json.load(f)

# Create and run graph
graph = create_conversation_graph(config, llm, prompt_functions)
result = await graph.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
```

## 🔧 Development Guide

### Project Structure
```
├── README.md                 # This documentation
├── graph_builder.py          # Graph construction
├── nodes.py                  # Node implementations
├── tools.py                  # Tool implementations
├── prompts.py                # Prompt building functions
├── models.py                 # State definitions
├── validation.py             # Input validation
├── llm_client.py             # LLM configuration
├── agents.json               # Example configurations
├── interactive_test.py       # Interactive testing
└── requirements.txt          # Dependencies
```

### Adding New Tools

#### 1. Implement Tool Function
```python
def custom_tool(extracted_data: Dict[str, Any], config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Custom tool implementation."""
    try:
        # Your tool logic here
        result = perform_operation(extracted_data, config)
        
        return {
            "success": True,
            "data": result,
            "type": "custom_tool",
            "message": "Operation completed successfully",
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": extracted_data,
            "type": "custom_tool",
            "message": f"Operation failed: {str(e)}",
            "error": str(e)
        }
```

#### 2. Register Tool
```python
# In tools.py
AVAILABLE_TOOLS["custom_tool"] = custom_tool
```

#### 3. Add to Configuration
```json
{
  "name": "custom_tool",
  "type": "input",
  "impl": "custom_tool",
  "config": {"option": "value"},
  "input_schema": {
    "type": "object",
    "properties": {"input_field": {"type": "string"}},
    "required": ["input_field"]
  }
}
```

### Customizing Prompts
Modify prompt templates in `prompts.py`:
- `build_decision_prompt`: How the agent decides what to do
- `build_input_extraction_prompt`: How data is extracted
- `build_conversational_response_prompt`: How chat responses are generated
- `build_tool_use_prompt`: How tools are executed
- `build_tool_answer_prompt`: How tool results are presented

## 🐛 Troubleshooting

### Common Issues



#### 1. Tool Not Found
```bash
# Error: Tool not found in AVAILABLE_TOOLS
# Solution: Check tool registration in tools.py
# Make sure tool name matches configuration
```

#### 2. Configuration Errors
```bash
# Error: Invalid tool configuration
# Solution: Validate JSON schema
# Check required fields: name, type, impl, config, input_schema
```

#### 3. Google Sheets Authentication
```bash
# Error: Google Sheets authentication failed
# Solution: Place credentials file in project directory
# Or set credentials_path in tool configuration
```


### Getting Help
- Check the error messages for specific guidance
- Enable verbose logging to see execution flow
- Verify configuration syntax and required fields
- Check environment variables and credentials

## 📖 Configuration Reference

### Agent Configuration Schema
```json
{
  "name": "string",                    // Required: Agent identifier
  "agent_prompt": "string",            // Required: Agent behavior instructions
  "tools": [                           // Required: Array of tool configurations
    {
      "name": "string",                // Required: Tool identifier
      "type": "input|retrieval",       // Required: Tool category
      "impl": "string",                // Required: Implementation name
      "config": {},                    // Required: Tool-specific settings
      "input_schema": {},              // Required: Input requirements
      "description": "string",         // Optional: Human-readable description
      "enabled": true                  // Optional: Enable/disable tool
    }
  ],
  "exit_conditions": [                 // Optional: When to end conversation
    {"type": "prompt", "expression": "thank you|goodbye"},
    {"type": "max_turns", "expression": 25}
  ]
}
```

### Tool Configuration Examples

#### Google Sheets Tool
```json
{
  "name": "data_saver",
  "type": "input",
  "impl": "sheets",
  "config": {
    "spreadsheet_title": "MyData",
    "worksheet_name": "main"
  },
  "input_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "email"]
  }
}
```

#### Web Search Tool
```json
{
  "name": "web_researcher",
  "type": "retrieval",
  "impl": "web_search",
  "config": {
    "max_results": 5,
    "include_snippets": true
  },
  "input_schema": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  }
}
```

## 🎯 Example Use Cases

### Contact Management
- Collect contact information
- Validate email addresses
- Save to Google Sheets
- Handle updates and corrections

### Research Assistant
- Search the web for information
- Query academic APIs
- Compile research summaries
- Track information sources

### Data Collection
- Form-based data entry
- Multi-step data collection
- Validation and error handling
- Data export and reporting

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints where possible
- Include docstrings for functions
- Test your changes thoroughly

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create an issue in the repository
- **Documentation**: Check this README and code comments
- **Examples**: See `agents.json` for configuration examples
- **Testing**: Use `interactive_test.py` for testing agents

---

**Ready to build your first agent?** Start with the [Quick Start](#-quick-start-5-minutes) section above!
