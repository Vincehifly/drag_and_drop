# LlamaIndex Workflow Checkpointing Guide

This guide covers the essential patterns for using LlamaIndex WorkflowCheckpointer to create resumable workflows.

## 1. How to Initiate a Checkpointer

### Basic Setup
```python
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from llama_index.core.workflow import Workflow

# Create your workflow class
class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> MyEvent:
        return MyEvent(data="step one complete")

# Initialize checkpointer
workflow = MyWorkflow()
checkpointer = WorkflowCheckpointer(workflow=workflow)
```

### With Custom Configuration
```python
# Advanced checkpointer setup
checkpointer = WorkflowCheckpointer(
    workflow=workflow,
    # Additional configuration options can be added here
)
```

## 2. How to Run a Workflow

### Start New Workflow
```python
# Start a new workflow execution
handler = checkpointer.run()
result = await handler

# Get the run_id for later reference
run_id = handler.run_id
print(f"Workflow started with run_id: {run_id}")
```

### Run with Input Data
```python
# Start workflow with specific input
handler = checkpointer.run(input_data={"user_id": "123", "task": "analysis"})
result = await handler
```

### Monitor Execution
```python
# Run and monitor progress
handler = checkpointer.run()

try:
    result = await handler
    print(f"Workflow completed: {result}")
except Exception as e:
    print(f"Workflow failed: {e}")
    # Checkpoint is still saved for resume
```

## 3. Checkpoint Data Structure When Interrupted

### Complete Checkpoint Object
```python
checkpoint = {
    'id': '049ffb7b-595d-40f3-b6da-3a50ba1ebcc7',
    'last_completed_step': 'start_process',
    'input_event': 'StartEvent()',
    'output_event': 'ObjectiveEvent(objective="", message="What is your main objective?")',
    'ctx_state': {
        'state': {
            'state_data': {'_data': {}},
            'state_type': 'DictState',
            'state_module': 'workflows.context.state_store'
        },
        'streaming_queue': '[]',
        'queues': {
            '_done': '[]',
            'gather_approach': '[]',
            'gather_notes': '[]',
            'gather_objective': '[]',
            'gather_resources': '[]',
            'start_process': '[]'
        },
        'stepwise': False,
        'event_buffers': {},
        'in_progress': {'start_process': []},
        'accepted_events': [
            ('start_process', 'StartEvent'),
            ('gather_objective', 'ObjectiveEvent'),
            ('gather_resources', 'ResourcesEvent'),
            ('gather_approach', 'ApproachEvent'),
            ('gather_notes', 'NotesEvent')
        ],
        'broker_log': [
            '{"__is_pydantic": true, "value": {}, "qualified_name": "workflows.events.StartEvent"}'
        ],
        'is_running': True
    }
}
```

### Key Checkpoint Fields Explained

| Field | Description | Resume Usage |
|-------|-------------|--------------|
| `id` | Unique checkpoint identifier | Reference for resume |
| `last_completed_step` | Last successfully completed step | Determines next step to run |
| `input_event` | Event that triggered the last step | Context for step execution |
| `output_event` | Event produced by last step | Input for next step |
| `accepted_events` | Step routing map | Determines step sequence |
| `ctx_state` | Complete workflow state | Restores execution context |
| `in_progress` | Currently running steps | Identifies active steps |
| `broker_log` | Event history | Debugging and audit trail |

### Resume Decision Logic
```python
def determine_resume_point(checkpoint):
    last_completed = checkpoint['last_completed_step']
    accepted_events = checkpoint['accepted_events']
    
    # Find next step in sequence
    for i, (step_name, event_type) in enumerate(accepted_events):
        if step_name == last_completed:
            if i + 1 < len(accepted_events):
                return accepted_events[i + 1][0]  # Next step
            else:
                return "completed"
    return "error"
```

## 4. How to Resume a Workflow

### Resume from Latest Checkpoint
```python
# Get all checkpoints for a run
run_id = "your-run-id"
checkpoints = checkpointer.checkpoints[run_id]

# Resume from the latest checkpoint
latest_checkpoint = checkpoints[-1]
handler = checkpointer.run_from(checkpoint=latest_checkpoint)
result = await handler
```

### Resume from Specific Checkpoint
```python
# Resume from a specific checkpoint by index
checkpoint_index = 2  # Third checkpoint
checkpoint = checkpointer.checkpoints[run_id][checkpoint_index]
handler = checkpointer.run_from(checkpoint=checkpoint)
result = await handler
```

### Resume with User Input
```python
# For human-in-the-loop workflows
checkpoint = checkpointer.checkpoints[run_id][-1]

# Modify the output event with user input
if hasattr(checkpoint.output_event, 'user_input'):
    checkpoint.output_event.user_input = "User provided input"

# Resume with modified event
handler = checkpointer.run_from(checkpoint=checkpoint)
result = await handler
```

### Complete Resume Example
```python
async def resume_workflow_demo():
    # 1. Get checkpoint
    run_id = "abc123"
    checkpoints = checkpointer.checkpoints[run_id]
    checkpoint = checkpoints[-1]
    
    # 2. Resume workflow
    handler = checkpointer.run_from(checkpoint=checkpoint)
    
    # 3. Continue execution
    try:
        result = await handler
        print(f"Workflow resumed and completed: {result}")
    except Exception as e:
        print(f"Resume failed: {e}")
```

## 5. Practical Patterns

### List All Checkpoints
```python
# List checkpoints for a specific run
run_id = "your-run-id"
if run_id in checkpointer.checkpoints:
    checkpoints = checkpointer.checkpoints[run_id]
    for i, checkpoint in enumerate(checkpoints):
        print(f"Checkpoint {i}: {checkpoint.last_completed_step}")
```

### List All Runs
```python
# List all workflow runs
for run_id, checkpoints in checkpointer.checkpoints.items():
    print(f"Run {run_id}: {len(checkpoints)} checkpoints")
```

### Checkpoint Management
```python
# Get checkpoint by ID
def get_checkpoint_by_id(run_id, checkpoint_id):
    checkpoints = checkpointer.checkpoints[run_id]
    for checkpoint in checkpoints:
        if checkpoint.id == checkpoint_id:
            return checkpoint
    return None

# Resume from specific checkpoint ID
checkpoint = get_checkpoint_by_id(run_id, "049ffb7b-595d-40f3-b6da-3a50ba1ebcc7")
if checkpoint:
    handler = checkpointer.run_from(checkpoint=checkpoint)
    result = await handler
```

## 6. Error Handling

### Safe Resume
```python
async def safe_resume(run_id, checkpoint_index=0):
    try:
        if run_id not in checkpointer.checkpoints:
            raise ValueError(f"No checkpoints found for run {run_id}")
        
        checkpoints = checkpointer.checkpoints[run_id]
        if checkpoint_index >= len(checkpoints):
            raise ValueError(f"Checkpoint index {checkpoint_index} out of range")
        
        checkpoint = checkpoints[checkpoint_index]
        handler = checkpointer.run_from(checkpoint=checkpoint)
        result = await handler
        return result
        
    except Exception as e:
        print(f"Resume failed: {e}")
        return None
```

## 7. Best Practices

1. **Always check if checkpoints exist** before attempting resume
2. **Handle exceptions** during resume operations
3. **Store run_id** when starting workflows for later reference
4. **Use latest checkpoint** unless you need to resume from a specific point
5. **Monitor checkpoint count** to avoid memory issues in long-running applications
6. **Clean up old checkpoints** periodically to manage storage

This guide provides the essential patterns for implementing resumable workflows with LlamaIndex checkpoints.
