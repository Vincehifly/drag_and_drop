import asyncio
import json
from typing import Any, Dict, List
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context, Event
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer

# Create distinct event types for proper step routing
class ObjectiveEvent(Event):
    objective: str
    message: str = ""

class ResourcesEvent(Event):
    resources: List[str]
    objective: str
    message: str = ""

class ApproachEvent(Event):
    approach: str
    resources: List[str]
    objective: str
    message: str = ""

class NotesEvent(Event):
    notes: str
    approach: str
    resources: List[str]
    objective: str
    message: str = ""

class HumanInTheLoopWorkflow(Workflow):
    """Human-in-the-loop workflow with checkpointing support"""
    
    @step
    async def start_process(self, ev: StartEvent, ctx: Context) -> ObjectiveEvent:
        print("=== LlamaIndex Human-in-the-Loop Workflow with Checkpointing ===")
        print("This workflow can be paused and resumed at any point.")
        return ObjectiveEvent(
            objective="",
            message="What is your main objective for this workflow?"
        )
    
    @step  
    async def gather_objective(self, ev: ObjectiveEvent, ctx: Context) -> ResourcesEvent:
        print(f"\n[Step: gather_objective]")
        user_input = input(f"{ev.message} ")
        print(f"✅ Objective recorded: {user_input}")
        
        return ResourcesEvent(
            resources=[],
            objective=user_input,
            message=f"Your objective: '{user_input}'\nWhat resources will you need? (comma-separated)"
        )
    
    @step
    async def gather_resources(self, ev: ResourcesEvent, ctx: Context) -> ApproachEvent:
        print(f"\n[Step: gather_resources]")
        user_input = input(f"{ev.message} ")
        resources = [r.strip() for r in user_input.split(',')]
        print(f"✅ Resources recorded: {', '.join(resources)}")
        
        return ApproachEvent(
            approach="",
            resources=resources,
            objective=ev.objective,
            message=f"Resources: {', '.join(resources)}\nDescribe your approach:"
        )
    
    @step
    async def gather_approach(self, ev: ApproachEvent, ctx: Context) -> NotesEvent:
        print(f"\n[Step: gather_approach]")
        user_input = input(f"{ev.message} ")
        print(f"✅ Approach recorded: {user_input}")
        
        return NotesEvent(
            notes="",
            approach=user_input,
            resources=ev.resources,
            objective=ev.objective,
            message=f"Approach: {user_input}\nAny additional notes or constraints?"
        )
    
    @step
    async def gather_notes(self, ev: NotesEvent, ctx: Context) -> StopEvent:
        print(f"\n[Step: gather_notes]")
        user_input = input(f"{ev.message} ")
        print(f"✅ Notes recorded: {user_input}")
        
        # Final summary
        summary = {
            'objective': ev.objective,
            'resources': ev.resources,
            'approach': ev.approach,
            'notes': user_input,
            'status': 'completed',
            'timestamp': asyncio.get_event_loop().time()
        }
        
        print("\n=== Workflow Summary ===")
        print(f"Objective: {summary['objective']}")
        print(f"Resources: {', '.join(summary['resources'])}")
        print(f"Approach: {summary['approach']}")
        print(f"Notes: {summary['notes']}")
        print(f"Status: {summary['status']}")
        
        return StopEvent(result=summary)
    

class HumanInTheLoopManager:
    """Manager class for human-in-the-loop workflow with checkpointing"""
    
    def __init__(self):
        self.workflow = HumanInTheLoopWorkflow()
        self.checkpointer = WorkflowCheckpointer(workflow=self.workflow)
        self.current_handler = None
        self.current_run_id = None
    
    async def start_workflow(self):
        """Start a new workflow session"""
        print("🚀 Starting new workflow session...")
        self.current_handler = self.checkpointer.run()
        self.current_run_id = self.current_handler.run_id
        print(f"Run ID: {self.current_run_id}")
        return await self.run_current_session()
    
    async def resume_workflow(self, run_id: str = None):
        """Resume a workflow from a checkpoint"""
        if run_id is None:
            run_id = self.current_run_id
        
        if run_id not in self.checkpointer.checkpoints:
            print(f"❌ No checkpoints found for run ID: {run_id}")
            return None
        
        print(f"🔄 Resuming workflow from run ID: {run_id}")
        checkpoints = self.checkpointer.checkpoints[run_id]
        
        if not checkpoints:
            print("❌ No checkpoints available for this run")
            return None
        
        # Resume from the latest checkpoint
        latest_checkpoint = checkpoints[-1]
        print(f"📍 Resuming from checkpoint: {latest_checkpoint}")
        
        self.current_handler = self.checkpointer.run_from(checkpoint=latest_checkpoint)
        self.current_run_id = run_id
        return await self.run_current_session()
    
    async def run_current_session(self):
        """Run the current workflow session"""
        if not self.current_handler:
            print("❌ No active workflow session")
            return None
        
        try:
            # Run the workflow and get the result
            result = await self.current_handler
            if hasattr(result, 'result'):
                print(f"\n✅ Workflow completed with result: {result.result}")
                return result.result
            else:
                print(f"\n✅ Workflow completed: {result}")
                return result
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return None
    
    def list_checkpoints(self, run_id: str = None):
        """List all available checkpoints"""
        if run_id is None:
            run_id = self.current_run_id
        
        if run_id not in self.checkpointer.checkpoints:
            print(f"❌ No checkpoints found for run ID: {run_id}")
            return []
        
        checkpoints = self.checkpointer.checkpoints[run_id]
        print(f"\n📋 Available checkpoints for run {run_id}:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"  {i}: {checkpoint}")
        return checkpoints
    
    def list_all_runs(self):
        """List all workflow runs"""
        if not self.checkpointer.checkpoints:
            print("❌ No workflow runs found")
            return []
        
        print("\n📋 All workflow runs:")
        for run_id, checkpoints in self.checkpointer.checkpoints.items():
            print(f"  Run {run_id}: {len(checkpoints)} checkpoints")
        return list(self.checkpointer.checkpoints.keys())

async def interactive_menu():
    """Interactive menu for managing human-in-the-loop workflow"""
    manager = HumanInTheLoopManager()
    
    while True:
        print("\n" + "="*60)
        print("🤖 Human-in-the-Loop Workflow Manager")
        print("="*60)
        print("1. Start new workflow")
        print("2. Resume workflow")
        print("3. List checkpoints")
        print("4. List all runs")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            result = await manager.start_workflow()
            if result:
                print(f"\n🎉 Workflow completed successfully!")
        
        elif choice == '2':
            runs = manager.list_all_runs()
            if runs:
                run_id = input(f"Enter run ID to resume (available: {', '.join(runs)}): ").strip()
                if run_id in runs:
                    result = await manager.resume_workflow(run_id)
                    if result:
                        print(f"\n🎉 Workflow resumed and completed successfully!")
                else:
                    print("❌ Invalid run ID")
            else:
                print("❌ No runs available to resume")
        
        elif choice == '3':
            manager.list_checkpoints()
        
        elif choice == '4':
            manager.list_all_runs()
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")

async def demo_workflow():
    """Simple demo of the workflow without interactive menu"""
    print("=== Simple Workflow Demo ===")
    manager = HumanInTheLoopManager()
    result = await manager.start_workflow()
    
    if result:
        print(f"\n🎉 Demo completed! Result: {result}")
    
    # Show checkpoints
    print("\n📋 Checkpoints created during demo:")
    manager.list_checkpoints()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        asyncio.run(demo_workflow())
    else:
        asyncio.run(interactive_menu())
