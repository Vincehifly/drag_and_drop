"""
Simple demonstration of LlamaIndex WorkflowCheckpointer with human-in-the-loop workflow
"""
import asyncio
from llamaindex_human_in_loop_checkpointed import HumanInTheLoopManager

async def demo_checkpointing():
    """Demonstrate workflow checkpointing capabilities"""
    print("🚀 LlamaIndex WorkflowCheckpointer Demo")
    print("=" * 50)
    
    # Create manager
    manager = HumanInTheLoopManager()
    
    # Start workflow
    print("\n1️⃣ Starting new workflow...")
    result = await manager.start_workflow()
    
    if result:
        print(f"\n✅ Workflow completed: {result}")
        
        # Show checkpoints created
        print("\n2️⃣ Checkpoints created during execution:")
        manager.list_checkpoints()
        
        # Show all runs
        print("\n3️⃣ All workflow runs:")
        manager.list_all_runs()
        
        # Demonstrate resuming (if user wants to)
        print("\n4️⃣ Resume workflow? (y/n)")
        if input().lower() == 'y':
            print("Resuming from latest checkpoint...")
            result2 = await manager.resume_workflow()
            if result2:
                print(f"✅ Resumed workflow completed: {result2}")

if __name__ == '__main__':
    asyncio.run(demo_checkpointing())
