"""
Simple test client for the LlamaIndex Chatbot API (No OpenAI required)
"""
import asyncio
import httpx
import json
from datetime import datetime

class SimpleChatbotTestClient:
    """Simple test client for the chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def send_message(self, message: str) -> dict:
        """Send a message to the chatbot"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json={
                        "session_id": self.session_id,
                        "message": message
                    }
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"error": f"HTTP error: {e}"}
            except Exception as e:
                return {"error": f"Error: {e}"}
    
    async def get_session_info(self) -> dict:
        """Get session information"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/sessions/{self.session_id}/info")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"error": f"HTTP error: {e}"}

async def demo_conversation():
    """Demo conversation showing checkpointing"""
    client = SimpleChatbotTestClient()
    
    print("🎬 Demo Conversation with Checkpointing")
    print("=" * 50)
    print(f"Session ID: {client.session_id}")
    print("🤖 Using simple chatbot (no OpenAI required)")
    print()
    
    # Demo messages
    messages = [
        "Hello! How are you today?",
        "Can you help me with Python programming?",
        "What's the difference between a list and a tuple?",
        "Can you give me an example of a function?",
        "Thank you for your help!"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"💬 Message {i}: {message}")
        response = await client.send_message(message)
        
        if "error" not in response:
            print(f"🤖 Bot: {response['bot_response']}")
            print(f"📊 Status: {response['status']}")
        else:
            print(f"❌ Error: {response['error']}")
        
        print()
        
        # Show checkpoints after each message
        info = await client.get_session_info()
        if "error" not in info:
            print(f"🔍 Checkpoints after message {i}: {len(info['checkpoints'])}")
            for j, checkpoint in enumerate(info['checkpoints']):
                print(f"  {j}: {checkpoint['last_completed_step']} → {checkpoint['output_event']}")
        print("-" * 30)
    
    # Final session info
    print("\n📊 Final Session Info:")
    info = await client.get_session_info()
    if "error" not in info:
        print(f"Checkpoints: {len(info['checkpoints'])}")
        print(f"Messages: {len(info['conversation_history'])}")
        
        print("\n📜 Full Conversation History:")
        for msg in info['conversation_history']:
            print(f"  User: {msg['user_message']}")
            print(f"  Bot:  {msg['bot_response']}")
            print()

async def interactive_chat():
    """Interactive chat session"""
    client = SimpleChatbotTestClient()
    
    print("🤖 Simple LlamaIndex Chatbot Test Client")
    print("=" * 50)
    print(f"Session ID: {client.session_id}")
    print("🤖 Using simple chatbot (no OpenAI required)")
    print("Type 'quit' to exit, 'info' for session info")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'info':
                info = await client.get_session_info()
                print(f"\n📊 Session Info:")
                print(json.dumps(info, indent=2))
                print()
                continue
            elif not user_input:
                continue
            
            print("🤖 Bot: ", end="", flush=True)
            response = await client.send_message(user_input)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(response["bot_response"])
                if response.get("conversation_summary"):
                    print(f"\n📝 Summary: {response['conversation_summary']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

async def main():
    """Main function"""
    print("Choose test mode:")
    print("1. Interactive chat")
    print("2. Demo conversation")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        await interactive_chat()
    elif choice == "2":
        await demo_conversation()
    else:
        print("Invalid choice. Starting interactive chat...")
        await interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())
