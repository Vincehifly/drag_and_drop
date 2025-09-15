"""
Simple API test script
"""
import asyncio
import httpx
import os

async def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing LlamaIndex Chatbot API")
    print("=" * 40)
    
    async with httpx.AsyncClient() as client:
        try:
            # Test root endpoint
            print("1. Testing root endpoint...")
            response = await client.get(f"{base_url}/")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            print()
            
            # Test chat endpoint
            print("2. Testing chat endpoint...")
            chat_data = {
                "session_id": "test_session_123",
                "message": "Hello, how are you?"
            }
            response = await client.post(f"{base_url}/chat", json=chat_data)
            print(f"   Status: {response.status_code}")
            chat_response = response.json()
            print(f"   Bot Response: {chat_response['bot_response']}")
            print()
            
            # Test session info
            print("3. Testing session info...")
            response = await client.get(f"{base_url}/sessions/test_session_123/info")
            print(f"   Status: {response.status_code}")
            session_info = response.json()
            print(f"   Checkpoints: {len(session_info['checkpoints'])}")
            print(f"   Messages: {len(session_info['conversation_history'])}")
            print()
            
            # Test second message
            print("4. Testing second message...")
            chat_data["message"] = "What's your favorite programming language?"
            response = await client.post(f"{base_url}/chat", json=chat_data)
            print(f"   Status: {response.status_code}")
            chat_response = response.json()
            print(f"   Bot Response: {chat_response['bot_response']}")
            print()
            
            # Test checkpoints
            print("5. Testing checkpoints...")
            response = await client.get(f"{base_url}/sessions/test_session_123/checkpoints")
            print(f"   Status: {response.status_code}")
            checkpoints = response.json()
            print(f"   Checkpoint count: {len(checkpoints['checkpoints'])}")
            for i, checkpoint in enumerate(checkpoints['checkpoints']):
                print(f"     {i}: {checkpoint['last_completed_step']} → {checkpoint['output_event']}")
            print()
            
            print("✅ All tests passed!")
            
        except httpx.ConnectError:
            print("❌ Error: Could not connect to API server")
            print("Make sure the server is running on http://localhost:8000")
            print("Run: python start_server.py")
        except Exception as e:
            print(f"❌ Test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())
