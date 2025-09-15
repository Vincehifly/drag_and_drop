"""
Quick start script for LlamaIndex Chatbot API
"""
import subprocess
import sys
import time
import asyncio
import httpx

async def test_server():
    """Test if server is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/")
            return response.status_code == 200
    except:
        return False

async def main():
    """Main function"""
    print("🚀 LlamaIndex Chatbot API Quick Start")
    print("=" * 50)
    
    print("Choose version:")
    print("1. Simple version (no OpenAI required)")
    print("2. Full version (requires OpenAI API key)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        print("\n🤖 Starting Simple Chatbot API...")
        print("📚 No OpenAI API key required!")
        print("🌐 Server will start on http://localhost:8000")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        try:
            subprocess.run([sys.executable, "simple_main.py"])
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
    
    elif choice == "2":
        print("\n🤖 Starting Full Chatbot API...")
        print("🔑 Requires OpenAI API key")
        print("🌐 Server will start on http://localhost:8000")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        try:
            subprocess.run([sys.executable, "start_server.py"])
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
    
    else:
        print("Invalid choice. Starting simple version...")
        subprocess.run([sys.executable, "simple_main.py"])

if __name__ == "__main__":
    asyncio.run(main())

