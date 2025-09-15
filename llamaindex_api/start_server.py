"""
Startup script for the LlamaIndex Chatbot API
"""
import os
import sys
import uvicorn

def check_environment():
    """Check if required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  or")
        print("  set OPENAI_API_KEY=your-api-key-here  # Windows")
        return False
    return True

def main():
    """Main startup function"""
    print("🚀 Starting LlamaIndex Chatbot API...")
    
    if not check_environment():
        sys.exit(1)
    
    print("✅ Environment check passed")
    print("🌐 Starting server on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

