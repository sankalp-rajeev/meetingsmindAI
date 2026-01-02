"""Quick test script for Gemini client."""
import sys
sys.path.insert(0, ".")

print("Testing Gemini client...")

try:
    from src.app.gemini_client import gemini_generate, GeminiClient
    
    # Test 1: Simple text generation
    print("\n1. Testing text generation...")
    response = gemini_generate("What is 2 + 2? Answer in one word.", temperature=0.1)
    print(f"   Response: {response.strip()}")
    
    # Test 2: GeminiClient class
    print("\n2. Testing GeminiClient class...")
    client = GeminiClient(temperature=0.3)
    response = client.generate_text("Say 'Hello MeetingMind!' exactly.")
    print(f"   Response: {response.strip()}")
    
    print("\n✅ Gemini API is working correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
