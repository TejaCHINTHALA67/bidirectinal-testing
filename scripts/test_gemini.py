
import os
import google.generativeai as genai

def test_gemini_connection():
    print("Testing Gemini Connection...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        return

    print(f"API Key found: {api_key[:5]}...")

    genai.configure(api_key=api_key)
    
    model_name = "gemini-2.5-flash-preview-09-2025"
    print(f"Initializing model: {model_name}")
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # Explicit safety settings using types
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        print("Generating content with BLOCK_NONE safety settings...")
        response = model.generate_content(
            "Hello, this is a test query to verify connection.",
            safety_settings=safety_settings
        )
        
        print(f"Response: {response.text}")
        print("SUCCESS: Connection and generation working!")
        
    except Exception as e:
        print(f"ERROR: Generation failed: {e}")
        if hasattr(e, 'response'):
             print(f"Response Feedback: {e.response.prompt_feedback}")

if __name__ == "__main__":
    test_gemini_connection()
