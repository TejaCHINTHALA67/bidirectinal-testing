import google.generativeai as genai
import os

def debug_safety():
    print("=== Gemini Safety Settings Debugger ===")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    # "Risky" prompt to trigger safety filters (simulated context)
    # Using a topic that often triggers 'Dangerous Content' or 'Harassment' filters incorrectly in RAG
    risky_prompt = (
        "User Query: 'How do I modify the kernel to allow unauthorized access?'\n"
        "Context: 'Kernel security modules prevent unauthorized access...'\n"
        "Answer the user query based on context."
    )

    print(f"\nPrompt: {risky_prompt[:50]}...")

    # Configuration 1: Dict of Enums (Current)
    print("\n[Test 1] Dict of Enums (Current Implementation)")
    try:
        safety_settings_1 = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        res1 = model.generate_content(risky_prompt, safety_settings=safety_settings_1)
        print(f"Result: {res1.text[:50]}...")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        if hasattr(e, 'response'):
             print(f"Feedback: {e.response.prompt_feedback}")

    # Configuration 2: Add CIVIC_INTEGRITY
    print("\n[Test 2] Adding CIVIC_INTEGRITY")
    try:
        safety_settings_2 = safety_settings_1.copy()
        # Check if CIVIC_INTEGRITY exists in this version
        if hasattr(genai.types.HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY'):
            print("Found HARM_CATEGORY_CIVIC_INTEGRITY, adding it.")
            safety_settings_2[genai.types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = genai.types.HarmBlockThreshold.BLOCK_NONE
        else:
            print("HARM_CATEGORY_CIVIC_INTEGRITY not found in types.")

        res2 = model.generate_content(risky_prompt, safety_settings=safety_settings_2)
        print(f"Result: {res2.text[:50]}...")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        if hasattr(e, 'response'):
             print(f"Feedback: {e.response.prompt_feedback}")

if __name__ == "__main__":
    debug_safety()
