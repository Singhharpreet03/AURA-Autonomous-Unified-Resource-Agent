import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

_GEMINI_CLIENT = None

# --- INITIALIZATION LOGIC (Runs once when the module is imported) ---
load_dotenv()
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    print("Gemini Client initialized successfully!")
    
except Exception as e:
    # Set the client to None on failure, and handle the error gracefully
    _GEMINI_CLIENT = None
    print(f"Error initializing Gemini client: {e}")
    print("WARNING: Gemini features will be disabled. Check 'GEMINI_API_KEY'.")

# my_agent/gemini_script_generator.py (Updated)

# ... (Existing imports and _GEMINI_CLIENT initialization remain the same) ...

def generate_script_from_prompt(
    instructions: str, 
    output_path: str, 
    script_type: str = 'bash'
) -> str:
    """
    Generates a script using the Gemini API, saves it to the specified path, 
    and returns a success message or an ERROR.

    Args:
        instructions: The steps or goal in natural language.
        output_path: The full file path (e.g., /var/agent/scripts/task_1.sh) 
                     where the script should be saved.
        script_type: The desired output script language ('python', 'bash', or 'powershell').
    
    Returns:
        A success message or an ERROR message string.
    """
    global _GEMINI_CLIENT
    
    # 1. Check client initialization
    if _GEMINI_CLIENT is None:
        return "ERROR: Gemini client is not initialized. Cannot generate script."
    
    script_type = script_type.lower()
    
    # ... (System Instruction and Prompt definitions remain the same) ...
    system_instruction = (
        "You are an expert AI code generator. Your sole purpose is to convert natural language "
        "instructions into a functional script in the requested language. "
        "The entire output MUST be ONLY the raw code wrapped in a single Markdown code block, "
        "and contain nothing else. Do not add any explanation, commentary, or conversational text. "
        f"The markdown language tag MUST be the requested type: '{script_type}'."
    )
    prompt = (
        f"Generate a robust, ready-to-execute {script_type} script for the following task:\n"
        f"TASK: {instructions}"
    )

    print(f"Sending request to Gemini for a {script_type.upper()} script...")
    
    try:
        # 2. Call the API
        response = _GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2 
            )
        )
        
        generated_text = response.text.strip()
        
        # 3. Extract the clean code content
        start_tag = f"```{script_type}"
        end_tag = "```"
        
        if generated_text.startswith(start_tag) and generated_text.endswith(end_tag):
            code_content = generated_text[len(start_tag):].lstrip('\n').rstrip('\n').rstrip(end_tag)
        else:
            # Fallback
            code_content = generated_text
        
        # 4. Save the generated script to the specified path
        try:
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(code_content)
                
            return f"SUCCESS: Script generated and saved to: {output_path}"
            
        except Exception as file_error:
            return f"ERROR: Failed to save script to {output_path}: {file_error}"


    except Exception as api_error:
        return f"ERROR: Gemini API call failed: {api_error}"


# Example Usage (in your main agent logic):
if __name__ == '__main__':
    # Define where the agent stores its scripts
    SCRIPTS_FOLDER = "Scripts_folder" 
    INSTRUCTIONS = "Create a bash script that installs Nginx and ensures it starts on boot."
    OUTPUT_SCRIPT_TYPE = 'bash'
    
    # Define the final path, ensuring the extension matches the script type
    ext_map = {'python': 'py', 'bash': 'sh', 'powershell': 'ps1'}
    file_extension = ext_map.get(OUTPUT_SCRIPT_TYPE.lower(), 'txt')
    output_filename = f"1-script.{file_extension}" 
    full_path = os.path.join(SCRIPTS_FOLDER, output_filename)
    
    result = generate_script_from_prompt(INSTRUCTIONS, full_path, OUTPUT_SCRIPT_TYPE)
    
    print(f"\n--- Result ---\n{result}")

    if result.startswith("SUCCESS"):
        # The script is saved, you can now proceed to execute the file at full_path
        with open(full_path, 'r') as f:
            print("\n--- Saved Script Content ---")
            print(f.read())