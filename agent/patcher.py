import os
from google import genai
from google.genai import types

# Initialize the Gemini client.
# It automatically looks for the GEMINI_API_KEY environment variable.
try:
    client = genai.Client(api_key="")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure you have the 'google-genai' library installed and your 'GEMINI_API_KEY' environment variable is set.")
    exit()


def generate_script_with_gemini(instructions: str, script_type: str = 'python') -> str:
    """
    Generates a script (Python, Bash, or PowerShell) using the Gemini API 
    based on natural language instructions.

    Args:
        instructions: The steps or goal in natural language.
        script_type: The desired output script language ('python', 'bash', or 'powershell').
    
    Returns:
        The generated script content as a raw string.
    """
    
    # 1. Define the System Instruction for tight control over the output format
    # This instructs the model to act as a code-generation engine and use markdown.
    system_instruction = (
        "You are an expert AI code generator. Your sole purpose is to convert natural language "
        "instructions into a functional script in the requested language. "
        "The entire output MUST be ONLY the raw code wrapped in a single Markdown code block, "
        "and contain nothing else. Do not add any explanation, commentary, or conversational text. "
        f"The markdown language tag MUST be the requested type: '{script_type}'."
    )

    # 2. Define the User Prompt
    # This is the actual task the user wants to accomplish.
    prompt = (
        f"Generate a robust, ready-to-execute {script_type} script for the following task:\n"
        f"TASK: {instructions}"
    )

    print(f"Sending request to Gemini for a {script_type.upper()} script...")
    
    try:
        # Call the API with the System Instruction
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Fast model suitable for code generation
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # Low temperature for deterministic, factual (code) output
                temperature=0.2 
            )
        )
        
        # 3. Extract and clean the generated code from the Markdown block
        generated_text = response.text.strip()
        
        # Expected format is ```language\ncode\n```. We need to strip the wrapping.
        start_tag = f"```{script_type.lower()}"
        end_tag = "```"
        
        if generated_text.startswith(start_tag) and generated_text.endswith(end_tag):
            # Remove the start tag (and the following newline) and the end tag
            code_content = generated_text[len(start_tag):].lstrip('\n').rstrip('\n').rstrip(end_tag)
            return code_content
        else:
            # Fallback in case the model deviates from the strict format
            print("Warning: Generated text did not match the expected Markdown format. Returning raw text.")
            return generated_text

    except Exception as e:
        return f"ERROR: Gemini API call failed: {e}"

# ====================================================================

## USER-CONFIGURABLE VARIABLES
# 1. Variable to set for giving output script in either python bash or powershell.
OUTPUT_SCRIPT_TYPE = 'python' # Change to 'python', 'bash', or 'powershell'

# 2. Natural language instructions for the script
INSTRUCTIONS = "Create a script that runs sudo apt update and sudo apt upgrade in bash for my ubuntu"

# --- Script Execution ---

print(f"--- Requested Script Type: {OUTPUT_SCRIPT_TYPE.upper()} ---")

# 1. Generate the script
generated_script_content = generate_script_with_gemini(INSTRUCTIONS, OUTPUT_SCRIPT_TYPE)

if generated_script_content.startswith("ERROR"):
    print(generated_script_content)
    exit()

# 2. Determine the file extension for saving
ext_map = {'python': 'py', 'bash': 'sh', 'powershell': 'ps1'}
file_extension = ext_map.get(OUTPUT_SCRIPT_TYPE.lower(), 'txt')

# 3. Define the output file name and path
SCRIPTS_FOLDER = "Scripts_folder" 
output_filename = f"1-script.{file_extension}" 
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)
full_path = os.path.join(SCRIPTS_FOLDER, output_filename)

# 4. Save the generated script
with open(full_path, 'w') as f:
    f.write(generated_script_content)
    
# --- Output ---
print("\n--- Generated Script Content ---")
print(generated_script_content)

print(f"\n✅ Script successfully generated and saved to: {full_path}")