import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import subprocess
from . import cryptographic
from . import patcher

_GEMINI_CLIENT = None

# --- INITIALIZATION LOGIC (Runs once when the module is imported) ---
load_dotenv()
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
    patcher._GEMINI_CLIENT = genai.Client(api_key=api_key)
    print("Gemini Client initialized successfully!")
    
except Exception as e:
    # Set the client to None on failure, and handle the error gracefully
    patcher._GEMINI_CLIENT = None
    print(f"Error initializing Gemini client: {e}")
    print("WARNING: Gemini features will be disabled. Check 'GEMINI_API_KEY'.")


# decrypt and run
def execute_encrypted_script(script_path: str) -> dict:
    """
    Reads an encrypted file, decrypts it, executes the content, and cleans up.
    """
    if not cryptographic._CIPHER: # Check if the cipher failed to initialize
        return {"status": "error", "message": "Execution environment not secure (Cipher failed to load)."}

    # 1. Read the encrypted script content
    try:
        with open(script_path, 'rb') as f: # Use 'rb' for reading bytes
            encrypted_content = f.read()
    except FileNotFoundError:
        return {"status": "error", "message": f"Script file not found: {script_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file: {e}"}

    # 2. Decrypt the content
    decrypted_script_content = cryptographic.decrypt_data(encrypted_content)
    if decrypted_script_content is None:
        return {"status": "error", "message": "Failed to decrypt script content."}

    # 3. Create a temporary, unencrypted executable script for execution
    temp_script_path = f"/tmp/agent_exec_{os.getpid()}_{os.urandom(4).hex()}.sh"
    
    try:
        with open(temp_script_path, 'w') as f:
            f.write(decrypted_script_content)
        
        # Make it executable
        os.chmod(temp_script_path, 0o700) # Only owner (root) can read/write/execute

        # 4. Execute the temporary script (as root)
        result = subprocess.run(
            [temp_script_path],
            capture_output=True,
            text=True,
            check=False # Do not raise error, handle it below
        )

        # 5. Return results
        if result.returncode == 0:
            return {"status": "success", "stdout": result.stdout}
        else:
            return {"status": "failed", "stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}

    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {e}"}

    finally:
        # 6. Cleanup the temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            
        # Optional: Remove the encrypted file after a successful one-time execution
        # os.remove(script_path)


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
    
    result = patcher.generate_script_from_prompt(INSTRUCTIONS, full_path, OUTPUT_SCRIPT_TYPE)
    
    print(f"\n--- Result ---\n{result}")

    if result.startswith("SUCCESS"):
        # The script is saved, you can now proceed to execute the file at full_path
        with open(full_path, 'r') as f:
            print("\n--- Saved Script Content ---")
            print(f.read())