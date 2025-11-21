import os
from cryptography.fernet import Fernet
from typing import Optional
import sys
# Configuration for the key file location
KEY_FILE_PATH = "secret.key" 

# Global variable to hold the initialized Fernet cipher object
_CIPHER = None

def _initialize_cipher():
    """Reads the key from the file and initializes the Fernet cipher."""
    global _CIPHER
    
    if _CIPHER is not None:
        return True # Already initialized

    print(f"Attempting to load encryption key from: {KEY_FILE_PATH}")
    
    try:
        # The agent MUST be running as root to read this protected file!
        with open(KEY_FILE_PATH, 'rb') as key_file:
            key = key_file.read()
            _CIPHER = Fernet(key)
            print("Encryption key loaded successfully.")
            return True
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Encryption key file not found at {KEY_FILE_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize cipher: {e}")
        
    _CIPHER = None
    return False

def encrypt_data(data: str) -> Optional[bytes]:
    """Encrypts a string (script content) and returns bytes."""
    if _CIPHER is None and not _initialize_cipher():
        return None
        
    try:
        # Encode the string data to bytes before encrypting
        return _CIPHER.encrypt(data.encode())
    except Exception as e:
        print(f"Encryption failed: {e}")
        return None

def decrypt_data(token: bytes) -> Optional[str]:
    """Decrypts bytes (encrypted script content) and returns a string."""
    if _CIPHER is None and not _initialize_cipher():
        return None
    try:
        # Decrypt the bytes and decode back to a string
        return _CIPHER.decrypt(token).decode()
    except Exception as e:
        print(f"Decryption failed: {e}")
        return None

# Attempt to initialize the cipher when the module is first imported
_initialize_cipher()




if __name__ == '__main__':
    # Define a known path for manual testing
    TEST_SCRIPT_PATH = './EncryptedScripts_folder/50186440-f7d5-4d89-8faa-74d775552819.sh'
    
    print(f"--- Starting Manual Decryption Test ---")
    
    # 1. Check Cipher Initialization
    try:
        if not _CIPHER: 
            print("ERROR: Execution environment not secure (_CIPHER failed to load).")
            # Use sys.exit or raise an error to stop execution
            import sys
            sys.exit(1) 
    except NameError:
        print("ERROR: _CIPHER variable not found. Is it defined globally?")
        sys.exit(1)

    # 2. Read the encrypted script content
    encrypted_content = None
    try:
        with open(TEST_SCRIPT_PATH, 'rb') as f:
            encrypted_content = f.read()
        print(f"SUCCESS: Read {len(encrypted_content)} bytes from {TEST_SCRIPT_PATH}.")
    except FileNotFoundError:
        # FIX: Changed print{...} to a valid print statement
        print(f"ERROR: Script file not found: {TEST_SCRIPT_PATH}") 
        sys.exit(1)
    except Exception as e:
        # FIX: Changed print{...} to a valid print statement
        print(f"ERROR: Failed to read file: {e}")
        sys.exit(1)

    # 3. Decrypt the content
    print("Attempting decryption...")
    # NOTE: You must use the internal function name, e.g., decrypt_data(encrypted_content)
    decrypted_script_content = decrypt_data(encrypted_content) 
    
    if decrypted_script_content is None:
        print("CRITICAL ERROR: Failed to decrypt script content.")
        # This is where your previous "failed to decrypt file" logic originates
        sys.exit(1)
    else:
        print("SUCCESS: Decryption successful!")
        # Print the decrypted content for verification (CAUTION: don't log secrets)
        print("\n--- Decrypted Content Preview ---")
        print(decrypted_script_content[:500] + ('...' if len(decrypted_script_content) > 500 else ''))
        print("---------------------------------")
