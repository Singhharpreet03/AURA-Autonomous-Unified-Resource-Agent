import os
from cryptography.fernet import Fernet
from typing import Optional

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