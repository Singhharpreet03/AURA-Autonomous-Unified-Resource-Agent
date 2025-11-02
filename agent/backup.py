import tarfile, gzip, io, pathlib, datetime, subprocess
import gnupg


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SRC = pathlib.Path(os.getenv('ROOT_DIR', 'demo/app'))
KEY_F = pathlib.Path(os.getenv('KEYS_DIR', 'keys')) / 'age.key'
PUB_F = pathlib.Path(os.getenv('KEYS_DIR', 'keys')) / 'age.pub'
OUT = pathlib.Path(os.getenv('BACKUPS_DIR', 'backups'))
KEY_F.parent.mkdir(exist_ok=True)
# RAGE  = r"C:\Users\RASHMEET SINGH\tools\rage\rage\rage.exe"       

# ---------- create age key-pair if missing (uses rage-keygen) ---------------
def _ensure_keys():
    """Create a symmetric passphrase if missing."""
    KEY_F.parent.mkdir(exist_ok=True)
    if KEY_F.exists() and KEY_F.stat().st_size > 0:
        return
    # 256-bit random passphrase
    KEY_F.write_text(subprocess.check_output("openssl rand -base64 32".split()).decode().strip())
# ---------- encrypt + authenticate in one call ------------------------------
def _rage_encrypt_sign(data: bytes, pubkey=None) -> bytes:
    """Encrypt+MAC with GnuPG symmetric AES-256."""
    gpg = gnupg.GPG()
    cipher = gpg.encrypt_file(
        io.BytesIO(data),
        recipients=None,
        symmetric="AES256",
        passphrase=KEY_F.read_text().strip(),
        armor=True
    )
    return str(cipher).encode()

# ---------- main backup -----------------------------------------------------
def snapshot():
    _ensure_keys()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. tar + gzip
    tarbuf = io.BytesIO()
    with tarfile.open(fileobj=tarbuf, mode="w") as tf:
        tf.add(SRC, arcname=SRC.name)
    tarbuf.seek(0)
    compressed = gzip.compress(tarbuf.read())

    cipher = _rage_encrypt_sign(compressed, pubkey=None)

    # 3. write single authenticated blob
    blob = OUT / f"{stamp}.tar.gz.age"
    blob.write_bytes(cipher)
    print("backup →", blob)
    return blob

if __name__ == "__main__":
    snapshot()