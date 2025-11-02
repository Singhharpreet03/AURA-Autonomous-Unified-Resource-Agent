# ---------- restore.py  (drop-in replacement) ------------------------------
import tarfile, gzip, io, pathlib, shutil, tempfile, gnupg
import os
from dotenv import load_dotenv


load_dotenv()
BACKUPS = pathlib.Path(os.getenv('BACKUPS_DIR', 'backups'))
ROOT = pathlib.Path(os.getenv('ROOT_DIR', 'demo/app'))
KEY_F = pathlib.Path(os.getenv('KEYS_DIR', 'keys')) / 'age.key'  # same symmetric passphrase file

def latest_backup():
    return next(iter(sorted(BACKUPS.glob("*.tar.gz.age"), reverse=True)), None)

def restore():
    blob = latest_backup()
    if not blob:
        print("no backup found"); return

    passphrase = KEY_F.read_text().strip()
    gpg = gnupg.GPG()

    # 1. decrypt
    with blob.open("rb") as f:
        plain = gpg.decrypt_file(f, passphrase=passphrase)
    if not plain.ok:
        raise RuntimeError("GPG decrypt failed: " + plain.stderr)

    # 2. unpack into temp dir
    tmp = ROOT.with_name("restore_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(plain.data))) as tar:
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if m.isdir()}
        if len(top_dirs) == 1:               # strip single top-level dir
            strip = top_dirs.pop() + "/"
            for m in tar.getmembers():
                if m.name.startswith(strip):
                    m.name = m.name[len(strip):]
                    tar.extract(m, tmp)
        else:
            tar.extractall(tmp)

    # 3. atomic swap
    backup_dir = ROOT.with_suffix(".bak")
    if ROOT.exists():
        ROOT.rename(backup_dir)
    tmp.rename(ROOT)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print("restored from", blob)