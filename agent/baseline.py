import json, pathlib, blake3

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


BASELINE = pathlib.Path(os.getenv('BASELINE_FILE', 'baseline.json'))
ROOT = pathlib.Path(os.getenv('ROOT_DIR', 'demo/app'))

def _hash_file(rel_path: str) -> str:
    full_path = ROOT / rel_path
    return blake3.blake3(full_path.read_bytes()).hexdigest()

def build() -> dict:
    golden = {}
    for p in ROOT.rglob("*"):
        if p.is_file():
            rel = p.relative_to(ROOT)
            golden[str(rel)] = _hash_file(str(rel))
    BASELINE.write_text(json.dumps(golden, indent=2))
    print("baseline built with", len(golden), "files")
    return golden

def check(rel_path: str, table: dict | None = None) -> str:
    if table is None:
        with BASELINE.open() as f:
            table = json.load(f)

    h = _hash_file(rel_path)
    if rel_path not in table:
        return "new_file"
    if table[rel_path] != h:
        return "drift"
    return "ok"

if __name__ == "__main__":
    build()