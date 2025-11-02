#!/usr/bin/env python3
"""
FastAPI wrapper around the existing drift scripts.
This version integrates the drift pipeline daemon directly and adds start/stop controls.
Run: uvicorn drift_api:app --host 0.0.0.0 --port 8000 --reload
"""
import json, pathlib, datetime, asyncio, time, threading
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---- import the existing micro-services -----------------------------------
import baseline
import backup
import restore
import alert

# ---- config ---------------------------------------------------------------
ROOT     = pathlib.Path("demo/app")
BASELINE = pathlib.Path("baseline.json")
BACKUPS  = pathlib.Path("backups")
MAX_ALERT_MEMORY = 200          # keep last N alerts in RAM

# ---- in-memory ring buffer and state management ---------------------------
_alerts = deque(maxlen=MAX_ALERT_MEMORY)
is_pipeline_running = False
pipeline_lock = threading.Lock()

# ---- Global variables for the event loop and handler -----------------------
main_loop = None
drift_handler = None
observer = None

# ---- Integrated Drift Handler ---------------------------------------------
class DriftHandler(FileSystemEventHandler):
    def __init__(self):
        self.refresh_truth()

    def refresh_truth(self):
        """Write baseline.json and load it into self.table."""
        print("INFO: Refreshing baseline truth table...")
        baseline.build()
        with baseline.BASELINE.open() as f:
            self.table = json.load(f)

    def on_any_event(self, event):
        if not is_pipeline_running:
            return

        if event.is_directory:
            return

        try:
            rel = str(pathlib.Path(event.src_path).relative_to(ROOT))
        except ValueError:
            return

        if event.event_type == "deleted":
            alert.send("info", f"file deleted {rel}", {"file": rel})
            restore.restore()
            self.refresh_truth()
            return

        status = baseline.check(rel, self.table)
        if status == "drift":
            alert.send("warn", f"drift on {rel}", {"file": rel})
            restore.restore()
            self.refresh_truth()
        elif status == "new_file":
            alert.send("info", f"new file {rel}", {"file": rel})

# ---- Pipeline Control Functions -------------------------------------------
def _start_pipeline():
    """Internal function to start the pipeline thread."""
    global is_pipeline_running, observer, drift_handler, main_loop
    with pipeline_lock:
        if is_pipeline_running:
            print("Pipeline is already running.")
            return

        print("Starting drift pipeline...")
        if not observer or not observer.is_alive():
            observer = Observer()
            drift_handler = DriftHandler()
            observer.schedule(drift_handler, str(ROOT.parent), recursive=True)
            observer.start()
        is_pipeline_running = True
        print("Drift pipeline started.")

def _stop_pipeline():
    """Internal function to stop the pipeline thread."""
    global is_pipeline_running, observer
    with pipeline_lock:
        if not is_pipeline_running:
            print("Pipeline is already stopped.")
            return

        print("Stopping drift pipeline...")
        is_pipeline_running = False
        if observer and observer.is_alive():
            observer.stop()
            observer.join()
        print("Drift pipeline stopped.")

# ---- FastAPI App Definition -----------------------------------------------
app = FastAPI(title="Drift-Management API", version="1.0.0")

# ---- ADD THE CORS MIDDLEWARE -------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- monkey-patch alert.send so it also feeds the ring ----------------------
_original_alert_send = alert.send
def _capture_alert(level, msg, extra=None):
    _original_alert_send(level, msg, extra)
    item = {
        "ts":  datetime.datetime.utcnow().isoformat()+"Z",
        "level": level,
        "msg": msg,
        "extra": extra or {}
    }
    _alerts.append(item)
    
    # *** ENHANCED DEBUGGING ***
    print(f"DEBUG: Alert captured. Active WebSocket clients: {len(manager.active)}")
    if main_loop and manager.active:
        try:
            future = asyncio.run_coroutine_threadsafe(manager.broadcast(item), main_loop)
            # We can add a callback to see the result of the broadcast
            def log_broadcast_result(fut):
                try:
                    fut.result() # This will re-raise any exception from the coroutine
                    print("DEBUG: Broadcast task completed successfully.")
                except Exception as e:
                    print(f"ERROR: Broadcast task failed with exception: {e}")
            future.add_done_callback(log_broadcast_result)
        except Exception as e:
            print(f"ERROR: Failed to schedule broadcast task: {e}")
    else:
        print("DEBUG: No active clients or main loop, skipping broadcast.")

alert.send = _capture_alert

# ---- Pydantic models ------------------------------------------------------
class FileStatus(BaseModel):
    path: str
    status: Literal["ok", "drift", "new_file", "missing"]
    hash: str = None

class BackupMeta(BaseModel):
    name: str
    size: int
    ctime: datetime.datetime

class Alert(BaseModel):
    ts: str
    level: str
    msg: str
    extra: dict

# ---- FastAPI Events --------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the baseline and start the file watcher on startup."""
    global main_loop
    print("API Starting up...")
    main_loop = asyncio.get_running_loop()

    if not BASELINE.exists():
        print("Baseline not found, building initial baseline...")
        baseline.build()
    
    _start_pipeline()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the file watcher on shutdown."""
    print("API Shutting down...")
    _stop_pipeline()

# ---- API Endpoints --------------------------------------------------------
# ---- health ---------------------------------------------------------------
@app.get("/api/status")
def status():
    return {
        "status": "alive",
        "baseline_exists": BASELINE.exists(),
        "pipeline_status": "running" if is_pipeline_running else "stopped"
    }

# ---- pipeline control -----------------------------------------------------
@app.post("/api/pipeline/stop")
def stop_pipeline_endpoint():
    """Stops the automatic drift detection and restoration."""
    _stop_pipeline()
    return {"status": "stopped", "message": "Drift management pipeline has been stopped. You can now modify files."}

@app.post("/api/pipeline/start")
def start_pipeline_endpoint():
    """Starts the automatic drift detection and restoration."""
    _start_pipeline()
    return {"status": "running", "message": "Drift management pipeline has been started."}

# ---- file inventory -------------------------------------------------------
@app.get("/api/files", response_model=List[FileStatus])
def list_files():
    if not BASELINE.exists():
        raise HTTPException(404, "baseline.json missing – run /api/baseline/rebuild")
    golden = json.loads(BASELINE.read_text())
    out = []
    for rel, h in golden.items():
        st = baseline.check(rel)
        out.append(FileStatus(path=rel, status=st, hash=h))
    for p in ROOT.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(ROOT))
            if rel not in golden:
                out.append(FileStatus(path=rel, status="new_file",
                                      hash=baseline.blake3.blake3(p.read_bytes()).hexdigest()))
    return out

# ---- single-file check ----------------------------------------------------
@app.get("/api/files/{rel_path:path}/status", response_model=FileStatus)
def file_status(rel_path: str):
    if not BASELINE.exists():
        raise HTTPException(404, "baseline missing")
    st = baseline.check(rel_path)
    if st == "new_file":
        h = baseline.blake3.blake3((ROOT/rel_path).read_bytes()).hexdigest()
    else:
        golden = json.loads(BASELINE.read_text())
        h = golden.get(rel_path)
    return FileStatus(path=rel_path, status=st, hash=h)

@app.post("/api/files/{rel_path:path}/check")
def check_file(rel_path: str):
    st = baseline.check(rel_path)
    if st == "drift":
        alert.send("warn", f"drift on {rel_path}", {"file": rel_path})
    return {"status": st}

# ---- baseline management --------------------------------------------------
@app.post("/api/baseline/rebuild")
def rebuild_baseline():
    baseline.build()
    if drift_handler:
        drift_handler.refresh_truth()
    return {"ok": True, "message": "Baseline rebuilt successfully."}

# ---- backup management ----------------------------------------------------
@app.get("/api/backups", response_model=List[BackupMeta])
def list_backups():
    ages = sorted(BACKUPS.glob("*.tar.gz.age"), reverse=True)
    return [BackupMeta(name=b.name, size=b.stat().st_size,
                       ctime=datetime.datetime.fromtimestamp(b.stat().st_ctime))
            for b in ages]

@app.post("/api/backups", status_code=201)
def create_backup():
    blob = backup.snapshot()
    return {"created": blob.name, "message": "Backup created successfully."}

@app.post("/api/backups/latest/restore")
def restore_latest():
    restore.restore()
    if drift_handler:
        drift_handler.refresh_truth()
    return {"restored": True, "message": "Successfully restored from the latest backup."}

# ---- alert history --------------------------------------------------------
@app.get("/api/alerts", response_model=List[Alert])
def get_alerts(limit: int = 50):
    return list(reversed(_alerts))[:limit]

# ---- optional websocket ---------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"DEBUG: WebSocket connected. Total clients: {len(self.active)}")
    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
            print(f"DEBUG: WebSocket disconnected. Total clients: {len(self.active)}")
    async def broadcast(self, msg: dict):
        print(f"DEBUG: Broadcasting message to {len(self.active)} clients.")
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception as e:
                # *** ENHANCED DEBUGGING ***
                print(f"ERROR: Failed to send to WebSocket. Client: {ws.client}. Error: {e}")
                dead.append(ws)
        if dead:
            print(f"DEBUG: Found {len(dead)} dead connections, removing them.")
            for ws in dead:
                self.disconnect(ws)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def live_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # wait until the client goes away
        await websocket.receive_text()   # raises WebSocketDisconnect
    except WebSocketDisconnect:
        manager.disconnect(websocket)