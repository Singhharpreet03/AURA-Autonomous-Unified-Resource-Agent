import json, pathlib, asyncio, time, threading
from collections import deque
import logging
import os
import sys
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uvicorn
from json_utils import dumps
import psycopg2
from dotenv import load_dotenv
import uuid
import subprocess
import pathlib
from starlette import status

#load_dotenv()
import executioner
import patcher
import cryptographic

WORKFLOW_DIR = os.getenv('WORKFLOW_DIR', './EncryptedScripts_folder')
# ---- import the existing micro-services for drift-----------------------------------
import baseline
import backup
import restore
import alert

# load env

SCRIPT_DIR = pathlib.Path(__file__).parent
ENV_PATH = SCRIPT_DIR / ".env" # This assumes .env is in the same directory as patcher.py

# --- INITIALIZATION LOGIC (Runs once when the module is imported) ---
load_dotenv(dotenv_path=ENV_PATH)
# ---- config ---------------------------------------------------------------
ROOT = pathlib.Path(os.getenv('ROOT_DIR', 'demo/app'))
BASELINE = pathlib.Path(os.getenv('BASELINE_FILE', 'baseline.json'))
BACKUPS = pathlib.Path(os.getenv('BACKUPS_DIR', 'backups'))
MAX_ALERT_MEMORY = 200          # keep last N alerts in RAM

# ---- in-memory ring buffer and state management ---------------------------
_alerts = deque(maxlen=MAX_ALERT_MEMORY)
is_pipeline_running = False
pipeline_lock = threading.Lock()

# ---- Global variables for the event loop and handler -----------------------
main_loop = None
drift_handler = None
observer = None

# ---- logging -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, # Set the minimum logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("alert_api")
executioner_logger = logging.getLogger("executioner")
# Define a logger for the API endpoint (for clarity)
api_logger = logging.getLogger("fastapi_endpoint")

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
app = FastAPI(title="Aura Agent API", version="1.0.0")


# Ensure the workflow directory exists when the app starts
os.makedirs(WORKFLOW_DIR, exist_ok=True)
print(f"Workflow directory checked: {WORKFLOW_DIR}")
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
        "ts":  datetime.utcnow().isoformat()+"Z",
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
    ctime: datetime

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
    await alert_startup()

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
                       ctime=datetime.fromtimestamp(b.stat().st_ctime))
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


#--------------------------intelligent alerting service -------------------------


DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 5432))
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'sonit03')
DB_NAME = os.environ.get('DB_NAME', 'metrics_db')


API_CONFIG = {
    'host': os.environ.get('API_HOST', '0.0.0.0'),
    'port': int(os.environ.get('API_PORT', 8040)),
    'debug': os.environ.get('API_DEBUG', 'False').lower() == 'true',
    'cors_origins': os.environ.get('CORS_ORIGINS', '*').split(',')
}


class DatabaseConnection:
    """Context manager for database connections"""
    def __enter__(self):
        self.conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

# Pydantic models for API
class Alert(BaseModel):
    id: int
    application: str
    metric_name: str
    detection_time: datetime
    severity: str
    enriched_details: Dict[str, Any]
    notified: bool
    notified_at: Optional[datetime] = None

class AlertAcknowledge(BaseModel):
    acknowledged: bool
    notes: Optional[str] = None

class AlertFilter(BaseModel):
    application: Optional[str] = None
    severity: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    acknowledged: Optional[bool] = None

# Initialize FastAPI app
# app = FastAPI(title="AURA Alert API", description="API for managing and monitoring AURA alerts")

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=API_CONFIG['cors_origins'],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed, remove it
                self.active_connections.remove(connection)

# manager = ConnectionManager()

# API Endpoints
@app.get("/api/v1/alerts", response_model=List[Alert])
async def get_alerts(
    application: Optional[str] = Query(None, description="Filter by application name"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    start_time: Optional[datetime] = Query(None, description="Filter alerts after this time"),
    end_time: Optional[datetime] = Query(None, description="Filter alerts before this time"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip")
):
    """Get alerts with optional filtering"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT id, application, metric_name, detection_time, severity, 
                       enriched_details, notified, notified_at
                FROM enriched_alerts
                WHERE 1=1
            """
            params = []
            
            if application:
                query += " AND application = %s"
                params.append(application)
                
            if severity:
                query += " AND severity = %s"
                params.append(severity)
                
            if start_time:
                query += " AND detection_time >= %s"
                params.append(start_time)
                
            if end_time:
                query += " AND detection_time <= %s"
                params.append(end_time)
                
            if acknowledged is not None:
                if acknowledged:
                    query += " AND id IN (SELECT alert_id FROM remediation_actions)"
                else:
                    query += " AND id NOT IN (SELECT alert_id FROM remediation_actions)"
            
            query += " ORDER BY detection_time DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            
            columns = [desc[0] for desc in cur.description]
            alerts = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            # Convert datetime objects to strings for JSON serialization
            for alert in alerts:
                if alert['detection_time']:
                    alert['detection_time'] = alert['detection_time'].isoformat()
                if alert['notified_at']:
                    alert['notified_at'] = alert['notified_at'].isoformat()
            
            return alerts
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/{alert_id}", response_model=Alert)
async def get_alert(alert_id: int):
    """Get a specific alert by ID"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT id, application, metric_name, detection_time, severity, 
                       enriched_details, notified, notified_at
                FROM enriched_alerts
                WHERE id = %s
            """, (alert_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            columns = [desc[0] for desc in cur.description]
            alert = dict(zip(columns, row))
            
            # Convert datetime objects to strings for JSON serialization
            if alert['detection_time']:
                alert['detection_time'] = alert['detection_time'].isoformat()
            if alert['notified_at']:
                alert['notified_at'] = alert['notified_at'].isoformat()
            
            return alert
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, acknowledge_data: AlertAcknowledge):
    """Acknowledge an alert"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Check if alert exists
            cur.execute("SELECT id FROM enriched_alerts WHERE id = %s", (alert_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Alert not found")
            
            if acknowledge_data.acknowledged:
                # Add acknowledgment record
                cur.execute("""
                    INSERT INTO remediation_actions (alert_id, time, application, action, success, details)
                    SELECT id, NOW(), application, 'acknowledged', TRUE, %s
                    FROM enriched_alerts
                    WHERE id = %s
                """, (dumps({"notes": acknowledge_data.notes}), alert_id))
            else:
                # Remove acknowledgment record
                cur.execute("""
                    DELETE FROM remediation_actions
                    WHERE alert_id = %s AND action = 'acknowledged'
                """, (alert_id,))
            
            return {"status": "success", "message": f"Alert {alert_id} acknowledged status updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete an alert"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Check if alert exists
            cur.execute("SELECT id FROM enriched_alerts WHERE id = %s", (alert_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Alert not found")
            
            # Delete related records first
            cur.execute("DELETE FROM remediation_actions WHERE alert_id = %s", (alert_id,))
            
            # Delete the alert
            cur.execute("DELETE FROM enriched_alerts WHERE id = %s", (alert_id,))
            
            return {"status": "success", "message": f"Alert {alert_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/stats")
async def get_alert_stats():
    """Get alert statistics"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Get total alerts by severity
            cur.execute("""
                SELECT severity, COUNT(*) as count
                FROM enriched_alerts
                GROUP BY severity
            """)
            
            severity_stats = dict(cur.fetchall())
            
            # Get alerts in the last 24 hours
            cur.execute("""
                SELECT COUNT(*) as count
                FROM enriched_alerts
                WHERE detection_time >= NOW() - INTERVAL '24 hours'
            """)
            
            last_24h = cur.fetchone()[0]
            
            # Get unacknowledged alerts
            cur.execute("""
                SELECT COUNT(*) as count
                FROM enriched_alerts
                WHERE id NOT IN (SELECT alert_id FROM remediation_actions WHERE action = 'acknowledged')
            """)
            
            unacknowledged = cur.fetchone()[0]
            
            # Get top applications with alerts
            cur.execute("""
                SELECT application, COUNT(*) as count
                FROM enriched_alerts
                WHERE detection_time >= NOW() - INTERVAL '7 days'
                GROUP BY application
                ORDER BY count DESC
                LIMIT 5
            """)
            
            top_applications = dict(cur.fetchall())
            
            return {
                "by_severity": severity_stats,
                "last_24_hours": last_24h,
                "unacknowledged": unacknowledged,
                "top_applications": top_applications
            }
    except Exception as e:
        logger.error(f"Error fetching alert stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/applications")
async def get_applications():
    """Get list of all applications with alerts"""
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT DISTINCT application
                FROM enriched_alerts
                ORDER BY application
            """)
            
            applications = [row[0] for row in cur.fetchall()]
            
            return {"applications": applications}
    except Exception as e:
        logger.error(f"Error fetching applications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time alerts
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for now, could be used for client messages
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task to broadcast new alerts
async def broadcast_new_alerts():
    """Check for new alerts and broadcast them to connected WebSocket clients"""
    last_check_time = datetime.now()
    
    while True:
        try:
            with DatabaseConnection() as conn:
                cur = conn.cursor()
                
                # Get alerts created since last check
                cur.execute("""
                    SELECT id, application, metric_name, detection_time, severity, enriched_details
                    FROM enriched_alerts
                    WHERE detection_time > %s
                    ORDER BY detection_time DESC
                """, (last_check_time,))
                
                columns = [desc[0] for desc in cur.description]
                new_alerts = [dict(zip(columns, row)) for row in cur.fetchall()]
                
                if new_alerts:
                    # Update last check time
                    last_check_time = datetime.now()
                    
                    # Broadcast each new alert
                    for alert in new_alerts:
                        # Convert datetime objects to strings for JSON serialization
                        if alert['detection_time']:
                            alert['detection_time'] = alert['detection_time'].isoformat()
                        
                        # Create alert message
                        alert_message = {
                            "type": "new_alert",
                            "data": alert
                        }
                        
                        # Broadcast to all connected clients
                        await manager.broadcast(json.dumps(alert_message))
                        logger.info(f"Broadcasted new alert {alert['id']} to WebSocket clients")
            
            # Check every 5 seconds
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in broadcast task: {e}")
            await asyncio.sleep(10)  # Wait longer if there's an error

# Start the background task when the app starts
@app.on_event("startup")
async def alert_startup():
    # Start the background task
    asyncio.create_task(broadcast_new_alerts())
    logger.info("Alert API service started")



#-------------------------------------patcher and executioner APIs ---------------------------------------


# ====================================================================
# --- PYDANTIC MODELS ---
# ====================================================================

class WorkflowCreateRequest(BaseModel):
    """Payload for creating a new workflow."""
    instructions: str = Field(..., description="The natural language prompt or script content.")
    script_type: Literal["bash", "powershell", "python"] = Field("bash", description="The type of script to generate/use.")
    encrypted: bool = Field(True, description="Whether the resulting script should be encrypted at rest.")

class WorkflowListResponse(BaseModel):
    """Response for GET /workflows/"""
    workflows: List[str]

class WorkflowCreateResponse(BaseModel):
    """Response after creating a new workflow."""
    script_type: str
    workflow_id: str
    message: str
    
class WorkflowStatusResponse(BaseModel):
    """Generic response for status updates (execute, delete)."""
    status: Literal["success", "error"]
    workflow_id: Optional[str] = None
    message: str
    
class AppListResponse(BaseModel):
    """Response for GET /managed/apps"""
    apps: Dict[str, str] # Maps app name to version (e.g., {"nginx": "1.20.1"})
    
class ExecutionResult(BaseModel):
    """Response after executing a workflow."""
    workflow_id: str
    status: Literal["success", "failed", "error"]
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    code: Optional[int] = None


# ====================================================================
# --- API ENDPOINTS ---
# ====================================================================

# Helper function
def get_workflow_path(workflow_id: str) -> str:
    """Returns the full file path for a workflow ID."""
    return os.path.join(WORKFLOW_DIR, f"{workflow_id}.sh")


# --- GET /api/v1/workflows/ ---
@app.get("/api/v1/workflows/", response_model=WorkflowListResponse)
async def list_workflows():
    """Gets list of all current workflows ready to be applied."""
    try:
        files = os.listdir(WORKFLOW_DIR)
        workflow_ids = [f.split('.')[0] for f in files if f.endswith(".sh")]
        return {"workflows": workflow_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read workflow directory: {e}")


# --- POST /api/v1/workflow/create ---
@app.post("/api/v1/workflow/create", response_model=WorkflowCreateResponse)
async def create_workflow(request: WorkflowCreateRequest):
    """Creates a new workflow by generating a script from instructions."""
    
    workflow_id = str(uuid.uuid4())
    output_path = get_workflow_path(workflow_id)
    
    # Calls your actual script generation/saving/encryption function
    # You would replace the placeholder call with: 
    # result_message = gemini_script_generator.patcher.generate_script_from_prompt(...)
    result_message = patcher.generate_script_from_prompt(
        instructions=request.instructions,
        output_path=output_path,
        script_type=request.script_type
    )

    if result_message.startswith("SUCCESS"):
        return WorkflowCreateResponse(
            script_type=request.script_type,
            workflow_id=workflow_id,
            message=f"Workflow created and saved. Encrypted: {request.encrypted}"
        )
    else:
        raise HTTPException(status_code=500, detail=result_message)


# --- POST /api/v1/workflow/patch/{workflow_id} ---
@app.post("/api/v1/workflow/patch/{workflow_id}", response_model=ExecutionResult)
async def execute_workflow(workflow_id: str):
    """Executes a workflow (decrypts and runs the script)."""
    api_logger.info(f"--- START: Workflow execution requested for ID: {workflow_id}")

    workflow_path = get_workflow_path(workflow_id)
    
    if not os.path.exists(workflow_path):
        api_logger.warning(f"Workflow file not found at path: {workflow_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Workflow ID '{workflow_id}' not found.")
    
    # Calls your execution function
    # You would replace the placeholder call with: 
    # execution_data = executioner.executioner.execute_encrypted_script(workflow_path)
    execution_data = executioner.execute_encrypted_script(workflow_path)
    status_val = execution_data.get("status", "error")

    if status_val == "success":
        api_logger.info(f"--- SUCCESS: Workflow '{workflow_id}' executed successfully.")
    else:
        # Log critical errors with details for quick visibility
        api_logger.error(
            f"--- FAILED: Workflow '{workflow_id}' execution failed.", 
            extra={"details": execution_data}
        )
    return ExecutionResult(
        workflow_id=workflow_id,
        status=execution_data.get("status", "error"),
        stdout=execution_data.get("stdout"),
        stderr=execution_data.get("stderr"),
        code=execution_data.get("code")
    )


# --- DELETE /api/v1/workflow/{workflow_id} ---
@app.delete("/api/v1/workflow/{workflow_id}", response_model=WorkflowStatusResponse)
async def delete_workflow(workflow_id: str):
    """Deletes an workflow file."""
    
    workflow_path = get_workflow_path(workflow_id)
    
    if not os.path.exists(workflow_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Workflow ID '{workflow_id}' not found.")
                            
    try:
        os.remove(workflow_path)
        return WorkflowStatusResponse(
            status="success", 
            workflow_id=workflow_id, 
            message=f"Workflow {workflow_id} deleted successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, 
                            detail=f"Failed to delete workflow {workflow_id}: {e}")


# --- GET /api/v1/managed/apps ---
@app.get("/api/v1/managed/apps", response_model=AppListResponse)
async def list_managed_apps():
    """Gets list of all managed apps and their versions (Placeholder)."""
    
    # NOTE: In a functional agent, this would dynamically query system packages.
    managed_apps = {
        "nginx": "1.20.1",
        "docker": "24.0.5",
        "python3": "3.10.12"
    }
    
    return {"apps": managed_apps}
