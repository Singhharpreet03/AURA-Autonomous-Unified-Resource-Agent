# agent_api_combined.py

import os
import uuid
import subprocess

from . import executioner
from . import patcher
from . import cryptographic

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional



# --- PACKAGE IMPORTS (Assumed) ---
# Replace these with your actual package structure imports if needed.
# We'll use placeholder functions that mirror your package structure.

# Placeholder functions for your core logic modules
# NOTE: In a real environment, these would be 'from agent import gemini_script_generator' etc.
def generate_script_from_prompt(*args, **kwargs) -> str:
    # Dummy implementation: SUCCESS or ERROR
    # Simulate a successful script creation and encryption
    return "SUCCESS: Script generated and saved." 

def execute_encrypted_script(path: str) -> Dict[str, Any]:
    # Dummy implementation: Simulates execution result structure
    return {"status": "success", "stdout": "Patch applied successfully.", "stderr": "", "code": 0}


# --- CONFIGURATION ---
WORKFLOW_DIR = "./EncryptedScripts_folder"

# Initialize FastAPI app
app = FastAPI(
    title="AURA Agent API", 
    version="v1", 
    description="Root-privileged service for dynamic patch and workflow management."
)

# Ensure the workflow directory exists when the app starts
os.makedirs(WORKFLOW_DIR, exist_ok=True)
print(f"Workflow directory checked: {WORKFLOW_DIR}")


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
    # result_message = gemini_script_generator.generate_script_from_prompt(...)
    result_message = generate_script_from_prompt(
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
    
    workflow_path = get_workflow_path(workflow_id)
    
    if not os.path.exists(workflow_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Workflow ID '{workflow_id}' not found.")
    
    # Calls your execution function
    # You would replace the placeholder call with: 
    # execution_data = executioner.execute_encrypted_script(workflow_path)
    execution_data = execute_encrypted_script(workflow_path)
    
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