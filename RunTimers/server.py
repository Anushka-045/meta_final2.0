"""
FastAPI server exposing the Compliance Monitor OpenEnv environment.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional, Literal
import uvicorn

from environment import ComplianceEnvironment, TASK_CONFIGS

app = FastAPI(
    title="Compliance Monitor — OpenEnv",
    description="AI compliance monitoring environment (OpenEnv compatible)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# One global environment instance (stateful)
env = ComplianceEnvironment()


class ResetRequest(BaseModel):
    task_id: Literal["task_easy", "task_medium", "task_hard"] = "task_easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok", "environment": "compliance-monitor", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        result = env.reset(task_id=req.task_id, seed=req.seed or 42)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        result = env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    return [
        {
            "id": tid,
            "name": cfg["name"],
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
        }
        for tid, cfg in TASK_CONFIGS.items()
    ]


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
