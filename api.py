import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from env import EmailEnv
from tasks import get_tasks, grade_ambiguous, grade_easy, grade_medium
from models import Email, Action, ActionType, ResetRequest, Observation, State, StepResult
import baseline

# Global environment state
global_env: Optional[EmailEnv] = None

app = FastAPI(title="MailMind AI API", version="1.2.0")

# Action Schema for /tasks
ACTION_SCHEMA = {
    "type": "string - One of: reply, mark_important, archive, schedule_followup, no_op",
    "email_id": "string (optional) - ID of the email being acted on",
    "reasoning": "string - Explain the hidden context and business impact",
    "reply_text": "string (optional) - Call to action if reply is selected"
}

class GraderRequest(BaseModel):
    task_name: str
    history: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"status": "ok", "message": "MailMind AI OpenEnv Server Running", "version": "1.2.0"}

@app.get("/tasks")
async def list_tasks():
    tasks = get_tasks()
    return {
        "tasks": [{"id": t.name.lower(), "name": t.name, "difficulty": t.difficulty} for t in tasks],
        "action_schema": ACTION_SCHEMA
    }

@app.get("/baseline")
async def run_baseline_endpoint():
    try:
        baseline.run_baseline()
        with open("results.json", "r") as f:
            results = json.load(f)

        scores: Dict[str, List[float]] = {}
        for res in results:
            task_key = res["Task"].lower()
            if task_key not in scores:
                scores[task_key] = []
            scores[task_key].append(res["Score"])

        avg_scores = {k: round(sum(v) / len(v), 2) for k, v in scores.items()}
        return avg_scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grader")
async def grader_endpoint(request: GraderRequest):
    """Exposes the grading logic for automated evaluation."""
    graders = {
        "EASY": grade_easy,
        "MEDIUM": grade_medium,
        "AMBIGUOUS": grade_ambiguous
    }

    grader_fn = graders.get(request.task_name.upper())
    if not grader_fn:
        raise HTTPException(status_code=400, detail=f"Unknown task: {request.task_name}. Valid: EASY, MEDIUM, AMBIGUOUS")

    try:
        formatted_history = []
        for entry in request.history:
            email_obj = Email(**entry["email"])
            action_obj = Action(**entry["action"])
            formatted_history.append({"email": email_obj, "action": action_obj})

        score = grader_fn(formatted_history)
        return {"score": float(score)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"History parsing error: {str(e)}")

@app.post("/reset")
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    """Resets the environment (POST)."""
    global global_env
    task_id = request.task_id if request else None
    
    tasks = get_tasks()
    selected_task = None
    if task_id:
        selected_task = next((t for t in tasks if t.name.lower() == task_id.lower()), None)
    
    # Default to EASY if not found or not provided
    if not selected_task:
        selected_task = next(t for t in tasks if t.name.upper() == "EASY")
        
    global_env = EmailEnv(task=selected_task)
    obs, _ = global_env.reset()
    return obs.model_dump()

@app.post("/step", response_model=StepResult)
async def step_endpoint(action: Action):
    """Executes a step (POST)."""
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    return global_env.step(action)

@app.get("/state", response_model=State)
async def get_state_endpoint():
    """Returns the current state (GET)."""
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    return global_env.state()

def main():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
