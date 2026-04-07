import os
import sys
import json
from typing import Optional, List

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from env import EmailEnv
from agent import OpenEnvAgent
from tasks import get_tasks

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_evaluation():
    print("=== Starting MailMind AI Hackathon Evaluation ===", flush=True)
    tasks = get_tasks()
    agent = OpenEnvAgent()
    
    results = []
    
    for task in tasks:
        log_start(task=task.name, env="MailMindEnv", model=MODEL_NAME)
        
        env = EmailEnv(task=task)
        obs, _ = env.reset()
        done = False
        
        rewards = []
        step_count = 0
        error = None
        
        try:
            while not done:
                step_count += 1
                action = agent.act(obs)
                
                step_res = env.step(action)
                obs = step_res.observation
                reward = float(step_res.reward.value)
                done = step_res.done
                
                rewards.append(reward)
                
                # Format action strictly to a simpler string representation
                action_str = f"Action({action.type.value}, {action.email_id})"
                log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)
                
        except Exception as e:
            error = str(e)
            print(f"[DEBUG] Error during environment step: {error}", flush=True)
        finally:
            score = float(task.grader(env.state().history))
            success = bool(score >= 0.5) 
            log_end(success=success, steps=step_count, score=score, rewards=rewards)
            
            results.append({
                "Task": task.name,
                "Reward": round(sum(rewards), 2),
                "Score": round(score, 4)
            })

    print("=== Evaluation Complete ===", flush=True)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json", flush=True)

if __name__ == "__main__":
    run_evaluation()
