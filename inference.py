import sys
from env import EmailEnv
from agent import OpenEnvAgent
from tasks import get_tasks
from baseline import run_agent

def run_evaluation():
    print("=== Starting MailMind AI Hackathon Evaluation ===", flush=True)
    tasks = get_tasks()
    agent = OpenEnvAgent()
    
    results = []
    for task in tasks:
        # The run_agent function now handles [START], [STEP], and [END] blocks
        reward, score = run_agent("Hackathon-Model", agent, task)
        results.append({
            "Task": task.name,
            "Reward": round(reward, 2),
            "Score": round(float(score), 4)
        })

    print("=== Evaluation Complete ===", flush=True)

    # Save results to JSON for compliance with README
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json", flush=True)

if __name__ == "__main__":
    run_evaluation()
