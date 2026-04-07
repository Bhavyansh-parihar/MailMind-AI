import os
import json
import pandas as pd
from env import EmailEnv
from agent import RandomAgent, HeuristicAgent, OpenAI_Agent, Gemini_Agent
from learning import QLearningAgent, LearningManager
from tasks import get_tasks

def run_agent(agent_name, agent, task):
    print(f"[START] task={task.name}", flush=True)
    env = EmailEnv(task=task)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        step_count += 1
        action = agent.act(obs)
        step_res = env.step(action)
        obs = step_res.observation
        reward = step_res.reward.value
        total_reward += reward
        done = step_res.done
        print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)
        
    final_score = task.grader(env.state().history)
    print(f"[END] task={task.name} score={final_score:.4f} steps={step_count}", flush=True)
    return total_reward, final_score

def run_baseline():
    print("=== Running OpenEnv AI Email Assistant Multi-Agent Baseline ===", flush=True)
    tasks = get_tasks()
    
    # Initialize agents
    agents = {
        "Random": RandomAgent(),
        "Heuristic (Naive)": HeuristicAgent(),
        "Hackathon-Model": OpenAI_Agent(), # Pointing to the new compliant class
    }
    
    # Add Q-Learning Agent (Trained for 100 episodes on random data)
    q_agent = QLearningAgent()
    q_manager = LearningManager(q_agent)
    print("Training Q-Learning Agent...", flush=True)
    for _ in range(100):
        dummy_env = EmailEnv()
        q_manager.run_episode(dummy_env)
    agents["Q-Learning (Trained)"] = q_agent
    results = []

    for task in tasks:
        print(f"\n--- Task: {task.name} ({task.difficulty}) ---", flush=True)
        for agent_name, agent in agents.items():
            reward, score = run_agent(agent_name, agent, task)
            results.append({
                "Task": task.name,
                "Difficulty": task.difficulty,
                "Agent": agent_name,
                "Reward": round(reward, 2),
                "Score": round(float(score), 4)
            })
            print(f"{agent_name:20}: Reward={reward:6.2f}, Score={score:6.4f}", flush=True)

    # Output Comparison Table
    df = pd.DataFrame(results)
    print("\n=== Agent Comparison Table ===", flush=True)
    print(df.to_string(index=False), flush=True)

    # Save to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results.json", flush=True)

    # Winner Calculation
    summary = df.groupby("Agent")["Score"].mean().reset_index()
    winner = summary.loc[summary["Score"].idxmax()]
    print(f"\n🏆 Best Performing Agent: {winner['Agent']} with Average Score: {winner['Score']:.4f}", flush=True)

if __name__ == "__main__":
    run_baseline()
