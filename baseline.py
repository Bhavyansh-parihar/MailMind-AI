import os
import json
import pandas as pd
from env import EmailEnv
from agent import RandomAgent, HeuristicAgent, OpenAI_Agent, Gemini_Agent
from learning import QLearningAgent, LearningManager
from tasks import get_tasks

def run_agent(agent_name, agent, task):
    env = EmailEnv(task=task)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        action = agent.act(obs)
        step_res = env.step(action)
        obs = step_res.observation
        total_reward += step_res.reward.value
        done = step_res.done
        
    final_score = task.grader(env.state().history)
    return total_reward, final_score

def run_baseline():
    print("=== Running OpenEnv AI Email Assistant Multi-Agent Baseline ===")
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
    print("Training Q-Learning Agent...")
    for _ in range(100):
        dummy_env = EmailEnv()
        q_manager.run_episode(dummy_env)
    agents["Q-Learning (Trained)"] = q_agent

    results = []

    for task in tasks:
        print(f"\n--- Task: {task.name} ({task.difficulty}) ---")
        for agent_name, agent in agents.items():
            reward, score = run_agent(agent_name, agent, task)
            results.append({
                "Task": task.name,
                "Difficulty": task.difficulty,
                "Agent": agent_name,
                "Reward": round(reward, 2),
                "Score": round(float(score), 4)
            })
            print(f"{agent_name:20}: Reward={reward:6.2f}, Score={score:6.4f}")

    # Output Comparison Table
    df = pd.DataFrame(results)
    print("\n=== Agent Comparison Table ===")
    print(df.to_string(index=False))

    # Save to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results.json")

    # Winner Calculation
    summary = df.groupby("Agent")["Score"].mean().reset_index()
    winner = summary.loc[summary["Score"].idxmax()]
    print(f"\n🏆 Best Performing Agent: {winner['Agent']} with Average Score: {winner['Score']:.4f}")

if __name__ == "__main__":
    run_baseline()
