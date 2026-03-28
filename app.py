import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from env import EmailEnv
from agent import RandomAgent, HeuristicAgent, OpenAI_Agent, Gemini_Agent
from learning import QLearningAgent, LearningManager
from tasks import get_tasks
import time
import os

st.set_page_config(page_title="MailMind AI- AI Email Assistant", layout="wide")

# Theme & Styling
st.markdown("""
<style>
    .stMetric { background: #f8f9fa; padding: 15px; border-radius: 12px; border: 1px solid #dee2e6; color: #1a1a1a !important; }
    .stMetric [data-testid="stMetricValue"] { color: #007bff !important; }
    .explain-box { 
        border-left: 5px solid #007bff; 
        padding: 15px; 
        background: #f1f3f5; 
        color: #212529 !important;
        margin: 15px 0; 
        border-radius: 4px; 
    }
</style>
""", unsafe_allow_html=True)

st.title("🏆 MailMind AI - Executive Edition")
st.markdown("### Hybrid Strategy: LLM Reasoning + RL Multi-Feature State Optimization")

if 'tasks' not in st.session_state:
    st.session_state.tasks = get_tasks()
    st.session_state.agents = {
        "Gemini (Pro)": Gemini_Agent(),
        "OpenAI (Pro)": OpenAI_Agent(),
        "Q-Learning (Pro)": QLearningAgent(),
        "Heuristic (Naive)": HeuristicAgent(),
        "Random (Noise)": RandomAgent()
    }
    st.session_state.q_manager = LearningManager(st.session_state.agents["Q-Learning (Pro)"])
    st.session_state.trained = False

# Sidebar for Training
st.sidebar.header("🧠 RL Training")
st.sidebar.info("Train the Q-Learning agent first to see data in the Analysis tabs.")
train_episodes = st.sidebar.slider("Training Episodes", 10, 500, 100)
if st.sidebar.button("Train Q-Learning Agent"):
    with st.spinner(f"Training on {train_episodes} episodes..."):
        for _ in range(train_episodes):
            st.session_state.q_manager.run_episode(EmailEnv())
    st.session_state.trained = True
    st.sidebar.success("Training Complete!")

# Simulation Controls
st.sidebar.divider()
st.sidebar.header("🎮 Live Simulation")
selected_task = st.sidebar.selectbox("Select Scenario", [t.name for t in st.session_state.tasks])
task = next(t for t in st.session_state.tasks if t.name == selected_task)
selected_agent = st.sidebar.selectbox("Select Agent", list(st.session_state.agents.keys()))

if st.sidebar.button("🚀 Run AI Agent"):
    env = EmailEnv(task=task)
    agent = st.session_state.agents[selected_agent]
    obs, _ = env.reset()
    done = False
    
    col_main, col_stats = st.columns([3, 1])
    
    with col_main:
        status_card = st.empty()
        email_card = st.empty()
        reasoning_card = st.empty()
        progress_bar = st.progress(0)

    with col_stats:
        m1 = st.empty()
        m2 = st.empty()

    step = 0
    total_reward = 0
    rewards_log = []

    def update_metrics(reward, pending):
        m1.markdown(f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:10px; border:1px solid #dee2e6; margin-bottom:10px;">
                <p style="color:#6c757d; font-size:14px; margin:0;">Total Reward</p>
                <h2 style="color:#007bff; margin:0;">{reward:.1f}</h2>
            </div>
        """, unsafe_allow_html=True)
        m2.markdown(f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:10px; border:1px solid #dee2e6;">
                <p style="color:#6c757d; font-size:14px; margin:0;">Emails Left</p>
                <h2 style="color:#007bff; margin:0;">{pending}</h2>
            </div>
        """, unsafe_allow_html=True)

    update_metrics(0.0, len(task.emails))

    while not done:
        action = agent.act(obs)
        step_res = env.step(action)
        obs = step_res.observation
        total_reward += step_res.reward.value
        rewards_log.append(step_res.reward.value)
        done = step_res.done
        step += 1

        with status_card:
            st.markdown(f"**Step {step}:** Agent `{selected_agent}` performed `{action.type.value.upper()}`")
            st.markdown(f"**Action Taken:** {action.type.value}")
        
        with email_card:
            st.info(f"📬 **{obs.current_email.subject if obs.current_email else 'Done'}**\n\nFrom: {obs.current_email.sender if obs.current_email else 'N/A'}")
        
        with reasoning_card:
            st.markdown(f"**Reasoning Insight:** {action.reasoning}")
            st.markdown(f'<div class="explain-box"><b>AI Reasoning Protocol:</b><br>{action.reasoning}</div>', unsafe_allow_html=True)
        
        update_metrics(total_reward, obs.emails_pending)
        progress_bar.progress(min(1.0, step / len(task.emails)))
        
        time.sleep(0.4)
    
    st.success(f"Simulation Complete! Final Task Score: {task.grader(env.state().history):.2%}")

# Dashboard Tabs
tab1, tab2 = st.tabs(["📈 Performance Analysis", "🤖 Agent Internals"])

with tab1:
    st.subheader("Learning Curves & Comparisons")
    if st.session_state.trained:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        history = st.session_state.agents["Q-Learning (Pro)"].learning_history
        ax1.plot(history, color='#007bff', alpha=0.6, label="Episode Reward")
        # Moving average
        if len(history) > 10:
            ma = pd.Series(history).rolling(10).mean()
            ax1.plot(ma, color='red', linewidth=2, label="10-Ep Moving Avg")
        ax1.set_title("Q-Learning Improvement Over Time")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend()
        st.pyplot(fig1)
    else:
        st.warning("No performance data yet. Please train the agent using the sidebar.")

with tab2:
    st.subheader("State-Action Decisions (Q-Table)")
    agent_q = st.session_state.agents["Q-Learning (Pro)"]
    if agent_q.q_table:
        st.write(f"Current Q-Table Knowledge Base: {len(agent_q.q_table)} entries.")
        # Show top 10 learned state-action pairs
        top_pairs = sorted(agent_q.q_table.items(), key=lambda x: x[1], reverse=True)[:10]
        st.write("Top Learned Strategies (State Tuple -> Optimal Action):")
        for (st_val, act_val), res in top_pairs:
            st.write(f"- 🧠 State {st_val} ➔ Action `{act_val.value}` | **Strength: {res:.2f}**")
    else:
        st.warning("Agent has not learned any strategies yet. Run training to populate the Q-Table.")
