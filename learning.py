import random
import numpy as np
from typing import Dict, Tuple, List, Optional
from models import Observation, Action, ActionType, EmailUrgency

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Q-table: (StateTuple, ActionType) -> Value
        # State: (UrgencyInt, IsBossBool, HasKeywordsBool)
        self.q_table = {}
        self.actions = list(ActionType)
        self.learning_history = [] # Track total reward per episode

    def _get_state(self, obs: Observation) -> Optional[Tuple[int, bool, bool]]:
        if not obs.current_email:
            return None
        
        email = obs.current_email
        urgency = int(email.urgency)
        is_boss = "boss" in email.sender.lower()
        
        keywords = ["urgent", "invoice", "meeting", "critical", "important"]
        content_lower = (email.subject + " " + email.content).lower()
        has_keywords = any(kw in content_lower for kw in keywords)
        
        return (urgency, is_boss, has_keywords)

    def act(self, obs: Observation) -> Action:
        state = self._get_state(obs)
        if state is None:
            return Action(type=ActionType.NO_OP, reasoning="Wait mode.")

        if random.random() < self.epsilon:
            action_type = random.choice(self.actions)
            reasoning = f"Exploring {action_type.value} for long-term optimization."
        else:
            action_type = self._get_best_action(state)
            q_val = self.q_table.get((state, action_type), 0.0)
            reasoning = f"Optimized action based on learned Q-value: {q_val:.2f}."
        
        return Action(
            type=action_type, 
            email_id=obs.current_email.id,
            reasoning=reasoning
        )

    def _get_best_action(self, state: Tuple[int, bool, bool]) -> ActionType:
        q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
        max_q = max(q_values)
        indices = [i for i, q in enumerate(q_values) if q == max_q]
        return self.actions[random.choice(indices)]

    def learn(self, state: Tuple, action: ActionType, reward: float, next_state: Optional[Tuple]):
        current_q = self.q_table.get((state, action), 0.0)
        
        if next_state is not None:
            max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in self.actions])
        else:
            max_next_q = 0.0
            
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

class LearningManager:
    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def run_episode(self, env):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state = self.agent._get_state(obs)
            action = self.agent.act(obs)
            step_res = env.step(action)
            
            total_reward += step_res.reward.value
            next_state = self.agent._get_state(step_res.observation)
            
            if state is not None:
                self.agent.learn(state, action.type, step_res.reward.value, next_state)
            
            obs = step_res.observation
            done = step_res.done
            
        self.agent.learning_history.append(total_reward)
        return total_reward
