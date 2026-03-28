import random
from typing import List, Tuple, Optional, Any
import gymnasium as gym
from models import (
    Email, Action, ActionType, Observation, Reward, StepResult, State, EmailUrgency
)
from tasks import Task, generate_sample_emails

class EmailEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, task: Optional[Task] = None):
        super(EmailEnv, self).__init__()
        self.task = task
        self.inbox: List[Email] = []
        self.history: List[dict] = []
        self.current_email_index = 0
        self.time_limit = 1500.0
        self.time_elapsed = 0.0
        self.emails_handled = 0
        self.last_action_status = "Waiting for first action."
        
        if self.task:
            self.inbox = self.task.emails
        else:
            self.inbox = generate_sample_emails(10)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Observation, dict]:
        super().reset(seed=seed)
        self.current_email_index = 0
        self.history = []
        self.time_elapsed = 0.0
        self.emails_handled = 0
        self.last_action_status = "Environment Reset."
        return self._get_observation(), {}

    def _get_observation(self) -> Observation:
        current_email = self.inbox[self.current_email_index] if self.current_email_index < len(self.inbox) else None
        pending = self.inbox[self.current_email_index:]
        
        return Observation(
            pending_emails=pending,
            current_email=current_email,
            emails_handled=self.emails_handled,
            emails_pending=len(pending),
            time_remaining=max(0.0, self.time_limit - self.time_elapsed),
            last_action_status=self.last_action_status
        )

    def step(self, action: Action) -> StepResult:
        if self.current_email_index >= len(self.inbox):
            return StepResult(
                observation=self._get_observation(),
                reward=Reward(value=0.0, reason="No more emails."),
                done=True,
                info={"status": "completed"}
            )

        email = self.inbox[self.current_email_index]
        reward_value, reason = self._calculate_reward(email, action)
        
        self.history.append({"email": email, "action": action})
        self.emails_handled += 1
        self.current_email_index += 1
        self.time_elapsed += random.uniform(5.0, 15.0) # Simulate time taken
        self.last_action_status = f"Action {action.type.value} performed on {email.id}"

        done = (self.current_email_index >= len(self.inbox)) or (self.time_elapsed >= self.time_limit)
        
        return StepResult(
            observation=self._get_observation(),
            reward=Reward(value=reward_value, reason=reason),
            done=done,
            info={"time_elapsed": self.time_elapsed}
        )

    def state(self) -> State:
        return State(
            inbox=self.inbox,
            history=self.history,
            stats={
                "emails_handled": self.emails_handled,
                "time_elapsed": self.time_elapsed,
                "total_emails": len(self.inbox)
            }
        )

    def _calculate_reward(self, email: Email, action: Action) -> Tuple[float, str]:
        reward = 0.0
        reason = ""
        is_boss = "boss" in email.sender.lower()

        # Timely action reward
        if self.time_elapsed < self.time_limit * 0.5:
            reward += 0.5
            reason += "Timely. "

        # Boss Priority (Wow Factor)
        if is_boss:
            if action.type == ActionType.REPLY:
                reward += 5.0
                reason += "EXCELLENT: Boss email prioritized with reply. "
            elif action.type == ActionType.MARK_IMPORTANT:
                reward += 2.0
                reason += "GOOD: Boss email flagged. "
            elif action.type == ActionType.ARCHIVE:
                reward -= 5.0
                reason += "CRITICAL FAILURE: Archived boss email. "

        # Correctness based on urgency
        elif email.urgency >= EmailUrgency.URGENT:
            if action.type in [ActionType.MARK_IMPORTANT, ActionType.REPLY]:
                reward += 2.0
                reason += "Correct prioritization for urgent email. "
            elif action.type == ActionType.ARCHIVE:
                reward -= 2.0
                reason += "Archived an urgent email without action. "
            elif action.type == ActionType.NO_OP:
                reward -= 1.0
                reason += "Ignored urgent email. "
        
        # Patch 4: Hidden Consequence Penalty
        content_lower = (email.subject + " " + email.content).lower()
        if any(word in content_lower for word in ["delay", "deadline", "impact"]):
            if action.type == ActionType.ARCHIVE:
                reward -= 3.0
                reason += "Ignored high-impact consequence. "
        
        elif email.urgency == EmailUrgency.LOW:
            if action.type == ActionType.ARCHIVE:
                reward += 1.0
                reason += "Correctly archived low priority email. "
            elif action.type == ActionType.REPLY:
                reward -= 0.5
                reason += "Wasted time replying to low priority. "
        
        else:
            if action.type != ActionType.NO_OP:
                reward += 0.5
                reason += "Handled email. "
            else:
                reward -= 0.5
                reason += "Skipped necessary email. "

        return reward, reason

    def render(self):
        obs = self._get_observation()
        print(f"\n--- Inbox Status ---")
        print(f"Handled: {obs.emails_handled} | Pending: {obs.emails_pending}")
        print(f"Current Email: {obs.current_email.subject if obs.current_email else 'None'}")
        print(f"Time Remaining: {obs.time_remaining:.1f}")
