import os
import random
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import openai
from models import Observation, Action, ActionType, EmailUrgency

# Load environment variables from .env
load_dotenv()

def get_compliant_client():
    """Returns an OpenAI client configured via hackathon environment variables."""
    api_base = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN")
    return openai.OpenAI(base_url=api_base, api_key=api_key)

def call_llm(prompt: str, json_mode: bool = False):
    """Unified LLM call function for all agents."""
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    client = get_compliant_client()
    
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    if json_mode:
        # Some providers might not support response_format, 
        # so we ensure the prompt asks for JSON as well.
        try:
            kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            pass

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Agent LLM Error: {e}", flush=True)
        return None

class HeuristicAgent:
    def act(self, obs: Observation) -> Action:
        if not obs.current_email:
            return Action(type=ActionType.NO_OP, reasoning="No email.")
        email = obs.current_email
        content_lower = (email.subject + " " + email.content).lower()
        if "boss" in email.sender.lower() or "urgent" in content_lower or "critical" in content_lower:
            return Action(type=ActionType.REPLY, email_id=email.id, reasoning="Explicit priority keyword detected.")
        elif "invoice" in content_lower:
            return Action(type=ActionType.MARK_IMPORTANT, email_id=email.id, reasoning="Finance related keyword.")
        elif "newsletter" in email.sender.lower() or "sale" in content_lower:
            return Action(type=ActionType.ARCHIVE, email_id=email.id, reasoning="Likely noise.")
        else:
            return Action(type=ActionType.SCHEDULE_FOLLOWUP, email_id=email.id, reasoning="Default fallback.")

class OpenEnvAgent:
    """Hackathon-compliant agent using the unified OpenAI pathway."""
    def __init__(self, name: str = "Hackathon-Model"):
        self.name = name
        self.fallback = HeuristicAgent()

    def act(self, obs: Observation) -> Action:
        if not obs.current_email:
            return self.fallback.act(obs)
        
        email = obs.current_email
        prompt = f"""
You are a senior executive assistant.

EMAIL:
Sender: {email.sender}
Subject: {email.subject}
Content: {email.content}
Urgency Level: {email.urgency.name}
Time Remaining: {obs.time_remaining}

TASK:
Decide the best action considering:
- Hidden urgency (implicit deadlines)
- Business consequences (delays, impact)
- Sender importance (boss, client)
- Misleading signals (fake urgency)

IMPORTANT:
- Do NOT rely on keywords alone
- Think step-by-step before deciding

Return JSON:
{{
  "action": "reply | mark_important | archive | schedule_followup | no_op",
  "reasoning": "Explain deep reasoning including consequences",
  "reply_text": "Required for 'reply' action, otherwise empty"
}}
"""
        response_text = call_llm(prompt, json_mode=True)
        if not response_text:
            return self.fallback.act(obs)

        try:
            # Clean up response if it has markdown blocks
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            return Action(
                type=ActionType(data["action"]), 
                email_id=email.id, 
                reply_text=data.get("reply_text"), 
                reasoning=data.get("reasoning", "LLM Analysis Completed.")
            )
        except Exception as e:
            print(f"Parsing error: {e}", flush=True)
            return self.fallback.act(obs)

# Keep these for legacy compatibility in baseline.py but redirect to compliant logic
class OpenAI_Agent(OpenEnvAgent):
    def __init__(self):
        super().__init__(name="OpenAI-Compatible")

class Gemini_Agent(OpenEnvAgent):
    def __init__(self):
        super().__init__(name="Gemini-via-OpenAI")

class RandomAgent:
    def act(self, obs: Observation) -> Action:
        if not obs.current_email: return Action(type=ActionType.NO_OP, reasoning="Idle.")
        return Action(type=random.choice(list(ActionType)), email_id=obs.current_email.id, reasoning="Stochastic exploration.")
