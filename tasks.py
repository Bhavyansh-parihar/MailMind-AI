import random
from datetime import datetime, timedelta
from typing import List, Dict, Callable
from models import Email, EmailUrgency, Action, ActionType

def generate_sample_emails(count: int = 10) -> List[Email]:
    senders = ["boss@company.com", "client@external.com", "newsletter@spam.com", "hr@company.com", "colleague@company.com"]
    subjects = ["Urgent: Meeting", "Invoice Due", "Weekly Newsletter", "Holiday Policy", "Lunch?", "Project Update"]
    contents = [
        "Please attend the meeting at 3 PM today. This is critical for the project.",
        "Your invoice for the last month is due. Please pay immediately.",
        "Check out our latest news and updates from the industry.",
        "We have updated the holiday policy. Please review the attached document.",
        "Are you free for lunch today? Let's catch up.",
        "The project update is ready. Please take a look at the slides."
    ]
    
    emails = []
    base_time = datetime.now()
    for i in range(count):
        sender = random.choice(senders)
        subject = random.choice(subjects)
        content = random.choice(contents)
        urgency = EmailUrgency.LOW
        if "Urgent" in subject or "Critical" in content or "boss" in sender:
            urgency = EmailUrgency.CRITICAL
        elif "Invoice" in subject or "client" in sender:
            urgency = EmailUrgency.HIGH
        elif "newsletter" in sender:
            urgency = EmailUrgency.LOW
        else:
            urgency = random.choice([EmailUrgency.MEDIUM, EmailUrgency.LOW])

        emails.append(Email(
            id=f"email_{i}",
            sender=sender,
            subject=subject,
            content=content,
            urgency=urgency,
            timestamp=base_time - timedelta(minutes=random.randint(1, 100))
        ))
    return emails

def generate_ambiguous_emails() -> List[Email]:
    raw_data = [
        ("boss@company.com", "Quick chat", "Hey, when you get a moment, let’s finalize the proposal before tomorrow.", EmailUrgency.URGENT),
        ("client@external.com", "Small clarification", "Just a tiny doubt about the invoice — we’re closing accounts today.", EmailUrgency.URGENT),
        ("colleague@company.com", "Lunch?", "Are you free later? Also, did you get a chance to review the contract?", EmailUrgency.HIGH),
        ("hr@company.com", "Policy Update", "We’ve updated leave policies. Please review sometime this week.", EmailUrgency.LOW),
        ("boss@company.com", "No rush", "Not urgent, but we might need those numbers for the board meeting.", EmailUrgency.HIGH),
        ("client@external.com", "Follow-up", "Checking in again — we’ll proceed based on your response.", EmailUrgency.MEDIUM),
        ("newsletter@spam.com", "Weekly Insights", "Top 10 productivity hacks for 2026.", EmailUrgency.LOW),
        ("boss@company.com", "FYI", "Forwarding this. Might be useful for tomorrow’s discussion.", EmailUrgency.MEDIUM),
        ("unknown@random.com", "Opportunity", "We have an exciting offer for you!!!", EmailUrgency.LOW),
        ("client@external.com", "Re: Project Timeline", "If we don’t finalize today, it may delay our launch.", EmailUrgency.CRITICAL),
        ("colleague@company.com", "Quick help", "Can you quickly check this? It’s blocking my deployment.", EmailUrgency.HIGH),
        ("boss@company.com", "Update", "Let’s ensure everything is ready before the morning.", EmailUrgency.URGENT),
        ("hr@company.com", "Reminder", "Please submit your documents by end of week.", EmailUrgency.LOW),
        ("client@external.com", "Gentle reminder", "We’re waiting on your confirmation to move forward.", EmailUrgency.HIGH),
        ("colleague@company.com", "Meeting notes", "Sharing notes. Action items include finalizing budget.", EmailUrgency.MEDIUM),
        ("boss@company.com", "Idea", "Had a thought — let’s discuss this before the client call.", EmailUrgency.HIGH),
        ("spam@ads.com", "SALE!!!", "Limited time discount on electronics!", EmailUrgency.LOW),
        ("client@external.com", "Urgent-ish", "Not critical, but would appreciate a quick response.", EmailUrgency.MEDIUM),
        ("colleague@company.com", "Heads up", "System might go down tonight for updates.", EmailUrgency.HIGH),
        ("boss@company.com", "Check this", "This might impact our quarterly results. Take a look.", EmailUrgency.CRITICAL),
        # Patch 1: Break Heuristic Reliance
        ("client@external.com", "Status Update", "Everything looks fine. No immediate action needed right now.", EmailUrgency.LOW),
        ("boss@company.com", "Quick thought", "This isn’t urgent, but missing this could affect tomorrow’s board review.", EmailUrgency.CRITICAL),
        ("colleague@company.com", "FYI", "No action required unless deployment fails later.", EmailUrgency.LOW),
        ("client@external.com", "Check-in", "Just making sure we’re aligned before the deadline hits.", EmailUrgency.HIGH),
        ("boss@company.com", "Casual note", "We might want to double-check those numbers before presenting.", EmailUrgency.HIGH),
        # Patch 2: False Positive Trap
        ("newsletter@spam.com", "URGENT: Read Now", "This is just promotional content, no action required.", EmailUrgency.LOW),
    ]
    
    emails = []
    base_time = datetime.now()
    for i, (sender, subject, content, urgency) in enumerate(raw_data):
        emails.append(Email(
            id=f"amb_email_{i}",
            sender=sender,
            subject=subject,
            content=content,
            urgency=urgency,
            timestamp=base_time - timedelta(minutes=random.randint(1, 100))
        ))
    return emails

class Task:
    def __init__(self, name: str, difficulty: str, objective: str, emails: List[Email], grader: Callable):
        self.name = name
        self.difficulty = difficulty
        self.objective = objective
        self.emails = emails
        self.grader = grader

def grade_easy(history: List[Dict]) -> float:
    if not history: return 0.0
    score = 0
    for entry in history:
        email, action = entry['email'], entry['action']
        if email.urgency >= EmailUrgency.HIGH:
            if action.type == ActionType.MARK_IMPORTANT: score += 1
        else:
            if action.type == ActionType.ARCHIVE: score += 1
    return score / len(history)

def grade_medium(history: List[Dict]) -> float:
    if not history: return 0.0
    score = 0
    for entry in history:
        email, action = entry['email'], entry['action']
        if email.urgency >= EmailUrgency.HIGH:
            if action.type == ActionType.REPLY: score += 1
        elif email.urgency == EmailUrgency.LOW:
            if action.type == ActionType.ARCHIVE: score += 1
        else:
            if action.type in [ActionType.MARK_IMPORTANT, ActionType.SCHEDULE_FOLLOWUP]: score += 1
    return score / len(history)

def grade_ambiguous(history: List[Dict]) -> float:
    """Grader for Ambiguous Task: Rewards semantic intent and reasoning."""
    if not history: return 0.0
    score = 0.0
    for entry in history:
        email, action = entry['email'], entry['action']
        reasoning = action.reasoning.lower() if action.reasoning else ""
        
        # Action Core Score
        if email.urgency >= EmailUrgency.URGENT:
            if action.type == ActionType.REPLY: 
                score += 1.0
                # Semantic Reasoning Bonus (Patch 3)
                if any(word in reasoning for word in ["deadline", "important", "critical", "impact"]):
                    score += 0.3
            elif action.type == ActionType.MARK_IMPORTANT: score += 0.5
            else: score -= 0.5 
        elif email.urgency == EmailUrgency.HIGH:
            if action.type in [ActionType.REPLY, ActionType.MARK_IMPORTANT]: 
                score += 1.0
                # Semantic Reasoning Bonus (Patch 3)
                if "important" in reasoning or "follow" in reasoning:
                    score += 0.2
            elif action.type == ActionType.SCHEDULE_FOLLOWUP: score += 0.5
        elif email.urgency == EmailUrgency.LOW:
            if action.type == ActionType.ARCHIVE: score += 1.0
            elif action.type != ActionType.NO_OP: score -= 0.5
        else:
            if action.type != ActionType.NO_OP: score += 0.5
            
    return max(0.0, min(1.0, score / len(history)))

def get_tasks() -> List[Task]:
    return [
        Task("EASY", "Easy", "Correctly classify emails based on urgency.", generate_sample_emails(5), grade_easy),
        Task("MEDIUM", "Medium", "Prioritize and respond to important sender/subjects.", generate_sample_emails(10), grade_medium),
        Task("AMBIGUOUS", "Stress Test", "Handle subtle and ambiguous email signals through reasoning.", generate_ambiguous_emails(), grade_ambiguous),
    ]
