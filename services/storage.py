import json
import os
import datetime
from services.detector import Assessment

LOG_FILE = os.path.join("data", "audit_log.jsonl")

def log_assessment(assessment: Assessment):
    """
    Appends an assessment result to the JSONL log file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Create a dictionary from the assessment object
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "score": assessment.score,
        "label": assessment.label,
        "categories": assessment.categories,
        "explanation": assessment.explanation,
        "rule_score": assessment.rule_score,
        "model_score": assessment.model_score,
        "context_score": assessment.context_score,
        "mode": assessment.mode,
        "translated_reply": assessment.translated_reply
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def get_logs() -> list[dict]:
    """
    Reads all logs from the JSONL file.
    """
    if not os.path.exists(LOG_FILE):
        return []
        
    logs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs
