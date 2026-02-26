import json
import os
from datetime import datetime

# Always write file in project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "emergency_logs.json")


def log_emergency_case(user_id, name, nationality, incident_location, original_message):

    new_entry = {
        "user_id": user_id,
        "name": name,
        "nationality": nationality,
        "incident_location": incident_location,
        "original_message": original_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "OPEN"
    }

    # Create file if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    # Load existing data
    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    # Append new entry
    data.append(new_entry)

    # Save back
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)
