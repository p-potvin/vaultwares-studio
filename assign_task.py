import sys
import os
import json
import redis

# Add vaultwares-adk to sys.path
sys.path.insert(0, os.path.abspath("..\\vaultwares-adk"))

if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    agent_id = "main_orchestrator"
    target = "video-specialist"
    task = "sample_frames"
    details = {
        "source": "room_capture.mp4",
        "fps": 2,
        "description": "Extract frames from the initial room capture video."
    }
    
    msg = {
        "agent": agent_id,
        "action": "ASSIGN",
        "task": task,
        "target": target,
        "details": details
    }
    
    r.publish("tasks", json.dumps(msg))
    print(f"Task '{task}' assigned to {target}")
