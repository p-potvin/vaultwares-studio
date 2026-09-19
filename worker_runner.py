import os
import sys
import threading
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add vaultwares_agentciation to sys.path
sys.path.insert(0, os.path.abspath("vaultwares_agentciation"))

try:
    from vaultwares_adk.telemetry import ModelRun
except ImportError:
    try:
        _adk = os.path.abspath(os.path.join(os.path.dirname(__file__), "vaultwares-adk"))
        if _adk not in sys.path:
            sys.path.insert(0, _adk)
        from vaultwares_adk.telemetry import ModelRun
    except Exception:
        ModelRun = None

import hook_registry
from agents.omni_agent import OmniAgent
from agents.reconstruction_agent import ReconstructionAgent
from agents.text_agent import TextAgent
from agents.video_agent import VideoAgent


def start_video_agent():
    print("Starting Video Agent...")
    agent = VideoAgent(agent_id="video-specialist")
    agent.start()
    while True:
        time.sleep(1)

def start_text_agent():
    print("Starting Text Agent...")
    agent = TextAgent(agent_id="text-analyst")
    agent.start()
    while True:
        time.sleep(1)

def start_reconstruction_agent():
    print("Starting Reconstruction Agent...")
    if ModelRun:
        with ModelRun(provider="local", runtime="cosmos-reason2", model="cosmos-reason2", task="3d-reconstruction", project="vaultwares-studio"):
            agent = ReconstructionAgent(agent_id="recon-professional")
            agent.start()
            while True:
                time.sleep(1)
    else:
        agent = ReconstructionAgent(agent_id="recon-professional")
        agent.start()
        while True:
            time.sleep(1)

def start_omni_agent():
    print("Starting Omni Agent...")
    agent = OmniAgent(agent_id="omni-specialist")
    agent.start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    threads = [
        threading.Thread(target=start_video_agent, daemon=True),
        threading.Thread(target=start_text_agent, daemon=True),
        threading.Thread(target=start_reconstruction_agent, daemon=True),
        threading.Thread(target=start_omni_agent, daemon=True),
    ]
    
    for thread in threads:
        thread.start()
    
    print("\nReady for Digital Twin Pipeline!")
    print("Agents online: Video, Text, Reconstruction, Omni (USD)")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down workers...")
        for thread in threads:
            thread.join(timeout=0.2)
