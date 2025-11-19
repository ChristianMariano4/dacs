import os
import numpy as np

# --- Audio & API Configuration ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = np.int16

# --- Server Configuration ---
FLASK_PORT = 50000
GRADIO_PORT = 50001

# --- Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FOLDER = os.path.join(CURRENT_DIR, 'cache')
GRAPH_LOG_PATH = os.path.join("graph_logs", "graph_history.jsonl")
FLYZONE_PATH = "controller/assets/tello/flyzone/flyzone_plot.png"

# --- Prompts ---
VOICE_SYSTEM_PROMPT = (
    "You are an interactive tutor for the TypeFly drone system. "
    "Guide the user step by step: explain what you are doing, "
    "wait for their response before continuing, and confirm their input. "
    "Speak clearly and concisely. Use audio and text so the user both hears "
    "and sees the tutorial. Begin by introducing yourself and explaining how "
    "to give a simple drone command."
)

# --- UI Templates (Embedded to remove file dependencies) ---
HEADER_HTML = """
<div style="text-align: center; max-width: 800px; margin: 0 auto;">
    <h1>TypeFly Controller</h1>
</div>
"""

DRONE_POV_HTML = """
<div style="text-align: center;">
    <h3>🎥 Drone POV</h3>
    <img src="http://localhost:50000/drone-pov/" style="width: 100%; border-radius: 8px; border: 1px solid #ddd;">
</div>
"""

GRAPH_IFRAME_TEMPLATE = """
<div style="width: 100%; height: 750px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
    <iframe src="http://localhost:50000/graph?t={timestamp}" 
            style="width: 100%; height: 100%; border: none;"
            sandbox="allow-scripts allow-same-origin">
    </iframe>
</div>
"""