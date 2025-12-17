import asyncio
import json
import queue
import random
import sys
import os
import io
import time
import gradio as gr
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
from threading import Thread
import argparse
import sounddevice as sd
import numpy as np
import base64
import wave
from openai import OpenAI
from queue import Empty as _QEmpty
from PIL import Image
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

from controller.utils.constants import FLYZONE_USER_IMAGE_PATH
from controller.context_map.graph_manager import GraphManager
from controller.llm.llm_controller import LLMController
from controller.utils.general_utils import print_t
from controller.llm.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType

# OpenAI Realtime API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Audio configuration
SAMPLE_RATE = 24000  # OpenAI Realtime API expects 24kHz
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = np.int16

class AudioHandler:
    def __init__(self):
        self.recording = False
        self.recording_buffer = bytearray()
        
    def audio_input_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio input status: {status}")
        if self.recording:
            # Convert to int16 and add to queue
            audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            self.recording_buffer.extend(audio_bytes)

    def get_wav_bytes(self):
        """Wrap the buffered PCM16 @ 24kHz mono into a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(CHANNELS)   # 1
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(SAMPLE_RATE)  # 24000
            wf.writeframes(bytes(self.recording_buffer))
        return buf.getvalue()

class SpeechToText:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self._client = OpenAI(api_key=api_key) if (api_key and OpenAI) else None
        self._local_model = None  # lazy-init for faster-whisper optional fallback

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """
        Try OpenAI transcription first (gpt-4o-* Transcribe).
        Fall back to local faster-whisper if available.
        """
        # --- OpenAI path (preferred) ---
        if self._client:
            import io as _io
            bio = _io.BytesIO(wav_bytes)
            bio.name = "speech.wav"  # some SDK versions require a filename
            last_err = None
            # try a couple of current STT models; fall back to whisper-1 if needed
            try:
                bio.seek(0)  # reset buffer 
                resp = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=bio,
                    **({"language": language} if language else {})
                )
                text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
                if text and text.strip():
                    return text.strip()
            except Exception as e:
                last_err = e
            raise RuntimeError(f"OpenAI transcription failed: {last_err}")

        # --- Local fallback (optional) ---
        try:
            if self._local_model is None:
                from faster_whisper import WhisperModel
                # Small or base are good starters; adjust as you like
                self._local_model = WhisperModel("base", device="cpu", compute_type="int8")
            # faster-whisper wants a file path; write temp
            tmp_path = os.path.join(CURRENT_DIR, "last_recording.wav")
            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)
            segments, _info = self._local_model.transcribe(tmp_path, language=language)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise RuntimeError(
                "No transcription provider available. "
                "Set OPENAI_API_KEY or install faster-whisper."
            ) from e

class VoiceAgent:
    def __init__(self, typefly_instance):
        self.typefly = typefly_instance
        self.audio_handler = AudioHandler()
        self.voice_active = False
        
    def start_audio_streams(self):
        """Start audio input/output streams"""
        try:
            self.input_stream = sd.InputStream(
                callback=self.audio_handler.audio_input_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype=np.float32
            )
            self.input_stream.start()
            return "Audio streams started successfully"
            
        except Exception as e:
            return f"Failed to start audio streams: {e}"

    def stop_audio_streams(self):
        """Stop audio streams"""
        try:
            if hasattr(self, 'input_stream'):
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            return "Audio streams stopped"
        except Exception as e:
            return f"Error stopping audio streams: {e}"

class TypeFly:
    def __init__(self, robot_type, use_http=False):
        # Create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.message_queue = queue.Queue()
        self.message_queue.put(self.cache_folder)
        self.user_answer_queue = queue.Queue()
        self.user_question_answer = [] # list of last question-answer pair between LLM and user
        self.graph_manager = GraphManager()
        self.llm_controller = LLMController(robot_type, self.graph_manager, use_http, self.message_queue, self.user_answer_queue)

        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        
        # Initialize Voice Agent
        self.voice_agent = VoiceAgent(self)
        self.transcriber = SpeechToText(OPENAI_API_KEY)
        
        # Graph log file path
        self.graph_log_path = os.path.join("graph_logs", "graph_history.jsonl")
        default_sentences = [
            "Find an apple",
            "Come back to region_0",
            "Where is a banana?",
            "Find a banana",
        ]
        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())
            
            # Create tabs for different views
            with gr.Tabs():
                with gr.TabItem("🤖 Robot Control"):
                    # Username display at the top with auto-refresh
                    def _ui_get_username():
                        # Always read from the controller so any backend change is reflected
                        return f"### 👤 Pilot: {self.llm_controller.get_username()}"

                    self.username_display = gr.Markdown(
                        _ui_get_username(),
                        elem_id="username-display"
                    )

                    # Poll every 1s to refresh the label if it changed
                    gr.Timer(1.0).tick(
                        fn=_ui_get_username,
                        outputs=self.username_display
                    )
                    # Create two-column layout for drone POV and flyzone
                    with gr.Row():
                        # Left column: Drone POV
                        with gr.Column(scale=1):
                            gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
                        
                        # Right column: Flyzone Map
                        with gr.Column(scale=1):
                            gr.Markdown("### 🗺️ Current Flyzone")
                            flyzone_display = gr.Image(
                                value="controller/assets/tello/flyzone/flyzone_plot.png",
                                label="Flyzone Map",
                                show_label=False,
                                interactive=False,
                                container=True
                            )
                            
                            def refresh_flyzone_image():
                                flyzone_path = "controller/assets/tello/flyzone/flyzone_plot.png"
                                if os.path.exists(flyzone_path):
                                    # Force reload by returning the path with timestamp
                                    return flyzone_path
                                return None
                            
                            gr.Timer(1.0).tick(
                                fn=refresh_flyzone_image,
                                outputs=flyzone_display
                            )
                    
                    # Chat interface below the two-column layout
                    with gr.Column():
                        # Chat display area
                        chatbot = gr.Chatbot(
                            value=[],
                            elem_id="chatbot",
                            height=500,
                            label="Chat"
                        )
                        
                        # Unified input area (WhatsApp-style)
                        with gr.Row():
                            # Voice recording button (left)
                            voice_btn = gr.Button(
                                "🎤",
                                variant="secondary",
                                size="sm",
                                scale=1,
                                min_width=60
                            )

                            image_input = gr.Image(
                                label="Upload Image",
                                type="numpy",   # returns numpy array
                                visible=True,
                                scale=2
                            )
                            
                            # Text input (center - takes most space)
                            msg_input = gr.Textbox(
                                placeholder="Type a message or use voice...",
                                show_label=False,
                                scale=8,
                                container=False
                            )
                            
                            # Send button (right)
                            send_btn = gr.Button(
                                "📤",
                                variant="primary",
                                size="sm",
                                scale=1,
                                min_width=60
                            )
                        
                        # Recording status indicator (hidden by default)
                        recording_status = gr.Markdown(
                            "",
                            visible=False
                        )
                        
                        # Examples
                        gr.Examples(
                            examples=default_sentences,
                            inputs=msg_input
                        )
                    
                    # Store state for recording
                    recording_state = gr.State(False)
                    
                    # Event handlers
                    def toggle_recording(is_recording):
                        """Toggle between start/stop recording"""
                        if not is_recording:
                            # Start recording
                            status = self.start_recording()
                            return (
                                True, 
                                "⏹️",  # Change button to stop icon
                                gr.Markdown("🔴 Recording... Click again to stop", visible=True),
                                gr.Textbox(interactive=False)  # Disable text input while recording
                            )
                        else:
                            # Stop recording and transcribe
                            result = self.stop_and_transcribe()
                            
                            # Check if transcription was successful
                            if result.startswith("✅"):
                                # Extract transcript from result
                                transcript = result.split("'")[1] if "'" in result else ""
                                return (
                                    False,
                                    "🎤",  # Change back to mic icon
                                    gr.Markdown("", visible=False),
                                    gr.Textbox(value=transcript, interactive=True)  # Put transcript in text box
                                )
                            else:
                                # Error case
                                return (
                                    False,
                                    "🎤",
                                    gr.Markdown(result, visible=True),
                                    gr.Textbox(interactive=True)
                                )

                    voice_btn.click(
                            fn=toggle_recording,
                            inputs=[recording_state],
                            outputs=[recording_state, voice_btn, recording_status, msg_input]
                        )
                    
                def send_message(message, history, image_array):
                    """Send message and get response - generator for streaming"""
                    if not message.strip() and not image_array:
                        yield history, ""
                        return
                    user_entry = message

                    img_b64 = None
                    # Handle image uploads (convert numpy → base64) and save it to use it later
                    if image_array is not None:
                        img = Image.fromarray(image_array.astype('uint8'))
                        img.save(FLYZONE_USER_IMAGE_PATH)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                        user_entry += f"\n\n![uploaded image](data:image/png;base64,{img_b64})"
                        
                    history = history + [[user_entry, None]]
                    yield history, ""

                    # Process message as before
                    for partial_response in self.process_message(message, history, img_b64):
                        if partial_response:
                            clean_response = partial_response.split("\nCommand Complete!")[0]
                            history[-1][1] = clean_response
                            yield history, ""

                send_btn.click(
                    send_message,
                    inputs=[msg_input, chatbot, image_input],
                    outputs=[chatbot, msg_input]
                )

                msg_input.submit(
                    send_message,
                    inputs=[msg_input, chatbot, image_input],
                    outputs=[chatbot, msg_input]
                )
                
                with gr.TabItem("📊 Graph Visualization", id="graph_tab") as graph_tab:
                    self.setup_graph_tab()
                    # Connect the tab select event to refresh the graph

    def start_recording(self):
        try:
            # Ensure clean state
            if hasattr(self.voice_agent, 'input_stream') and self.voice_agent.input_stream:
                self.voice_agent.stop_audio_streams() # Use the helper we cleaned up

            # Start stream
            self.voice_agent.start_audio_streams() 
            self.voice_agent.audio_handler.recording = True
            self.voice_agent.audio_handler.recording_buffer.clear()
            print("[DEBUG] Recording started")
            return "Recording started"
        except Exception as e:
            return f"Error: {e}"

    def stop_and_transcribe(self):
        """Stop recording and return transcription"""
        try:
            # Stop recording
            self.voice_agent.audio_handler.recording = False
            try:
                if getattr(self.voice_agent, "input_stream", None):
                    self.voice_agent.input_stream.stop()
                    self.voice_agent.input_stream.close()
                    self.voice_agent.input_stream = None
            except Exception as _:
                pass

            # Get WAV bytes
            wav_bytes = self.voice_agent.audio_handler.get_wav_bytes()

            if len(wav_bytes) <= 44:
                return "⚠️ No audio captured. Try again and speak clearly."

            # Transcribe
            print("[DEBUG] Transcribing audio...")
            transcript = self.transcriber.transcribe(wav_bytes)
            
            if not transcript.strip():
                return "⚠️ Could not understand audio. Please try again."

            # Clear buffer for next recording
            self.voice_agent.audio_handler.recording_buffer.clear()
            
            return f"✅ Transcribed: '{transcript}'"

        except Exception as e:
            print(f"[DEBUG] Error: {e}")
            return f"❌ Error: {e}"
        
    def stop_recording(self):
        """Stop recording audio"""
        return self.voice_agent.audio_handler.stop_recording()

    def setup_graph_tab(self):
        """Setup the graph visualization tab"""
        with gr.Column():
            gr.Markdown("## 🌐 Dynamic Object Detection Graph")
            
            # Static Iframe - no timestamp parameter needed anymore
            graph_html = """
            <div style="width: 100%; height: 750px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                <iframe src="http://localhost:50000/graph" 
                        style="width: 100%; height: 100%; border: none;"
                        sandbox="allow-scripts allow-same-origin">
                </iframe>
            </div>
            """
            gr.HTML(graph_html)

    # def checkbox_llama3(self):
    #     self.use_llama3 = not self.use_llama3
    #     if self.use_llama3:
    #         print_t(f"Switch to llama3")
    #         self.llm_controller.planner.set_model(LLAMA3)
    #     else:
    #         print_t(f"Switch to gpt4")
    #         self.llm_controller.planner.set_model(GPT4)

    def _format_log_message(self, msg: str) -> str:
        """
        Parses backend protocol markers ([LOG], [Q], \\) into readable text.
        """
        formatted_out = ""
        
        # Handle protocol markers
        is_log = msg.startswith('[LOG]')
        is_question = msg.startswith('[Q]')
        
        # Visual formatting: Add newline before logs or questions for readability
        if is_log or is_question:
            formatted_out += '\n'
            
        # Strip protocol suffixes (like line continuations)
        clean_content = msg.rstrip('\\\\')
        
        # Add content
        formatted_out += clean_content
        
        # Regular messages get a newline, line-continued ones don't
        if not msg.endswith('\\\\'):
            formatted_out += '\n'
            
        return formatted_out

    def process_message(self, message, history, img_b64):
        """
        Unified handler for tasks and answers using a single consumer loop.
        """
        print_t(f"[S] Processing input: {message}")

        if not message or message == "exit":
            if message == "exit":
                self.llm_controller.stop_controller()
                self.system_stop = True
                yield "Shutting down..."
            else:
                yield "[WARNING] Empty command!"
            return

        # --- 1. DISPATCH PHASE ---
        # Check if we are waiting for an answer to a question
        if self.user_question_answer:
            print_t(f"[DEBUG] Treating as answer to: {self.user_question_answer[0]}")
            # Combine the stored question with the new answer
            conversation_pair = self.user_question_answer + [message]
            # Reset state immediately
            self.user_question_answer = [] 
            # Send to backend
            self.user_answer_queue.put(conversation_pair)
        
        else:
            # It is a new task
            # Start the controller on a separate thread so we don't block here
            task_thread = Thread(
                target=self.llm_controller.execute_task_description, 
                args=(message, img_b64,)
            )
            task_thread.start()

        # --- 2. CONSUMPTION PHASE ---
        complete_response = ""
        start_time = time.time()
        timeout = 60  # seconds
        
        while True:
            try:
                # Poll queue with short timeout to allow UI updates
                msg = self.message_queue.get(timeout=0.1)
                
                # A. Check for End of Stream
                if msg == 'end':
                    yield complete_response + "\nCommand Complete!"
                    return

                # B. Check for Protocol Strings
                if isinstance(msg, str):
                    # Detect if the backend is asking a new question
                    if msg.startswith('[Q]'):
                        self.user_question_answer = [msg] # Save state
                        
                    # Format and append
                    complete_response += self._format_log_message(msg)
                    yield complete_response
                    
                    # If it was a question, we stop listening and wait for user input
                    if msg.startswith('[Q]'):
                        return

            except _QEmpty:

                continue

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def _create_graph_figure(self):
            """
            Helper: Generates the Plotly Figure. 
            Calculates object positions dynamically relative to their parent regions.
            """
            import plotly.graph_objects as go
            from shapely import Polygon

            graph_json_str = self.graph_manager.get_graph()
            if not graph_json_str:
                return go.Figure()

            graph_data: dict = json.loads(graph_json_str)

            # --- Helper to safely parse XY ---
            def parse_xy(v):
                if isinstance(v, str):
                    try: return json.loads(v)
                    except: return (0.0, 0.0)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    return float(v[0]), float(v[1])
                return (0.0, 0.0)

            # 1. Parse Regions (Metric: they have coords)
            regions = {r["name"]: parse_xy(r["coords"]) for r in graph_data.get("regions", [])}
            
            # 2. Parse Objects (Topological: we just need their names)
            # Note: We no longer look for 'coords' here to prevent KeyErrors
            object_names = {o["name"] for o in graph_data.get("objects", [])}

            region_connections = graph_data.get("region_connections", [])
            object_connections = graph_data.get("object_connections", [])
            current_pos = graph_data.get("current_position", {}) # Note: JSON key is usually "current_position"

            # --- Calculate Visual Positions for Objects ---
            object_display_positions = {}
            
            # Find which region acts as the anchor for each object
            for a, b in object_connections:
                # Determine which node is the region and which is the object
                if a in regions and b in object_names:
                    region_node, obj_node = a, b
                elif b in regions and a in object_names:
                    region_node, obj_node = b, a
                else:
                    continue # Skip object-object or region-region links in this loop

                # Generate a stable pseudo-random position around the region
                if region_node in regions:
                    rx, ry = regions[region_node]
                    
                    # Use hash of name to ensure the object stays in the same place 
                    # every time the graph refreshes
                    seed_value = hash(obj_node) % 10000
                    random.seed(seed_value)
                    
                    angle = random.uniform(0, 2 * np.pi)
                    distance = random.uniform(20, 40) # Distance from region center (cm)
                    
                    object_display_positions[obj_node] = (
                        rx + distance * np.cos(angle), 
                        ry + distance * np.sin(angle)
                    )
                    random.seed() # Reset seed

            # --- Build Figure ---
            fig = go.Figure()

            # 1. Flyzones (Polygon)
            try:
                flyzones = self.llm_controller.middle_layer.get_flyzone_polygon()
                for idx, poly in enumerate(flyzones):
                    if isinstance(poly, Polygon):
                        x, y = poly.exterior.xy
                        fig.add_trace(go.Scatter(
                            x=list(x), y=list(y), fill='toself', fillcolor='rgba(255,165,0,0.1)',
                            line=dict(color='orange', width=2), name=f"Flyzone {idx+1}", hoverinfo='skip'
                        ))
            except Exception as e:
                print(f"[WARN] Could not plot flyzone: {e}")

            # 2. Region Circles (Visual anchor zones)
            radius = 75
            theta = np.linspace(0, 2 * np.pi, 30)
            for region_name, (cx, cy) in regions.items():
                fig.add_trace(go.Scatter(
                    x=cx + radius * np.cos(theta), y=cy + radius * np.sin(theta),
                    mode='lines', line=dict(color='lightgray', width=1, dash='dash'),
                    fill='toself', fillcolor='rgba(128,128,128,0.35)', opacity=0.5,
                    hoverinfo='skip', showlegend=False
                ))

            # 3. Region-Region Connections
            for r1, r2 in region_connections:
                if r1 in regions and r2 in regions:
                    x1, y1 = regions[r1]
                    x2, y2 = regions[r2]
                    fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='lines',
                        line=dict(color='gray', dash='dot'), opacity=0.5, showlegend=False))

            # 4. Object-Region Connections (Visual lines)
            for obj_name, (ox, oy) in object_display_positions.items():
                # Find parent region again to draw line
                # (In a real app, you might optimize by storing parent in the loop above)
                parent_region = None
                for r_name, r_coords in regions.items():
                    # This is a visual heuristic; strictly we should check edges again, 
                    # but drawing to the nearest is usually visually sufficient or 
                    # you can re-iterate object_connections if strictness is needed.
                    pass 
                
                # Re-find the connected region for drawing the line
                for a, b in object_connections:
                    other = a if b == obj_name else (b if a == obj_name else None)
                    if other and other in regions:
                        rx, ry = regions[other]
                        fig.add_trace(go.Scatter(x=[rx, ox], y=[ry, oy], mode='lines',
                            line=dict(color='lightblue', dash='solid', width=1), opacity=0.6, showlegend=False))
                        break

            # 5. Markers: Regions
            if regions:
                fig.add_trace(go.Scatter(
                    x=[x for x, _ in regions.values()], y=[y for _, y in regions.values()],
                    mode='markers+text', text=list(regions.keys()), textposition='top center',
                    marker=dict(size=14, color='skyblue', line=dict(color='black', width=1)), name='Regions'
                ))

            # 6. Markers: Objects
            if object_display_positions:
                fig.add_trace(go.Scatter(
                    x=[x for x, _ in object_display_positions.values()], 
                    y=[y for _, y in object_display_positions.values()],
                    mode='markers+text', 
                    text=list(object_display_positions.keys()), 
                    textposition='top center',
                    marker=dict(size=9, color='lightgreen', line=dict(color='black', width=1)), 
                    name='Objects'
                ))

            # 7. Marker: Drone
            if "coords" in current_pos and current_pos["coords"]:
                x, y, *_ = current_pos["coords"]
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers+text', text=[f"Drone"], textposition='top right',
                    marker=dict(symbol='x', size=12, color='red', line=dict(color='black', width=1)), name='You'
                ))

            # --- Layout ---
            fig.update_layout(
                autosize=True,   
                height=None,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="X (cm)", 
                yaxis_title="Y (cm)", 
                template="plotly_white",
                showlegend=True,
                dragmode='pan',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange=True)
            
            return fig

    def setup_graph_server(self, app):
        import plotly
        
        @app.route('/graph-json')
        def graph_json():
            fig = self._create_graph_figure()
            return fig.to_json()

        @app.route('/graph')
        def graph_page():
            # CSS: width/height 100vw/vh ensures it fills the iframe
            # JS: captures current range before update
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
                <style>
                    body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
                    #graph-div { width: 100vw; height: 100vh; }
                </style>
            </head>
            <body>
                <div id="graph-div"></div>
                <script>
                    var graphDiv = document.getElementById('graph-div');
                    
                    async function updateGraph() {
                        try {
                            const response = await fetch('/graph-json');
                            const newFig = await response.json();
                            
                            // --- PRESERVE ZOOM STATE ---
                            // If the graph is already initialized, grab the current user's view
                            if (graphDiv.layout && graphDiv.layout.xaxis && graphDiv.layout.xaxis.range) {
                                // Overwrite the incoming server layout with the user's current ranges
                                newFig.layout.xaxis.range = graphDiv.layout.xaxis.range;
                                newFig.layout.yaxis.range = graphDiv.layout.yaxis.range;
                                
                                // Also preserve the 'autorange' setting if the user hasn't zoomed yet
                                if (graphDiv.layout.xaxis.autorange === false) {
                                    newFig.layout.xaxis.autorange = false;
                                    newFig.layout.yaxis.autorange = false;
                                }
                            }
                            // ---------------------------

                            Plotly.react(graphDiv, newFig.data, newFig.layout).then(function() {
                                // Ensure it fills the screen after render
                                Plotly.Resize(graphDiv); 
                            });

                        } catch (err) {
                            console.error("Graph update failed:", err);
                        }
                    }

                    // Update every 1 second
                    updateGraph();
                    setInterval(updateGraph, 1000);
                    
                    // Handle window resizing
                    window.onresize = function() {
                        Plotly.Plots.resize(graphDiv);
                    };
                </script>
            </body>
            </html>
            """
            return html
    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.start()

        app = Flask(__name__)
        CORS(app)  # allow all origins
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Add graph visualization endpoints
        self.setup_graph_server(app)

        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': False, 'use_reloader': False})
        flask_thread.start()
        
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)
        
        while True:
            time.sleep(1)
            if self.system_stop:
                break

        llmc_thread.join()
        asyncio_thread.join()
        self.llm_controller.stop_robot()

        # clean self.cache_folder
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')
    parser.add_argument('--crazyflie', action='store_true')

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    if args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR
    elif args.crazyflie:
        robot_type = RobotType.CRAZYFLIE
    typefly = TypeFly(robot_type, use_http=args.use_http)
    typefly.run()