import json
import queue
import random
import sys, os
import asyncio
import io, time
from fastapi.responses import JSONResponse
import gradio as gr
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
from threading import Thread
import argparse
from shapely import Polygon
import websockets
import sounddevice as sd
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt
import base64
from collections import deque
import wave
from openai import OpenAI
from queue import Empty as _QEmpty
import mpld3
from PIL import Image



PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(PARENT_DIR)
from controller.utils.constants import FLYZONE_TXT_PATH, FLYZONE_USER_IMAGE_PATH
from controller.context_map.graph_manager import GraphManager
from controller.llm.llm_controller import LLMController
from controller.utils.general_utils import print_t
from controller.llm.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenAI Realtime API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio configuration
SAMPLE_RATE = 24000  # OpenAI Realtime API expects 24kHz
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = np.int16

class AudioHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.recording = False
        self.playing = False
        self.audio_buffer = deque()
        self.recording_buffer = bytearray()
        
    def audio_input_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio input status: {status}")
        if self.recording:
            # Convert to int16 and add to queue
            audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            self.input_queue.put(audio_bytes)
            self.recording_buffer.extend(audio_bytes)
    
    def audio_output_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        if status:
            print(f"Audio output status: {status}")
        
        try:
            # Try to get audio data from buffer
            if self.audio_buffer:
                data = self.audio_buffer.popleft()
                # Convert bytes back to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                # print(f"🔊 Playing {len(audio_array)} samples")
                # Ensure we have the right amount of data
                if len(audio_array) >= frames:
                    outdata[:, 0] = (audio_array[:frames] / 32767.0).astype(np.float32)
                    # Put remaining data back if any
                    if len(audio_array) > frames:
                        remaining = audio_array[frames:].tobytes()
                        self.audio_buffer.appendleft(remaining)
                else:
                    # Pad with zeros if not enough data
                    padded = np.zeros(frames, dtype=np.float32)
                    padded[:len(audio_array)] = (audio_array / 32767.0).astype(np.float32)
                    outdata[:, 0] = padded
            else:
                # No audio data available, output silence
                outdata.fill(0)
        except queue.Empty:
            outdata.fill(0)
    
    def start_recording(self):
        """Start recording audio"""
        self.recording_buffer.clear()
        self.recording = True
        print("[DEBUG] Recording started — buffer cleared.")
        return "🎤 Recording started - speak your command..."
    
    def stop_recording(self):
        """Stop recording audio"""
        try:
            self.recording = False
            if getattr(self.voice_agent, "input_stream", None):
                self.voice_agent.input_stream.stop()
                self.voice_agent.input_stream.close()
                self.voice_agent.input_stream = None
            return "🎤 Recording stopped - processing..."
        except Exception as e:
            return f"❌ Could not stop recording: {e}"
        
    def add_audio_output(self, audio_bytes):
        """Add audio data to output buffer"""
        self.audio_buffer.append(audio_bytes)

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
            for model in ("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"):
                try:
                    bio.seek(0)  # reset buffer 
                    resp = self._client.audio.transcriptions.create(
                        model=model,
                        file=bio,
                        **({"language": language} if language else {})
                    )
                    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
                    if text and text.strip():
                        return text.strip()
                except Exception as e:
                    last_err = e
                    continue
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
        self.ws = None
        self.voice_active = False
        self.current_transcript = ""
        self.ai_response = ""
        
    async def send_audio_data(self):
        """Send audio data from input queue to websocket"""
        while self.voice_active and self.ws:
            try:
                if not self.audio_handler.input_queue.empty():
                    audio_data = self.audio_handler.input_queue.get_nowait()
                    # Encode audio data as base64
                    audio_b64 = base64.b64encode(audio_data).decode()
                    
                    await self.ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error sending audio: {e}")

    async def start_voice_session(self):
        """Start a voice session with OpenAI Realtime API"""
        if not OPENAI_API_KEY:
            return "Error: OPENAI_API_KEY not set"
        
        try:
            self.ws = await websockets.connect(
                REALTIME_URL,
                additional_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            
            # Send session configuration
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": (
                        "You are an interactive tutor for the TypeFly drone system. "
                        "Guide the user step by step: explain what you are doing, "
                        "wait for their response before continuing, and confirm their input. "
                        "Speak clearly and concisely. Use audio and text so the user both hears "
                        "and sees the tutorial. Begin by introducing yourself and explaining how "
                        "to give a simple drone command."
                    ),
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    },
                    "tools": [],
                    "tool_choice": "auto",
                    "temperature": 0.8,
                    "max_response_output_tokens": "inf"
                }
            }))
            
            self.voice_active = True
            
            # Start audio sending task
            audio_task = asyncio.create_task(self.send_audio_data())
            
            try:
                # Listen for responses
                async for msg in self.ws:
                    if not self.voice_active:
                        break
                        
                    try:
                        event = json.loads(msg)
                        event_type = event.get('type', 'unknown')
                        
                        if event_type == 'session.created':
                            print("✅ Voice session created")
                            
                        elif event_type == 'session.updated':
                            print("✅ Voice session configured")
                            
                        elif event_type == 'input_audio_buffer.speech_started':
                            print("🗣️ Speech detected")
                            
                        elif event_type == 'input_audio_buffer.speech_stopped':
                            print("🤫 Speech ended")
                            
                        elif event_type == 'conversation.item.input_audio_transcription.completed':
                            transcript = event.get('transcript', '')
                            print(f"📝 Voice command: {transcript}")
                            self.current_transcript = transcript
                            
                            # Process the voice command through TypeFly
                            if transcript.strip():
                                # Execute the command in TypeFly system
                                await self.execute_typefly_command(transcript)
                            
                        elif event_type == 'response.audio.delta':
                            # Stream audio output
                            audio_b64 = event.get('delta', '')
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                self.audio_handler.add_audio_output(audio_bytes)
                                
                        elif event_type == 'response.text.delta':
                            # Collect AI response text
                            delta = event.get('delta', '')
                            self.ai_response += delta
                            
                        elif event_type == 'response.text.done':
                            print(f"🤖 AI: {self.ai_response}")
                            self.ai_response = ""
                    
                        elif event_type == "input_audio_buffer.committed":
                            print("✅ Server committed audio buffer:", event.get("item_id"))
                                
                        elif event_type == 'error':
                            print(f"❌ Voice Error: {event}")
                        # else:
                        #     print("ℹ️ Unhandled event:", event_type, event)
                            
                    except json.JSONDecodeError:
                        print(f"Failed to parse voice message: {msg}")
                        
            except Exception as e:
                print(f"Voice session error: {e}")
            finally:
                audio_task.cancel()
                
        except Exception as e:
            return f"Voice session failed: {e}"
    
    async def execute_typefly_command(self, command):
        """Execute a voice command in the TypeFly system"""
        try:
            # Put the command into TypeFly's processing system
            # This simulates typing the command in the chat interface
            print(f"🎯 Executing voice command: {command}")
            
            # You can integrate this with TypeFly's message processing
            # For now, we'll just print it, but you could call:
            # self.typefly.process_message(command, [])
            
        except Exception as e:
            print(f"Error executing command: {e}")

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
            
            self.output_stream = sd.OutputStream(
                callback=self.audio_handler.audio_output_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype=np.float32
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            return "Audio streams started successfully"
            
        except Exception as e:
            return f"Failed to start audio streams: {e}"

    def stop_audio_streams(self):
        """Stop audio streams"""
        try:
            if hasattr(self, 'input_stream'):
                self.input_stream.stop()
                self.input_stream.close()
            if hasattr(self, 'output_stream'):
                self.output_stream.stop()
                self.output_stream.close()
            return "Audio streams stopped"
        except Exception as e:
            return f"Error stopping audio streams: {e}"

    async def stop_voice_session(self):
        """Stop the voice session"""
        self.voice_active = False
        if self.ws:
            await self.ws.close()
        return "Voice session stopped"

class TypeFly:
    def __init__(self, robot_type, use_http=False):
         # create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.message_queue = queue.Queue()
        self.message_queue.put(self.cache_folder)
        self.user_answer_queue = queue.Queue()
        self.user_question_answer = [] # list of last question-answer pair between LLM and user
        self.llm_controller = LLMController(robot_type, use_http, self.message_queue, self.user_answer_queue)

        # Initialize GraphManager with the sample data
        self.graph_manager = GraphManager(
            llm_controller=self.llm_controller, 
        )
        # self.graph_manager = GraphManager(llm_controller=self.llm_controller, init_graph_json=)
        self.llm_controller.set_graph_manager(self.graph_manager)
        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False
        
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
                
                with gr.TabItem("📊 Graph Visualization"):
                    self.setup_graph_tab()
    

    def change_username(self):
        return gr.update(
            value=f"### 👤 Pilot: {self.llm_controller.get_username()}"
        )

    def transcribe_and_send(self):
        """Transcribe last recording and run it through the same controller pipeline."""
        try:
            # Stop mic while packaging to avoid races
            try:
                if getattr(self.voice_agent, "input_stream", None):
                    self.voice_agent.input_stream.stop()
                    self.voice_agent.input_stream.close()
                    self.voice_agent.input_stream = None
            except Exception as _:
                pass

            wav_bytes = self.voice_agent.audio_handler.get_wav_bytes()

            # Robust empty/short take detection (44 bytes = header only)
            if len(wav_bytes) <= 44:
                return "⚠️ No audio captured. Click Start, speak, Stop, then Transcribe.", ""
            
            if not wav_bytes:
                print("[DEBUG] No WAV bytes returned.")
                return "⚠️ No audio captured. Record first.", ""

            # Inspect audio
            import numpy as _np
            header = 44
            pcm = _np.frombuffer(wav_bytes[header:], dtype=_np.int16)
            seconds = (pcm.size) / float(SAMPLE_RATE)
            rms = float(_np.sqrt(_np.mean((_np.asarray(pcm, _np.float32) ** 2)))) if pcm.size else 0.0
            dbfs = 20.0 * _np.log10(max(rms / 32768.0, 1e-9))
            print(f"[DEBUG] WAV size={len(wav_bytes)} bytes | samples={pcm.size} | dur={seconds:.2f}s | RMS={rms:.1f} | dBFS={dbfs:.1f}")

            if seconds < 0.25:
                print("[DEBUG] Too short — likely clicked too fast or callback not firing.")
            if dbfs < -45.0:
                print("[DEBUG] Very quiet (near silence). Check mic device/permissions/input gain.")

        except Exception as e:
            print(f"[DEBUG] Error while reading WAV: {e}")
            return f"❌ Could not read audio: {e}", ""

        try:
            # Transcribe
            print("[DEBUG] Sending audio to STT...")
            transcript = self.transcriber.transcribe(wav_bytes)
            print(f"[DEBUG] Transcript: {transcript!r}")

            if not transcript.strip():
                print("[DEBUG] Empty transcript returned.")
                return "⚠️ Transcription produced empty text.", ""

            # Clear buffer so next take is fresh
            self.voice_agent.audio_handler.recording_buffer.clear()
            print("[DEBUG] Cleared recording buffer after transcription.")

            # Kick off the same pipeline as typed chat
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(transcript,))
            task_thread.start()
            print("[DEBUG] Started llm_controller thread.")

            complete_response = f"📝 {transcript}\n"
            deadline = time.time() + 60  # hard stop to avoid hanging the event
            while True:
                try:
                    msg = self.message_queue.get(timeout=0.5)
                except _QEmpty:
                    if time.time() > deadline:
                        yield complete_response + "\n⏱️ Timed out waiting for completion.", transcript
                        return
                    continue

                if isinstance(msg, tuple):
                    pass
                elif isinstance(msg, str):
                    if msg == 'end':
                        print("[DEBUG] Got 'end' from message_queue.")
                        return "✅ Command Complete!", transcript
                    if msg.startswith('[LOG]') or msg.startswith('[Q]'):
                        complete_response += '\n'
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                    yield complete_response, transcript

        except Exception as e:
            print(f"[DEBUG] Error during transcription or execution: {e}")
            return f"❌ STT or execution error: {e}", ""

        
    def start_voice_control(self):
        """Start voice control system"""
        try:
            # Start audio streams
            status = self.voice_agent.start_audio_streams()
            self.voice_agent.audio_handler.recording = True  
            # Start voice session in background
            asyncio.run_coroutine_threadsafe(
                self.voice_agent.start_voice_session(), self.asyncio_loop
            )
                
            return "🎤 Voice control started! You can now speak commands."
        except Exception as e:
            return f"❌ Failed to start voice control: {e}"

    def stop_voice_control(self):
        """Stop voice control system"""
        try:
            # Stop voice session
            asyncio.run_coroutine_threadsafe(
                self.voice_agent.stop_voice_session(), self.asyncio_loop
            )
                
            # Stop audio streams
            status = self.voice_agent.stop_audio_streams()
            
            return "⏹️ Voice control stopped."
        except Exception as e:
            return f"❌ Error stopping voice control: {e}"

    def start_recording(self):
        """Start recording audio"""
        try:
            # Stop/close any prior stream
            try:
                if hasattr(self.voice_agent, 'input_stream') and self.voice_agent.input_stream:
                    self.voice_agent.input_stream.stop()
                    self.voice_agent.input_stream.close()
            except Exception as e:
                print(f"[DEBUG] Could not stop previous InputStream: {e}")

            # Fresh stream
            self.voice_agent.input_stream = sd.InputStream(
                callback=self.voice_agent.audio_handler.audio_input_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype=np.float32
            )
            self.voice_agent.input_stream.start()

            # Fresh buffer and flag
            self.voice_agent.audio_handler.recording_buffer.clear()
            self.voice_agent.audio_handler.recording = True
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
            gr.Markdown("Visualize the evolution of detected objects and their relationships over time.")
            
            # Graph visualization iframe
            graph_html = self.get_graph_html()
            self.graph_component = gr.HTML(graph_html)
            
            # Controls
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Graph", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Graph Data", variant="secondary")
                auto_refresh_btn = gr.Button("🔁 Auto Refresh", variant="secondary")
                
            # Status
            self.graph_status = gr.Textbox(
                label="Graph Status", 
                value="Ready to display graph data...",
                interactive=False
            )
            
            # Event handlers
            refresh_btn.click(self.refresh_graph, outputs=[self.graph_component, self.graph_status])
            clear_btn.click(self.clear_graph_data, outputs=[self.graph_status])
            auto_refresh_btn.click(self.toggle_auto_refresh, outputs=[self.graph_status])

    def get_graph_html(self):
        """Generate the HTML for the graph visualization"""
        return """
        <div style="width: 100%; height: 750px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
            <iframe src="http://localhost:50000/graph" 
                    style="width: 100%; height: 100%; border: none;"
                    sandbox="allow-scripts allow-same-origin">
            </iframe>
        </div>
        """
    
    def refresh_graph(self):
        """Refresh the graph visualization"""
        try:
            # Get current graph state from graph manager
            current_graph = self.graph_manager.get_graph()
            if current_graph:
                status = "Graph refreshed! Displaying current graph state."
            else:
                status = "No graph data available. Start robot operations to generate data."
            
            return self.get_graph_html(), status
        except Exception as e:
            return self.get_graph_html(), f"Error refreshing graph: {str(e)}"

    def clear_graph_data(self):
        """Clear the graph log data"""
        try:
            if os.path.exists(self.graph_log_path):
                os.remove(self.graph_log_path)
                return "Graph data cleared successfully!"
            else:
                return "No graph data to clear."
        except Exception as e:
            return f"Error clearing graph data: {str(e)}"

    def toggle_auto_refresh(self):
        """Toggle auto-refresh mode"""
        return "Auto-refresh toggled! Graph will update automatically as new data arrives."
    
    def checkbox_llama3(self):
        self.use_llama3 = not self.use_llama3
        if self.use_llama3:
            print_t(f"Switch to llama3")
            self.llm_controller.planner.set_model(LLAMA3)
        else:
            print_t(f"Switch to gpt4")
            self.llm_controller.planner.set_model(GPT4)

    def process_message(self, message, history, img_b64):
        print("Message to be processed is: " + message)
        """Process message and yield responses"""
        print_t(f"[S] Receiving task description: {message}")
        
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
            return
        
        if len(message) == 0:
            yield "[WARNING] Empty command!"
            return
        
        # Handle user answers to LLM questions
        if len(self.user_question_answer) == 1:
            print_t(f"[DEBUG] Treating as answer to: {self.user_question_answer[0]}")
            self.user_question_answer.append(message)
            temp = self.user_question_answer.copy()

            self.user_question_answer = []

            # Now put the answer - background thread can continue
            self.user_answer_queue.put(temp)

            complete_response = ""

            deadline = time.time() + 60
            while True:
                try:
                    msg = self.message_queue.get(timeout=0.5)
                except _QEmpty:
                    # keep UI alive; optionally show a spinner every few seconds
                    if time.time() > deadline:
                        yield complete_response + "\n⏳ Still working..."
                        deadline = time.time() + 60
                    continue
                    
                if isinstance(msg, str):
                    if msg == 'end':
                        # Clear the question state before yielding final response
                        self.user_question_answer = []
                        yield complete_response + "\nCommand Complete!"
                        return
                    
                    if msg.startswith('[LOG]') or msg.startswith('[Q]'):
                        complete_response += '\n'
                        
                    if msg.startswith('[Q]'):
                        # New question arrived - store it for next user input
                        question_text = msg[3:].strip()  # Remove [Q] prefix
                        self.user_question_answer = [msg]
                        # Add the question to the response
                        if msg.endswith('\\\\'):
                            complete_response += msg.rstrip('\\\\')
                        else:
                            complete_response += msg + '\n'
                        yield complete_response
                        return  # Exit and wait for user's answer

                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                    yield complete_response
                    
        else:
            # New task execution
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message, img_b64,))
            task_thread.start()
            complete_response = ''
            
            deadline = time.time() + 60
            while True:
                try:
                    msg = self.message_queue.get(timeout=0.5)
                except _QEmpty:
                    if time.time() > deadline:
                        yield complete_response + "\n⏳ Still processing..."
                        deadline = time.time() + 60
                    continue

                if isinstance(msg, str):
                    if msg == 'end':
                        self.user_question_answer = []
                        yield complete_response + "\nCommand Complete!"
                        return

                    # Append a newline *before* log/Q lines for readability
                    if msg.startswith('[LOG]') or msg.startswith('[Q]'):
                        complete_response += '\n'

                    if msg.startswith('[Q]'):
                        # Store the question so the next user message is treated as the answer
                        self.user_question_answer = [msg]

                        # Show the question to the user
                        if msg.endswith('\\\\'):
                            complete_response += msg.rstrip('\\\\')
                        else:
                            complete_response += msg + '\n'
                        yield complete_response

                        return

                    # Normal streaming line
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                    yield complete_response


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


    def setup_graph_server(self, app):
        """Setup graph visualization endpoints (matplotlib style)"""

        from shapely import Polygon
        import plotly.graph_objects as go

        @app.route('/graph')
        def graph_page():
            """Render the coordinate-based graph visualization aligned with flyzone and auto-offset objects (interactive Plotly version)."""
            try:
                graph_json_str = self.graph_manager.get_graph()
                if not graph_json_str:
                    return "<h3>No graph data available</h3>"

                graph_data: dict = json.loads(graph_json_str)

                # --- Helper to safely parse XY ---
                def parse_xy(v):
                    if isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except Exception:
                            return (0.0, 0.0)
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        return float(v[0]), float(v[1])
                    return (0.0, 0.0)

                regions = {r["name"]: parse_xy(r["coords"]) for r in graph_data.get("regions", [])}
                objects = {r["name"]: parse_xy(r["coords"]) for r in graph_data.get("objects", [])}
                region_connections = graph_data.get("region_connections", [])
                object_connections = graph_data.get("object_connections", [])
                current_pos = graph_data.get("current_pos", {})

                # --- AUTO OFFSET for clustered (0,0) objects ---
                if objects and all((x == 0.0 and y == 0.0) for (_, (x, y)) in objects.items()):
                    region_lookup = {reg: obj for reg, obj in graph_data.get("object_connections", []) if reg in regions}
                    for name in objects:
                        if name in region_lookup:
                            rx, ry = regions[region_lookup[name]]
                        else:
                            rx, ry = 0, 0
                        angle = random.uniform(0, 2 * np.pi)
                        dist = random.uniform(10, 40)
                        objects[name] = (rx + dist * np.cos(angle), ry + dist * np.sin(angle))

                # --- Build interactive Plotly figure ---
                fig = go.Figure()

                # 🟧 Flyzones (background polygons)
                try:
                    flyzones = self.llm_controller.middle_layer.get_flyzone_polygon()
                    for idx, poly in enumerate(flyzones):
                        if isinstance(poly, Polygon):
                            x, y = poly.exterior.xy
                            x, y = list(x), list(y)

                            fig.add_trace(go.Scatter(
                                x=x, y=y,
                                fill='toself',
                                fillcolor='rgba(255,165,0,0.1)',
                                line=dict(color='orange', width=2),
                                name=f"Flyzone {idx+1}",
                                hoverinfo='skip'
                            ))
                except Exception as e:
                    print(f"[WARN] Could not plot flyzone: {e}")

                # ⚫ Region–Region links
                for r1, r2 in region_connections:
                    if r1 in regions and r2 in regions:
                        x1, y1 = regions[r1]
                        x2, y2 = regions[r2]
                        fig.add_trace(go.Scatter(
                            x=[x1, x2], y=[y1, y2],
                            mode='lines',
                            line=dict(color='gray', dash='dot'),
                            opacity=0.5,
                            hoverinfo='none',
                            showlegend=False
                        ))

                # 🔵 Object–Region links
                for a, b in object_connections:
                    if a in regions and b in objects:
                        x1, y1 = regions[a]
                        x2, y2 = objects[b]
                    elif a in objects and b in regions:
                        x1, y1 = objects[a]
                        x2, y2 = regions[b]
                    else:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[x1, x2], y=[y1, y2],
                        mode='lines',
                        line=dict(color='lightblue', dash='dot'),
                        opacity=0.7,
                        hoverinfo='none',
                        showlegend=False
                    ))

                # 🔷 Regions (blue circles)
                if regions:
                    fig.add_trace(go.Scatter(
                        x=[x for x, _ in regions.values()],
                        y=[y for _, y in regions.values()],
                        mode='markers+text',
                        text=[n for n in regions.keys()],
                        textposition='top center',
                        marker=dict(size=14, color='skyblue', line=dict(color='black', width=1)),
                        name='Regions'
                    ))

                # 🟢 Objects (green circles)
                if objects:
                    fig.add_trace(go.Scatter(
                        x=[x for x, _ in objects.values()],
                        y=[y for _, y in objects.values()],
                        mode='markers+text',
                        text=[n for n in objects.keys()],
                        textposition='top center',
                        marker=dict(size=9, color='lightgreen', line=dict(color='black', width=1)),
                        name='Objects'
                    ))

                # 🔴 Current position ("you")
                if "coords" in current_pos and current_pos["coords"]:
                    x, y, *_ = current_pos["coords"]
                    region = current_pos.get("region", "unknown")
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        text=[f"you ({region})"],
                        textposition='top right',
                        marker=dict(symbol='x', size=12, color='red', line=dict(color='black', width=1)),
                        name='You'
                    ))

                # --- Layout and style ---
                fig.update_layout(
                    title="📈 Coordinate Graph View",
                    xaxis_title="X Coordinates",
                    yaxis_title="Y Coordinates",
                    template="plotly_white",
                    height=750,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_layout(dragmode='pan')  # default: panning mode

                # Return self-contained HTML (with zoom/pan controls)
                return fig.to_html(include_plotlyjs='cdn', full_html=True)

            except Exception as e:
                return f"<h3>Error rendering graph: {e}</h3>"

        
        @app.route('/graph-data')
        def graph_data():
            """Serve the current graph data as JSON"""
            try:
                graph_json_str = self.graph_manager.get_graph()
                
                if not graph_json_str:
                    return jsonify({
                        "success": False, 
                        "data": None, 
                        "message": "No graph data available - start robot operations to generate data"
                    })

                try:
                    graph_dict = json.loads(graph_json_str)
                except json.JSONDecodeError as e:
                    return jsonify({
                        "success": False, 
                        "data": None, 
                        "message": f"Invalid graph JSON format: {str(e)}"
                    })

                return jsonify({
                    "success": True, 
                    "data": graph_json_str,
                    "message": "Current graph state retrieved successfully"
                })

            except Exception as e:
                return jsonify({
                    "success": False, 
                    "data": None, 
                    "message": f"Error retrieving graph: {str(e)}"
                })
            
        @app.route('/flyzone-image')
        def flyzone_image():
            """Serve the latest flyzone background image (PNG)."""
            path = os.path.abspath("controller/assets/tello/flyzone/flyzone_plot.png")
            if not os.path.exists(path):
                return jsonify({"success": False, "message": "Flyzone image not found"}), 404
            resp = send_file(path, mimetype='image/png')
            # prevent caching so updates show immediately
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
     
    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.start()

        app = Flask(__name__)
        CORS(app)  # allow all origins

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