import json
import queue
import sys, os
import asyncio
import io, time
from fastapi.responses import JSONResponse
import gradio as gr
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
from threading import Thread
import argparse
import websockets
import sounddevice as sd
import pyttsx3
import numpy as np
import base64
from collections import deque
import wave
from openai import OpenAI
from queue import Empty as _QEmpty


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(PARENT_DIR)
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

        # Convert the JSON to a string
        init_graph_path = "controller/assets/tello/memory/graph.txt"

        # Initialize GraphManager with the sample data
        self.graph_manager = GraphManager(
            llm_controller=self.llm_controller, 
            init_graph_json=init_graph_path,
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
                    gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
                    
                    # Create custom chat interface with integrated audio controls
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

                    
                def send_message(message, history):
                    """Send message and get response - generator for streaming"""
                    if not message.strip():
                        yield history, ""
                        return
                    
                    # Add user message to chat
                    history = history + [[message, None]]
                    yield history, ""  # Clear input immediately and show user message
                    
                    # Process message and stream response
                    for partial_response in self.process_message(message, history):
                        if partial_response:
                            # Extract just the response text (remove "Command Complete!" etc)
                            clean_response = partial_response.split("\nCommand Complete!")[0]
                            # Update the last message's response
                            history[-1][1] = clean_response

                            if clean_response.strip().startswith("[Q]"):
                                yield history, ""   # show question
                                return              # exit stream to accept next user input
                            
                            yield history, ""

                # Wire up events - CORRECTED VERSION
                send_btn.click(
                    send_message,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )

                msg_input.submit(
                    send_message,  # Use the same function, not a wrapper
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )
                
                with gr.TabItem("📊 Graph Visualization"):
                    self.setup_graph_tab()
                
                with gr.TabItem("⚙️ Settings"):
                    self.setup_settings_tab()

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

    def setup_settings_tab(self):
        """Setup the Settings tab with flyzone generation and feedback."""
        with gr.Column():
            gr.Markdown("## 🌐 User Settings")
            gr.Markdown("Personalize your experience by adjusting options to suit your preferences.")

            # --- User name section ---
            username = gr.Textbox(
                label="Enter your name",
                placeholder="Type your name here...",
                lines=1,
                value="Christian",
            )
            
            # Add event handler to update username when changed
            def update_username(new_name):
                self.llm_controller.set_username(new_name)
                return f"✅ Username updated to: {new_name}"
            
            username_status = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                visible=False
            )
            
            # Connect the change event
            username.change(
                fn=update_username,
                inputs=username,
                outputs=username_status
            )
            
            # Set initial username
            self.llm_controller.set_username(username.value)
            
            # --- Voice Settings ---
            gr.Markdown("### 🎤 Voice Settings")
            with gr.Row():
                audio_devices_btn = gr.Button("🔍 Show Audio Devices")
                test_audio_btn = gr.Button("🔊 Test Audio")
            
            audio_info = gr.Textbox(
                label="Audio Device Info",
                value="Click 'Show Audio Devices' to see available devices",
                interactive=False,
                lines=5
            )
            
            # --- Flyzone generation section ---
            gr.Markdown("### ✈️ Generate Flyzone")
            flyzone_prompt = gr.Textbox(
                label="Enter prompt to generate a flyzone",
                placeholder="Describe the area and shape of the flyzone...",
                lines=2
            )
            generate_btn = gr.Button("🚀 Generate Flyzone")

            # --- Feedback and result section ---
            status_output = gr.Label(label="Request Status")
            flyzone_image = gr.Image(
                label="Generated Flyzone Preview",
                type="filepath",
                visible=True
            )

            # Absolute path for flyzone image
            flyzone_img_path = os.path.abspath("controller/assets/tello/flyzone/flyzone_plot.png")

            # --- Handler functions ---
            def show_audio_devices():
                try:
                    devices = sd.query_devices()
                    device_info = "Available Audio Devices:\n\n"
                    for i, device in enumerate(devices):
                        device_info += f"Device {i}: {device['name']}\n"
                        device_info += f"  Max Input Channels: {device['max_input_channels']}\n"
                        device_info += f"  Max Output Channels: {device['max_output_channels']}\n"
                        device_info += f"  Default Sample Rate: {device['default_samplerate']}\n\n"
                    return device_info
                except Exception as e:
                    return f"Error getting audio devices: {e}"

            def test_audio():
                try:
                    # Generate a test tone
                    duration = 1.0  # seconds
                    frequency = 440  # Hz (A note)
                    sample_rate = 44100
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    test_tone = np.sin(frequency * 2 * np.pi * t) * 0.3
                    
                    sd.play(test_tone, sample_rate)
                    sd.wait()  # Wait until the sound is finished
                    
                    return "✅ Audio test completed - you should have heard a tone"
                except Exception as e:
                    return f"❌ Audio test failed: {e}"

            def handle_flyzone_request(instruction: str):
                yield "⏳ Generating flyzone, please wait...", None

                try:
                    # Call controller to request flyzone
                    self.llm_controller.get_flyzone_manager().request_new_flyzone(instruction=instruction)

                    # Show image if it exists
                    if os.path.exists(flyzone_img_path):
                        yield "✅ Flyzone generated successfully!", flyzone_img_path
                    else:
                        yield "⚠️ Flyzone generated, but no image found.", None

                except Exception as e:
                    yield f"❌ Failed to generate flyzone: {str(e)}", None

            # --- Connect UI events ---
            audio_devices_btn.click(show_audio_devices, outputs=[audio_info])
            test_audio_btn.click(test_audio, outputs=[audio_info])
            
            generate_btn.click(
                fn=handle_flyzone_request,
                inputs=flyzone_prompt,
                outputs=[status_output, flyzone_image]
            )

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

    def process_message(self, message, history):
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

            question = self.user_question_answer[0]
            is_feedback_question = "feedback" in question.lower()
            
            self.user_question_answer = []

            # Now put the answer - background thread can continue
            self.user_answer_queue.put(temp)

            if is_feedback_question:
                yield_msg = "Thank you for your feedback. I am elaborating it."
                self.llm_controller.text_to_speech(yield_msg)
                complete_response = yield_msg + "\n"
            else:
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
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message,))
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
        """Setup the graph visualization server endpoint with enhanced pan/zoom"""
        @app.route('/graph')
        def graph_page():
            return '''<!DOCTYPE html>
    <html>
    <head>
        <title>TypeFly Graph Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
        <style>
            body {
                margin: 0;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #f5f5f5;
            }
            #container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }
            #graph {
                width: 100%;
                height: 650px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #fafafa;
                cursor: grab;
                position: relative;
            }
            #graph:active {
                cursor: grabbing;
            }
            .node circle {
                stroke: #fff;
                stroke-width: 3px;
                cursor: pointer;
                transition: r 0.2s;
            }
            .node:hover circle {
                r: 25;
            }
            .node text {
                font-size: 11px;
                pointer-events: none;
                font-weight: 500;
            }
            .link {
                stroke: #999;
                stroke-opacity: 0.4;
                stroke-width: 1.5px;
            }
            #controls {
                margin-bottom: 20px;
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            button {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                background: #007bff;
                color: white;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }
            button:hover {
                background: #0056b3;
            }
            #status {
                margin-left: auto;
                color: #666;
                font-size: 14px;
                font-weight: 500;
            }
            .zoom-controls {
                position: absolute;
                right: 30px;
                top: 20px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                z-index: 1000;
            }
            .zoom-btn {
                width: 36px;
                height: 36px;
                border: 2px solid #ddd;
                background: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .zoom-btn:hover {
                background: #f0f0f0;
                border-color: #007bff;
            }
            .zoom-btn:active {
                transform: scale(0.95);
            }
            .zoom-level {
                text-align: center;
                font-size: 11px;
                color: #666;
                padding: 4px;
                background: white;
                border-radius: 4px;
                border: 2px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend {
                margin-top: 15px;
                display: flex;
                gap: 20px;
                font-size: 13px;
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .legend-color {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                border: 3px solid #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            }
            .stats {
                margin-top: 15px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 13px;
                color: #666;
            }
            .hint {
                margin-top: 10px;
                padding: 8px 12px;
                background: #e3f2fd;
                border-left: 3px solid #2196F3;
                border-radius: 4px;
                font-size: 12px;
                color: #1976d2;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <h2>🌐 Object Detection Graph</h2>
            <div id="controls">
                <button onclick="loadGraph()">🔄 Refresh</button>
                <button onclick="resetZoom()">🔍 Reset View</button>
                <button onclick="centerGraph()">📍 Center</button>
                <button onclick="fitToScreen()">⛶ Fit All</button>
                <button onclick="toggleFlyzone()">🗺️ Flyzone</button>
                <span id="status">Loading graph data...</span>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50"></div>
                    <span>Regions</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3"></div>
                    <span>Objects</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800"></div>
                    <span>Current Position</span>
                </div>
            </div>
            <div class="hint">
                💡 <strong>Navigation:</strong> Drag to pan • Scroll to zoom • Double-click to center on node • Use +/- buttons for precise zoom control
            </div>
            <div class="stats" id="stats"></div>
            <div style="position: relative;">
                <svg id="graph"></svg>
                <div class="zoom-controls">
                    <div class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</div>
                    <div class="zoom-level" id="zoom-level">100%</div>
                    <div class="zoom-btn" onclick="zoomOut()" title="Zoom Out">−</div>
                    <div class="zoom-btn" onclick="resetZoom()" title="Reset Zoom">⟲</div>
                </div>
            </div>
        </div>

        <script>
            function computeSize() {
                const graphEl = document.getElementById('graph');
                const containerEl = document.getElementById('container');
                // try svg width first, then container, then window; never allow <= 0
                let w = graphEl?.clientWidth || containerEl?.clientWidth || (window.innerWidth - 80) || 0;
                w = Math.max(700, Math.min(1360, w));  // keep it reasonable and > 0
                return { width: w, height: 650 };
            }

            let { width, height } = computeSize();
            let currentZoom = 1;
            // --- coordinate mapping config ---
            const DEFAULT_WORLD = { width: 1000, height: 1000 }; // fallback flyzone extents
            const ORIGIN_TOP_LEFT = true; // set false if your coords are bottom-left origin

            let showFlyzone = true;          // toggle visibility
            let flyzoneOpacity = 0.28;       // background transparency
            let lastGraphObj = null;         // keep parsed data for re-render


            const svg = d3.select("#graph")
                // make the SVG responsive but keep internal coords stable
                .attr("viewBox", `0 0 ${width} ${height}`)
                .attr("preserveAspectRatio", "xMidYMid meet");

            // keep the event-capturing rect in sync with the logical size
            const rect = svg.append("rect")
                .attr("width", width)
                .attr("height", height)
                .style("fill", "none")
                .style("pointer-events", "all");

            const g = svg.append("g");

            // Enhanced zoom with smoother transitions
            const zoom = d3.zoom()
                .scaleExtent([0.1, 8])  // Allow more zoom range
                .filter((event) => {
                    // Allow zoom on background, prevent on nodes
                    return !event.button && event.target === rect.node();
                })
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                    currentZoom = event.transform.k;
                    updateZoomLevel();
                });

            rect.call(zoom);
            
            // Also enable wheel zoom on entire SVG
            svg.on("wheel", (event) => {
                event.preventDefault();
                const transform = d3.zoomTransform(svg.node());
                const k = transform.k * (event.deltaY < 0 ? 1.2 : 0.8);
                const constrained_k = Math.max(0.1, Math.min(8, k));
                
                svg.transition()
                    .duration(100)
                    .call(zoom.transform, transform.scale(constrained_k / transform.k));
            });

            
            function addFlyzoneBackground() {
                if (!showFlyzone) return;
                // insert as FIRST child of g so it sits under links/nodes and moves with zoom/pan
                g.insert("image", ":first-child")
                .attr("class", "flyzone-img")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", width)
                .attr("height", height)
                .attr("preserveAspectRatio", "none")
                .attr("opacity", flyzoneOpacity)
                // cache-bust so updated image shows when regenerated
                .attr("href", `/flyzone-image?ts=${Date.now()}`);
            }

            function toggleFlyzone() {
                showFlyzone = !showFlyzone;
                if (lastGraphObj) renderGraph(lastGraphObj);
            }

            // Enable panning with mouse drag (already enabled by zoom behavior)
            // Enable zoom with mouse wheel (already enabled by zoom behavior)

            function updateZoomLevel() {
                document.getElementById('zoom-level').textContent = Math.round(currentZoom * 100) + '%';
            }

            function zoomIn() {
                const transform = d3.zoomTransform(svg.node());
                rect.transition()
                    .duration(300)
                    .call(zoom.transform, transform.scale(1.3));
            }

            function zoomOut() {
                const transform = d3.zoomTransform(svg.node());
                rect.transition()
                    .duration(300)
                    .call(zoom.transform, transform.scale(0.77));
            }

            function resetZoom() {
                rect.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity);
            }

            window.addEventListener('resize', () => {
                const size = computeSize();
                width = size.width;
                height = size.height;
                svg.attr("viewBox", `0 0 ${width} ${height}`);
                rect.attr("width", width).attr("height", height);
                d3.select(".flyzone-img")
                .attr("width", width)
                .attr("height", height);
            });


            function centerGraph() {
                const bounds = g.node().getBBox();
                const fullWidth = bounds.width;
                const fullHeight = bounds.height;
                const midX = bounds.x + fullWidth / 2;
                const midY = bounds.y + fullHeight / 2;

                const scale = 0.9 / Math.max(fullWidth / width, fullHeight / height);
                const translate = [width / 2 - scale * midX, height / 2 - scale * midY];

                rect.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            }

            function fitToScreen() {
                const bounds = g.node().getBBox();
                const fullWidth = bounds.width;
                const fullHeight = bounds.height;
                const midX = bounds.x + fullWidth / 2;
                const midY = bounds.y + fullHeight / 2;

                // Calculate scale to fit everything with padding
                const scale = 0.85 / Math.max(fullWidth / width, fullHeight / height);
                const translate = [width / 2 - scale * midX, height / 2 - scale * midY];

                rect.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            }

            function loadGraph() {
                document.getElementById('status').textContent = 'Loading...';
                
                fetch('/graph-data')
                    .then(response => response.json())
                    .then(result => {
                        if (result.success && result.data) {
                            lastGraphData = result.data;
                            const graphData = JSON.parse(result.data);
                            lastGraphObj = graphData;
                            renderGraph(graphData);
                            
                            const nodeCount = (graphData.objects?.length || 0) + (graphData.regions?.length || 0);
                            const edgeCount = graphData.object_connections?.length || 0;
                            document.getElementById('status').textContent = 
                                `✅ Loaded: ${nodeCount} nodes, ${edgeCount} connections`;
                            
                            document.getElementById('stats').innerHTML = 
                                `<strong>Objects:</strong> ${graphData.objects?.length || 0} | ` +
                                `<strong>Regions:</strong> ${graphData.regions?.length || 0} | ` +
                                `<strong>Current Region:</strong> ${graphData.current_position?.region || 'unknown'}`;
                        } else {
                            document.getElementById('status').textContent = 
                                '⚠️ ' + (result.message || 'No data available');
                            renderEmptyState();
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('status').textContent = '❌ Failed to load';
                        renderEmptyState();
                    });
            }

            function renderEmptyState() {
                g.selectAll("*").remove();
                g.append("text")
                    .attr("x", width / 2)
                    .attr("y", height / 2)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#999")
                    .style("font-size", "16px")
                    .text("No graph data available. Start robot operations to generate data.");
            }

            function renderGraph(graphData) {
                g.selectAll("*").remove();

                // add flyzone image first so it's behind the links/nodes and participates in zoom/pan
                addFlyzoneBackground && addFlyzoneBackground();
                const nodes = [];
                const links = [];
                const nodeMap = new Map();

                // --- determine world extents for scaling ---
                const allCoords = [];
                (graphData.regions || []).forEach(r => Array.isArray(r.coords) && allCoords.push(r.coords));
                (graphData.objects || []).forEach(o => Array.isArray(o.coords) && allCoords.push(o.coords));

                let maxX = Math.max(0, ...allCoords.map(c => +c[0] || 0));
                let maxY = Math.max(0, ...allCoords.map(c => +c[1] || 0));

                if (!isFinite(maxX) || maxX <= 0) maxX = DEFAULT_WORLD.width;
                if (!isFinite(maxY) || maxY <= 0) maxY = DEFAULT_WORLD.height;

                const xScale = d3.scaleLinear().domain([0, maxX]).range([0, width]);
                const yScale = ORIGIN_TOP_LEFT
                    ? d3.scaleLinear().domain([0, maxY]).range([0, height])        // y grows downward
                    : d3.scaleLinear().domain([0, maxY]).range([height, 0]);

                // --- regions (fixed at coords) ---
                if (graphData.regions) {
                    graphData.regions.forEach(region => {
                    const node = {
                        id: region.name,
                        label: region.name,
                        type: 'region',
                        coords: region.coords
                    };
                    // place & fix at world coords if present
                    if (Array.isArray(region.coords) && region.coords.length >= 2) {
                        const [rx, ry] = region.coords;
                        const px = xScale(+rx || 0);
                        const py = yScale(+ry || 0);
                        node.x = px; node.y = py;
                        node.fx = px; node.fy = py;   // <- fixed
                        node.fixed = true;
                    }
                    nodes.push(node);
                    nodeMap.set(region.name, node);
                    });
                }

                // --- objects (not fixed; just seeded) ---
                if (graphData.objects) {
                    graphData.objects.forEach(obj => {
                    const node = {
                        id: obj.name,
                        label: obj.name,
                        type: 'object',
                        coords: obj.coords
                    };
                    if (Array.isArray(obj.coords) && obj.coords.length >= 2) {
                        const [ox, oy] = obj.coords;
                        node.x = xScale(+ox || 0);
                        node.y = yScale(+oy || 0);
                    } else {
                        // fallback: scatter near center
                        const angle = Math.random() * 2 * Math.PI;
                        const radius = Math.random() * Math.min(width, height) / 3;
                        node.x = width / 2 + radius * Math.cos(angle);
                        node.y = height / 2 + radius * Math.sin(angle);
                    }
                    nodes.push(node);
                    nodeMap.set(obj.name, node);
                    });
                }

                // Add current position
                if (graphData.current_position) {
                    const posNode = {
                    id: '__robot_position__',
                    label: 'Robot',
                    type: 'robot',
                    coords: graphData.current_position.coords
                    };
                    if (Array.isArray(posNode.coords) && posNode.coords.length >= 2) {
                    const [rx, ry] = posNode.coords;
                    posNode.x = xScale(+rx || 0);
                    posNode.y = yScale(+ry || 0);
                    } else {
                    posNode.x = width * 0.5; posNode.y = height * 0.2;
                    }
                    nodes.push(posNode);
                    nodeMap.set('__robot_position__', posNode);
                    if (graphData.current_position.region) {
                    links.push({ source: '__robot_position__', target: graphData.current_position.region });
                    }
                }


                // Add connections
                if (graphData.object_connections) {
                    graphData.object_connections.forEach(conn => {
                        if (conn.length === 2) {
                            const [source, target] = conn;
                            if (nodeMap.has(source) && nodeMap.has(target)) {
                                links.push({ source, target });
                            }
                        }
                    });
                }

                if (graphData.region_connections) {
                    graphData.region_connections.forEach(conn => {
                        if (conn.length === 2) {
                            const [source, target] = conn;
                            if (nodeMap.has(source) && nodeMap.has(target)) {
                                links.push({ source, target });
                            }
                        }
                    });
                }

                if (nodes.length === 0) {
                    renderEmptyState();
                    return;
                }

                // Give nodes random initial positions spread across the canvas
                nodes.forEach(d => {
                    // Use a wider spread for initial positions
                    const angle = Math.random() * 2 * Math.PI;
                    const radius = Math.random() * Math.min(width, height) / 3;
                    d.x = width / 2 + radius * Math.cos(angle);
                    d.y = height / 2 + radius * Math.sin(angle);
                });

                // Create force simulation with balanced forces
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id)
                    .distance(d => {
                        const st = d.source.type || d.source;
                        const tt = d.target.type || d.target;
                        if (st === 'region' && tt === 'region') return 200;
                        return 150;
                    })
                    .strength(0.3)
                    )
                    .force("charge", d3.forceManyBody().strength(d => {
                    if (d.type === 'region') return -1200;
                    if (d.type === 'robot')  return -1000;
                    return -600;
                    }).distanceMax(400))
                    .force("collision", d3.forceCollide().radius(d => {
                    if (d.type === 'robot')  return 35;
                    if (d.type === 'region') return 30;
                    return 25;
                    }).strength(0.7))
                    // pull **unfixed** nodes gently toward their current x/y; fixed nodes already have fx/fy
                    .force("x", d3.forceX(d => d.fx ?? width / 2).strength(d => d.fx != null ? 0.0 : 0.03))
                    .force("y", d3.forceY(d => d.fy ?? height / 2).strength(d => d.fy != null ? 0.0 : 0.03))
                    .velocityDecay(0.4)
                    .alphaDecay(0.01);

                // Create links
                const link = g.append("g")
                    .selectAll("line")
                    .data(links)
                    .join("line")
                    .attr("class", "link");

                // Create nodes
                const node = g.append("g")
                    .selectAll("g")
                    .data(nodes)
                    .join("g")
                    .attr("class", "node")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended))
                    .on("dblclick", function(event, d) {
                        // Double-click to center on node
                        event.stopPropagation();
                        const transform = d3.zoomTransform(rect.node());
                        const x = transform.applyX(d.x);
                        const y = transform.applyY(d.y);
                        
                        rect.transition()
                            .duration(500)
                            .call(zoom.transform, 
                                d3.zoomIdentity
                                    .translate(width / 2, height / 2)
                                    .scale(transform.k)
                                    .translate(-d.x, -d.y)
                            );
                    });

                node.append("circle")
                    .attr("r", d => d.type === 'robot' ? 25 : 18)
                    .attr("fill", d => {
                        if (d.type === 'region') return '#4CAF50';
                        if (d.type === 'object') return '#2196F3';
                        if (d.type === 'robot') return '#FF9800';
                        return '#9E9E9E';
                    })
                    .attr("stroke-width", d => d.type === 'robot' ? 4 : 3);

                node.append("text")
                    .attr("dy", 35)
                    .attr("text-anchor", "middle")
                    .text(d => d.label)
                    .style("font-size", d => d.type === 'robot' ? "13px" : "11px")
                    .style("fill", "#333")
                    .style("font-weight", d => d.type === 'robot' ? "bold" : "500");

                node.append("title")
                    .text(d => `${d.label}\nType: ${d.type}\nCoords: ${d.coords}\n\nDouble-click to center`);

                simulation.on("tick", () => {
                    const PAD = 50;
                    const safeW = Math.max(2 * PAD, width);
                    const safeH = Math.max(2 * PAD, height);

                    nodes.forEach(d => {
                    if (d.fx != null && d.fy != null) {
                        // fixed (regions): keep them exactly at fx/fy
                        d.x = d.fx; d.y = d.fy;
                    } else {
                        d.x = Math.max(PAD, Math.min(safeW - PAD, d.x));
                        d.y = Math.max(PAD, Math.min(safeH - PAD, d.y));
                    }
                    });

                    link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                    node.attr("transform", d => `translate(${d.x},${d.y})`);
                });


                function dragstarted(event) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }

                function dragged(event) {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }

                function dragended(event) {
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }

                // Let simulation run longer for better spreading
                setTimeout(() => {
                    fitToScreen();
                }, 2000);

                simulation.on("tick", () => {
                    // Use safe inner dimensions so we never collapse to a single x=50 column
                    const PAD = 50;
                    const safeW = Math.max(2 * PAD, width);
                    const safeH = Math.max(2 * PAD, height);

                    nodes.forEach(d => {
                        d.x = Math.max(PAD, Math.min(safeW - PAD, d.x));
                        d.y = Math.max(PAD, Math.min(safeH - PAD, d.y));
                    });

                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);

                    node.attr("transform", d => `translate(${d.x},${d.y})`);
                });
            }

            let lastGraphData = null;
            let autoRefreshEnabled = true;

            function checkForUpdates() {
                if (!autoRefreshEnabled) return;

                fetch('/graph-data')
                    .then(response => response.json())
                    .then(result => {
                        if (result.success && result.data) {
                            // Only reload if data actually changed
                            if (lastGraphData !== result.data) {
                                lastGraphData = result.data;
                                const graphData = JSON.parse(result.data);
                                lastGraphObj = graphData;
                                renderGraph(graphData);
                                
                                const nodeCount = (graphData.objects?.length || 0) + (graphData.regions?.length || 0);
                                const edgeCount = graphData.object_connections?.length || 0;
                                document.getElementById('status').textContent = 
                                    `✅ Loaded: ${nodeCount} nodes, ${edgeCount} connections`;
                                
                                document.getElementById('stats').innerHTML = 
                                    `<strong>Objects:</strong> ${graphData.objects?.length || 0} | ` +
                                    `<strong>Regions:</strong> ${graphData.regions?.length || 0} | ` +
                                    `<strong>Current Region:</strong> ${graphData.current_position?.region || 'unknown'}`;
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Background update error:', error);
                    });
            }

            // Load graph on page load
            loadGraph();

            // Check for updates every 2 seconds (only re-renders if changed)
            setInterval(checkForUpdates, 2000);

            // Initialize zoom level display
            updateZoomLevel();
        </script>
    </body>
    </html>'''
        
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