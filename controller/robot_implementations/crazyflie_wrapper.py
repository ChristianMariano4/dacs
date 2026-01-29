"""
CrazyflieWrapper with multiprocessing odometry.

Uses a separate process for the Crazyflie connection to avoid GIL contention
and isolate hardware failures. Position data flows via shared memory, commands
via queue.
"""

import math
import time
import logging
import ctypes
import multiprocessing as mp
from multiprocessing import Process, Array, Queue
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import socket
import struct
import threading

from controller.robot_implementations.abs.robot_wrapper import RobotWrapper, CommandResult
from controller.context_map.graph_manager import GraphManager
from controller.utils.constants import REGION_THRESHOLD

# Use 'spawn' to avoid fork+threads deadlock (cflib creates internal threads)
_spawn_ctx = mp.get_context('spawn')

logger = logging.getLogger(__name__)

# Constants
MOVEMENT_MIN_CM = 20
MOVEMENT_MAX_CM = 500
DEFAULT_HEIGHT_CM = 50
DEFAULT_VELOCITY = 0.3


def adjust_exposure(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def cap_distance_cm(distance_cm: int, min_cm: int = MOVEMENT_MIN_CM,
                    max_cm: int = MOVEMENT_MAX_CM) -> int:
    if abs(distance_cm) < min_cm:
        return 0
    return max(-max_cm, min(distance_cm, max_cm))


def calculate_duration(distance_m: float, velocity: float = DEFAULT_VELOCITY) -> float:
    if distance_m == 0:
        return 0.1
    return max(abs(distance_m) / velocity, 0.5)


class AIDeckClient:
    """UDP client for AI-deck image streaming."""
    CPX_HEADER_SIZE = 4
    IMG_HEADER_MAGIC = 0xBC
    IMG_HEADER_SIZE = 11
    MAGIC_BYTE = b'FER'

    def __init__(self, esp32_ip="172.16.0.151", esp32_port=5000, listen_port=5001):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.listen_port = listen_port
        self.socket = None
        self.buffer = bytearray()
        self.expected_size = None
        self.receiving = False

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", self.listen_port))
        self.socket.settimeout(5.0)
        # Send magic byte to initiate streaming
        self.socket.sendto(self.MAGIC_BYTE, (self.esp32_ip, self.esp32_port))
        logger.info(f"AI-deck UDP connection established on port {self.listen_port}")

    def receive_image_packet(self):
        try:
            while True:
                data, addr = self.socket.recvfrom(2048)

                # Check for image header
                if len(data) >= self.CPX_HEADER_SIZE + 1 and data[self.CPX_HEADER_SIZE] == self.IMG_HEADER_MAGIC:
                    payload = data[self.CPX_HEADER_SIZE:]
                    if len(payload) < self.IMG_HEADER_SIZE:
                        continue

                    _, width, height, depth, fmt, size = struct.unpack('<BHHBBI', payload[:self.IMG_HEADER_SIZE])
                    self.expected_size = size
                    self.buffer = bytearray(payload[self.IMG_HEADER_SIZE:])
                    self.receiving = True

                elif self.receiving:
                    self.buffer.extend(data[self.CPX_HEADER_SIZE:])

                if self.expected_size is not None and len(self.buffer) >= self.expected_size:
                    img_data = bytes(self.buffer[:self.expected_size])
                    self.receiving = False
                    self.expected_size = None
                    return img_data, 1, (width, height)

        except socket.timeout:
            return None, None, None
        except Exception as e:
            logger.error(f"Error receiving image: {e}")
            return None, None, None

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None


class AIDeckFrameReader:
    """Threaded frame reader for AI-deck."""
    def __init__(self, client: AIDeckClient):
        self.client = client
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _capture_loop(self):
        while self._running:
            try:
                img_data, img_format, dims = self.client.receive_image_packet()
                if img_data is None:
                    continue

                nparr = np.frombuffer(img_data, np.uint8)
                color_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                if color_img is not None:
                    with self._lock:
                        self._frame = color_img
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(0.1)

    @property
    def frame(self):
        with self._lock:
            if self._frame is None:
                return None
            frame = adjust_exposure(self._frame.copy(), alpha=1.3, beta=-30)
            return sharpen_image(frame)


class TrackingStatus:
    VALID = 1.0
    STALE = 0.0
    RECOVERING = -1.0
    FAILED = -2.0
    NOT_STARTED = -3.0


def _odometry_only_process_func(
    shared_position: Array,
    shared_status: Array,
    stop_event,
    link_uri: str,
    position_lock,
):
    """
    Odometry-only process for external position tracking (e.g., TelloWrapper).
    
    Unlike _crazyflie_process_func, this doesn't handle flight commands - 
    it only reads position from the Lighthouse system and writes to shared memory.
    """
    import time
    import numpy as np
    
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
    
    print(f"[OdometryProcess] Starting (PID: {mp.current_process().pid})")
    
    # State
    pose = np.zeros(4)
    STALE_THRESHOLD_SEC = 0.5
    POSITION_CHANGE_THRESHOLD = 0.001
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_COOLDOWN_SEC = 5.0
    
    last_position = np.zeros(4)
    last_change_time = time.time()
    last_recovery_attempt = 0.0
    recovery_count = 0
    
    # Connect
    try:
        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache='./cache')
        scf = SyncCrazyflie(link_uri, cf=cf)
        scf.open_link()
        time.sleep(2.0)
        print("[OdometryProcess] Connected")
    except Exception as e:
        print(f"[OdometryProcess] Connection failed: {e}")
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return
    
    def reset_kalman():
        print("[OdometryProcess] Resetting Kalman...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.0)
        
        log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
        log_config.add_variable('kalman.varPX', 'float')
        log_config.add_variable('kalman.varPY', 'float')
        log_config.add_variable('kalman.varPZ', 'float')
        
        var_history = {'x': [1000]*10, 'y': [1000]*10, 'z': [1000]*10}
        threshold = 0.001
        
        try:
            with SyncLogger(scf, log_config) as sync_logger:
                for log_entry in sync_logger:
                    if stop_event.is_set():
                        return False
                    data = log_entry[1]
                    for axis in ['x', 'y', 'z']:
                        var_history[axis].append(data[f'kalman.varP{axis.upper()}'])
                        var_history[axis].pop(0)
                    ranges = {k: max(v) - min(v) for k, v in var_history.items()}
                    if all(r < threshold for r in ranges.values()):
                        print("[OdometryProcess] Kalman converged")
                        return True
        except Exception as e:
            print(f"[OdometryProcess] Kalman reset error: {e}")
            return False
        return False
    
    reset_kalman()
    
    # Position logging callback
    def pose_callback(timestamp, data, logconf):
        nonlocal pose
        pose[0] = data['stateEstimate.x'] * 100  # cm
        pose[1] = data['stateEstimate.y'] * 100
        pose[2] = (data['stateEstimate.z'] - 0.05) * 100
        pose[3] = data['stateEstimate.yaw'] % 360
    
    log_config = LogConfig(name='Position', period_in_ms=20)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')
    log_config.add_variable('stateEstimate.yaw', 'float')
    cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(pose_callback)
    log_config.start()
    
    # Main loop
    while not stop_event.is_set():
        try:
            current_time = time.time()
            
            # Check for stale tracking
            position_changed = np.any(np.abs(pose - last_position) > POSITION_CHANGE_THRESHOLD * 100)
            if position_changed:
                last_change_time = current_time
                last_position = pose.copy()
            
            time_since_change = current_time - last_change_time
            is_stale = time_since_change > STALE_THRESHOLD_SEC
            
            # Attempt recovery if stale
            if is_stale and recovery_count < MAX_RECOVERY_ATTEMPTS:
                if current_time - last_recovery_attempt > RECOVERY_COOLDOWN_SEC:
                    print(f"[OdometryProcess] Stale, recovery {recovery_count + 1}/{MAX_RECOVERY_ATTEMPTS}")
                    with position_lock:
                        shared_status[0] = TrackingStatus.RECOVERING
                        shared_status[2] = float(recovery_count + 1)
                    
                    if reset_kalman():
                        last_change_time = current_time
                        recovery_count = 0
                    else:
                        recovery_count += 1
                    last_recovery_attempt = current_time
            
            # Write to shared memory
            with position_lock:
                shared_position[0] = pose[0]
                shared_position[1] = pose[1]
                shared_position[2] = pose[2]
                shared_position[3] = pose[3]
                
                if recovery_count >= MAX_RECOVERY_ATTEMPTS:
                    shared_status[0] = TrackingStatus.FAILED
                elif is_stale:
                    shared_status[0] = TrackingStatus.STALE
                else:
                    shared_status[0] = TrackingStatus.VALID
                shared_status[1] = last_change_time
                shared_status[2] = float(recovery_count)
            
            time.sleep(0.02)  # 50Hz
            
        except Exception as e:
            print(f"[OdometryProcess] Error: {e}")
            time.sleep(0.1)
    
    # Cleanup
    print("[OdometryProcess] Stopping...")
    try:
        log_config.stop()
        scf.close_link()
    except:
        pass


class CommandType(Enum):
    TAKEOFF = "takeoff"
    LAND = "land"
    GO_TO = "go_to"
    STOP = "stop"
    RESET_KALMAN = "reset_kalman"
    SOUND_EFFECT = "sound_effect"


@dataclass
class Command:
    type: CommandType
    args: tuple = ()
    kwargs: dict = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


def _crazyflie_process_func(
    shared_position: Array,
    shared_status: Array,
    command_queue: Queue,
    stop_event,
    link_uri: str,
    position_lock,
):
    """
    Crazyflie subprocess entry point.
    
    Must be module-level (not a method) for pickling with 'spawn' mode.
    """
    import time
    import math
    import numpy as np
    
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
    
    print(f"[CrazyflieProcess] Starting (PID: {mp.current_process().pid})")
    
    # State
    pose = np.zeros(4)
    last_pose_time = time.time()
    recovery_count = 0
    is_flying = False
    
    STALE_THRESHOLD_SEC = 0.5
    POSITION_CHANGE_THRESHOLD = 0.001
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_COOLDOWN_SEC = 5.0
    
    last_position = np.zeros(4)
    last_change_time = time.time()
    last_recovery_attempt = 0.0
    
    # Connect
    try:
        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache='./cache')
        scf = SyncCrazyflie(link_uri, cf=cf)
        scf.open_link()
        time.sleep(2.0)
        
        cf.param.set_value("commander.enHighLevel", "1")
        time.sleep(0.5)
        commander = cf.high_level_commander
        
        print("[CrazyflieProcess] Connected")
    except Exception as e:
        print(f"[CrazyflieProcess] Connection failed: {e}")
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return
    
    def reset_kalman():
        """Reset Kalman filter and wait for convergence."""
        print("[CrazyflieProcess] Resetting Kalman...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.0)
        
        log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
        log_config.add_variable('kalman.varPX', 'float')
        log_config.add_variable('kalman.varPY', 'float')
        log_config.add_variable('kalman.varPZ', 'float')
        
        var_history = {'x': [1000]*10, 'y': [1000]*10, 'z': [1000]*10}
        threshold = 0.001
        
        try:
            with SyncLogger(scf, log_config) as logger:
                for log_entry in logger:
                    if stop_event.is_set():
                        return False
                    
                    data = log_entry[1]
                    for axis in ['x', 'y', 'z']:
                        var_history[axis].append(data[f'kalman.varP{axis.upper()}'])
                        var_history[axis].pop(0)
                    
                    ranges = {k: max(v) - min(v) for k, v in var_history.items()}
                    if all(r < threshold for r in ranges.values()):
                        print("[CrazyflieProcess] Kalman converged")
                        return True
        except Exception as e:
            print(f"[CrazyflieProcess] Kalman reset error: {e}")
            return False
        
        return False
    
    reset_kalman()
    
    def pose_callback(timestamp, data, logconf):
        nonlocal pose, last_pose_time
        pose[0] = data['stateEstimate.x'] * 100
        pose[1] = data['stateEstimate.y'] * 100
        pose[2] = (data['stateEstimate.z'] - 0.05) * 100  # Ground offset
        pose[3] = data['stateEstimate.yaw'] % 360
        last_pose_time = time.time()
    
    log_config = LogConfig(name='Position', period_in_ms=20)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')
    log_config.add_variable('stateEstimate.yaw', 'float')
    cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(pose_callback)
    log_config.start()
    
    def handle_command(cmd: Command):
        nonlocal is_flying
        try:
            if cmd.type == CommandType.TAKEOFF:
                height = cmd.kwargs.get('height', 0.5)
                duration = cmd.kwargs.get('duration', 3.0)
                commander.takeoff(height, duration)
                time.sleep(duration)
                is_flying = True
                
            elif cmd.type == CommandType.LAND:
                height = cmd.kwargs.get('height', 0.0)
                duration = cmd.kwargs.get('duration', 4.5)
                commander.land(height, duration)
                time.sleep(duration)
                is_flying = False
                
            elif cmd.type == CommandType.GO_TO:
                commander.go_to(
                    cmd.kwargs.get('x', 0),
                    cmd.kwargs.get('y', 0),
                    cmd.kwargs.get('z', 0),
                    cmd.kwargs.get('yaw', 0),
                    cmd.kwargs.get('duration', 2.0),
                    cmd.kwargs.get('relative', True)
                )
                
            elif cmd.type == CommandType.RESET_KALMAN:
                reset_kalman()
                
            elif cmd.type == CommandType.SOUND_EFFECT:
                cf.param.set_value("sound.effect", str(cmd.kwargs.get('effect', 0)))
                
        except Exception as e:
            print(f"[CrazyflieProcess] Command error: {e}")
    
    # Main loop
    while not stop_event.is_set():
        try:
            current_time = time.time()
            
            # Process pending commands (non-blocking)
            try:
                while True:
                    cmd = command_queue.get_nowait()
                    if cmd.type == CommandType.STOP:
                        stop_event.set()
                        break
                    handle_command(cmd)
            except:
                pass
            
            # Check for stale tracking
            position_changed = np.any(np.abs(pose - last_position) > POSITION_CHANGE_THRESHOLD * 100)
            if position_changed:
                last_change_time = current_time
                last_position = pose.copy()
            
            time_since_change = current_time - last_change_time
            is_stale = time_since_change > STALE_THRESHOLD_SEC
            
            if is_stale and recovery_count < MAX_RECOVERY_ATTEMPTS:
                if current_time - last_recovery_attempt > RECOVERY_COOLDOWN_SEC:
                    print(f"[CrazyflieProcess] Stale tracking, recovery {recovery_count + 1}/{MAX_RECOVERY_ATTEMPTS}")
                    with position_lock:
                        shared_status[0] = TrackingStatus.RECOVERING
                        shared_status[2] = float(recovery_count + 1)
                    
                    if reset_kalman():
                        last_change_time = current_time
                        recovery_count = 0
                    else:
                        recovery_count += 1
                    last_recovery_attempt = current_time
            
            # Write to shared memory (lock ensures atomic read from main process)
            with position_lock:
                shared_position[0] = pose[0]
                shared_position[1] = pose[1]
                shared_position[2] = pose[2]
                shared_position[3] = pose[3]
                
                if recovery_count >= MAX_RECOVERY_ATTEMPTS:
                    shared_status[0] = TrackingStatus.FAILED
                elif is_stale:
                    shared_status[0] = TrackingStatus.STALE
                else:
                    shared_status[0] = TrackingStatus.VALID
                shared_status[1] = last_change_time
                shared_status[2] = float(recovery_count)
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"[CrazyflieProcess] Error: {e}")
            time.sleep(0.1)
    
    # Cleanup
    print("[CrazyflieProcess] Shutting down...")
    try:
        log_config.stop()
        if is_flying:
            commander.land(0.0, 3.0)
            time.sleep(3.5)
        scf.close_link()
    except Exception as e:
        print(f"[CrazyflieProcess] Cleanup error: {e}")


class CrazyflieWrapperMP(RobotWrapper):
    """Crazyflie wrapper using multiprocessing. API-compatible with RobotWrapper."""
    
    def __init__(
        self, 
        graph_manager: GraphManager,
        move_enable: bool = False, 
        link_uri: str = 'radio://0/40/2M/BADF00D003',
        esp32_ip: str = "172.16.0.151",
        esp32_port: int = 5000,
        listen_port: int = 5001
    ):
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        self.link_uri = link_uri
        self.connected = False
        
        self._shared_position = _spawn_ctx.Array(ctypes.c_double, 4)  # [x, y, z, yaw] in cm/deg
        self._shared_status = _spawn_ctx.Array(ctypes.c_double, 3)    # [status, time, recovery_count]
        
        for i in range(4):
            self._shared_position[i] = 0.0
        self._shared_status[0] = TrackingStatus.NOT_STARTED
        self._shared_status[1] = 0.0
        self._shared_status[2] = 0.0
        
        self._position_lock = _spawn_ctx.Lock()
        self._stop_event = _spawn_ctx.Event()
        self._command_queue = _spawn_ctx.Queue()
        self._crazyflie_process: Optional[Process] = None
        self._is_flying = False
        
        # AI-deck streaming (runs in main process)
        self.ai_deck_client = AIDeckClient(
            esp32_ip=esp32_ip, 
            esp32_port=esp32_port, 
            listen_port=listen_port
        )
        self.frame_reader: Optional[AIDeckFrameReader] = None
        self.stream_on = False
        
        # Graph update thread
        self._thread_stop_event = threading.Event()
        self._threads = []
    
    def connect(self):
        if self._crazyflie_process is not None:
            print("[CrazyflieWrapperMP] Already connected")
            return
        
        self._crazyflie_process = _spawn_ctx.Process(
            target=_crazyflie_process_func,
            args=(
                self._shared_position,
                self._shared_status,
                self._command_queue,
                self._stop_event,
                self.link_uri,
                self._position_lock,
            ),
            name="crazyflie-process",
            daemon=True
        )
        self._crazyflie_process.start()
        print(f"[CrazyflieWrapperMP] Process started (PID: {self._crazyflie_process.pid})")
        
        if self.wait_for_tracking(timeout=15.0):
            print("[CrazyflieWrapperMP] Tracking active")
        else:
            print("[CrazyflieWrapperMP] Warning: tracking not yet valid")
        self.connected = True
        
        # Start graph update thread if graph_manager is available
        if self.graph_manager is not None:
            self._start_thread(self._graph_update_loop, "graph-updater")
    
    def disconnect(self):
        if self._crazyflie_process is None:
            return
        
        # Stop threads
        self._thread_stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        
        # Stop streaming
        self.stop_stream()
        
        # Stop crazyflie process
        self._command_queue.put(Command(type=CommandType.STOP))
        self._stop_event.set()
        self._crazyflie_process.join(timeout=5.0)
        
        if self._crazyflie_process.is_alive():
            self._crazyflie_process.terminate()
            self._crazyflie_process.join(timeout=2.0)
        
        self._crazyflie_process = None
        self.connected = False
        print("[CrazyflieWrapperMP] Disconnected")
    
    def _start_thread(self, target, name: str):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)
    
    def _graph_update_loop(self):
        while not self._thread_stop_event.is_set():
            try:
                if self.graph_manager is not None and self.is_position_valid():
                    pos = self.get_position()
                    self.graph_manager.update_pose(
                        np.array([pos[0], pos[1], pos[2]]),
                        pos[3]
                    )
            except Exception as e:
                logger.debug(f"Graph update error: {e}")
            time.sleep(0.1)
    
    def get_tracking_status(self) -> str:
        with self._position_lock:
            status_code = self._shared_status[0]
            recovery_count = int(self._shared_status[2])
        
        status_map = {
            TrackingStatus.VALID: "VALID",
            TrackingStatus.STALE: "STALE",
            TrackingStatus.RECOVERING: f"RECOVERING ({recovery_count})",
            TrackingStatus.FAILED: f"FAILED ({recovery_count} attempts)",
            TrackingStatus.NOT_STARTED: "NOT_STARTED",
        }
        return status_map.get(status_code, f"UNKNOWN ({status_code})")
    
    def is_position_valid(self) -> bool:
        with self._position_lock:
            return self._shared_status[0] == TrackingStatus.VALID
    
    def wait_for_tracking(self, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.is_position_valid():
                return True
            time.sleep(0.1)
        return False
    
    def get_pose(self) -> np.ndarray:
        with self._position_lock:
            return np.array([
                self._shared_position[0],
                self._shared_position[1],
                self._shared_position[2],
                self._shared_position[3],
            ])
    
    def get_position(self) -> Tuple[float, float, float, float]:
        """Returns (x_cm, y_cm, z_cm, yaw_deg)."""
        pose = self.get_pose()
        return (pose[0], pose[1], pose[2], pose[3])
    
    def takeoff(self, height: float = 0.5, duration: float = 3.0) -> CommandResult:
        if not self.move_enable:
            print(f"[CrazyflieWrapperMP] Takeoff (simulated)")
            return CommandResult(value=True, replan=False)
        
        self._command_queue.put(Command(
            type=CommandType.TAKEOFF,
            kwargs={'height': height, 'duration': duration}
        ))
        self._is_flying = True
        time.sleep(duration)
        return CommandResult(value=True, replan=False)
    
    def land(self, height: float = 0.0, duration: float = 4.5) -> CommandResult:
        if not self.move_enable:
            print(f"[CrazyflieWrapperMP] Land (simulated)")
            return CommandResult(value=True, replan=False)
        
        self._command_queue.put(Command(
            type=CommandType.LAND,
            kwargs={'height': height, 'duration': duration}
        ))
        self._is_flying = False
        time.sleep(duration)
        return CommandResult(value=True, replan=False)
    
    def go_to(
        self, 
        x: float, 
        y: float, 
        z: float, 
        yaw: float, 
        duration: float,
        relative: bool = True
    ) -> CommandResult:
        if not self.move_enable:
            print(f"[CrazyflieWrapperMP] go_to({x}, {y}, {z}) (simulated)")
            return CommandResult(value=True, replan=False)
        
        self._command_queue.put(Command(
            type=CommandType.GO_TO,
            kwargs={
                'x': x, 'y': y, 'z': z, 
                'yaw': yaw, 'duration': duration, 
                'relative': relative
            }
        ))
        time.sleep(duration)
        return CommandResult(value=True, replan=False)
    
    # --- RobotWrapper abstract methods ---
    
    def _execute_relative_move(self, dx_cm: float, dy_cm: float, dz_cm: float,
                               dyaw_deg: float = 0.0) -> CommandResult:
        """Helper for relative movements with validation."""
        if not self.connected:
            return CommandResult(value=False, replan=False)
        if not self._is_flying:
            return CommandResult(value=False, replan=True)
        
        dx_cm = cap_distance_cm(int(dx_cm))
        dy_cm = cap_distance_cm(int(dy_cm))
        dz_cm = cap_distance_cm(int(dz_cm))
        
        if dx_cm == 0 and dy_cm == 0 and dz_cm == 0 and dyaw_deg == 0:
            return CommandResult(value=True, replan=False)
        
        dx_m = dx_cm / 100.0
        dy_m = dy_cm / 100.0
        dz_m = dz_cm / 100.0
        dyaw_rad = math.radians(dyaw_deg)
        
        distance_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        duration = calculate_duration(distance_m)
        
        return self.go_to(dx_m, dy_m, dz_m, dyaw_rad, duration, relative=True)
    
    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float) -> CommandResult:
        """Navigate to absolute position."""
        if not self.connected:
            return CommandResult(value=False, replan=False)
        if not self._is_flying:
            return CommandResult(value=False, replan=True)
        
        curr = self.get_position()
        dx = target_x_cm - curr[0]
        dy = target_y_cm - curr[1]
        dz = target_z_cm - curr[2]
        
        if abs(dx) < 20 and abs(dy) < 20 and abs(dz) < 20:
            return CommandResult(value=True, replan=False)
        
        dx_m = dx / 100.0
        dy_m = dy / 100.0
        dz_m = dz / 100.0
        
        distance_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        duration = calculate_duration(distance_m)
        
        return self.go_to(dx_m, dy_m, dz_m, 0, duration, relative=True)
    
    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._execute_relative_move(dx_cm=distance_cm, dy_cm=0, dz_cm=0)
    
    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._execute_relative_move(dx_cm=-distance_cm, dy_cm=0, dz_cm=0)
    
    def move_east(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._execute_relative_move(dx_cm=0, dy_cm=distance_cm, dz_cm=0)
    
    def move_west(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._execute_relative_move(dx_cm=0, dy_cm=-distance_cm, dz_cm=0)
    
    def move_up(self, distance_cm: int) -> CommandResult:
        return self._execute_relative_move(dx_cm=0, dy_cm=0, dz_cm=distance_cm)
    
    def move_down(self, distance_cm: int) -> CommandResult:
        return self._execute_relative_move(dx_cm=0, dy_cm=0, dz_cm=-distance_cm)
    
    def move_direction(self, direction_deg: int, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Move along a heading. First rotate, then move forward along that heading."""
        if not self._is_flying:
            return CommandResult(value=False, replan=True)
        
        distance_cm = cap_distance_cm(distance_cm)
        if distance_cm == 0:
            return CommandResult(value=True, replan=False)
        
        current_yaw = self.get_position()[3] % 360
        target_yaw = direction_deg % 360
        delta_yaw = (target_yaw - current_yaw + 180) % 360 - 180
        
        # Rotate first if needed
        if abs(delta_yaw) > 5:
            result = self._execute_relative_move(0, 0, 0, delta_yaw)
            if not result.value:
                return result
        
        # Move along heading direction
        direction_rad = math.radians(direction_deg)
        dx_cm = distance_cm * math.cos(direction_rad)
        dy_cm = distance_cm * math.sin(direction_rad)
        
        return self._execute_relative_move(dx_cm, dy_cm, 0, 0)
    
    def turn_cw(self, degree: int) -> CommandResult:
        return self._execute_relative_move(0, 0, 0, dyaw_deg=-degree)
    
    def turn_ccw(self, degree: int) -> CommandResult:
        return self._execute_relative_move(0, 0, 0, dyaw_deg=degree)
    
    def reset_kalman_estimator(self):
        self._command_queue.put(Command(type=CommandType.RESET_KALMAN))
    
    def sound_effect(self, effect: int):
        self._command_queue.put(Command(
            type=CommandType.SOUND_EFFECT,
            kwargs={'effect': effect}
        ))
    
    # Compatibility methods
    def get_move_enable(self) -> bool:
        return self.move_enable
    
    def get_is_flying(self) -> bool:
        return self._is_flying
    
    def set_is_flying(self, value: bool):
        self._is_flying = value
    
    def start_stream(self):
        if self.stream_on:
            return
        try:
            self.ai_deck_client.connect()
            self.frame_reader = AIDeckFrameReader(self.ai_deck_client)
            self.frame_reader.start()
            self.stream_on = True
            logger.info("AI-deck stream started")
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            self.stream_on = False
    
    def stop_stream(self):
        if not self.stream_on:
            return
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader = None
        self.ai_deck_client.close()
        self.stream_on = False
    
    def get_frame_reader(self) -> Any:
        return self.frame_reader
    
    def keep_active(self):
        pass
    
    # Motion helpers
    def no_motion(self, duration: float = 0.4):
        if not self.move_enable:
            return
        self.go_to(0, 0, 0, -math.pi / 5, duration)
        time.sleep(duration)
        self.go_to(0, 0, 0, 2 * math.pi / 5, duration * 2)
        time.sleep(duration * 2)
        self.go_to(0, 0, 0, -math.pi / 5, duration)
        time.sleep(duration)
    
    def rotate_360(self, duration: float = 2.0):
        if not self.move_enable:
            return
        self.go_to(0, 0, 0, math.pi, duration / 2)
        time.sleep(duration)
        self.go_to(0, 0, 0, -math.pi, duration / 2)
        time.sleep(duration)
    
    def victory_motion(self, duration: float = 0.5):
        if not self.move_enable:
            return
        for _ in range(2):
            self.go_to(0, 0, 0.2, 0, duration)
            time.sleep(duration)
            self.go_to(0, 0, -0.2, 0, duration)
            time.sleep(duration)


if __name__ == "__main__":
    # Example usage - requires a GraphManager instance in real usage
    print("CrazyflieWrapperMP requires a GraphManager instance.")
    print("Example instantiation:")
    print("  wrapper = CrazyflieWrapperMP(graph_manager=gm, move_enable=False)")
    print("  wrapper.connect()")
    print("  pos = wrapper.get_position()")
    print("  wrapper.disconnect()")