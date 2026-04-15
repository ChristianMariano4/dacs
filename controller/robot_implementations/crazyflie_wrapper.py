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
from queue import Empty as QueueEmpty
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

from controller.abs.robot_wrapper import RobotWrapper, CommandResult
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

def cap_distance_cm(distance_cm: int, min_cm: int = MOVEMENT_MIN_CM,
                    max_cm: int = MOVEMENT_MAX_CM) -> int:
    if abs(distance_cm) < min_cm:
        return 0
    return max(-max_cm, min(distance_cm, max_cm))


def calculate_duration(distance_m: float, velocity: float = DEFAULT_VELOCITY) -> float:
    if distance_m == 0:
        return 0.1
    return max(abs(distance_m) / velocity, 0.5)


def _wait_for_param_toc(cf: Any, timeout_sec: float = 3.0):
    """Wait briefly for param TOC sync when cflib exposes the flag."""
    param_iface = getattr(cf, "param", None)
    if param_iface is None or not hasattr(param_iface, "is_updated"):
        return

    deadline = time.time() + timeout_sec
    while time.time() < deadline and not getattr(param_iface, "is_updated", False):
        time.sleep(0.05)


def _cf_param_exists(cf: Any, full_name: str) -> bool:
    """Best-effort check for a parameter in the Crazyflie TOC."""
    group, sep, name = full_name.partition(".")
    if not sep or not group or not name:
        return False

    param_iface = getattr(cf, "param", None)
    toc_iface = getattr(param_iface, "toc", None) if param_iface else None
    if toc_iface is None:
        return False

    getter = getattr(toc_iface, "get_element_by_complete_name", None)
    if callable(getter):
        try:
            return getter(full_name) is not None
        except Exception:
            pass

    toc_map = getattr(toc_iface, "toc", None)
    if isinstance(toc_map, dict):
        if full_name in toc_map:
            return True
        group_map = toc_map.get(group)
        if isinstance(group_map, dict):
            return name in group_map

    return False


def _is_param_toc_missing_error(error: Exception) -> bool:
    msg = str(error).lower()
    return ("not in param toc" in msg) or ("not in the toc" in msg)


def _is_not_connected_error(error: Exception) -> bool:
    msg = str(error).lower()
    return (
        "without being connected to a crazyflie" in msg
        or "not connected" in msg
        or "link is down" in msg
    )


def _cf_set_param(cf: Any, full_name: str, value: Any, log_prefix: str) -> Tuple[bool, bool]:
    """Returns (ok, missing_in_toc)."""
    try:
        cf.param.set_value(full_name, str(value))
        return True, False
    except Exception as e:
        print(f"{log_prefix} Failed to set '{full_name}': {e}")
        return False, _is_param_toc_missing_error(e)


class AIDeckFrameReader:
    """Threaded frame reader for AI-deck using the same UDP loop as test/cf.py."""

    CPX_HEADER_SIZE = 4
    IMG_HEADER_MAGIC = 0xBC
    IMG_HEADER_SIZE = 11
    MAGIC_BYTE = b'FER'

    def __init__(self, esp32_ip: str, esp32_port: int, listen_port: int):
        self._esp32_ip = esp32_ip
        self._esp32_port = esp32_port
        self._listen_port = listen_port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame = None
        self._lock = threading.Lock()

        self._buffer = bytearray()
        self._expected_size = None
        self._receiving = False
        self._frame_count = 0
        self._last_time = None

    def start(self):
        if self._running:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._listen_port))
        self._sock.settimeout(3.0)

        print("=" * 60)
        print("AI-deck UDP Video Stream Client (Station Mode)")
        print("=" * 60)
        print("Your computer must be on the same network as the AI-deck")
        print(f"AI-deck IP: {self._esp32_ip}")
        print(f"Listening on port: {self._listen_port}")
        print("-" * 60)
        print(f"Sending magic byte to {self._esp32_ip}:{self._esp32_port}...")
        self._sock.sendto(self.MAGIC_BYTE, (self._esp32_ip, self._esp32_port))
        print("Waiting for video frames... Press Ctrl+C to exit")
        print("-" * 60)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="aideck-stream", daemon=True)
        self._thread.start()
        print("[AIDeckFrameReader] Thread started")

    def _capture_loop(self):
        got_any_packet = False
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                print("[AIDeckProcess] No packets yet, resending magic byte...")
                try:
                    self._sock.sendto(self.MAGIC_BYTE, (self._esp32_ip, self._esp32_port))
                except Exception as e:
                    print(f"[AIDeckProcess] Failed to resend magic byte: {e}")
                continue
            except OSError:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

            if not got_any_packet:
                got_any_packet = True
                print(f"[AIDeckProcess] First UDP packet received from {addr[0]}:{addr[1]}")

            if len(data) >= self.CPX_HEADER_SIZE + 1 and data[self.CPX_HEADER_SIZE] == self.IMG_HEADER_MAGIC:
                payload = data[self.CPX_HEADER_SIZE:]
                if len(payload) >= self.IMG_HEADER_SIZE:
                    magic, width, height, depth, fmt, size = struct.unpack('<BHHBBI', payload[:self.IMG_HEADER_SIZE])
                    self._expected_size = size
                    self._buffer = bytearray(payload[self.IMG_HEADER_SIZE:])
                    self._receiving = True
            elif self._receiving:
                self._buffer.extend(data[self.CPX_HEADER_SIZE:])

            if self._expected_size and len(self._buffer) >= self._expected_size:
                self._frame_count += 1
                now = time.time()
                if self._last_time:
                    fps = 1.0 / (now - self._last_time)
                    if self._frame_count % 10 == 0:
                        print(f"Frame {self._frame_count}, FPS: {fps:.1f}")
                self._last_time = now

                try:
                    np_data = np.frombuffer(self._buffer[:self._expected_size], np.uint8)
                    decoded = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                    if decoded is not None:
                        with self._lock:
                            self._latest_frame = decoded
                    else:
                        print(f"Frame {self._frame_count}: Trying raw decode...")
                except Exception as e:
                    print(f"Error: {e}")

                self._receiving = False
                self._expected_size = None

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def frame(self):
        """Return the latest frame as RGB, or None if none received yet."""
        with self._lock:
            img = None if self._latest_frame is None else self._latest_frame.copy()
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if channels == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if channels == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img


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
    has_pose = False
    last_pose_time = time.time()
    STALE_THRESHOLD_SEC = 1.5
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_COOLDOWN_SEC = 5.0
    
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

    kalman_reset_supported = None
    log_config = None

    def _reopen_link() -> bool:
        nonlocal has_pose, last_pose_time
        try:
            scf.close_link()
        except Exception:
            pass

        time.sleep(0.5)
        try:
            scf.open_link()
            time.sleep(2.0)
            has_pose = False
            last_pose_time = time.time()
            return True
        except Exception as e:
            print(f"[OdometryProcess] Link restart failed: {e}")
            return False

    def reset_kalman():
        nonlocal kalman_reset_supported

        if kalman_reset_supported is None:
            _wait_for_param_toc(cf)
            kalman_reset_supported = _cf_param_exists(cf, 'kalman.resetEstimation')
            if not kalman_reset_supported:
                print("[OdometryProcess] 'kalman.resetEstimation' not in TOC, disabling Kalman reset recovery")
        if not kalman_reset_supported:
            return False

        print("[OdometryProcess] Resetting Kalman...")
        ok, missing_in_toc = _cf_set_param(cf, 'kalman.resetEstimation', '1', "[OdometryProcess]")
        if not ok:
            if missing_in_toc:
                kalman_reset_supported = False
                print("[OdometryProcess] Disabling Kalman reset recovery (parameter missing in TOC)")
            return False
        time.sleep(0.1)
        ok, missing_in_toc = _cf_set_param(cf, 'kalman.resetEstimation', '0', "[OdometryProcess]")
        if not ok:
            if missing_in_toc:
                kalman_reset_supported = False
                print("[OdometryProcess] Disabling Kalman reset recovery (parameter missing in TOC)")
            return False
        time.sleep(1.0)

        kalman_log = LogConfig(name='Kalman Variance', period_in_ms=500)
        kalman_log.add_variable('kalman.varPX', 'float')
        kalman_log.add_variable('kalman.varPY', 'float')
        kalman_log.add_variable('kalman.varPZ', 'float')

        var_history = {'x': [1000] * 10, 'y': [1000] * 10, 'z': [1000] * 10}
        threshold = 0.001

        try:
            with SyncLogger(scf, kalman_log) as sync_logger:
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
    
    # Position logging callback
    def pose_callback(timestamp, data, logconf):
        nonlocal pose, last_pose_time, has_pose
        pose[0] = data['stateEstimate.x'] * 100  # cm
        pose[1] = data['stateEstimate.y'] * 100
        pose[2] = (data['stateEstimate.z'] - 0.05) * 100
        pose[3] = data['stateEstimate.yaw'] % 360
        last_pose_time = time.time()
        has_pose = True

    def _create_and_start_position_log() -> bool:
        nonlocal log_config
        cfg = LogConfig(name=f'Position_{int(time.time() * 1000) % 1000000}', period_in_ms=20)
        cfg.add_variable('stateEstimate.x', 'float')
        cfg.add_variable('stateEstimate.y', 'float')
        cfg.add_variable('stateEstimate.z', 'float')
        cfg.add_variable('stateEstimate.yaw', 'float')
        cf.log.add_config(cfg)

        # cflib may only print the connection error and keep cfg.cf unset.
        if getattr(cfg, "cf", None) is None:
            raise RuntimeError("Cannot add configs without being connected to a Crazyflie!")

        cfg.data_received_cb.add_callback(pose_callback)
        cfg.start()
        log_config = cfg
        return True

    def start_position_logging() -> bool:
        try:
            return _create_and_start_position_log()
        except Exception as e:
            print(f"[OdometryProcess] Failed to start position logging: {e}")
            if not _is_not_connected_error(e):
                return False

        if not _reopen_link():
            return False

        try:
            return _create_and_start_position_log()
        except Exception as e:
            print(f"[OdometryProcess] Failed to start position logging after reconnect: {e}")
            return False

    def reconnect_tracking() -> bool:
        nonlocal kalman_reset_supported, log_config
        print("[OdometryProcess] Restarting link/logging...")
        try:
            if log_config is not None:
                log_config.stop()
        except Exception:
            pass
        log_config = None

        if not _reopen_link():
            return False

        kalman_reset_supported = None
        return start_position_logging()

    reset_kalman()
    if not start_position_logging():
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return
    
    # Main loop
    print("[CrazyflieProcess] Entering main loop", flush=True)
    while not stop_event.is_set():
        try:
            current_time = time.time()

            if not has_pose:
                with position_lock:
                    shared_position[0] = pose[0]
                    shared_position[1] = pose[1]
                    shared_position[2] = pose[2]
                    shared_position[3] = pose[3]
                    shared_status[0] = TrackingStatus.NOT_STARTED
                    shared_status[1] = last_pose_time
                    shared_status[2] = float(recovery_count)
                time.sleep(0.02)
                continue
            
            # Staleness must be based on missing pose updates, not on movement.
            # A hovering drone can keep the same position while tracking is valid.
            time_since_pose = current_time - last_pose_time
            is_stale = time_since_pose > STALE_THRESHOLD_SEC
            
            # Attempt recovery if stale
            if is_stale and recovery_count < MAX_RECOVERY_ATTEMPTS:
                if current_time - last_recovery_attempt > RECOVERY_COOLDOWN_SEC:
                    print(f"[OdometryProcess] Stale, recovery {recovery_count + 1}/{MAX_RECOVERY_ATTEMPTS}")
                    with position_lock:
                        shared_status[0] = TrackingStatus.RECOVERING
                        shared_status[2] = float(recovery_count + 1)

                    recovered = reset_kalman()
                    if not recovered:
                        recovered = reconnect_tracking()

                    if recovered:
                        last_pose_time = current_time
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
                shared_status[1] = last_pose_time
                shared_status[2] = float(recovery_count)
            
            time.sleep(0.02)  # 50Hz
            
        except Exception as e:
            print(f"[OdometryProcess] Error: {e}")
            time.sleep(0.1)
    
    # Cleanup
    print("[OdometryProcess] Stopping...")
    try:
        if log_config is not None:
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
    has_pose = False
    last_pose_time = time.time()
    recovery_count = 0
    is_flying = False
    
    STALE_THRESHOLD_SEC = 1.5
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_COOLDOWN_SEC = 5.0
    
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
    
    kalman_reset_supported = None

    def reset_kalman():
        """Reset Kalman filter and wait for convergence."""
        nonlocal kalman_reset_supported

        if kalman_reset_supported is None:
            _wait_for_param_toc(cf)
            kalman_reset_supported = _cf_param_exists(cf, 'kalman.resetEstimation')
            if not kalman_reset_supported:
                print("[CrazyflieProcess] 'kalman.resetEstimation' not in TOC, disabling Kalman reset recovery")
        if not kalman_reset_supported:
            return False

        print("[CrazyflieProcess] Resetting Kalman...")
        ok, missing_in_toc = _cf_set_param(cf, 'kalman.resetEstimation', '1', "[CrazyflieProcess]")
        if not ok:
            if missing_in_toc:
                kalman_reset_supported = False
                print("[CrazyflieProcess] Disabling Kalman reset recovery (parameter missing in TOC)")
            return False
        time.sleep(0.1)
        ok, missing_in_toc = _cf_set_param(cf, 'kalman.resetEstimation', '0', "[CrazyflieProcess]")
        if not ok:
            if missing_in_toc:
                kalman_reset_supported = False
                print("[CrazyflieProcess] Disabling Kalman reset recovery (parameter missing in TOC)")
            return False
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
    
    if not reset_kalman():
        print("[CrazyflieProcess] Kalman reset failed, checking connection...")
        if cf.link is None:
            print("[CrazyflieProcess] Connection lost during Kalman reset")
            with position_lock:
                shared_status[0] = TrackingStatus.FAILED
            return

    # Verify connection is still alive before setting up logging
    if cf.link is None:
        print("[CrazyflieProcess] Connection lost after Kalman reset")
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return

    def pose_callback(timestamp, data, logconf):
        nonlocal pose, last_pose_time, has_pose
        pose[0] = data['stateEstimate.x'] * 100
        pose[1] = data['stateEstimate.y'] * 100
        pose[2] = (data['stateEstimate.z'] - 0.05) * 100  # Ground offset
        pose[3] = data['stateEstimate.yaw'] % 360
        last_pose_time = time.time()
        has_pose = True

    try:
        log_config = LogConfig(name='Position', period_in_ms=20)
        log_config.add_variable('stateEstimate.x', 'float')
        log_config.add_variable('stateEstimate.y', 'float')
        log_config.add_variable('stateEstimate.z', 'float')
        log_config.add_variable('stateEstimate.yaw', 'float')
        cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(pose_callback)
        log_config.start()
    except Exception as e:
        print(f"[CrazyflieProcess] Failed to start position logging: {e}")
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return
    
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
    print("[CrazyflieProcess] Entering main loop", flush=True)
    while not stop_event.is_set():
        try:
            # Check if connection is still alive
            if cf.link is None:
                print("[CrazyflieProcess] Connection lost, exiting...")
                with position_lock:
                    shared_status[0] = TrackingStatus.FAILED
                break

            current_time = time.time()

            # Process pending commands (non-blocking)
            try:
                while True:
                    cmd = command_queue.get_nowait()
                    print(f"[CrazyflieProcess] Got command: {cmd}", flush=True)
                    if cmd.type == CommandType.STOP:
                        stop_event.set()
                        break
                    handle_command(cmd)
            except QueueEmpty:
                pass
            except Exception as e:
                print(f"[CrazyflieProcess] Command queue error: {e}", flush=True)

            if not has_pose:
                with position_lock:
                    shared_position[0] = pose[0]
                    shared_position[1] = pose[1]
                    shared_position[2] = pose[2]
                    shared_position[3] = pose[3]
                    shared_status[0] = TrackingStatus.NOT_STARTED
                    shared_status[1] = last_pose_time
                    shared_status[2] = float(recovery_count)
                time.sleep(0.02)
                continue

            # Staleness must be based on missing pose updates, not movement.
            time_since_pose = current_time - last_pose_time
            is_stale = time_since_pose > STALE_THRESHOLD_SEC
            
            if is_stale and recovery_count < MAX_RECOVERY_ATTEMPTS:
                if current_time - last_recovery_attempt > RECOVERY_COOLDOWN_SEC:
                    print(f"[CrazyflieProcess] Stale tracking, recovery {recovery_count + 1}/{MAX_RECOVERY_ATTEMPTS}")
                    with position_lock:
                        shared_status[0] = TrackingStatus.RECOVERING
                        shared_status[2] = float(recovery_count + 1)
                    
                    if reset_kalman():
                        last_pose_time = current_time
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
                shared_status[1] = last_pose_time
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
        link_uri: str = 'radio://0/40/2M/BADF00D000',
        esp32_ip: str = "172.16.0.39",
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
        
        # AI-deck streaming (runs in separate process)
        self._esp32_ip = esp32_ip
        self._esp32_port = esp32_port
        self._listen_port = listen_port
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
        print(f"[First] Takeoff (simulated)")
        print(self.move_enable)
        if not self.move_enable:
            self._is_flying = True
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
            self._is_flying = False
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
            print(
                "[CrazyflieWrapperMP] Starting AI-deck stream "
                f"{self._esp32_ip}:{self._esp32_port} "
                f"-> 0.0.0.0:{self._listen_port}"
            )
            self.frame_reader = AIDeckFrameReader(
                esp32_ip=self._esp32_ip,
                esp32_port=self._esp32_port,
                listen_port=self._listen_port,
            )
            self.frame_reader.start()
            self.stream_on = True
            logger.info("AI-deck stream started")
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            print(f"[CrazyflieWrapperMP] Failed to start AI-deck stream: {e}")
            self.stream_on = False

    def stop_stream(self):
        if not self.stream_on:
            return
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader = None
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
