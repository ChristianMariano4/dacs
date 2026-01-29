"""
TelloWrapper with Crazyflie Lighthouse odometry via multiprocessing.

Uses a separate process for Crazyflie position tracking while Tello handles flight.
"""

import logging
import threading
import multiprocessing as mp
from multiprocessing import Process, Array
import time
from typing import Tuple, Optional, Any
import ctypes

import numpy as np
from djitellopy import Tello

from controller.context_map.graph_manager import GraphManager
from controller.abs.robot_wrapper import RobotWrapper, CommandResult
from controller.robot_implementations.crazyflie_wrapper import (
    _spawn_ctx,
    _odometry_only_process_func,
    TrackingStatus,
)
from controller.utils.constants import REGION_THRESHOLD
from controller.utils.general_utils import adjust_exposure, sharpen_image

Tello.LOGGER.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class FrameReader:
    """Lazy-eval wrapper for Tello video frames."""
    def __init__(self, background_frame_reader):
        self.bfr = background_frame_reader

    @property
    def frame(self):
        raw_frame = self.bfr.frame
        if raw_frame is None:
            return None
        frame = adjust_exposure(raw_frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)


def _cap_distance_positive(distance: int) -> int:
    distance = abs(distance)
    return max(20, min(distance, 500))


def _cap_distance_signed(distance: int) -> int:
    if -20 < distance < 20:
        return 0
    return max(-500, min(distance, 500))


class TelloWrapper(RobotWrapper):
    """
    Tello drone with Crazyflie Lighthouse positioning.
    
    Architecture:
    - Main process: Tello flight control, video streaming
    - Subprocess: Crazyflie radio connection for Lighthouse position tracking
    - Shared memory: Position data [x, y, z, yaw] flows from subprocess to main
    """
    
    KEEPALIVE_PERIOD = 4.0
    GRAPH_UPDATE_PERIOD = 0.1

    def __init__(
        self, 
        graph_manager: GraphManager,
        move_enable: bool = True,
        use_crazyflie_lighthouse: bool = True,
        cf_uri: str = 'radio://0/40/2M/BADF00D003'
    ):
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)

        self.drone = Tello()
        self.use_lighthouse = use_crazyflie_lighthouse
        self._cf_uri = cf_uri
        
        self.stream_on = False
        self._is_flying = False

        # Shared memory for position (updated by odometry subprocess)
        self._shared_position = _spawn_ctx.Array(ctypes.c_double, 4)
        self._shared_status = _spawn_ctx.Array(ctypes.c_double, 3)
        
        for i in range(4):
            self._shared_position[i] = 0.0
        self._shared_status[0] = TrackingStatus.NOT_STARTED
        self._shared_status[1] = 0.0
        self._shared_status[2] = 0.0
        
        # Locks
        self._position_lock = _spawn_ctx.Lock()
        self._command_lock = threading.Lock()  # For Tello commands (main process only)
        
        # Events
        self._stop_event = _spawn_ctx.Event()
        self._thread_stop_event = threading.Event()
        
        # Process/thread handles
        self._odometry_process: Optional[Process] = None
        self._threads = []

    # -------------------------------------------------------------------------
    # Tracking Status
    # -------------------------------------------------------------------------
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
            time.sleep(0.5)
        return False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def connect(self, retries: int = 3) -> None:
        # Connect to Tello
        for attempt in range(retries):
            try:
                time.sleep(0.5)
                self.drone.connect()
                logger.info("Tello connected")
                break
            except Exception as e:
                logger.warning(f"Tello connection attempt {attempt + 1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(1.0)
        
        # Start keepalive thread
        self._start_thread(self._keepalive_loop, "tello-keepalive")
        
        # Start odometry subprocess if using Lighthouse
        if self.use_lighthouse:
            self._odometry_process = _spawn_ctx.Process(
                target=_odometry_only_process_func,
                args=(
                    self._shared_position,
                    self._shared_status,
                    self._stop_event,
                    self._cf_uri,
                    self._position_lock,
                ),
                name="odometry-crazyflie",
                daemon=True
            )
            self._odometry_process.start()
            print(f"[TelloWrapper] Odometry process started (PID: {self._odometry_process.pid})")
            
            if self.wait_for_tracking(timeout=15.0):
                print("[TelloWrapper] Lighthouse tracking active")
            else:
                print("[TelloWrapper] Warning: tracking not yet valid")
        
        # Start graph update thread
        if self.graph_manager is not None:
            self._start_thread(self._graph_update_loop, "graph-updater")

    def disconnect(self) -> None:
        self._stop_event.set()
        self._thread_stop_event.set()
        
        # Stop odometry process
        if self._odometry_process is not None:
            print("[TelloWrapper] Stopping odometry process...")
            self._odometry_process.join(timeout=2.0)
            if self._odometry_process.is_alive():
                self._odometry_process.terminate()
                self._odometry_process.join(timeout=1.0)
        
        # Stop threads
        for t in self._threads:
            t.join(timeout=1.0)
        
        self.drone.end()
        print("[TelloWrapper] Disconnected")

    def _start_thread(self, target, name: str):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # -------------------------------------------------------------------------
    # Video Stream
    # -------------------------------------------------------------------------
    def start_stream(self) -> None:
        if not self.stream_on:
            self.drone.streamon()
            self.stream_on = True

    def stop_stream(self) -> None:
        if self.stream_on:
            self.drone.streamoff()
            self.stream_on = False

    def get_frame_reader(self) -> Any:
        if not self.stream_on:
            return None
        return FrameReader(self.drone.get_frame_read())

    # -------------------------------------------------------------------------
    # Position
    # -------------------------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float, float]:
        """Returns (x_cm, y_cm, z_cm, yaw_deg) from Lighthouse tracking."""
        with self._position_lock:
            return (
                self._shared_position[0],
                self._shared_position[1],
                self._shared_position[2],
                self._shared_position[3],
            )

    # -------------------------------------------------------------------------
    # Flight Commands
    # -------------------------------------------------------------------------
    def takeoff(self) -> CommandResult:
        if not self._is_battery_good():
            return CommandResult(value=False, replan=False)
        
        if self.move_enable:
            with self._command_lock:
                self.drone.takeoff()
            self._is_flying = True
        else:
            print("[Tello] Takeoff (simulated)")
            self._is_flying = True
        return CommandResult(value=True, replan=False)

    def land(self) -> CommandResult:
        if self.move_enable:
            with self._command_lock:
                self.drone.land()
            self._is_flying = False
        else:
            print("[Tello] Land (simulated)")
            self._is_flying = False
        return CommandResult(value=True, replan=False)

    def _move_relative(self, forward=0, backward=0, left=0, right=0, 
                       up=0, down=0, yaw_cw=0, yaw_ccw=0) -> CommandResult:
        if not self.move_enable:
            print(f"[Tello] Move: F:{forward} B:{backward} L:{left} R:{right} (simulated)")
            return CommandResult(value=True, replan=False)

        try:
            with self._command_lock:
                if forward:
                    self.drone.move_forward(_cap_distance_positive(forward))
                if backward:
                    self.drone.move_back(_cap_distance_positive(backward))
                if left:
                    self.drone.move_left(_cap_distance_positive(left))
                if right:
                    self.drone.move_right(_cap_distance_positive(right))
                if up:
                    self.drone.move_up(_cap_distance_positive(up))
                if down:
                    self.drone.move_down(_cap_distance_positive(down))
                if yaw_cw:
                    self.drone.rotate_clockwise(max(1, min(yaw_cw, 360)))
                if yaw_ccw:
                    self.drone.rotate_counter_clockwise(max(1, min(yaw_ccw, 360)))
        except Exception as e:
            logger.error(f"Movement error: {e}")
            return CommandResult(value=False, replan=True)
        
        time.sleep(0.5)
        return CommandResult(value=True, replan=False)

    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(forward=int(distance_cm))

    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(backward=int(distance_cm))

    def move_west(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(left=int(distance_cm))

    def move_east(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(right=int(distance_cm))

    def move_up(self, distance_cm: int) -> CommandResult:
        return self._move_relative(up=int(distance_cm))

    def move_down(self, distance_cm: int) -> CommandResult:
        return self._move_relative(down=int(distance_cm))

    def turn_cw(self, degree: int) -> CommandResult:
        return self._move_relative(yaw_cw=degree)

    def turn_ccw(self, degree: int) -> CommandResult:
        return self._move_relative(yaw_ccw=degree)

    def move_direction(self, direction_deg: int, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Rotate to heading, then move forward."""
        if not self._is_flying:
            return CommandResult(value=False, replan=True)
        
        current_yaw = self.get_position()[3] % 360
        target_yaw = direction_deg % 360
        delta = (target_yaw - current_yaw + 180) % 360 - 180
        
        if delta > 5:
            self.turn_ccw(abs(int(delta)))
        elif delta < -5:
            self.turn_cw(abs(int(delta)))
        
        time.sleep(0.5)
        return self.move_north(distance_cm)

    def go_to_position(self, target_x_cm: float, target_y_cm: float, 
                       target_z_cm: float) -> CommandResult:
        curr = self.get_position()
        
        dx = _cap_distance_signed(int(target_x_cm - curr[0]))
        dy = _cap_distance_signed(int(target_y_cm - curr[1]))
        dz = _cap_distance_signed(int(target_z_cm - curr[2]))
        
        if dx == 0 and dy == 0 and dz == 0:
            return CommandResult(value=True, replan=False)
        
        if not self.move_enable:
            print(f"[Tello] go_to_position Δ({dx}, {dy}, {dz}) (simulated)")
            return CommandResult(value=True, replan=False)
        
        with self._command_lock:
            self.drone.go_xyz_speed(dx, dy, dz, speed=20)
        
        return CommandResult(value=True, replan=False)

    # -------------------------------------------------------------------------
    # Background Loops
    # -------------------------------------------------------------------------
    def _keepalive_loop(self):
        while not self._thread_stop_event.is_set():
            try:
                with self._command_lock:
                    self.drone.send_control_command("command")
            except Exception as e:
                logger.debug(f"Keepalive error: {e}")
            self._thread_stop_event.wait(self.KEEPALIVE_PERIOD)

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
            self._thread_stop_event.wait(self.GRAPH_UPDATE_PERIOD)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _is_battery_good(self) -> bool:
        try:
            with self._command_lock:
                bat = self.drone.get_battery()
            print(f"> Battery: {bat}% {'[LOW]' if bat < 20 else '[OK]'}")
            return bat >= 20
        except:
            return True

    def keep_active(self):
        pass

    def get_is_flying(self) -> bool:
        return self._is_flying
    
    def set_is_flying(self, value: bool):
        self._is_flying = value
    
    def get_move_enable(self) -> bool:
        return self.move_enable