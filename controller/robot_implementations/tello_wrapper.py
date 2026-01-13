import logging
import math
import threading
import time
from typing import Tuple, Optional

import numpy as np
from djitellopy import Tello

from controller.context_map.graph_manager import GraphManager
from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper, cap_distance
from controller.robot_implementations.virtual_robot_wrapper import CommandResult
from controller.utils.constants import REGION_THRESHOLD
from controller.utils.general_utils import adjust_exposure, print_debug, sharpen_image
from ..abs.robot_wrapper import RobotWrapper

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
Tello.LOGGER.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class FrameReader:
    """
    Lazy-eval wrapper for video frames. 
    Applies image processing only when .frame is accessed.
    """
    def __init__(self, background_frame_reader):
        self.bfr = background_frame_reader

    @property
    def frame(self):
        raw_frame = self.bfr.frame
        if raw_frame is None:
            return None
        # Educational Note: Alpha > 1 increases contrast, Beta < 0 decreases brightness
        frame = adjust_exposure(raw_frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)

# -----------------------------------------------------------------------------
# Tello Implementation
# -----------------------------------------------------------------------------
class TelloWrapper(RobotWrapper):
    """
    RobotWrapper implementation for DJI Tello.
    Handles connection, video streaming, and position estimation (Odometry).
    """
    KEEPALIVE_PERIOD = 4.0  # Seconds

    def __init__(self, move_enable: bool, graph_manager: GraphManager, 
                 use_crazyflie_lighthouse: bool = True, 
                 cf_uri: str = 'radio://0/40/2M/BADF00D002'):
        
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        
        self.drone = Tello()
        self.lock = threading.Lock()  # Protect SDK access across threads
        self.use_lighthouse = use_crazyflie_lighthouse
        
        # Crazyflie Integration for Lighthouse positioning
        self.crazyflie = None
        if self.use_lighthouse:
            try:
                self.crazyflie = CrazyflieWrapper(move_enable=False, link_uri=cf_uri)
            except Exception as e:
                logger.error(f"Crazyflie init failed: {e}")
                self.use_lighthouse = False

        self.stream_on = False
        self.active_count = 0

        # Odometry State (Meters internally, exposed as cm)
        self.position = np.zeros(4, dtype=float) 
        
        # Threading Events
        self._stop_event = threading.Event()
        self._threads = []

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def connect(self):
        with self.lock:
            self.drone.connect()
        
        # Start background tasks
        self._start_thread(self._keepalive_loop, "tello-keepalive")
        
        if self.use_lighthouse and self.crazyflie:
            self._start_thread(self._odometry_loop_crazyflie, "odo-crazyflie")
        else:
            self._start_thread(self._odometry_loop_dead_reckoning, "odo-dead-reckoning")

    def disconnect(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        
        with self.lock:
            self.drone.end()

    def _start_thread(self, target, name):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # -------------------------------------------------------------------------
    # Video Stream
    # -------------------------------------------------------------------------
    def start_stream(self):
        if not self.stream_on:
            with self.lock:
                self.drone.streamon()
            self.stream_on = True

    def stop_stream(self):
        if self.stream_on:
            with self.lock:
                self.drone.streamoff()
            self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on: 
            return None
        return FrameReader(self.drone.get_frame_read())

    # -------------------------------------------------------------------------
    # Movement Commands
    # -------------------------------------------------------------------------

    def _cap_distance_positive(distance: int) -> int:
        """
        Cap distance for directional movement commands.
        Used by: forward, back, left, right, up, down
        Range: 20-500 cm (positive only)
        """
        distance = abs(distance)
        return max(20, min(distance, 500))

    def _cap_distance_signed(distance: int) -> int:
        """
        Cap distance for go command (preserves sign).
        Used by: go_xyz_speed
        Range: -500 to 500 cm (cannot all be -20 to 20)
        """
        if -20 < distance < 20:
            return 0  # Too small for go command
        return max(-500, min(distance, 500))

    def takeoff(self) -> CommandResult:
        if not self.is_battery_good(): 
            return CommandResult(value=False, replan=False)
            
        if self.move_enable:
            with self.lock:
                self.drone.takeoff()
        else:
            print("[Drone] Takeoff (Simulated)")
        return CommandResult(value=True, replan=False)

    def land(self) -> CommandResult:
        if self.move_enable:
            with self.lock:
                self.drone.land()
        else:
            print("[Drone] Land (Simulated)")
        return CommandResult(value=True, replan=False)

    def _move_relative(self, forward=0, backward=0, left=0, right=0, up=0, down=0, yaw_cw=0, yaw_ccw=0):
        if not self.move_enable:
            print(f"[Drone] Move: F:{forward} B:{backward} L:{left} R:{right}")
            return CommandResult(value=True, replan=False)

        with self.lock:
            # Use _cap_distance_positive for directional commands
            if forward: 
                self.drone.move_forward(self._cap_distance_positive(forward))
            if backward: 
                self.drone.move_back(self._cap_distance_positive(backward))
            if left: 
                self.drone.move_left(self._cap_distance_positive(left))
            if right: 
                self.drone.move_right(self._cap_distance_positive(right))
            if up: 
                self.drone.move_up(self._cap_distance_positive(up))
            if down: 
                self.drone.move_down(self._cap_distance_positive(down))
            
            # Rotation: 1-360 degrees
            if yaw_cw: 
                self.drone.rotate_clockwise(max(1, min(yaw_cw, 360)))
            if yaw_ccw: 
                self.drone.rotate_counter_clockwise(max(1, min(yaw_ccw, 360)))
        
        time.sleep(0.5)
        return CommandResult(value=True, replan=False)

    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(forward=int(distance_cm))

    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        # Tello move_back is standard, but rotating 180 keeps "head forward" logic if desired.
        # Reverting to standard move_back for cleaner flight patterns.
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
        """Rotate to target yaw (shortest path), then move forward."""

        if not self.move_enable:
            print(f"[Drone] Rotate to {direction_deg}°, then move {distance_cm}cm")
            return CommandResult(value=True, replan=False)

        with self.lock:
            # Read and normalize current yaw
            current_yaw = self.get_position()[3] % 360
            target_yaw = direction_deg % 360

            # Compute shortest signed delta in range [-180, 180]
            delta = (target_yaw - current_yaw + 180) % 360 - 180

            # Rotate in correct direction
            if delta > 0:
                self.drone.rotate_clockwise(abs(int(delta)))
            elif delta < 0:
                self.drone.rotate_counter_clockwise(abs(int(delta)))

            time.sleep(1.0)

            # Move forward
            self.drone.move_forward(cap_distance(int(distance_cm)))

            return CommandResult(value=True, replan=False)

    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float):
        curr_x, curr_y, curr_z = self.get_position()[:3]
        
        dx = int(target_x_cm - curr_x)
        dy = int(target_y_cm - curr_y)
        dz = int(target_z_cm - curr_z)
        
        # Check SDK constraint: can't all be between -20 and 20
        if abs(dx) < 20 and abs(dy) < 20 and abs(dz) < 20:
            return CommandResult(value=True, replan=False)
        
        # Use cap_distance_signed for go command
        dx = self._cap_distance_signed(dx)
        dy = self._cap_distance_signed(dy)
        dz = self._cap_distance_signed(dz)

        if not self.move_enable:
            print(f"[Drone] GoTo: Δ({dx}, {dy}, {dz}) cm")
            return CommandResult(value=True, replan=False)

        with self.lock:
            self.drone.go_xyz_speed(dx, dy, dz, speed=20)
        
        return CommandResult(value=True, replan=False)

    # -------------------------------------------------------------------------
    # Odometry & State
    # -------------------------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float, float]:
        """Return position in cm and yaw in degrees (x, y, z. yaw)."""
        return (
            self.position[0] * 100.0,
            self.position[1] * 100.0,
            self.position[2] * 100.0,
            self.position[3],
        )

    def is_battery_good(self) -> bool:
        try:
            bat = self.drone.get_battery() # Cached value is faster
            print(f"> Battery: {bat}% {'[LOW]' if bat < 20 else '[OK]'}")
            return bat >= 20
        except:
            return True

    def _keepalive_loop(self):
        while not self._stop_event.is_set():
            try:
                with self.lock:
                    self.drone.send_control_command("command")
            except Exception as e:
                logger.debug(f"Keepalive error: {e}")
            self._stop_event.wait(self.KEEPALIVE_PERIOD)

    def _odometry_loop_crazyflie(self):
        """Position tracking via external Lighthouse system."""
        print_debug("_odometry_loop_crazyflie started")
        while not self._stop_event.is_set():
            if self.crazyflie:
                pos_m = self.crazyflie.get_pose()
                self.position = pos_m
                self.position[3] = math.degrees(self.position[3])
                if self.position[3] < 0:
                    self.position[3] = self.position[3] + 360
                if self.graph_manager:
                    self.graph_manager.update_pose(self.position[:3] * 100.0, self.position[3])
            time.sleep(0.02) # 50Hz