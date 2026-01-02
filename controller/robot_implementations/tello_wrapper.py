import logging
import math
import threading
import time
from typing import Tuple, Optional

import numpy as np
from djitellopy import Tello

from controller.context_map.graph_manager import GraphManager
from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper, cap_distance
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
                 use_crazyflie_lighthouse: bool = False, 
                 cf_uri: str = 'radio://0/40/2M/BADF00D002'):
        
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        
        self.drone = Tello()
        self.lock = threading.Lock()  # Protect SDK access across threads
        self.use_lighthouse = use_crazyflie_lighthouse
        
        # Crazyflie Integration (Optional Lighthouse positioning)
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
        self.position = np.zeros(3, dtype=float) 
        self._yaw_offset = 0.0
        self._last_integration_ts = None
        
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
    def takeoff(self) -> Tuple[bool, bool]:
        if not self.is_battery_good(): 
            return False, False
            
        if self.move_enable:
            with self.lock:
                self.drone.takeoff()
                # Zero yaw on takeoff for relative odometry
                self._yaw_offset = self.drone.get_current_state().get("yaw", 0.0)
                self._last_integration_ts = time.time()
        else:
            print("[Drone] Takeoff (Simulated)")
        return True, False

    def land(self) -> Tuple[bool, bool]:
        if self.move_enable:
            with self.lock:
                self.drone.land()
        else:
            print("[Drone] Land (Simulated)")
        return True, False

    def _move_relative(self, forward=0, backward=0, left=0, right=0, up=0, down=0, yaw_cw=0, yaw_ccw=0):
        """Unified movement handler."""
        if not self.move_enable:
            print(f"[Drone] Move: F:{forward} B:{backward} L:{left} R:{right} U:{up} D:{down} Y:{yaw_cw}")
            return True, False

        with self.lock:
            if forward or backward or left or right or up or down:
                # Tello SDK expects cm
                self.drone.send_rc_control(left - right, forward - backward, up - down, yaw_cw - yaw_ccw)
                # Duration is approximate based on speed; better to use specific move commands for precision
                # For simplicity here using SDK high-level commands which block:
                pass 
            
            # Using blocking commands for reliability in this implementation
            if forward: self.drone.move_forward(cap_distance(forward))
            if backward: self.drone.move_back(cap_distance(backward))
            if left: self.drone.move_left(cap_distance(left))
            if right: self.drone.move_right(cap_distance(right))
            if up: self.drone.move_up(cap_distance(up))
            if down: self.drone.move_down(cap_distance(down))
            if yaw_cw: self.drone.rotate_clockwise(yaw_cw)
            if yaw_ccw: self.drone.rotate_counter_clockwise(yaw_ccw)
            
        time.sleep(0.5) # Settle time
        return True, False

    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> Tuple[bool, bool]:
        return self._move_relative(forward=int(distance_cm))

    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> Tuple[bool, bool]:
        # Tello move_back is standard, but rotating 180 keeps "head forward" logic if desired.
        # Reverting to standard move_back for cleaner flight patterns.
        return self._move_relative(backward=int(distance_cm))

    def move_west(self, distance_cm: int = REGION_THRESHOLD) -> Tuple[bool, bool]:
        return self._move_relative(left=int(distance_cm))

    def move_east(self, distance_cm: int = REGION_THRESHOLD) -> Tuple[bool, bool]:
        return self._move_relative(right=int(distance_cm))

    def move_up(self, distance_cm: int) -> Tuple[bool, bool]:
        return self._move_relative(up=int(distance_cm))

    def move_down(self, distance_cm: int) -> Tuple[bool, bool]:
        return self._move_relative(down=int(distance_cm))

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        return self._move_relative(yaw_cw=degree)

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        return self._move_relative(yaw_ccw=degree)
    
    def move_direction(self, direction_deg: int, distance_cm: int = REGION_THRESHOLD) -> Tuple[bool, bool]:
        """Rotate to target yaw (shortest path), then move forward."""

        if not self.move_enable:
            print(f"[Drone] Rotate to {direction_deg}°, then move {distance_cm}cm")
            return True, False

        with self.lock:
            # Read and normalize current yaw
            current_yaw = self.drone.get_position()[3] % 360
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

        return True, False

    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float) -> Tuple[bool, bool]:
        """Move to absolute world coordinates (blocking)."""
        curr_x, curr_y, curr_z = self.get_position()
        
        dx = cap_distance(int(target_x_cm - curr_x))
        dy = cap_distance(int(target_y_cm - curr_y))
        dz = cap_distance(int(target_z_cm - curr_z))

        if not self.move_enable:
            print(f"[Drone] GoTo: Δ({dx}, {dy}, {dz})")
            return True, False

        with self.lock:
            # go_xyz_speed is relative to current position
            self.drone.go_xyz_speed(dx, dy, dz, speed=20)
        
        return True, False

    # -------------------------------------------------------------------------
    # Odometry & State
    # -------------------------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float, float]:
        """Return position in cm and yaw in degrees (x, y, z. yaw)."""
        return tuple(self.position[0] * 100.0, 
                     self.position[1] * 100.0,
                     self.position[2] * 100.0,
                     self.position[3])
        

    def keep_active(self):
        """Heartbeat to keep session alive during idle times."""
        self.active_count += 1
        if self.active_count % 20 == 0:
            # Queue a non-blocking read command to keep socket active
            # We don't use the thread lock here to avoid blocking the main loop
            try:
                self.drone.send_command_without_return("command") 
            except: pass

    def is_battery_good(self) -> bool:
        try:
            bat = self.drone.get_battery() # Cached value is faster
            print(f"> Battery: {bat}% {'[LOW]' if bat < 20 else '[OK]'}")
            return bat >= 20
        except:
            return True # Assume OK if read fails to avoid lock-out

    def _keepalive_loop(self):
        while not self._stop_event.is_set():
            try:
                with self.lock:
                    self.drone.send_control_command("command")
            except Exception as e:
                logger.debug(f"Keepalive error: {e}")
            self._stop_event.wait(self.KEEPALIVE_PERIOD)

    def _odometry_loop_dead_reckoning(self):
        """
        Integrates IMU velocity to estimate position.
        Math: v_world = R(yaw) * v_body
        """
        while not self._stop_event.is_set():
            try:
                state = self.drone.get_current_state()
                if not state:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if self._last_integration_ts is None:
                    self._last_integration_ts = now
                    continue

                dt = now - self._last_integration_ts
                self._last_integration_ts = now

                if dt > 0.2 or dt <= 0: continue # Skip time jumps

                # 1. Get Velocities (dm/s or cm/s -> convert to m/s)
                # Tello state vgx/vgy are usually in dm/s (decimeters/s) or cm/s depending on firmware.
                # Standardize to meters/s. Assuming cm/s here based on common Tello SDKs.
                vx_b = state.get("vgx", 0) / 100.0
                vy_b = state.get("vgy", 0) / 100.0
                
                # 2. Get Orientation (Yaw)
                # Tello yaw is CW positive in degrees. Convert to Radians CCW standard math or match frame.
                # Assuming standard: +X Forward, +Y Left. Tello: +X Forward, +Y Right?
                # We normalize yaw relative to takeoff heading.
                yaw_deg = state.get("yaw", 0) - self._yaw_offset
                yaw_rad = math.radians(yaw_deg)

                # 3. Rotate Body Velocity to World Velocity (2D Rotation)
                # vx_world = vx_b * cos(yaw) - vy_b * sin(yaw)
                # vy_world = vx_b * sin(yaw) + vy_b * cos(yaw)
                
                # Tello Reference: usually pitch/roll influence is small for hovering/slow flight, 
                # so full 3D rotation matrix (R) is often overkill for basic tracking.
                # Simplified 2D rotation:
                c, s = math.cos(yaw_rad), math.sin(yaw_rad)
                vx_w = vx_b * c - vy_b * s
                vy_w = vx_b * s + vy_b * c

                # 4. Integrate
                self.position[0] += vx_w * dt
                self.position[1] += vy_w * dt
                self.position[2] = state.get("h", 0) / 100.0 # Barometer height is absolute

                if self.graph_manager:
                    self.graph_manager.update_pose(self.get_position())

            except Exception as e:
                pass # Suppress transient math errors
            
            time.sleep(0.01) # 100Hz update

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