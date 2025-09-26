import logging
import math
import threading
import time
from typing import Tuple, Optional

import cv2
import numpy as np
from djitellopy import Tello

from controller.context_map.graph_manager import GraphManager
from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper, FrameReader, cap_distance

from ..abs.robot_wrapper import RobotWrapper

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
Tello.LOGGER.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants / Types
# -----------------------------------------------------------------------------
CRAZYFLIE_LH_ENABLED = False #TODO: Add as argument by cli
REGION_THRESHOLD: int = 100 # How distant in centimeters are the regions. After that distance, a new region is automatically created

CommandResult = Tuple[bool, bool]  # (ok, replan)

# Legacy scene-change heuristics (kept for compatibility)
SCENE_CHANGE_DISTANCE = 1000  # cm  TODO: consider removing if unused
SCENE_CHANGE_ANGLE = 1000     # deg  TODO: consider removing if unused

# -----------------------------------------------------------------------------
# Tello implementation
# -----------------------------------------------------------------------------
class TelloWrapper(RobotWrapper):
    """
    RobotWrapper implementation for DJI Tello.

    Notes
    -----
    - Position is maintained internally in meters and exposed in centimeters
      via `get_position()`.
    - Orientation (yaw) is used internally for dead-reckoning (disabled by default).
    """
    KEEPALIVE_PERIOD = 4.0  # s (safe margin under SDK ~15 s timeout)

    def __init__(self, move_enable: bool, graph_manager: GraphManager, link_uri_cf='radio://0/40/2M/BADF00D002'):
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        self.drone = Tello()
        self.crazyflie_drone: Optional[CrazyflieWrapper] = None
        if CRAZYFLIE_LH_ENABLED:
            self.crazyflie_drone = CrazyflieWrapper(move_enable=False, link_uri=link_uri_cf)

        self.active_count = 0 # used to send heartbeat to drone
        self.stream_on = False

        # keep-alive infrastructure
        self._ka_stop   = threading.Event()
        self._ka_thread: Optional[threading.Thread] = None

        # Odometry (internal position in meters)
        self.position = np.zeros(3, dtype=float)          # (x, y, z) in metres
        self._yaw0: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._odo_th: Optional[threading.Thread] = None
        self._odo_stop = threading.Event()
        self._inited = False

    
    # -------------------------------------------------------------------------
    # RobotWrapper interface
    # -------------------------------------------------------------------------
    def connect(self):
        """Establish SDK connection and start keep-alive."""
        self.drone.connect()
        self._start_keepalive()
        if not CRAZYFLIE_LH_ENABLED:
            self._start_odometry()  # Use dead reckoning inseatd of Crazyflie lighthouse

    def disconnect(self):
        """Gracefully stop background threads and close the SDK session."""
        # Stop keep-alive
        self._ka_stop.set()
        if self._ka_thread:
            self._ka_thread.join(timeout=2.0)

        # Stop odometry
        self._odo_stop.set()
        if self._odo_th:
            self._odo_th.join(timeout=2.0)

        self.drone.end()

    def keep_active(self):
        """Send periodic 'command' to keep the SDK session alive."""
        if self.active_count % 20 == 0:
            try:
                self.drone.send_control_command("command")
            except Exception as exc:
                logger.warning("Keep-active failed: %s", exc)
        self.active_count += 1

    # --- Stream control -----------------------------------------------------
    def start_stream(self):
        """Start video stream."""
        self.stream_on = True
        self.drone.streamon()

    def stop_stream(self):
        """Stop video stream."""
        self.stream_on = False
        self.drone.streamoff()

    def get_frame_reader(self):
        """Return a frame reader handle, or None if stream is off."""
        if not self.stream_on:
            return None
        return FrameReader(self.drone.get_frame_read())
    
    # --- Flight control -----------------------------------------------------
    def takeoff(self) -> bool:
        """Command takeoff (True if accepted)."""
        if not self.is_battery_good():
            return False
        if self.move_enable:
            self.drone.takeoff()
            # Initialise reference yaw only on the first take-off
            if not self._inited:
                st        = self.drone.get_current_state()
                self._yaw0 = st.get("yaw", 0.0)
                self._last_ts = None             # restart integration clock
                self._inited  = True
        else:
            print("[Drone] Takeoff")
        return True
    
    def land(self) -> bool:
        """Command landing."""
        if self.move_enable:
            self.drone.land()
        else:
            print("[Drone] Land")
        return True

    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Move forward/north by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.move_forward(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Forward")
        return True, d > SCENE_CHANGE_DISTANCE

    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Move backward/south by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            # Using rotate+forward to standardize motion direction
            self.drone.rotate_clockwise(180)
            self.drone.move_forward(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Backward")
        return True, d > SCENE_CHANGE_DISTANCE

    def move_west(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Move left/west by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.rotate_counter_clockwise(90)
            self.drone.move_forward(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Left")
        return True, d > SCENE_CHANGE_DISTANCE

    def move_east(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Move right/east by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.rotate_clockwise(90)
            self.drone.move_forward(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Right")
        return True, distance > SCENE_CHANGE_DISTANCE
    
    def move_direction(self, direction_deg: int, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """
        Rotate by `direction_deg` (deg, +CW) and move forward `distance_cm` (cm).
        """
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.rotate_clockwise(direction_deg)
            self.drone.move_forward(d)
            time.sleep(0.5)
        else:
            print(f"[Drone] Move after rotating by {direction_deg} degrees")
        return True, distance_cm > SCENE_CHANGE_DISTANCE

    def move_up(self, distance_cm: int) -> CommandResult:
        """Increase altitude by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.move_up(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Up")
        return True, False

    def move_down(self, distance_cm: int) -> CommandResult:
        """Decrease altitude by `distance_cm` centimeters."""
        d = cap_distance(int(distance_cm))
        if self.move_enable:
            self.drone.move_down(d)
            time.sleep(0.5)
        else:
            print("[Drone] Move Down")
        return True, False

    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float,) -> CommandResult:
        """
        Move to the absolute planar position (target_x_cm, target_y_cm) in the world frame.

        Note
        ----
        Current implementation computes Δ in world frame and sends it as drone-frame
        offsets.
        """
        x_cm, y_cm, z_cm = self.get_position()

        # Calculate displacement
        dx_world = target_x_cm - x_cm
        dy_world = target_y_cm - y_cm
        dz_world = target_z_cm - z_cm
        
        # Convert to integers
        dx_drone = cap_distance(int(round(dx_world)))
        dy_drone = cap_distance(int(round(dy_world)))
        dz_drone = cap_distance(int(round(dy_world)))

        print(f"Drone will move by Δx={dx_world:.1f} cm, Δy={dy_world:.1f} cm, Δz={dz_world:.1f} cm")
        print(f"Position before move: {self.get_position()}")
        
        # Move
        if self.move_enable:
            self.drone.go_xyz_speed(dx_drone, dy_drone, dz_drone, speed=20)
        else:
            print(f"[Drone] go_xyz_speed({dx_drone}, {dy_drone}, {dz_drone}, 20)")

        print(f"Position after moving {self.get_position()}")
        return True, False

    def turn_ccw(self, degree: int) -> CommandResult:
        """Rotate counter-clockwise by `degrees`."""
        if self.move_enable:
            self.drone.rotate_counter_clockwise(degree)
            time.sleep(1.0)
        else:
            print("[Drone] Turn CCW")
        return True, degree > SCENE_CHANGE_ANGLE

    def turn_cw(self, degree: int) -> CommandResult:
        """Rotate clockwise by `degrees`."""
        if self.move_enable:
            self.drone.rotate_clockwise(degree)
            time.sleep(1.0)
        else:
            print("[Drone] Turn CW")
        return True, degree > SCENE_CHANGE_ANGLE

    # --- State / pose ------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float]:
        """
        Current (x_cm, y_cm, z_cm) in centimeters.
        """
        x_m, y_m, z_m = self.position
        return (x_m * 100.0, y_m * 100.0, z_m * 100.0)
    
    # -------------------------------------------------------------------------
    # Public utilities (non-interface)
    # -------------------------------------------------------------------------
    def reset_origin(self) -> None:
        """Zero the local origin and capture current yaw as reference."""
        self.position[:] = 0.0
        st = self.drone.get_current_state()
        self._yaw0   = st.get("yaw", self._yaw0)

    def is_battery_good(self) -> bool:
        """Query battery and print a status line; True if >= 20%."""
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
            return False
        print('[OK]')
        return True
    
    # -------------------------------------------------------------------------
    # Background loops (private)
    # -------------------------------------------------------------------------
    def _keepalive_loop(self) -> None:
        """Send the blocking SDK command 'command' every KEEPALIVE_PERIOD seconds."""   
        while not self._ka_stop.is_set():
            try:
                self.drone.send_control_command("command", timeout=3)
            except Exception as exc:
                logger.warning("[Tello] keep-alive failed: %s", exc)
            self._ka_stop.wait(self.KEEPALIVE_PERIOD)

    def _start_keepalive(self) -> None:
        """Launch the keep-alive thread exactly once."""
        if self._ka_thread is None:
            self._ka_stop.clear()
            self._ka_thread = threading.Thread(
                target=self._keepalive_loop,
                name="tello-keepalive",
                daemon=True,
            )
            self._ka_thread.start()

    def _start_odometry(self) -> None:   
        """Launch the odometry thread exactly once (dead reckoning by default)."""            
        if self._odo_th is None:
            self._odo_stop.clear()
            self._odo_th = threading.Thread(target=self._odometry_loop_dead_reckoning,
                                            daemon=True)
            self._odo_th.start()
    
    def _odometry_loop_dead_reckoning(self) -> None:
        """
        Integrate body-frame velocities to estimate position in world frame (meters).
        Updates self.pose and notifies GraphManager in centimeters.
        """
        while not self._odo_stop.is_set():
            state = self.drone.get_current_state()
            if not state:
                time.sleep(0.01)
                continue

            now = time.time()
            if self._last_ts is None:
                self._last_ts = now
                self._yaw0 = 0.0
                continue

            dt = now - self._last_ts
            self._last_ts = now

            # Guard against stale packets or time jumps
            if dt > 0.2 or dt <= 0:
                continue
            
            try:
                # Velocities in body frame (cm/s -> m/s)
                v_body = np.array([
                        state.get("vgx", 0) / 100.0,
                        state.get("vgy", 0) / 100.0,
                        state.get("vgz", 0) / 100.0
                    ])
                
                # Orientation (deg -> rad), yaw relative to reference
                roll = math.radians(state.get("roll", 0))
                pitch = math.radians(state.get("pitch", 0))
                yaw = math.radians(state.get("yaw", 0) - self._yaw0)

                # Rotation matrix Z-Y-X (body -> world)
                cy, sy = math.cos(yaw), math.sin(yaw)
                cp, sp = math.cos(pitch), math.sin(pitch)
                cr, sr = math.cos(roll), math.sin(roll)

                # Full rotation matrix from body to world frame
                R = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp, cp*sr, cp*cr]
                ])

                # Transform velocity to world frame
                v_world = R @ v_body
                
                # Integrate x/y; z from direct height (cm -> m)
                self.position[0] += v_world[0] * dt
                self.position[1] += v_world[1] * dt
                self.position[2] = state.get("h", 0) / 100.0

                # Notify graph manager in centimeters
                if self.graph_manager:
                    self.graph_manager.update_pose(self.get_position())

            except Exception as e:
                logger.exception("Odometry update error: %s", e)
                
            time.sleep(0.001)

    def _odometry_loop_crazyflie(self):
        """Use Crazyflie lighthouse (if enabled) to update precise position."""
        if not self.crazyflie_drone:
            return
        while not self._odo_stop.is_set():
            self.position = self.crazyflie_drone.get_pose() # expected meters
            # Convert back to cm
            if self.graph_manager:
                self.graph_manager.update_pose(self.position * 100)
            time.sleep(0.001)