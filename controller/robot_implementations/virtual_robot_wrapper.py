import time
from typing import Tuple, Optional

import cv2
import numpy as np

from controller.context_map.graph_manager import GraphManager

from ..abs.robot_wrapper import RobotWrapper

CommandResult = Tuple[bool, bool]  # (ok, replan)

# -----------------------------------------------------------------------------
# Frame reader helper
# -----------------------------------------------------------------------------
class FrameReader:
    """
    Minimal frame reader that yields RGB frames from an OpenCV VideoCapture.
    """
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def frame(self):
        """Return the latest frame as RGB (or raise on failure)."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Virtual robot implementation
# -----------------------------------------------------------------------------
class VirtualRobotWrapper(RobotWrapper):
    """
    Lightweight mock robot for local testing.

    - Tracks a simple position state in centimeters: (x_cm, y_cm, z_cm).
    - Orientation is not explicitly modeled; `move_direction` uses a provided heading.
    - Video stream uses the default webcam via OpenCV when enabled.
    """
    def __init__(self, graph_manager: GraphManager, move_enable: bool = True) -> None:
        self.stream_on = False
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Pose in centimeters: (x, y, z)
        self._pos_cm = np.zeros(3, dtype=float)
        # Simple rotation accumulator for logging/telemetry only (degrees, +CW)
        self._yaw_deg = 0.0

    # -------------------------------------------------------------------------
    # RobotWrapper interface (listed first)
    # -------------------------------------------------------------------------
    def connect(self) -> None:
        """No-op for the virtual robot."""
        pass

    def disconnect(self) -> None:
        """No-op for the virtual robot."""
        pass

    def keep_active(self) -> None:
        """No-op heartbeat for the virtual robot."""
        pass

    # --- Stream control -----------------------------------------------------
    def start_stream(self):
        """Open the default webcam (device 0)."""
        self.cap = cv2.VideoCapture(0)
        self.stream_on = True

    def stop_stream(self):
        """Release the webcam if opened."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.stream_on = False

    def get_frame_reader(self):
        """Return a FrameReader if the stream is active; otherwise None."""
        if not self.stream_on or self.cap is None:
            return None
        return FrameReader(self.cap)

    # --- Flight control -----------------------------------------------------
    def takeoff(self) -> bool:
        """Always succeeds for the virtual robot."""
        print("[Virtual] Takeoff")
        return True

    def land(self) -> bool:
        """Land (no state changes other than log for the virtual robot)."""
        print("[Virtual] Land")
        return True

    def move_north(self, distance_cm: int = 10) -> CommandResult:
        """Increase Y by distance_cm."""
        d = int(distance_cm)
        print(f"-> Moving forward {d} cm")
        self._pos_cm[1] += d
        time.sleep(0.2)
        return True, False

    def move_south(self, distance_cm: int = 10) -> CommandResult:
        """Decrease Y by distance_cm."""
        d = int(distance_cm)
        print(f"-> Moving backward {d} cm")
        self._pos_cm[1] -= d
        time.sleep(0.2)
        return True, False
    
    def move_west(self, distance_cm: int = 10) -> CommandResult:
        """Increase X by distance_cm (left)."""
        d = int(distance_cm)
        print(f"-> Moving left {d} cm")
        self._pos_cm[0] += d
        time.sleep(0.2)
        return True, False
    
    def move_east(self, distance_cm: int = 10) -> CommandResult:
        """Decrease X by distance_cm (right)."""
        d = int(distance_cm)
        print(f"-> Moving right {d} cm")
        self._pos_cm[0] -= d
        time.sleep(0.2)
        return True, False
    
    def move_direction(self, direction_deg: int, distance_cm: int) -> CommandResult:
        """
        Move along a heading in the world frame.

        Parameters
        ----------
        direction_deg : int
            Heading in degrees (0° = north/+Y, +CW).
        distance_cm : int
            Travel distance in centimeters.
        """
        d = float(distance_cm)
        theta = np.deg2rad(direction_deg)
        dx = d * np.sin(theta)   # +90° (east) would decrease X in our convention
        dy = d * np.cos(theta)   # 0° points +Y
        print(f"-> Moving {d:.0f} cm at {direction_deg}° (dx={dx:.1f}, dy={dy:.1f})")
        self._pos_cm[0] += dx
        self._pos_cm[1] += dy
        time.sleep(0.2)
        return True, False

    def move_up(self, distance_cm: int) -> CommandResult:
        """Increase Z by distance_cm."""
        d = int(distance_cm)
        print(f"-> Moving up {d} cm")
        self._pos_cm[2] += d
        time.sleep(0.2)
        return True, False

    def move_down(self, distance_cm: int) -> CommandResult:
        """Decrease Z by distance_cm."""
        d = int(distance_cm)
        print(f"-> Moving down {d} cm")
        self._pos_cm[2] = max(0.0, self._pos_cm[2] - d)
        time.sleep(0.2)
        return True, False
    
    def go_to_position(self, target_x_cm: float, target_y_cm: float) -> CommandResult:
        """
        Teleport-style move to an absolute planar position (x, y) in centimeters.
        Z is unchanged.
        """
        print(f"-> Go to position ({target_x_cm:.1f}, {target_y_cm:.1f}) cm")
        self._pos_cm[0] = float(target_x_cm)
        self._pos_cm[1] = float(target_y_cm)
        time.sleep(0.2)
        return True, False
    
    def turn_ccw(self, degrees: int) -> CommandResult:
        """Increase yaw accumulator (CCW)."""
        print(f"-> Turning CCW {degrees} degrees")
        self._yaw_deg = (self._yaw_deg - degrees) % 360.0
        time.sleep(0.1 if degrees < 90 else 0.0)
        return True, False

    def turn_cw(self, degrees: int) -> CommandResult:
        """Decrease yaw accumulator (CW)."""
        print(f"-> Turning CW {degrees} degrees")
        self._yaw_deg = (self._yaw_deg + degrees) % 360.0
        time.sleep(0.1 if degrees < 90 else 0.0)
        return True, False
    
    # --- State / pose ------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float]:
        """Return (x_cm, y_cm, z_cm)."""
        x, y, z = self._pos_cm
        return (float(x), float(y), float(z))