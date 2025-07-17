import math
import threading
import time
import cv2
import numpy as np
from typing import Tuple, List, Optional
from djitellopy import Tello

from controller.constants import REGION_THRESHOLD
from controller.context_map.mapping.graph_manager import GraphManager

from .abs.robot_wrapper import RobotWrapper

import logging
Tello.LOGGER.setLevel(logging.WARNING)

MOVEMENT_MIN = 20
MOVEMENT_MAX = 500

SCENE_CHANGE_DISTANCE = 1000 #TODO: delete these
SCENE_CHANGE_ANGLE = 1000

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

class FrameReader:
    def __init__(self, fr):
        self.fr = fr

    @property
    def frame(self):
        frame = self.fr.frame
        frame = adjust_exposure(frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)
        
def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
    return distance

class DeadReckoningState:
    """Enhanced state tracking for dead reckoning"""
    def __init__(self):
        self.pose_history = []  # Store pose history for trajectory tracking
        self.velocity_history = []  # Store velocity history for smoothing
        self.confidence_score = 1.0  # Confidence in current position estimate
        self.drift_estimate = np.zeros(3)  # Estimated drift compensation
        self.last_correction_time = time.time()
        self.movement_commands = []  # Track movement commands for verification
        
    def add_pose(self, pose: np.ndarray, timestamp: float = None):
        """Add pose to history with timestamp"""
        if timestamp is None:
            timestamp = time.time()
        self.pose_history.append((timestamp, pose.copy()))
        
        # Keep only last 100 poses to prevent memory issues
        if len(self.pose_history) > 100:
            self.pose_history.pop(0)
    
    def add_velocity(self, velocity: np.ndarray, timestamp: float = None):
        """Add velocity to history for smoothing"""
        if timestamp is None:
            timestamp = time.time()
        self.velocity_history.append((timestamp, velocity.copy()))
        
        # Keep only last 50 velocities
        if len(self.velocity_history) > 50:
            self.velocity_history.pop(0)
    
    def get_smoothed_velocity(self, window_size: int = 5) -> np.ndarray:
        """Get smoothed velocity using moving average"""
        if len(self.velocity_history) < window_size:
            return np.zeros(3)
        
        recent_velocities = [v for _, v in self.velocity_history[-window_size:]]
        return np.mean(recent_velocities, axis=0)
    
    def estimate_drift(self):
        """Estimate position drift over time"""
        if len(self.pose_history) < 10:
            return np.zeros(3)
        
        # Simple linear drift estimation
        times = [t for t, _ in self.pose_history[-10:]]
        poses = [p for _, p in self.pose_history[-10:]]
        
        if len(times) >= 2:
            dt = times[-1] - times[0]
            if dt > 0:
                dp = poses[-1] - poses[0]
                expected_displacement = np.linalg.norm(dp)
                
                # If we've moved significantly, update drift estimate
                if expected_displacement > 0.1:  # 10cm threshold
                    self.drift_estimate = dp / dt * 0.01  # Small correction factor
        
        return self.drift_estimate

class TelloWrapper(RobotWrapper):
    KEEPALIVE_PERIOD = 4.0          # s – safely <15 s timeout of the SDK

    def __init__(self, move_enable, graph_manager: GraphManager):
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        self.drone = Tello()
        self.active_count = 0
        self.stream_on = False

        # --- keep-alive infrastructure ---------------------------------
        self._ka_stop = threading.Event()
        self._ka_thread = None

        # Enhanced odometry fields
        self.pose = np.zeros(3)          # (x, y, z) in metres
        self._yaw0 = 0.0                 # yaw at take-off
        self._last_ts = None
        self._odo_th = None
        self._odo_stop = threading.Event()
        self._inited = False
        
        # Dead reckoning enhancements
        self.dead_reckoning = DeadReckoningState()
        self.position_uncertainty = np.array([0.1, 0.1, 0.1])  # Position uncertainty (m)
        self.last_imu_update = time.time()
        self.calibration_samples = []
        
        # Movement tracking
        self.planned_movements = []  # Track planned vs actual movements
        self.movement_errors = []    # Track movement execution errors

    # ------------------------------------------------------------------
    # ENHANCED DEAD RECKONING METHODS
    # ------------------------------------------------------------------
    
    def add_planned_movement(self, movement_type: str, distance: float, expected_pose: np.ndarray):
        """Track planned movements for error analysis"""
        self.planned_movements.append({
            'type': movement_type,
            'distance': distance,
            'expected_pose': expected_pose.copy(),
            'timestamp': time.time()
        })
        
        # Keep only recent movements
        if len(self.planned_movements) > 20:
            self.planned_movements.pop(0)
    
    def analyze_movement_error(self):
        """Analyze difference between planned and actual movements"""
        if not self.planned_movements:
            return np.zeros(3)
        
        recent_movement = self.planned_movements[-1]
        current_pose = self.pose
        expected_pose = recent_movement['expected_pose']
        
        error = current_pose - expected_pose
        self.movement_errors.append(error)
        
        # Keep only recent errors
        if len(self.movement_errors) > 10:
            self.movement_errors.pop(0)
        
        return error
    
    def get_position_confidence(self) -> float:
        """Calculate confidence in current position estimate"""
        base_confidence = 1.0
        
        # Reduce confidence based on time since last known good position
        time_factor = min(1.0, (time.time() - self.dead_reckoning.last_correction_time) / 60.0)
        
        # Reduce confidence based on movement errors
        if self.movement_errors:
            error_magnitude = np.mean([np.linalg.norm(e) for e in self.movement_errors])
            error_factor = max(0.1, 1.0 - error_magnitude / 2.0)  # 2m max error
        else:
            error_factor = 1.0
        
        # Reduce confidence based on accumulated distance
        total_distance = self.get_total_distance_traveled()
        distance_factor = max(0.3, 1.0 - total_distance / 1000.0)  # 10m max distance
        
        confidence = base_confidence * (1.0 - time_factor * 0.3) * error_factor * distance_factor
        return max(0.1, min(1.0, confidence))
    
    def get_total_distance_traveled(self) -> float:
        """Calculate total distance traveled from pose history"""
        if len(self.dead_reckoning.pose_history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.dead_reckoning.pose_history)):
            prev_pose = self.dead_reckoning.pose_history[i-1][1]
            curr_pose = self.dead_reckoning.pose_history[i][1]
            total_distance += np.linalg.norm(curr_pose - prev_pose)
        
        return total_distance
    
    def apply_drift_correction(self):
        """Apply drift correction to current pose"""
        drift = self.dead_reckoning.estimate_drift()
        
        # Apply small correction to reduce accumulated drift
        correction_factor = 0.1  # Conservative correction
        self.pose -= drift * correction_factor
        
        # Update uncertainty based on correction
        self.position_uncertainty += np.abs(drift) * 0.1

    # ------------------------------------------------------------------
    # ENHANCED ODOMETRY LOOP
    # ------------------------------------------------------------------
    
    def _odometry_loop(self):
        """Enhanced odometry loop with better error handling and drift correction"""
        while not self._odo_stop.is_set():
            try:
                state = self.drone.get_current_state()
                if not state:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if self._last_ts is None:
                    self._last_ts = now
                    continue
                    
                dt = now - self._last_ts
                self._last_ts = now
                
                # Guard against stale or invalid packets
                if dt > 0.2 or dt <= 0:
                    continue

                # Get body-frame velocity and transform to world frame
                v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) / 100.0
                yaw = math.radians(state["yaw"] - self._yaw0)
                cy, sy = math.cos(yaw), math.sin(yaw)
                Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                v_world = Rz @ v_body

                # Store velocity for smoothing
                self.dead_reckoning.add_velocity(v_world, now)
                
                # Use smoothed velocity for integration
                v_smoothed = self.dead_reckoning.get_smoothed_velocity()
                
                # Integrate position
                self.pose[:2] += v_smoothed[:2] * dt
                self.pose[2] = state["h"] / 100.0  # Use direct height measurement
                
                # Add pose to history
                self.dead_reckoning.add_pose(self.pose, now)
                
                # Periodic drift correction
                if now - self.last_imu_update > 1.0:  # Every second
                    self.apply_drift_correction()
                    self.last_imu_update = now
                
                # Update graph manager with enhanced position data
                pose_cm = list(self.pose * 100.0)
                confidence = self.get_position_confidence()
                
                # Update graph manager 
                if hasattr(self.graph_manager, 'update_pose_with_confidence'):
                    self.graph_manager.update_pose_with_confidence(pose_cm, confidence) #TODO
                else:
                    self.graph_manager.update_pose(pose_cm)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[Tello] Odometry error: {e}")
                time.sleep(0.05)  # Wait longer on error

    # ------------------------------------------------------------------
    # ENHANCED MOVEMENT METHODS
    # ------------------------------------------------------------------
    
    def _execute_movement_with_tracking(self, movement_func, movement_type: str, distance: float):
        """Execute movement with enhanced tracking"""
        # Record expected pose before movement
        expected_pose = self.pose.copy()
        print(f"Position before movement: {expected_pose[0]} - {expected_pose[1]} - {expected_pose[2]}")
        
        # Calculate expected position change based on movement type
        if movement_type == "forward":
            yaw = math.radians(self.drone.get_current_state().get("yaw", 0) - self._yaw0)
            expected_pose[0] += (distance / 100.0) * math.cos(yaw)
            expected_pose[1] += (distance / 100.0) * math.sin(yaw)
        elif movement_type == "backward":
            yaw = math.radians(self.drone.get_current_state().get("yaw", 0) - self._yaw0)
            expected_pose[0] -= (distance / 100.0) * math.cos(yaw)
            expected_pose[1] -= (distance / 100.0) * math.sin(yaw)
        elif movement_type == "left":
            yaw = math.radians(self.drone.get_current_state().get("yaw", 0) - self._yaw0)
            expected_pose[0] += (distance / 100.0) * math.cos(yaw - math.pi/2)
            expected_pose[1] += (distance / 100.0) * math.sin(yaw - math.pi/2)
        elif movement_type == "right":
            yaw = math.radians(self.drone.get_current_state().get("yaw", 0) - self._yaw0)
            expected_pose[0] += (distance / 100.0) * math.cos(yaw + math.pi/2)
            expected_pose[1] += (distance / 100.0) * math.sin(yaw + math.pi/2)
        elif movement_type == "up":
            expected_pose[2] += distance / 100.0
        elif movement_type == "down":
            expected_pose[2] -= distance / 100.0
        
        # Track the planned movement
        self.add_planned_movement(movement_type, distance, expected_pose)
        
        # Execute the movement
        result = movement_func()
        
        # Wait for movement to complete and analyze error
        time.sleep(0.8)  # Allow time for odometry to update
        error = self.analyze_movement_error()
        
        # Update position uncertainty based on movement error
        self.position_uncertainty += np.abs(error) * 0.1

        print(f"Position after movement: {expected_pose[0]} - {expected_pose[1]} - {expected_pose[2]}")

        return result

    # ------------------------------------------------------------------
    # EXISTING METHODS (keeping your original implementation)
    # ------------------------------------------------------------------
    
    def _keepalive_loop(self):
        """
        Continuously send the blocking SDK command ``"command"`` every
        ``KEEPALIVE_PERIOD`` seconds.  Runs in its own daemon thread.
        """
        while not self._ka_stop.is_set():
            try:
                self.drone.send_control_command("command", timeout=3)
            except Exception as exc:
                print(f"[Tello] keep-alive failed: {exc}")
            self._ka_stop.wait(self.KEEPALIVE_PERIOD)

    def _start_keepalive(self):
        """Launch the keep-alive thread exactly once."""
        if self._ka_thread is None:
            self._ka_stop.clear()
            self._ka_thread = threading.Thread(
                target=self._keepalive_loop,
                name="tello-keepalive",
                daemon=True,
            )
            self._ka_thread.start()

    def _start_odometry(self):               
        if self._odo_th is None:
            self._odo_stop.clear()
            self._odo_th = threading.Thread(target=self._odometry_loop,
                                            daemon=True)
            self._odo_th.start()

    def get_pose(self):
        """Current (x, y, z) in cm, unchanged by land/take-off cycles."""
        return list(self.pose * 100.0)
    
    def get_pose_with_confidence(self):
        """Get current pose with confidence score"""
        return list(self.pose * 100.0), self.get_position_confidence()
    
    def get_position_uncertainty(self):
        """Get current position uncertainty in cm"""
        return list(self.position_uncertainty * 100.0)
    
    def reset_origin(self):
        self.pose[:] = 0.0
        self._yaw0 = self.drone.get_current_state().get("yaw", self._yaw0)
        self.dead_reckoning = DeadReckoningState()  # Reset dead reckoning state
        self.position_uncertainty = np.array([0.1, 0.1, 0.1])

    def keep_active(self):
        if self.active_count % 20 == 0:
            self.drone.send_control_command("command")
        self.active_count += 1

    def connect(self):
        self.drone.connect()
        self._start_keepalive()
        self._start_odometry()

    def disconnect(self):
        self._ka_stop.set()
        if self._ka_thread:
            self._ka_thread.join()

        self._odo_stop.set()
        if self._odo_th:
            self._odo_th.join()
        self.drone.end()

    def takeoff(self) -> bool:
        if not self.is_battery_good():
            return False
        if self.move_enable:
            self.drone.takeoff()
            if not self._inited:
                st = self.drone.get_current_state()
                self._yaw0 = st.get("yaw", 0.0)
                self._last_ts = None
                self._inited = True
                self.reset_origin()  # Reset dead reckoning on first takeoff
        else:
            print("[Drone] Takeoff")
        return True
    
    def land(self):
        if self.move_enable:
            self.drone.land()
        else:
            print("[Drone] Land")

    def start_stream(self):
        self.stream_on = True
        self.drone.streamon()

    def stop_stream(self):
        self.stream_on = False
        self.drone.streamoff()

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.drone.get_frame_read())

    # Enhanced movement methods with tracking
    def move_forward(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.move_forward(cap_distance(distance))
                self.movement_x_accumulator += distance
                time.sleep(0.5)
            else:
                print("[Drone] Move Forward")
            return True, distance > SCENE_CHANGE_DISTANCE
        
        return self._execute_movement_with_tracking(movement, "forward", distance)

    def move_backward(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.rotate_clockwise(180)
                self.drone.move_forward(cap_distance(distance))
                self.movement_x_accumulator -= distance
                time.sleep(0.5)
            else:
                print("[Drone] Move Backward")
            return True, distance > SCENE_CHANGE_DISTANCE
        
        return self._execute_movement_with_tracking(movement, "backward", distance)

    def move_left(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.rotate_counter_clockwise(90)
                self.drone.move_forward(cap_distance(distance))
                self.movement_y_accumulator += distance
                time.sleep(0.5)
            else:
                print("[Drone] Move Left")
            return True, distance > SCENE_CHANGE_DISTANCE
        
        return self._execute_movement_with_tracking(movement, "left", distance)

    def move_right(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.rotate_clockwise(90)
                self.drone.move_forward(50)
                self.movement_y_accumulator -= distance
                time.sleep(0.5)
            else:
                print("[Drone] Move Right")
            return True, distance > SCENE_CHANGE_DISTANCE
        
        return self._execute_movement_with_tracking(movement, "right", distance)

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.move_up(cap_distance(distance))
                time.sleep(0.5)
            else:
                print("[Drone] Move Up")
            return True, False
        
        return self._execute_movement_with_tracking(movement, "up", distance)

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        def movement():
            if self.move_enable:
                self.drone.move_down(cap_distance(distance))
                time.sleep(0.5)
            else:
                print("[Drone] Move Down")
            return True, False
        
        return self._execute_movement_with_tracking(movement, "down", distance)

    def go_to_position(self, target_x, target_y):
        """
        Move to target position in drone's current reference frame with enhanced tracking
        """
        current_yaw = self.drone.get_yaw()
        yaw_rad = np.radians(current_yaw)

        current_pos = self.get_pose()
        
        dx_world = target_x - current_pos[0]
        dy_world = target_y - current_pos[1]
        
        dx_drone = dx_world * np.cos(-yaw_rad) - dy_world * np.sin(-yaw_rad)
        dy_drone = dx_world * np.sin(-yaw_rad) + dy_world * np.cos(-yaw_rad)
        
        dx_drone = int(round(dx_drone))
        dy_drone = int(round(dy_drone))
        
        # Track this as a planned movement
        expected_pose = self.pose.copy()
        expected_pose[0] = target_x / 100.0
        expected_pose[1] = target_y / 100.0
        
        distance = np.sqrt(dx_drone**2 + dy_drone**2)
        self.add_planned_movement("go_to_position", distance, expected_pose)
        
        if self.move_enable:
            self.drone.go_xyz_speed(dx_drone, dy_drone, self.drone.get_height(), speed=20)
        else:
            print(f"[Drone] Go to position ({target_x}, {target_y})")

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.rotate_counter_clockwise(degree)
            self.rotation_accumulator += degree
            time.sleep(1)
        else:
            print("[Drone] Turn Ccw")
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.rotate_clockwise(degree)
            self.rotation_accumulator -= degree
            time.sleep(1)
        else:
            print("[Drone] Turn Cw")
        return True, False
    
    def is_battery_good(self) -> bool:
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False
    
    def create_new_trajectory(self):
        pass

    def start_trajectory(self):
        pass

    # ------------------------------------------------------------------
    # ENHANCED DIAGNOSTIC METHODS
    # ------------------------------------------------------------------
    
    def get_dead_reckoning_status(self):
        """Get comprehensive dead reckoning status"""
        return {
            'current_pose': self.get_pose(),
            'confidence': self.get_position_confidence(),
            'uncertainty': self.get_position_uncertainty(),
            'total_distance': self.get_total_distance_traveled() * 100,  # in cm
            'drift_estimate': list(self.dead_reckoning.drift_estimate * 100),  # in cm
            'pose_history_length': len(self.dead_reckoning.pose_history),
            'velocity_history_length': len(self.dead_reckoning.velocity_history),
            'movement_errors': len(self.movement_errors),
            'last_correction_time': self.dead_reckoning.last_correction_time
        }
    
    def print_dead_reckoning_status(self):
        """Print formatted dead reckoning status"""
        status = self.get_dead_reckoning_status()
        print("\n=== Dead Reckoning Status ===")
        print(f"Position: ({status['current_pose'][0]:.1f}, {status['current_pose'][1]:.1f}, {status['current_pose'][2]:.1f}) cm")
        print(f"Confidence: {status['confidence']:.2f}")
        print(f"Uncertainty: ±({status['uncertainty'][0]:.1f}, {status['uncertainty'][1]:.1f}, {status['uncertainty'][2]:.1f}) cm")
        print(f"Total distance: {status['total_distance']:.1f} cm")
        print(f"Drift estimate: ({status['drift_estimate'][0]:.1f}, {status['drift_estimate'][1]:.1f}, {status['drift_estimate'][2]:.1f}) cm")
        print(f"History length: {status['pose_history_length']} poses, {status['velocity_history_length']} velocities")
        print("============================\n")