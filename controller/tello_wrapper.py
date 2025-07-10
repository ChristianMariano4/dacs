import math
import threading
import math
import threading
from controller import utils
import time, cv2
import numpy as np
from typing import Tuple
from djitellopy import Tello

from controller.constants import REGION_THRESHOLD
from controller.context_map.mapping.graph_manager import GraphManager
from utils.utils import process_trajectory, save_trajectory_csv, save_waypoints_csv

from .abs.robot_wrapper import RobotWrapper

import logging
Tello.LOGGER.setLevel(logging.WARNING)

MOVEMENT_MIN = 20
MOVEMENT_MAX = 500

SCENE_CHANGE_DISTANCE = 120
SCENE_CHANGE_ANGLE = 90

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

class FrameReader:
    def __init__(self, fr):
        # Initialize the video capture
        self.fr = fr

    @property
    def frame(self):
        # Read a frame from the video capture
        frame = self.fr.frame
        frame = adjust_exposure(frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)
        
def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
    return distance

class TelloWrapper(RobotWrapper):
    KEEPALIVE_PERIOD   = 4.0          # s – safely <15 s timeout of the SDK

    def __init__(self, move_enable, graph_manager: GraphManager):
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)
        self.drone = Tello()
        self.active_count = 0
        self.stream_on = False

        # --- keep-alive infrastructure ---------------------------------
        self._ka_stop   = threading.Event()
        self._ka_thread = None

        # odometry fields
        self.pose = np.zeros(3)          # (x, y, z) in metres
        self._yaw0 = 0.0                 # yaw at take-off
        self._last_ts = None
        self._odo_th = None
        self._odo_stop = threading.Event()
        self._inited    = False

        # trajectory fields
        self._recording = False
        self._recorded_waypoints = []
    # ------------------------------------------------------------------
    # KEEP-ALIVE LOOP
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
            # wait, but bail out early if stop was requested
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
    
    def _odometry_loop(self):
        while not self._odo_stop.is_set():
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
            if dt > 0.2:                     # guard against stale packets
                continue

            # body-frame velocity → world frame → integrate
            v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) / 100.0
            yaw = math.radians(state["yaw"] - self._yaw0)
            cy, sy = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            v_world = Rz @ v_body

            self.pose[:2] += v_world[:2] * dt
            self.pose[2]   = state["h"] / 100.0

            self.graph_manager.update_pose(list(self.pose * 100.0))
            time.sleep(0.01)

    def _start_odometry(self):               
        if self._odo_th is None:             # start exactly *once*
            self._odo_stop.clear()
            self._odo_th = threading.Thread(target=self._odometry_loop,
                                            daemon=True)
            self._odo_th.start()

    def get_pose(self):
        """Current (x, y, z) in cm, unchanged by land/take-off cycles."""
        return list(self.pose * 100.0)
    
    def reset_origin(self):
        self.pose[:] = 0.0
        self._yaw0   = self.drone.get_current_state().get("yaw", self._yaw0)

    def keep_active(self):
        if self.active_count % 20 == 0:
            self.drone.send_control_command("command")
        self.active_count += 1

    def connect(self):
        self.drone.connect()
        self._start_keepalive()
        self._start_odometry()

    def disconnect(self):
        # stop keep-alive
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
            # initialise reference yaw only on the *first* take-off
            if not self._inited:                 #  <<< CHANGED
                st        = self.drone.get_current_state()
                self._yaw0 = st.get("yaw", 0.0)
                self._last_ts = None             # restart integration clock
                self._inited  = True
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

    def move_forward(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.move_forward(cap_distance(distance))
            self.movement_x_accumulator += distance
            time.sleep(0.5)
        else:
            print("[Drone] Move Forward")
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_backward(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        if self.move_enable:
            # self.drone.move_back(cap_distance(distance))
            self.drone.rotate_clockwise(180)
            self.drone.move_forward(cap_distance(distance))
            self.movement_x_accumulator -= distance
            time.sleep(0.5)
        else:
            print("[Drone] Move Backward")
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_left(self, distance: int =int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        if self.move_enable:
            # self.drone.move_left(cap_distance(distance))
            self.drone.rotate_counter_clockwise(90)
            self.drone.move_forward(cap_distance(distance))
            self.movement_y_accumulator += distance
            time.sleep(0.5)
        else:
            print("[Drone] Move Left")
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_right(self, distance: int = int(REGION_THRESHOLD)) -> Tuple[bool, bool]:
        if self.move_enable:
            # self.drone.move_right(cap_distance(distance))
            self.drone.rotate_clockwise(90)
            self.drone.move_forward(50)
            self.movement_y_accumulator -= distance
            time.sleep(0.5)
        else:
            print("[Drone] Move Right")
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.move_up(cap_distance(distance))
            time.sleep(0.5)
        else:
            print("[Drone] Move Up")
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.move_down(cap_distance(distance))
            time.sleep(0.5)
        else:
            print("[Drone] Move Down")
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.rotate_counter_clockwise(degree)
            self.rotation_accumulator += degree
            time.sleep(1)
        else:
            print("[Drone] Turn Ccw")
        # return True, degree > SCENE_CHANGE_ANGLE
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        if self.move_enable:
            self.drone.rotate_clockwise(degree)
            self.rotation_accumulator -= degree
            time.sleep(1)
        else:
            print("[Drone] Turn Cw")
        # return True, degree > SCENE_CHANGE_ANGLE
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
    
    def create_new_trajectory(self, gesture, duration_s=15)-> Tuple[bool, bool]:
        """Create a new trajectory by recording current flight path"""
        utils.print_t(f"New trajectory creation ... associated with gesture {gesture}")

        # Land if currently flying
        if self.move_enable and self.is_flying:
            self.land(height=0.0, duration=4.5)
            self.is_flying = False
            time.sleep(3.0)

        # Start recording
        time.sleep(62.0)
        self._record_trajectory(duration_s)

        # Process and store trajectory
        if self._recorded_waypoints:
            processed_trajectory = process_trajectory(self._recorded_waypoints)
            self.gesture_trajectory_mapping[gesture] = processed_trajectory
            print(f'Trajectory recorded with {len(processed_trajectory)} waypoints')
            print(f'DICTIONARY: {list(self.gesture_trajectory_mapping.keys())}')
            return True, False
        else:
            print("No trajectory data recorded")
            return False, False

    def _record_trajectory(self, duration_s):
        """Record trajectory using your odometry system"""
        self._recorded_waypoints = []

        # Countdown
        for count in ['3', '2', '1']:
            print(count)
            time.sleep(1)

        print('Start recording...')
        
        # Take off for recording
        # self.drone.takeoff()
        # self.is_flying = True
        # time.sleep(2.0)  # Wait for stable hover

        # Start recording
        self._recording = True
        start_time = time.time()
        end_time = start_time + duration_s

        while time.time() < end_time and self._recording:
            # Get current pose from your odometry system
            pose = self.get_pose() # Returns [x, y, z] in cm
            state = self.drone.get_current_state()

            if pose and state:
                timestamp = time.time() - start_time
                waypoint = {
                    't': timestamp,
                    'x': pose[0] / 100.0, # Convert to meters
                    'y': pose[1] / 100.0,
                    'z': pose[2] / 100.0,
                    'yaw': math.radians(state.get('yaw', 0))
                }
                self._recorded_waypoints.append(waypoint)
            
            time.sleep(0.1) # 10Hz recording rate
        
        self._recording = False
        print(f"Recording complete. Captured {len(self._recorded_waypoints)} waypoints")
        save_trajectory_csv(self._recorded_waypoints, 'recorded_trajectory.csv')

    def _execute_trajectory(self, trajectory):
        """Execute trajectory - handles both polynomial and waypoint formats"""
        
        # Check if trajectory is polynomial (uav_trajectory.Trajectory object)
        if hasattr(trajectory, 'polynomials') and hasattr(trajectory, 'duration'):
            self._execute_polynomial_trajectory(trajectory)
        else:
            # Fallback to waypoint execution
            self._execute_waypoint_trajectory(trajectory)


    def _execute_polynomial_trajectory(self, trajectory):
        """Execute polynomial trajectory with high precision"""
        print(f"Executing polynomial trajectory (duration: {trajectory.duration:.2f}s)")

        # Get starting position
        start_pose = self.get_pose()
        if not start_pose:
            print("Cannot get current position")
            return
        
        start_time = time.time()
        dt = 0.1 # 10Hz execution rate

        while True:
            elapsed = time.time() -start_time

            # Check if trajectory is complete
            if elapsed >= trajectory.duration:
                print("Polynomial trajectory execution complete")
                break

            # Evaluate trajectory at current time
            try:
                target_state = trajectory.eval(elapsed)

                # Get current position
                current_pose = self.get_pose()
                if not current_pose:
                    print("Lost position tracking")
                    break
                
                curr_x, curr_y, curr_z = current_pose[0]/100.0,  current_pose[1]/100.0, current_pose[2]/100.0
                # Calculate movement to target position (in cm for Tello)
                dx = int((target_state.pos[0] - curr_x) * 100)
                dy = int((target_state.pos[1] - curr_y) * 100)
                dz = int((target_state.pos[2] - curr_z) * 100)

                # Only send movement commands if significant displacement
                if abs(dx) > 15 or abs(dy) > 15 or abs(dz) > 15:
                    try:
                        # Calculate speed based on desired velocity
                        vel_magnitude = np.linalg.norm(target_state.vel) * 100  # m/s to cm/s
                        speed = max(20, min(100, int(vel_magnitude))) # Clamp between 20-100 cm/s

                        self.drone.go_xyz_speed(dx, dy, dz, speed)
                    except Exception as e:
                        print(f"Movement command failed: {e}")
                
                # Handle yaw
                current_state = self.drone.get_current_state()
                if current_state:
                    current_yaw = math.radians(current_state.get('yaw', 0))
                    target_yaw = target_state.yaw

                    yaw_diff = target_yaw - current_yaw
                    # Normalize yaw difference
                    while yaw_diff > math.pi:
                        yaw_diff -= 2 * math.pi
                    while yaw_diff < -math.pi:
                        yaw_diff += 2 * math.pi

                    yaw_degrees = int(math.degrees(yaw_diff))
                    if abs(yaw_degrees) > 15:
                        try:
                            if yaw_degrees > 0:
                                self.turn_ccw(abs(yaw_degrees))
                            else:
                                self.turn_cw(abs(yaw_degrees))
                        except Exception as e:
                            print(f"Rotation failed: {e}")

            except Exception as e:
                print(f"Trajectory evaluation failed at t={elapsed:.2f}: {e}")
                break
            
            time.sleep(dt)

    def _execute_waypoint_trajectory(self, trajectory):
        """Execute trajectory using discrete waypoints (fallback method)"""
        """Execute trajectory using discrete waypoints (fallback method)"""
        if len(trajectory) < 2:
            print("Trajectory too short to execute")
            return
        
        # Get starting position
        start_pose = self.get_pose()
        if not start_pose:
            print("Cannot get current position")
            return
        
        start_x, start_y, start_z = start_pose[0]/100.0, start_pose[1]/100.0, start_pose[2]/100.0
        
        print(f"Starting waypoint trajectory execution from ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")
        
        # Execute each waypoint
        for i, waypoint in enumerate(trajectory[1:], 1):  # Skip first waypoint (starting position)
            target_x = waypoint['x']
            target_y = waypoint['y'] 
            target_z = waypoint['z']
            target_yaw = waypoint['yaw']
            
            # Calculate relative movement from current position
            current_pose = self.get_pose()
            if not current_pose:
                print("Lost position tracking")
                break
                
            curr_x, curr_y, curr_z = current_pose[0]/100.0, current_pose[1]/100.0, current_pose[2]/100.0
            
            # Calculate movement in cm (Tello units)
            dx = int((target_x - curr_x) * 100)
            dy = int((target_y - curr_y) * 100)
            dz = int((target_z - curr_z) * 100)
            
            print(f"Waypoint {i}: Moving ({dx}, {dy}, {dz}) cm")
            
            # Execute movement if significant
            if abs(dx) > 20 or abs(dy) > 20 or abs(dz) > 20:
                try:
                    # Use go_xyz_speed for smooth movement
                    self.drone.go_xyz_speed(dx, dy, dz, 50)  # 50 cm/s speed
                    time.sleep(2.0)  # Wait for movement to complete
                except Exception as e:
                    print(f"Movement failed: {e}")
                    break
            
            # Handle yaw rotation
            current_state = self.drone.get_current_state()
            if current_state:
                current_yaw = math.radians(current_state.get('yaw', 0))
                yaw_diff = target_yaw - current_yaw
                
                # Normalize yaw difference to [-pi, pi]
                while yaw_diff > math.pi:
                    yaw_diff -= 2 * math.pi
                while yaw_diff < -math.pi:
                    yaw_diff += 2 * math.pi
                
                yaw_degrees = int(math.degrees(yaw_diff))
                if abs(yaw_degrees) > 10:  # Only rotate if significant
                    try:
                        if yaw_degrees > 0:
                            self.turn_ccw(abs(yaw_degrees))
                        else:
                            self.turn_cw(abs(yaw_degrees))
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"Rotation failed: {e}")
        
        print("Waypoint trajectory execution complete")

    def start_trajectory(self, gesture) -> Tuple[bool, bool]:
        """Execute a recorded trajectory"""
        print('[BASIC_COMMANDS] Start trajectory command received')
        
        trajectory = self.gesture_trajectory_mapping.get(gesture)
        if not trajectory:
            print(f"No trajectory found for gesture: {gesture}")
            return False, False
        print(f"Executing trajectory with {len(trajectory)} waypoints")

        # Take off if not flying
        if not self.is_flying:
            self.takeoff()
            self.is_flying = True
            time.sleep(2.0)

        # Execute trajectory

        duration = self._upload_trajectory(trajectory_id, traj) #duration=around 15s
        print('Done! duration: ', duration)
        self._run_sequence(trajectory_id=1, duration=duration)
        # time.sleep(duration)
        return True, False