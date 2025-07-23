#!/usr/bin/env python3
"""
Corrected Tello Drone Controller with improved dead reckoning
This script provides accurate position tracking for DJI Tello drone
All coordinates are in centimeters for consistency
"""

from djitellopy import Tello
import time
import cv2
import threading
import numpy as np
import math

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.is_flying = False
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, z in centimeters
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz in cm/s
        self.current_yaw = 0  # degrees
        self._yaw0 = 0  # initial yaw offset
        self._last_ts = None
        self._odo_stop = threading.Event()
        self._odo_thread = None
        
        # Dead reckoning improvements
        self.acceleration_history = []
        self.velocity_filter_alpha = 0.7  # Low-pass filter for velocity
        self.position_correction_factor = 0.98  # Drift correction
        
    def connect(self):
        """Connect to the Tello drone"""
        try:
            self.tello.connect()
            print(f"Connected to Tello")
            print(f"Battery: {self.tello.get_battery()}%")
            print(f"Temperature: {self.tello.get_temperature()}°C")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def takeoff(self):
        """Take off the drone"""
        if not self.is_flying:
            try:
                self.tello.takeoff()
                self.is_flying = True
                self.set_position_origin()
                self.start_odometry()
                print("Takeoff successful")
                time.sleep(3)  # Wait for stabilization
            except Exception as e:
                print(f"Takeoff failed: {e}")
    
    def land(self):
        """Land the drone"""
        if self.is_flying:
            try:
                self.stop_odometry()
                self.tello.land()
                self.is_flying = False
                print("Landing successful")
            except Exception as e:
                print(f"Landing failed: {e}")
    
    def move_forward(self, distance=30):
        """Move forward by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_forward(distance)
                print(f"Moved forward {distance}cm")
                time.sleep(1)  # Wait for movement completion
            except Exception as e:
                print(f"Forward movement failed: {e}")
    
    def move_backward(self, distance=30):
        """Move backward by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_back(distance)
                print(f"Moved backward {distance}cm")
                time.sleep(1)
            except Exception as e:
                print(f"Backward movement failed: {e}")
    
    def move_left(self, distance=30):
        """Move left by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_left(distance)
                print(f"Moved left {distance}cm")
                time.sleep(1)
            except Exception as e:
                print(f"Left movement failed: {e}")
    
    def move_right(self, distance=30):
        """Move right by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_right(distance)
                print(f"Moved right {distance}cm")
                time.sleep(1)
            except Exception as e:
                print(f"Right movement failed: {e}")
    
    def move_up(self, distance=30):
        """Move up by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_up(distance)
                print(f"Moved up {distance}cm")
                time.sleep(1)
            except Exception as e:
                print(f"Up movement failed: {e}")
    
    def move_down(self, distance=30):
        """Move down by specified distance (cm)"""
        if self.is_flying:
            try:
                self.tello.move_down(distance)
                print(f"Moved down {distance}cm")
                time.sleep(1)
            except Exception as e:
                print(f"Down movement failed: {e}")
    
    def rotate_clockwise(self, degrees=90):
        """Rotate clockwise by specified degrees"""
        if self.is_flying:
            try:
                self.tello.rotate_clockwise(degrees)
                print(f"Rotated clockwise {degrees}°")
                time.sleep(1)
            except Exception as e:
                print(f"Clockwise rotation failed: {e}")
    
    def rotate_counter_clockwise(self, degrees=90):
        """Rotate counter-clockwise by specified degrees"""
        if self.is_flying:
            try:
                self.tello.rotate_counter_clockwise(degrees)
                print(f"Rotated counter-clockwise {degrees}°")
                time.sleep(1)
            except Exception as e:
                print(f"Counter-clockwise rotation failed: {e}")
    
    def flip_forward(self):
        """Perform forward flip"""
        if self.is_flying:
            try:
                self.tello.flip_forward()
                print("Forward flip executed")
                time.sleep(2)  # Longer wait for flip recovery
            except Exception as e:
                print(f"Forward flip failed: {e}")
    
    def flip_backward(self):
        """Perform backward flip"""
        if self.is_flying:
            try:
                self.tello.flip_back()
                print("Backward flip executed")
                time.sleep(2)
            except Exception as e:
                print(f"Backward flip failed: {e}")
    
    def get_pose(self):
        """Get current position estimate (x, y, z in cm)"""
        return self.pose.copy()  # Return copy to prevent external modification
    
    def get_velocity(self):
        """Get current velocity estimate (vx, vy, vz in cm/s)"""
        return self.velocity.copy()
    
    def odometry_loop(self):
        """Improved odometry loop with better dead reckoning"""
        velocity_filtered = np.array([0.0, 0.0, 0.0])
        last_valid_state = None
        
        while not self._odo_stop.is_set():
            try:
                state = self.tello.get_current_state()
                if not state:
                    time.sleep(0.01)
                    continue
                    
                now = time.time()
                if self._last_ts is None:
                    self._last_ts = now
                    last_valid_state = state
                    continue
                    
                dt = now - self._last_ts
                self._last_ts = now
                
                # Guard against stale packets or too large time steps
                if dt > 0.5 or dt < 0.005:
                    continue
                
                # Get body-frame velocity (already in cm/s from djitellopy)
                v_body = np.array([state["vgx"], state["vgy"], state["vgz"]])
                
                # Check if we have significant velocity or position change
                has_velocity = np.linalg.norm(v_body) > 1.0  # > 1 cm/s
                
                # If no velocity but state changed, estimate movement from state differences
                if not has_velocity and last_valid_state:
                    # Use mission pad data if available for position updates
                    if ("mpx" in state and "mpy" in state and 
                        state["mpx"] != -1 and state["mpy"] != -1 and
                        "mpx" in last_valid_state and "mpy" in last_valid_state):
                        
                        # Mission pad coordinates (if available)
                        dx_mp = (state["mpx"] - last_valid_state["mpx"]) * 10  # Convert to cm
                        dy_mp = (state["mpy"] - last_valid_state["mpy"]) * 10
                        
                        self.pose[0] += dx_mp
                        self.pose[1] += dy_mp
                        print(f"Mission pad update: dx={dx_mp:.1f}, dy={dy_mp:.1f}")
                
                # Apply velocity filtering to reduce noise
                velocity_filtered = (self.velocity_filter_alpha * v_body + 
                                   (1 - self.velocity_filter_alpha) * velocity_filtered)
                
                # Transform to world frame
                yaw_rad = math.radians(state["yaw"] - self._yaw0)
                cos_yaw = math.cos(yaw_rad)
                sin_yaw = math.sin(yaw_rad)
                
                # Rotation matrix for yaw (Z-axis rotation)
                v_world = np.array([
                    velocity_filtered[0] * cos_yaw - velocity_filtered[1] * sin_yaw,
                    velocity_filtered[0] * sin_yaw + velocity_filtered[1] * cos_yaw,
                    velocity_filtered[2]
                ])
                
                # Store filtered velocity
                self.velocity = v_world.copy()
                
                # Integrate velocity to get position change
                position_delta = v_world * dt
                
                # Only integrate if we have meaningful velocity
                if np.linalg.norm(position_delta[:2]) > 0.1:  # > 1mm movement
                    self.pose[:2] += position_delta[:2] * self.position_correction_factor
                
                # Height comes directly from barometer (more accurate)
                self.pose[2] = state["h"]
                
                # Update current yaw
                self.current_yaw = state["yaw"]
                
                # Debug output for velocity integration
                if np.linalg.norm(v_body) > 5.0:  # Only show when moving
                    print(f"Velocity: {v_body} -> World: {v_world[:2]} -> Delta: {position_delta[:2]}")
                
                # Store last valid state
                last_valid_state = state.copy()
                
                # Optional: Store acceleration history for future improvements
                if len(self.acceleration_history) > 10:
                    self.acceleration_history.pop(0)
                
                # Calculate acceleration from velocity change
                if hasattr(self, '_last_velocity'):
                    acceleration = (v_world - self._last_velocity) / dt
                    self.acceleration_history.append(acceleration)
                
                self._last_velocity = v_world.copy()
                
            except Exception as e:
                print(f"Odometry error: {e}")
                
            time.sleep(0.01)  # 100 Hz update rate
    
    def start_odometry(self):
        """Start the odometry tracking thread"""
        if self._odo_thread is None or not self._odo_thread.is_alive():
            self._odo_stop.clear()
            self._odo_thread = threading.Thread(target=self.odometry_loop, daemon=True)
            self._odo_thread.start()
            print("Odometry tracking started")
    
    def stop_odometry(self):
        """Stop the odometry tracking thread"""
        if self._odo_thread and self._odo_thread.is_alive():
            self._odo_stop.set()
            self._odo_thread.join(timeout=1.0)
            print("Odometry tracking stopped")
    
    def go_to_position(self, target_x, target_y):
        """
        Move to target position in world coordinates (cm)
        CORRECTED VERSION with manual position update
        """
        if not self.is_flying:
            print("Cannot move to position: drone is not flying")
            return False
        
        try:
            # Get current position (already in cm)
            current_pos = self.get_pose()
            current_x = current_pos[0]  # NO coordinate swapping
            current_y = current_pos[1]  # NO unit conversion (already cm)
            
            print(f"Current position: ({current_x:.1f}, {current_y:.1f}, {current_pos[2]:.1f})cm")
            print(f"Target position: ({target_x}, {target_y})cm")
            
            # Calculate displacement in world frame (cm)
            dx = target_x - current_x
            dy = target_y - current_y
            
            print(f"Required displacement: dx={dx:.1f}cm, dy={dy:.1f}cm")
            
            # Convert to integers and clamp to safe ranges
            dx_clamped = int(round(np.clip(dx, -500, 500)))
            dy_clamped = int(round(np.clip(dy, -500, 500)))
            
            # Skip movement if already at target
            if abs(dx_clamped) < 10 and abs(dy_clamped) < 10:
                print(f"Already at target position ({target_x}, {target_y})")
                return True
            
            # Store position before movement
            pre_move_pos = self.pose.copy()
            
            # Use go_xyz_speed if available (more accurate)
            if hasattr(self.tello, 'go_xyz_speed'):
                print(f"Using go_xyz_speed: dx={dx_clamped}cm, dy={dy_clamped}cm")
                self.tello.go_xyz_speed(dx_clamped, dy_clamped, 0, speed=30)
                
                # Wait for movement with progress monitoring
                expected_duration = max(abs(dx_clamped), abs(dy_clamped)) / 30 + 1
                self._wait_for_movement_completion(expected_duration)
                
                # Manually update position since go_xyz_speed doesn't trigger velocity updates
                self.pose[0] += dx_clamped
                self.pose[1] += dy_clamped
                print(f"Position manually updated by: ({dx_clamped}, {dy_clamped})")
                
            else:
                # Fallback to individual moves
                print(f"Using individual moves: dx={dx_clamped}cm, dy={dy_clamped}cm")
                
                # Move in X direction (forward/backward)
                if abs(dx_clamped) >= 20:
                    if dx_clamped > 0:
                        self.tello.move_forward(abs(dx_clamped))
                        self.pose[0] += abs(dx_clamped)
                    else:
                        self.tello.move_back(abs(dx_clamped))
                        self.pose[0] -= abs(dx_clamped)
                    time.sleep(1)
                
                # Move in Y direction (left/right)
                if abs(dy_clamped) >= 20:
                    if dy_clamped > 0:
                        self.tello.move_right(abs(dy_clamped))
                        self.pose[1] += abs(dy_clamped)
                    else:
                        self.tello.move_left(abs(dy_clamped))
                        self.pose[1] -= abs(dy_clamped)
                    time.sleep(1)
            
            # Verify final position
            final_pos = self.get_pose()
            print(f"Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f}, {final_pos[2]:.1f})cm")
            
            return True
            
        except Exception as e:
            print(f"Failed to move to position: {e}")
            return False
    
    def _wait_for_movement_completion(self, expected_duration):
        """Wait for movement to complete with velocity monitoring"""
        start_time = time.time()
        last_check = start_time
        movement_detected = False
        
        while time.time() - start_time < expected_duration:
            # Check if drone is moving
            current_velocity = np.linalg.norm(self.velocity[:2])
            
            if current_velocity > 5.0:  # Moving faster than 5 cm/s
                movement_detected = True
                last_check = time.time()
            elif movement_detected and current_velocity < 2.0:  # Stopped moving
                print("Movement completed (velocity-based detection)")
                break
                
            time.sleep(0.1)
        
        # Additional small delay for stabilization
        time.sleep(0.5)
    
    def set_position_origin(self):
        """Reset position tracking to origin (0, 0, 0)"""
        self.pose = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Get current yaw as reference
        try:
            state = self.tello.get_current_state()
            if state:
                self._yaw0 = state["yaw"]
                self.current_yaw = state["yaw"]
                self.pose[2] = state["h"]  # Set current height in cm
        except:
            self._yaw0 = 0
            self.current_yaw = 0
            
        self._last_ts = None
        self.acceleration_history = []
        print("Position origin set to current location")
    
    def calibrate_position(self):
        """Calibrate position using known reference point"""
        print("Calibrating position... ensure drone is at known location")
        
        # Reset position tracking
        self.set_position_origin()
        
        # Allow time for stabilization
        time.sleep(2)
        
        # Could be extended to use visual markers or GPS for absolute positioning
        print("Calibration complete")
    
    def get_position_accuracy_estimate(self):
        """Estimate current position accuracy based on flight time and movements"""
        if not hasattr(self, '_flight_start_time'):
            return "Unknown"
        
        flight_time = time.time() - self._flight_start_time
        
        # Rough accuracy estimate (degrades over time)
        if flight_time < 30:
            return "High (±10cm)"
        elif flight_time < 60:
            return "Medium (±20cm)"
        else:
            return "Low (±50cm+)"
    
    def emergency_stop(self):
        """Emergency stop - immediately stop all motors"""
        try:
            self.tello.emergency()
            self.is_flying = False
            print("EMERGENCY STOP ACTIVATED")
        except Exception as e:
            print(f"Emergency stop failed: {e}")
    
    def get_status(self):
        """Get current drone status with improved accuracy info"""
        try:
            battery = self.tello.get_battery()
            height = self.tello.get_height()
            temp = self.tello.get_temperature()
            position = self.get_pose()
            velocity = self.get_velocity()
            accuracy = self.get_position_accuracy_estimate()
            
            print(f"Status - Battery: {battery}%, Height: {height}cm, Temp: {temp}°C")
            print(f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})cm")
            print(f"Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f})cm/s")
            print(f"Yaw: {self.current_yaw:.1f}°")
            print(f"Position accuracy: {accuracy}")
            print(f"Flying: {self.is_flying}")
            
            return {
                'battery': battery,
                'height': height,
                'temperature': temp,
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'yaw': self.current_yaw,
                'flying': self.is_flying,
                'accuracy': accuracy
            }
        except Exception as e:
            print(f"Failed to get status: {e}")
            return None
    
    def start_video_stream(self):
        """Start video stream from drone camera"""
        try:
            self.tello.streamon()
            print("Video stream started")
        except Exception as e:
            print(f"Failed to start video stream: {e}")
    
    def stop_video_stream(self):
        """Stop video stream"""
        try:
            self.tello.streamoff()
            print("Video stream stopped")
        except Exception as e:
            print(f"Failed to stop video stream: {e}")
    
    def basic_flight_demo(self):
        """Demonstrate basic flight patterns with position tracking"""
        print("Starting basic flight demonstration...")
        
        # Takeoff
        self.takeoff()
        self._flight_start_time = time.time()
        
        # Square pattern with position tracking
        print("Flying in square pattern...")
        waypoints = [(50, 0), (50, 50), (0, 50), (0, 0)]
        
        for i, (x, y) in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}: ({x}, {y})")
            self.go_to_position(x, y)
            time.sleep(2)
            
            # Show current position
            pos = self.get_pose()
            print(f"Reached: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})cm")
        
        # Return to origin
        print("Returning to origin...")
        self.go_to_position(0, 0)
        
        # Land
        self.land()
        print("Demo completed")

def main():
    """Main function with interactive controls"""
    controller = TelloController()
    
    # Connect to drone
    if not controller.connect():
        print("Failed to connect to drone. Exiting.")
        return
    
    print("\n=== Tello Drone Controller (Corrected Version) ===")
    print("Commands:")
    print("  takeoff    - Take off")
    print("  land       - Land")
    print("  forward    - Move forward 30cm")
    print("  backward   - Move backward 30cm") 
    print("  left       - Move left 30cm")
    print("  right      - Move right 30cm")
    print("  up         - Move up 30cm")
    print("  down       - Move down 30cm")
    print("  cw         - Rotate clockwise 90°")
    print("  ccw        - Rotate counter-clockwise 90°")
    print("  flip_f     - Flip forward")
    print("  flip_b     - Flip backward")
    print("  status     - Get drone status")
    print("  goto X Y   - Move to position (X, Y) in cm")
    print("  origin     - Set current position as origin")
    print("  calibrate  - Calibrate position tracking")
    print("  position   - Show current position")
    print("  velocity   - Show current velocity")
    print("  accuracy   - Show position accuracy estimate")
    print("  demo       - Run flight demonstration")
    print("  video      - Start video stream")
    print("  stop_video - Stop video stream")
    print("  emergency  - Emergency stop")
    print("  quit       - Exit program")
    print()
    
    try:
        while True:
            command = input("Enter command: ").strip().lower()
            
            if command == 'takeoff':
                controller.takeoff()
                controller._flight_start_time = time.time()
            elif command == 'land':
                controller.land()
            elif command == 'forward':
                controller.move_forward()
            elif command == 'backward':
                controller.move_backward()
            elif command == 'left':
                controller.move_left()
            elif command == 'right':
                controller.move_right()
            elif command == 'up':
                controller.move_up()
            elif command == 'down':
                controller.move_down()
            elif command == 'cw':
                controller.rotate_clockwise()
            elif command == 'ccw':
                controller.rotate_counter_clockwise()
            elif command == 'flip_f':
                controller.flip_forward()
            elif command == 'flip_b':
                controller.flip_backward()
            elif command == 'status':
                controller.get_status()
            elif command == 'demo':
                controller.basic_flight_demo()
            elif command.startswith('goto '):
                try:
                    parts = command.split()
                    if len(parts) == 3:
                        x = float(parts[1])
                        y = float(parts[2])
                        controller.go_to_position(x, y)
                    else:
                        print("Usage: goto X Y (where X and Y are coordinates in cm)")
                except ValueError:
                    print("Invalid coordinates. Use numbers only.")
            elif command == 'origin':
                controller.set_position_origin()
            elif command == 'calibrate':
                controller.calibrate_position()
            elif command == 'position':
                pos = controller.get_pose()
                print(f"Current position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})cm")
            elif command == 'velocity':
                vel = controller.get_velocity()
                print(f"Current velocity: ({vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f})cm/s")
            elif command == 'accuracy':
                acc = controller.get_position_accuracy_estimate()
                print(f"Position accuracy: {acc}")
            elif command == 'video':
                controller.start_video_stream()
            elif command == 'stop_video':
                controller.stop_video_stream()
            elif command == 'emergency':
                controller.emergency_stop()
            elif command == 'quit':
                if controller.is_flying:
                    print("Landing before exit...")
                    controller.land()
                break
            else:
                print("Unknown command.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if controller.is_flying:
            print("Emergency landing...")
            controller.land()
    
    print("Goodbye!")

if __name__ == "__main__":
    main()