#!/usr/bin/env python3
"""
Simple standalone test for Tello trajectory functionality.
Copy your TelloWrapper class code directly here to avoid import issues.
"""

import time
import math
import threading
import numpy as np
from typing import Tuple
from djitellopy import Tello
import logging

# Disable Tello logging
Tello.LOGGER.setLevel(logging.WARNING)

class MockGraphManager:
    """Mock graph manager for testing"""
    def update_pose(self, pose):
        pass

class SimpleTelloTest:
    """Simplified Tello wrapper for testing trajectory functionality"""
    
    def __init__(self, move_enable=True):
        self.drone = Tello()
        self.move_enable = move_enable
        self.is_flying = False
        
        # Odometry fields
        self.pose = np.zeros(3)  # (x, y, z) in metres
        self._yaw0 = 0.0
        self._last_ts = None
        self._odo_th = None
        self._odo_stop = threading.Event()
        self._inited = False
        
        # Trajectory fields
        self._recording = False
        self._recorded_waypoints = []
        self.gesture_trajectory_mapping = {}
        
        # Keep-alive
        self._ka_stop = threading.Event()
        self._ka_thread = None
    
    def _odometry_loop(self):
        """Simplified odometry loop"""
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
                if dt > 0.2:
                    continue

                # Simple odometry
                v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) / 100.0
                yaw = math.radians(state["yaw"] - self._yaw0)
                cy, sy = math.cos(yaw), math.sin(yaw)
                Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                v_world = Rz @ v_body

                self.pose[:2] += v_world[:2] * dt
                self.pose[2] = state["h"] / 100.0
                
            except Exception as e:
                print(f"Odometry error: {e}")
            
            time.sleep(0.01)

    def _keepalive_loop(self):
        """Keep-alive loop"""
        while not self._ka_stop.is_set():
            try:
                self.drone.send_control_command("command", timeout=3)
            except Exception as e:
                print(f"Keep-alive failed: {e}")
            self._ka_stop.wait(4.0)

    def connect(self):
        """Connect to Tello"""
        print("Connecting to Tello...")
        self.drone.connect()
        
        # Start keep-alive
        self._ka_stop.clear()
        self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka_thread.start()
        
        # Start odometry
        self._odo_stop.clear()
        self._odo_th = threading.Thread(target=self._odometry_loop, daemon=True)
        self._odo_th.start()
        
        print("Connected!")

    def disconnect(self):
        """Disconnect from Tello"""
        print("Disconnecting...")
        self._ka_stop.set()
        self._odo_stop.set()
        if self._ka_thread:
            self._ka_thread.join(timeout=1)
        if self._odo_th:
            self._odo_th.join(timeout=1)
        try:
            self.drone.end()
        except:
            pass
        print("Disconnected!")

    def get_pose(self):
        """Get current pose in cm"""
        return list(self.pose * 100.0)

    def is_battery_good(self):
        """Check battery level"""
        try:
            battery = self.drone.query_battery()
            print(f"Battery: {battery}%")
            return battery > 20
        except:
            print("Could not read battery")
            return True

    def takeoff(self):
        """Takeoff"""
        if not self.is_battery_good():
            return False
        
        if self.move_enable:
            print("Taking off...")
            self.drone.takeoff()
            if not self._inited:
                try:
                    st = self.drone.get_current_state()
                    self._yaw0 = st.get("yaw", 0.0)
                    self._last_ts = None
                    self._inited = True
                except:
                    pass
        else:
            print("[SIMULATION] Taking off...")
        
        self.is_flying = True
        return True

    def land(self):
        """Land"""
        if self.move_enable:
            print("Landing...")
            self.drone.land()
        else:
            print("[SIMULATION] Landing...")
        self.is_flying = False

    def record_trajectory(self, gesture_name, duration_s=10):
        """Record a simple trajectory"""
        print(f"\n=== Recording trajectory '{gesture_name}' ===")
        self._recorded_waypoints = []
        
        # Countdown
        for i in [3, 2, 1]:
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("🔴 RECORDING STARTED!")
        print("Fly the drone manually now...")
        
        # Take off if not flying
        # if not self.is_flying:
        #     self.takeoff()
        #     time.sleep(2)
        
        # Record waypoints
        start_time = time.time()
        end_time = start_time + duration_s
        
        while time.time() < end_time:
            try:
                pose = self.get_pose()
                state = self.drone.get_current_state()
                
                if pose and state:
                    timestamp = time.time() - start_time
                    waypoint = {
                        't': timestamp,
                        'x': pose[0] / 100.0,  # Convert to meters
                        'y': pose[1] / 100.0,
                        'z': pose[2] / 100.0,
                        'yaw': math.radians(state.get('yaw', 0))
                    }
                    self._recorded_waypoints.append(waypoint)
                    
                    # Print progress
                    remaining = end_time - time.time()
                    print(f"\rRecording... {remaining:.1f}s left | Position: ({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:.0f})cm", end='')
                
            except Exception as e:
                print(f"\nRecording error: {e}")
            
            time.sleep(0.1)  # 10Hz
        
        print(f"\n🟢 RECORDING STOPPED!")
        print(f"Captured {len(self._recorded_waypoints)} waypoints")
        
        # Save trajectory
        if self._recorded_waypoints:
            self.gesture_trajectory_mapping[gesture_name] = self._recorded_waypoints
            return True
        return False

    def execute_trajectory(self, gesture_name):
        """Execute a recorded trajectory"""
        if gesture_name not in self.gesture_trajectory_mapping:
            print(f"No trajectory found for '{gesture_name}'")
            return False
        
        trajectory = self.gesture_trajectory_mapping[gesture_name]
        print(f"\n=== Executing trajectory '{gesture_name}' ===")
        print(f"Trajectory has {len(trajectory)} waypoints")
        
        # Take off if not flying
        if not self.is_flying:
            self.takeoff()
            time.sleep(2)
        
        # Execute each waypoint
        for i, waypoint in enumerate(trajectory[1:], 1):  # Skip first waypoint
            try:
                current_pose = self.get_pose()
                if not current_pose:
                    print("Lost position tracking")
                    break
                
                # Calculate movement
                curr_x, curr_y, curr_z = current_pose[0]/100.0, current_pose[1]/100.0, current_pose[2]/100.0
                target_x, target_y, target_z = waypoint['x'], waypoint['y'], waypoint['z']
                
                dx = int((target_x - curr_x) * 100)
                dy = int((target_y - curr_y) * 100)
                dz = int((target_z - curr_z) * 100)
                
                print(f"Waypoint {i}/{len(trajectory)-1}: Moving ({dx}, {dy}, {dz})cm")
                
                # Execute movement if significant
                if abs(dx) > 20 or abs(dy) > 20 or abs(dz) > 20:
                    if self.move_enable:
                        self.drone.go_xyz_speed(dx, dy, dz, 50)
                        time.sleep(2.0)
                    else:
                        print(f"[SIMULATION] Would move ({dx}, {dy}, {dz})cm")
                        time.sleep(0.5)
                
            except Exception as e:
                print(f"Execution error at waypoint {i}: {e}")
                break
        
        print("🟢 Trajectory execution complete!")
        return True

def main():
    """Main test function"""
    print("=== Simple Tello Trajectory Test ===")
    
    # Ask about simulation mode
    sim_mode = input("Run in simulation mode? (y/N): ").strip().lower() == 'y'
    move_enable = not sim_mode
    
    if sim_mode:
        print("⚠️  Running in SIMULATION mode - no actual drone movement")
    else:
        print("🚁 Running with REAL drone movement")
        print("⚠️  Make sure you have enough space and safety precautions!")
        confirm = input("Continue with real drone? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    # Initialize Tello
    tello = SimpleTelloTest(move_enable=move_enable)
    
    try:
        # Connect
        tello.connect()
        time.sleep(2)
        
        while True:
            print("\n=== Test Menu ===")
            print("1. Check current position")
            print("2. Record trajectory")
            print("3. List trajectories")
            print("4. Execute trajectory")
            print("5. Emergency land")
            print("6. Exit")
            
            choice = input("Choice (1-6): ").strip()
            
            if choice == "1":
                pose = tello.get_pose()
                print(f"Current position: ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f})cm")
                
            elif choice == "2":
                name = input("Trajectory name: ").strip()
                if name:
                    duration = input("Duration (default 10s): ").strip()
                    duration = int(duration) if duration.isdigit() else 10
                    success = tello.record_trajectory(name, duration)
                    if success:
                        print(f"✓ Trajectory '{name}' recorded!")
                    else:
                        print(f"✗ Failed to record trajectory")
                
            elif choice == "3":
                trajectories = list(tello.gesture_trajectory_mapping.keys())
                if trajectories:
                    print("Recorded trajectories:")
                    for i, name in enumerate(trajectories, 1):
                        traj = tello.gesture_trajectory_mapping[name]
                        print(f"  {i}. {name} ({len(traj)} waypoints)")
                else:
                    print("No trajectories recorded")
                
            elif choice == "4":
                trajectories = list(tello.gesture_trajectory_mapping.keys())
                if not trajectories:
                    print("No trajectories available!")
                    continue
                
                print("Available trajectories:")
                for i, name in enumerate(trajectories, 1):
                    print(f"  {i}. {name}")
                
                try:
                    idx = int(input("Execute trajectory number: ")) - 1
                    if 0 <= idx < len(trajectories):
                        name = trajectories[idx]
                        if not sim_mode:
                            confirm = input(f"Execute '{name}'? This will move the drone! (y/N): ").strip().lower()
                            if confirm != 'y':
                                continue
                        tello.execute_trajectory(name)
                    else:
                        print("Invalid number!")
                except ValueError:
                    print("Invalid input!")
                
            elif choice == "5":
                tello.land()
                print("Emergency landing executed")
                
            elif choice == "6":
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if tello.is_flying:
            tello.land()
        tello.disconnect()

if __name__ == "__main__":
    main()