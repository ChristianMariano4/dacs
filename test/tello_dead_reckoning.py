from djitellopy import Tello
import time
import math

class TelloDeadReckoning:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        
        # Initial position (x, y, z) in cm
        self.x = 0
        self.y = 0 
        self.z = 0
        
        # Current heading in degrees (0 = north)
        self.heading = 0
        
        # Store last known velocities
        self.last_update_time = time.time()
        
    def update_position_from_movement(self, distance_cm, direction="forward"):
        """Update position based on movement command"""
        if direction == "forward":
            self.x += distance_cm * math.cos(math.radians(self.heading))
            self.y += distance_cm * math.sin(math.radians(self.heading))
        elif direction == "back":
            self.x -= distance_cm * math.cos(math.radians(self.heading))
            self.y -= distance_cm * math.sin(math.radians(self.heading))
        elif direction == "left":
            self.x += distance_cm * math.cos(math.radians(self.heading - 90))
            self.y += distance_cm * math.sin(math.radians(self.heading - 90))
        elif direction == "right":
            self.x += distance_cm * math.cos(math.radians(self.heading + 90))
            self.y += distance_cm * math.sin(math.radians(self.heading + 90))
        elif direction == "up":
            self.z += distance_cm
        elif direction == "down":
            self.z -= distance_cm
    
    def update_heading(self, rotation_degrees, direction="cw"):
        """Update heading based on rotation"""
        if direction == "cw":
            self.heading = (self.heading + rotation_degrees) % 360
        else:  # ccw
            self.heading = (self.heading - rotation_degrees) % 360

    def update_from_sensors(self):
        """Update position using Tello's built-in sensors"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Get velocity data (cm/s)
        vx = self.tello.get_speed_x()  # Forward/back velocity
        vy = self.tello.get_speed_y()  # Left/right velocity  
        vz = self.tello.get_speed_z()  # Up/down velocity
        
        # Update position based on velocity and time
        self.x += vx * dt
        self.y += vy * dt
        self.z += vz * dt
        
        # Update heading if available (some Tello models provide yaw)
        try:
            yaw = self.tello.get_yaw()
            self.heading = yaw
        except:
            pass  # Not all Tello models support yaw reading
        
        self.last_update_time = current_time

    def get_position(self):
        """Get current estimated position"""
        return (self.x, self.y, self.z, self.heading)


class TelloNavigator:
    def __init__(self):
        self.dead_reckoning = TelloDeadReckoning()
        self.tello = self.dead_reckoning.tello
        
    def move_and_track(self, distance, direction):
        """Move drone and update position tracking"""
        if direction == "forward":
            self.tello.move_forward(distance)
        elif direction == "back":
            self.tello.move_back(distance)
        elif direction == "left":
            self.tello.move_left(distance)
        elif direction == "right":
            self.tello.move_right(distance)
        elif direction == "up":
            self.tello.move_up(distance)
        elif direction == "down":
            self.tello.move_down(distance)
            
        # Update our position estimate
        self.dead_reckoning.update_position_from_movement(distance, direction)
        
    def rotate_and_track(self, degrees, direction="cw"):
        """Rotate drone and update heading"""
        if direction == "cw":
            self.tello.rotate_clockwise(degrees)
        else:
            self.tello.rotate_counter_clockwise(degrees)
            
        self.dead_reckoning.update_heading(degrees, direction)
    
    def continuous_tracking(self, duration_seconds=10):
        """Continuously update position using sensor data"""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            self.dead_reckoning.update_from_sensors()
            x, y, z, heading = self.dead_reckoning.get_position()
            
            print(f"Position: ({x:.1f}, {y:.1f}, {z:.1f}) cm, Heading: {heading:.1f}°")
            time.sleep(0.1)  # Update 10 times per second

# Usage example
if __name__ == "__main__":
    navigator = TelloNavigator()
    
    # Take off and start tracking
    navigator.tello.takeoff()
    print(f"{navigator.tello.get_battery()}")
    time.sleep(2)
    
    # Move in a square pattern while tracking position
    for _ in range(4):
        navigator.move_and_track(25, "forward")
        navigator.rotate_and_track(90, "cw")
        
        x, y, z, heading = navigator.dead_reckoning.get_position()
        print(f"Current position: ({x:.1f}, {y:.1f}, {z:.1f}) cm")
    
    # Land
    navigator.tello.land()