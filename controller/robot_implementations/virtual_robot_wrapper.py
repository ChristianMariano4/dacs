import cv2, time
from typing import Tuple

from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper
from ..abs.robot_wrapper import RobotWrapper

class FrameReader:
    def __init__(self, cap):
        # Initialize the video capture
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def frame(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        # self.crazyflie_drone = CrazyflieWrapper(move_enable=False, link_uri='radio://0/40/2M/BADF00D002')

    def keep_active(self):
        pass

    def connect(self):
        pass

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    def start_stream(self):
        self.cap = cv2.VideoCapture(0)
        self.stream_on = True

    def stop_stream(self):
        self.cap.release()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.cap)
    
    def get_pose(self):
        pass

    def create_new_trajectory(self):
        pass

    def start_trajectory(self):
        pass

    def move_north(self, distance: float = 10) -> Tuple[bool, bool]:
        distance = int(distance)
        print(f"-> Moving forward {distance} cm")
        self.movement_x_accumulator += distance
        time.sleep(1)
        return True, False

    def move_south(self, distance: float = 10) -> Tuple[bool, bool]:
        distance = int(distance)
        print(f"-> Moving backward {distance} cm")
        self.movement_x_accumulator -= distance
        time.sleep(1)
        return True, False

    def move_west(self, distance: float = 10) -> Tuple[bool, bool]:
        distance = int(distance)
        print(f"-> Moving left {distance} cm")
        self.movement_y_accumulator += distance
        time.sleep(1)
        return True, False

    def move_east(self, distance: float = 10) -> Tuple[bool, bool]:
        distance = int(distance)
        print(f"-> Moving right {distance} cm")
        self.movement_y_accumulator -= distance
        time.sleep(1)
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        time.sleep(1)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        time.sleep(1)
        return True, False
    
    def go_xy_speed(self, x: int, y: int, speed: int = 20) -> Tuple[bool, bool]:
        time.sleep(1)
        print(f"[Drone] Move by an offset of {x} - {y} - current_height with speed {speed}")
        return True, False
    
    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        self.rotation_accumulator += degree
        if degree >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        self.rotation_accumulator -= degree
        if degree >= 90:
            print("-> Turning CW over 90 degrees")
            return True, False
        time.sleep(1)
        return True, False
    
    def get_pose(self):
        pass

    def create_new_trajectory(self, gesture, duration_s=15)-> Tuple[bool, bool]:
        pass

    def start_trajectory(self, gesture) -> Tuple[bool, bool]:
        pass

    def go_to_position(self, current_pos, target_pos, speed=50):
        pass

    def move_direction(self, direction, distance):
        pass