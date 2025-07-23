from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from typing import Tuple

from controller.context_map.mapping.graph_manager import GraphManager

class RobotType(Enum):
    VIRTUAL = "virtual"
    TELLO = "tello"
    GEAR = "gear"
    CRAZYFLIE = "crazyflie"
    PX4_SIMULATOR = "px4_simulator"

class RobotWrapper(ABC):
    movement_x_accumulator = 0
    movement_y_accumulator = 0
    rotation_accumulator = 0

    def __init__(self, graph_manager: GraphManager, move_enable=False):
        self.graph_manager = graph_manager
        self.move_enable = move_enable
        self.gesture_trajectory_mapping = {}

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def keep_active(self):
        pass

    @abstractmethod
    def takeoff(self) -> bool:
        pass

    @abstractmethod
    def land(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_frame_reader(self):
        pass

    @abstractmethod
    def move_north(self, distance: int) -> Tuple[bool, bool]:
        pass
    
    @abstractmethod
    def move_south(self, distance: int) -> Tuple[bool, bool]:
        pass
    
    @abstractmethod
    def move_west(self, distance: int) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def move_east(self, distance: int) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def move_direction(self, direction: int, distance: int) -> Tuple[bool, bool]:
        '''The direction is given by the degrees used to rotate. Skill used internally and not given to LLM'''
        pass

    @abstractmethod
    def move_up(self, distance: int) -> Tuple[bool, bool]:
        pass
    
    @abstractmethod
    def move_down(self, distance: int) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def go_to_position(self, current_pos, target_pos, speed=50) -> Tuple[bool, bool]:
        pass

    # @abstractmethod
    # def go_xy_speed(self, x: int, y: int, z:int, speed: int) -> Tuple[bool, bool]:
    #     pass

    @abstractmethod
    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def get_pose(self):
        pass

    @abstractmethod
    def create_new_trajectory(self, gesture, duration_s=15)-> Tuple[bool, bool]:
        pass

    @abstractmethod
    def start_trajectory(self, gesture) -> Tuple[bool, bool]:
        pass