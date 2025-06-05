from abc import ABC, abstractmethod
from enum import Enum
import json
import os

SKILL_FILE = "controller/assets/tello/high_level_skills.json"

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
    def move_forward(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_backward(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_left(self, distance: int) -> bool:
        pass

    @abstractmethod
    def move_right(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_up(self, distance: int) -> bool:
        pass
    
    @abstractmethod
    def move_down(self, distance: int) -> bool:
        pass

    @abstractmethod
    def turn_ccw(self, degree: int) -> bool:
        pass

    @abstractmethod
    def turn_cw(self, degree: int) -> bool:
        pass

    def add_skill(self, skill_name: str, description: str, minispec_def: str):
        skill_name = skill_name.strip('\'"')
        minispec_def = minispec_def.strip('\'"').replace('\\;', ';')
        print(f"Skill added: {skill_name}: {minispec_def}")

        # Load existing skills
        if os.path.exists(SKILL_FILE):
            with open(SKILL_FILE, "r") as f:
                skills = json.load(f)
                if not isinstance(skills, list):
                    print("Error: Skill file is not a list. Resetting.")
                    skills = []
        else:
            skills = []

        # Remove old skill with same name if it exists
        skills = [s for s in skills if s.get("skill_name") != skill_name]

        # Add or update the skill
        skills.append({
            "skill_name": skill_name,
            "skill_description": description,
            "definition": minispec_def
        })

        # Write back to file
        with open(SKILL_FILE, "w") as f:
            json.dump(skills, f, indent=4)

        return True, False