from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple

from controller.context_map.graph_manager import GraphManager

# Common return type for motion/actuation commands
CommandResult = Tuple[bool, bool]  # (ok, replan)

class RobotType(Enum):
    """Supported robot backends."""
    VIRTUAL = "virtual"
    TELLO = "tello"

    # To be implemented
    GEAR = "gear"
    CRAZYFLIE = "crazyflie"
    PX4_SIMULATOR = "px4_simulator"

class RobotWrapper(ABC):
    """
    Abstract base class defining a common control interface for different robot types.

    Notes
    -----
    - All movement and action methods return a tuple `(ok, replan)`:
        ok   : bool -> whether the command was accepted/executed without error
        replan : bool -> whether a replan is required to fulfill the user's task
    """

    def __init__(self, graph_manager: GraphManager, move_enable: bool = False) -> None:
        """
        Parameters
        ----------
        graph_manager : GraphManager
            Manager for world graph representation.
        move_enable : bool, optional
            If False, motion commands may be ignored or simulated by implementations.
        """
        self.graph_manager = graph_manager
        self.move_enable = move_enable

    # --- Lifecycle / connectivity -------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the robot (pairing, links, SDK init, etc.)."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the robot and eventually stop threads."""
        pass

    @abstractmethod
    def keep_active(self) -> None:
        """Optionally send heartbeats to prevent robot timeout or sleep."""
        pass

    # --- Stream control --------------------------------------------------
    @abstractmethod
    def start_stream(self) -> None:
        """Start any available telemetry/video stream."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop any active telemetry/video stream."""
        pass

    @abstractmethod
    def get_frame_reader(self) -> Any:
        """Return a frame reader handle (implementation-defined)."""
        pass


    # --- Flight control --------------------------------------------------
    @abstractmethod
    def takeoff(self) -> bool:
        """Command the robot to take off. Returns True if accepted."""
        pass

    @abstractmethod
    def land(self) -> bool:
        """Command the robot to land. Returns True if accepted."""
        pass

    @abstractmethod
    def move_north(self, distance_cm: int) -> CommandResult:
        """Move forward/north by `distance_cm` centimeters. Returns (ok, replan)."""
        pass
    
    @abstractmethod
    def move_south(self, distance_cm: int) -> CommandResult:
        """Move backward/south by `distance_cm` centimeters. Returns (ok, replan)."""
        pass
    
    @abstractmethod
    def move_west(self, distance_cm: int) -> CommandResult:
        """Move left/west by `distance_cm` centimeters. Returns (ok, replan)."""
        pass

    @abstractmethod
    def move_east(self, distance_cm: int) -> CommandResult:
        """Move right/east by `distance_cm` centimeters. Returns (ok, replan)."""
        pass

    @abstractmethod
    def move_direction(self, direction_deg: int, distance_cm: int) -> CommandResult:
        """
        Move along a heading by `distance_cm` centimeters.

        Parameters
        ----------
        direction_deg : int
            Heading in degrees (convention: 0° = north, positive = clockwise).
            This is intended for internal use (not exposed to the LLM).
        distance_cm : int
            Travel distance in centimeters.

        Returns
        -------
        (ok, replan) : Tuple[bool, bool]
        """
        pass

    @abstractmethod
    def move_up(self, distance_cm: int) -> CommandResult:
        """Increase altitude by `distance_cm` centimeters. Returns (ok, replan)."""
        pass
    
    @abstractmethod
    def move_down(self, distance_cm: int) -> CommandResult:
        """Decrease altitude by `distance_cm` centimeters. Returns (ok, replan)."""
        pass

    @abstractmethod
    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float) -> CommandResult:
        """
        Navigate the robot to the absolute planar position (target_x_cm, target_y_cm, target_z_cm).

        Parameters
        ----------
        target_x_cm : float
            Target X coordinate (cm) in the world/map frame.
        target_y_cm : float
            Target Y coordinate (cm) in the world/map frame.
        target_z_cm : float
            Target Z coordinate (cm) in the world/map frame.
        Returns
        -------
        (ok, replan) : Tuple[bool, bool]
        """
        pass

    @abstractmethod
    def turn_ccw(self, degree: int) -> CommandResult:
        """Rotate counter-clockwise by `degree`. Returns (ok, replan)."""
        pass

    @abstractmethod
    def turn_cw(self, degree: int) -> CommandResult:
        """Rotate clockwise by `degree`. Returns (ok, replan)."""
        pass

    # --- State / pose -------------------------------------------------------------
    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the robot's current pose (position + orientation).

        Returns
        -------
        Tuple[float, float, float]
            (x, y, z) in centimeters; 
        """
        pass