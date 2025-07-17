from enum import Enum
from PIL import Image
import queue, time, os, json
from typing import Optional, Tuple
import asyncio
import uuid

from controller.constants import REGION_THRESHOLD
from controller.task import Task

import cv2
import numpy as np
import torch

from controller.context_map.mapping.graph_manager import GraphManager
from controller.visual_sensing.enviromental_analysis_module import EnvironmentalAnalysisModule


from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import input_t, print_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
import os

load_dotenv()

class IterationDecision(Enum):
    REPLAN = "replan",
    DONE = "done",
    IMPOSSIBLE = "impossible"

class LLMController():
    def __init__(self, robot_type, use_http=False, message_queue: Optional[queue.Queue]=None):
        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.graph_manager = GraphManager(self)
        # self.spine_agent = SPINE(self.graph_manager.graph_handler)
        self.vision = VisionSkillWrapper(self.shared_frame, graph_manager=self.graph_manager)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        self.current_task = None
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper(move_enable=True, graph_manager=self.graph_manager)
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case RobotType.CRAZYFLIE:
                print_t("[C] Start Crazyflie drone...")
                from .crazyflie_wrapper import CrazyflieWrapper
                self.drone: CrazyflieWrapper = CrazyflieWrapper(move_enable=True)
            case _:
                print_t("[C] Start virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper()
        
        self.planner = LLMPlanner(robot_type, self.current_task)

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("go_xy", self.drone.go_to_position, "Move to x y absolute position.", args=[SkillArg("x", int), SkillArg("y", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("explore_new_region", self.explore_new_region, "Explore a new region (forward, backward, left, right) by a given distance in cm", args=[SkillArg("direction", str), SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("name_region", self.name_region, "Give a meaningful name to current region node in context graph", args=[SkillArg("region_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("create_new_trajectory", self.drone.create_new_trajectory, "Create and save a new trajectory, mapping it to a gesture", args=[SkillArg("gesture", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("start_trajectory", self.drone.start_trajectory, "Start a trajectory mapped by a gesture", args=[SkillArg("gesture", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("choose_direction", self.skill_choose_direction, "Choose the direction to go to based on images and graph"))
        # self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))
        # Instead of replanning, at the end of each iteration, the LLM decides if the task:
        # - needs another iteration (replanning with context updated)
        # - has been fullfilled, so return
        # - can't be fullfilled, so return
        # self.low_level_skillset.add_skill(LowLevelSkillItem("end_iteration", self.planner.probe_end_iteration, "Decide what to do at the end of an iteration of planning (stop or continue)"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))
        #         self.low_level_skillset.add_skill(
        #     LowLevelSkillItem("flush_updates",
        #                       lambda: (self.graph_mgr.flush_prompt_updates(), False),
        #                       "Return accumulated graph diff and clear buffer")
        # )
        self.low_level_skillset.add_skill(LowLevelSkillItem("goto", self.skill_goto, "goto the object", args=[SkillArg("object_name[*x-value]", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'
        with open(os.path.join(CURRENT_DIR, f"assets/{type_folder_name}/new/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        # self.current_plan = None
        # self.execution_history = None
        self.execution_time = time.time()
        self.env_analysis_module = EnvironmentalAnalysisModule()
        self.images_counter = 0
        self.directions = {0: "forward", 1: "right", 2: "backward", 3: "left"}

    def get_drone(self) -> RobotWrapper:
        return self.drone
    
    def get_drone_pose(self):
        self.drone.get_pose()

    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        print(f'Goto {object_name}')
        if '[' in object_name:
            x = float(object_name.split('[')[1].split(']')[0])
        else:
            x = self.vision.object_x(object_name)[0]

        print(f'>> GOTO x {x} {type(x)}')

        if type(x) is float:
            if x > 0.55:
                self.drone.turn_cw(int((x - 0.5) * 70))
            elif x < 0.45:
                self.drone.turn_ccw(int((0.5 - x) * 70))

        self.drone.move_forward(110)
        return None, False
    
    def explore_new_region(self, direction: int, distance: int = REGION_THRESHOLD) -> Tuple[None, bool]:
        # next_yaw = {"forward":0,"right":90,"backward":180,"left":-90}[dir]
        match direction:
            case 0:
                print("forward")
                self.drone.move_forward(distance=distance)
            case 180:
                print("backward")
                self.drone.move_backward(distance=distance)
            case -90:
                print("left")
                self.drone.move_left(distance=distance)
            case 90:
                print("right")
                self.drone.move_right(distance=distance)
        return None, False
    
    # def add_region(self, region_name: str) -> Tuple[None, bool]:
    #     self.graph_manager.add_region(self.drone.get_pose(), region_name)

    def name_region(self, region_name: str) -> Tuple[None, bool]:
        self.graph_manager.name_region(region_name)

    def skill_take_picture(self) -> Tuple[None, bool]:
        time.sleep(1)
        img_path = os.path.join(self.cache_folder, f"{self.directions.get(self.images_counter)}.jpg")
        self.images_counter = (self.images_counter + 1) % 4
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self.append_message((img_path,))
        return None, False
    
    def skill_choose_direction(self) -> Tuple[int, bool]:
        """
        Finds all jpg images in cache_folder, sorts them (if possible), 
        and returns a list of file paths for LLM direction selection.
        """
        dir = self.env_analysis_module.choose_direction(self.current_task.get_task_description(), self.cache_folder)
        if dir in ["forward","right","backward","left"]:
            next_yaw = {"forward":0,"right":90,"backward":180,"left":-90}[dir]
            print(f"Next yaw {next_yaw}")
            return next_yaw, False
        return 0, False

    def skill_log(self, text: str) -> Tuple[None, bool]:
        self.append_message(f"[LOG] {text}")
        print_t(f"[LOG] {text}")
        return None, False
    
    def skill_re_plan(self) -> Tuple[None, bool]:
        print("[C] Start Replanning...")
        return None, True

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        time.sleep(s)
        return None, False

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update_obj_list()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        ret_val = interpreter.execute(minispec)
        self.current_task.update_execution_history(interpreter.execution_history)
        # self.execution_history = interpreter.execution_history
        # ret_val = interpreter.ret_queue.get()
        return ret_val

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.current_task = Task(task_description)
        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        while True:
            self.current_plan = self.planner.plan(task_description, execution_history=self.current_task.get_execution_history(), context_graph=self.graph_manager.get_graph(), current_position=self.graph_manager.get_drone_pose(), current_region=self.graph_manager.get_current_region())
            self.current_task.set_current_plan(self.current_plan)
            print_t(f"The plan is {self.current_task.get_current_plan()}.")
            input_t("Press a key to execute that plan\n")
            self.append_message(f'[Plan]: \\\\')
            try:
                self.execution_time = time.time()
                ret_val = self.execute_minispec(self.current_task.get_current_plan())
            except Exception as e:
                print_t(f"[C] Error: {e}")
            
            # TODO: enable. disable replan for debugging
            # break
            if ret_val is not None and ret_val.replan:
                print_t(f"[C] > Replanning <: {ret_val.value}")
                continue
            else:
                break
        self.append_message(f'\n[Task ended]')
        self.append_message('end')
        self.current_plan = None
        # self.execution_history = None

    def continue_execution(self): pass #TODO

    def start_robot(self):
        print_t("[C] Connecting to robot...")
        self.drone.connect()
        print_t("[C] Starting robot...")
        self.drone.takeoff()
        # self.drone.move_up(25)
        print_t("[C] Starting stream...")
        self.drone.start_stream()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True


    def _infer_depth_mm(self, bgr: np.ndarray,
                        max_depth_mm: int = 5000) -> np.ndarray:
        """
        Convert a BGR frame to a depth map (millimetres, int16) using MiDaS.
        """
        import cv2, torch

        if bgr is None:
            raise ValueError("Empty frame passed to _infer_depth_mm()")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # transform already RETURNS shape (1, 3, H, W) ⇢ no .unsqueeze(0)!
        input_tensor = self.depth_tf(rgb).to(self.depth_device)

        with torch.no_grad():
            pred = self.midas(input_tensor)

            # upscale back to original frame size
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),                  # (1, 1, H', W')
                size=rgb.shape[:2],                 # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)                            # (1, H, W)

        depth = pred.cpu().numpy()[0]               # (H, W) float32
        depth_norm = depth / depth.max()            # 0-1
        depth_mm   = (depth_norm * max_depth_mm).astype(np.int16)
        return depth_mm

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        while self.controller_active:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            frame = Frame(frame_reader.frame,
                          frame_reader.depth if hasattr(frame_reader, 'depth') else None)
            
            # depth_mm = self._infer_depth_mm(self.latest_frame)
            # frame.depth = depth_mm

            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
            time.sleep(0.10)

 
        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")