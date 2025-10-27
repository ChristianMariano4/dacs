from enum import Enum
import re
from PIL import Image
import queue, time, os, json
from typing import List, Optional, Tuple
import asyncio
import uuid

from controller.memory.short_memory import ShortMemoryModule
from controller.utils.constants import HIGH_LEVEL_SKILL_FILE, REGION_THRESHOLD, ROBOT_NAME
from controller.llm.llm_wrapper import GPT4, GPT5, GPT5_MINI, GPT5_NANO, LLMWrapper
from controller.memory.long_memory import LongMemoryModule
from controller.middle_layer.flyzone_manager import FlyzoneManager
from controller.middle_layer.middle_layer import MiddleLayer
from controller.task import Task

import cv2
import numpy as np
import torch

# From text to speech
from gtts import gTTS

from controller.context_map.graph_manager import GraphManager
from controller.visual_sensing.enviromental_analysis_module import EnvironmentalAnalysisModule


from ..shared_frame import SharedFrame, Frame
from ..yolo_client import YoloClient
from ..yolo_grpc_client import YoloGRPCClient
from ..robot_implementations.tello_wrapper import TelloWrapper
from ..robot_implementations.virtual_robot_wrapper import VirtualRobotWrapper
from ..abs.robot_wrapper import RobotWrapper
from ..visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from ..skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from ..utils.general_utils import input_t, print_t
from ..minispec_interpreter import MiniSpecInterpreter, Statement
from ..abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
import os

load_dotenv()

class IterationDecision(Enum):
    REPLAN = "replan",
    DONE = "done",
    IMPOSSIBLE = "impossible"

class LLMController():
    def __init__(self, robot_type, use_http=False, message_queue: Optional[queue.Queue]=None,  user_answer_queue: Optional[queue.Queue]=None):
        # shared middle layer that stores user settings
        self.middle_layer = MiddleLayer()
        self.short_memory = ShortMemoryModule()
        
        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.graph_manager : GraphManager = None
        self.vision = VisionSkillWrapper(self.shared_frame)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        self.user_answer_queue = user_answer_queue
        self.current_task = None
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # user preferences in middle layer
        self.username = "Christian" # name of the user, used to personalized his own experience (e.g. different feedbacks)

        self.long_memory_module = LongMemoryModule(username=self.username)

        # Flyzone section
        self.flyzone_manager = FlyzoneManager(middle_layer=self.middle_layer)
        
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper(move_enable=True, graph_manager=self.graph_manager)
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from ..robot_implementations.gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case RobotType.CRAZYFLIE:
                print_t("[C] Start Crazyflie drone...")
                from ..robot_implementations.crazyflie_wrapper import CrazyflieWrapper
                self.drone: CrazyflieWrapper = CrazyflieWrapper(move_enable=True)
            case _:
                print_t("[C] Start Virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper(graph_manager=self.graph_manager, move_enable=True)
        
        self.planner = LLMPlanner(robot_type, self.current_task, self.latest_frame)

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_off", self.drone.takeoff, "Take off", args=[]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("land", self.drone.land, "Land", args=[]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_north, "Move forward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_south, "Move backward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_west, "Move left by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_east, "Move right by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("go_xy", self.drone.go_to_position, "Move to x,y,z absolute position.", args=[SkillArg("x", int), SkillArg("y", int), SkillArg("z", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("explore_new_region", self.explore_new_region, "Explore a new region (forward, backward, left, right) by a given distance in cm", args=[SkillArg("direction", str), SkillArg("distance", float)]))
        # self.low_level_skillset.add_skill(LowLevelSkillItem("name_region", self._name_region, "Give a meaningful name to current region node in context graph", args=[SkillArg("region_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target YOLO-detectable objects", args=[SkillArg("objects_name", list)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get the object’s center X in normalized image coordinates (0,1) (left=0, right=1).", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get the object’s center Y in normalized image coordinates (0,1) (top=0, bottom=1).", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning. Add also what do you expect as returned value type", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log_user", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("explore_direction", self.skill_explore_direction, "Explore through a direction based on video streaming, graph, current task and an hint (if needed) given as argument. Assume this respects the flyzone while exploring. What is more, it names the current region based on what the drone can see around it.", args=[SkillArg("hint", Optional[str])]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("add_skill", self.skill_add_skill, "Define a new high-level skill through already existing low and high-level ones", args=[SkillArg("name", str), SkillArg("description", str), SkillArg("definition", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("ask_user", self.skill_ask_user, "Ask user a question in order to retrieve some missing information about his task. After this skill it automatically replans with new information", args=[SkillArg("question", str)]))
        
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
        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/skills/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        # self.current_plan = None
        # self.execution_history = None
        self.execution_time = time.time()
        self.env_analysis_module = EnvironmentalAnalysisModule(middle_layer=self.middle_layer)
        self.images_counter = 0
        self.directions  = {0: "north", 1: "north-east", 2: "east", 3:"south-east", 4: "south", 5: "south-west", 6: "west", 7: "north-west"}


    def set_username(self, username):
        self.username = username
        self.long_memory_module.change_username(self.username)

    def get_username(self) -> str:
        return self.username 

    def set_graph_manager(self, graph_manager):
        self.graph_manager = graph_manager
        self.vision.set_graph_manager(graph_manager)

    def get_drone(self) -> RobotWrapper:
        return self.drone
    
    def get_flyzone_manager(self) -> FlyzoneManager:
        return self.flyzone_manager
    
    def get_drone_pose(self):
        self.drone.get_position()

    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        # TODO: improve this skill to not be fixed of 110 cm moving forward
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

        self.drone.move_north(110)
        return None, False
    
    def explore_new_region(self, direction: int, distance: int = REGION_THRESHOLD) -> Tuple[None, bool]:
        # next_yaw = {"north":0, "north-east": 45, "east":90, "south-east": 135, "south":180, "south-west": -135, "west":-90, "north-west": -45}
        match direction:
            case 0:
                self.drone.move_north(distance_cm=distance)
            case 180:
                self.drone.move_south(distance_cm=distance)
            case -90:
                self.drone.move_west(distance_cm=distance)
            case 90:
                self.drone.move_east(distance_cm=distance)
            case _:
                self.drone.move_direction(direction, distance)
        return None, False
    
    # def add_region(self, region_name: str) -> Tuple[None, bool]:
    #     self.graph_manager.add_region(self.drone.get_pose(), region_name)

    def _name_region(self, region_name: str) -> Tuple[None, bool]:
        self.graph_manager.name_region(region_name)

    def skill_take_picture(self) -> Tuple[None, bool]:
        time.sleep(0.1)
        img_path = os.path.join(self.cache_folder, f"{self.directions.get(self.images_counter)}.jpg")
        self.images_counter = (self.images_counter + 1) % 8
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self.append_message((img_path,))
        return None, False
    
    def skill_explore_direction(self, hint: Optional[str]) -> Tuple[None, bool]:
        """
        Finds all jpg images in cache_folder, sorts them (if possible), 
        and returns a list of file paths for LLM direction selection.
        """
        (dir, distance, region_name) = self.env_analysis_module.choose_direction(self.current_task.get_task_description(), self.cache_folder, self.get_drone_pose(), hint)
        self._name_region(region_name)
        valid_directions = ["north", "east", "sourh", "west", "north-east", "north-west", "south-east", "south-west"]
        if dir in valid_directions:
            print(f"{dir} is a valide direction")
            next_yaw = {"north":0, "north-east": 45, "east":90, "south-east": 135, "south":180, "south-west": -135, "west":-90, "north-west": -45}[dir]
            print(f"Next yaw {next_yaw}")
            self.drone.move_direction(next_yaw, distance)
        return 0, False
    
    def text_to_speech(self, text: str):
        # Convert text to speech
        tts = gTTS(text, lang="en")
        # Save as mp3 and play
        tts.save("serving/webui/cache/speech.mp3")
        os.system("mpg123 serving/webui/cache/speech.mp3")

    def skill_log(self, text: str) -> Tuple[None, bool]:
        text = str(text)
        self.append_message(f"[LOG] {text}")
        self.text_to_speech(text)
        return None, False
    
    def skill_re_plan(self) -> Tuple[None, bool]:
        print("[C] Start Replanning...")
        return None, True

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        time.sleep(s)
        return None, False
    
    def skill_add_skill(self, skill_name: str, description: str, minispec_def: str):
        skill_name = skill_name.strip('\'"')
        minispec_def = minispec_def.strip('\'"').replace('\\;', ';')
        print(f"Skill added: {skill_name}: {minispec_def}")

        # Load existing skills
        if os.path.exists(HIGH_LEVEL_SKILL_FILE):
            with open(HIGH_LEVEL_SKILL_FILE, "r") as f:
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
        with open(HIGH_LEVEL_SKILL_FILE, "w") as f:
            json.dump(skills, f, indent=4)

        return True, False
    

    def skill_ask_user(self, question: str) -> Tuple[None, bool, bool]:
        '''Log to the user a question made by LLM and attach question-answer pair in plan history'''
        self.append_message(f"[Q] {question}")
        self.text_to_speech(question)
        print_t(f"[Q] {question}")
        return None, True, True
        

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            objects_list = self.vision.update_obj_list()
            # for obj in self.vision.get_objects_list():
                # print(f"{obj.name}: {self.vision.object_distance(obj.name)}")
            YoloClient.plot_results_oi(image, objects_list)
        return image
    
    def execute_minispec(self, minispec: str, iteration_description: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        ret_val = interpreter.execute(minispec)
        self.current_task.update_execution_history(interpreter.execution_history, iteration_description)
        # self.execution_history = interpreter.execution_history
        # ret_val = interpreter.ret_queue.get()
        return ret_val
    
    def _get_after_shortcut(self, text: str) -> str:
        '''
        Retrieve the words used after "shortcut" using regex.
        '''
        match = re.search(r"shortcut\s+(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        if "shortcut" in task_description:
            print(task_description)
            #TODO: use fast LLM to retrieve the sentence used as shortcut, instead of statically parsing it
            keywords = self._get_after_shortcut(task_description)
            print(keywords)
            current_task_dict = self.long_memory_module.get_shortcut_task(username=self.username, keywords=keywords)
            self.current_task = Task(task_description=current_task_dict['task_description'], 
                                     execution_history=current_task_dict['execution_history'],
                                     current_plan=current_task_dict['current_plan'],
                                     user_feedback=current_task_dict['user_feedback'],
                                     is_new=False
                                     )
            if not self.current_task:
                self.append_message("Sorry. Given shortcut is not existing")
                return
        else:
            self.current_task = Task(task_description)

        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        while True:
            # if ret_val is None: # This is the first iteration of the plan
            #     model_name = GPT5
            # else: # This is executed after a replanning
            #     model_name = GPT5_MINI
            model_name = GPT5_MINI
            print(f"Sending request to model {model_name}")
            # Request plan to the model chosen above
            print(self.current_task)
            self.current_plan, reason, iteration_description = self.planner.plan(self.current_task, 
                                                                execution_history=self.current_task.get_execution_history(), 
                                                                context_graph=self.graph_manager.get_graph(), 
                                                                current_position=self.graph_manager.get_drone_pose(), 
                                                                current_region=self.graph_manager.get_current_region(),
                                                                old_interactions_feedbacks = self.long_memory_module.retrieve_old_interactions(self.current_task.get_task_description()),
                                                                model_name=model_name
                                                                )
            if not self.current_plan: # Resend the request
                continue

            self.current_task.set_current_plan(self.current_plan)
            print_t(f"The plan is {self.current_task.get_current_plan()}. With reason: {reason}")
            # input_t("Press a key to execute that plan\n")
            self.append_message(f'[Plan]: \\\\')
            print_t("Message appended")
            try:
                self.execution_time = time.time()
                ret_val = self.execute_minispec(self.current_task.get_current_plan(), iteration_description)
            except Exception as e:
                print_t(f"[C] Error: {e}")
            
            if ret_val is not None and ret_val.wait_user_answer:
                user_question_answer = self.user_answer_queue.get(block=True)
                print(f"Inside if statement {user_question_answer}")
                user_question_answer_str = str(user_question_answer[0]) + " The user answer is: " + str(user_question_answer[1])
                self.current_task.update_execution_history(user_question_answer_str)
                print_t(f"[C] > Replanning after user answer<: {ret_val.value}, {user_question_answer_str}")
                continue
            elif ret_val is not None and ret_val.replan:
                print_t(f"[C] > Replanning <: {ret_val.value}")
                self.short_memory.generate_interaction_summary(self.current_task)
                continue
            else:
                # Ask user a feedback about the executed plan
                self.append_message("[Q] Write a feedback of the executed plan")
                self.text_to_speech("Write a feedback of the executed plan")
                user_feedback = self.user_answer_queue.get(block=True)
                # print(user_feedback) debug
                self.current_task.set_user_feedback(user_feedback[1])
                self.long_memory_module.save_task_summary(self.current_task)
                self.append_message("Ready again to execute your command.")
                self.text_to_speech("Ready again to execute your command.")

                # Ask user a shortcut to associate to a task and its own plan
                self.append_message(f"[Q] Do you want to save this plan with a shortcut? If yes, say a sentence to associate to this task, otherwise say 'No'")
                self.text_to_speech("Do you want to save this plan with a shortcut? If yes, say a sentence to associate to this task, otherwise say 'No'")
                shortcut = self.user_answer_queue.get(block=True)
                # print(shortcut[1])
                answer: str = shortcut[1]
                if answer.lower() == "no":
                    print("No shortcut")
                    break
                self.long_memory_module.save_shortcut_task(username=self.username,
                                                    keywords=answer,
                                                    task=self.current_task)
                break
        print("[Task ended]")
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
            self.planner.update_latest_frame(self.latest_frame)
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