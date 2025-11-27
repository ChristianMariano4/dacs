import os
import re
import sys
import json
import time
import queue
import asyncio
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from gtts import gTTS
from dotenv import load_dotenv

# Local project imports
from controller.utils.constants import (
    FLYZONE_USER_IMAGE_PATH, 
    GRAPH_TXT_PATH, 
    HIGH_LEVEL_SKILL_PATH, 
    REGION_THRESHOLD, 
    SKILL_PATH
)
from controller.memory.short_memory import ShortMemoryModule
from controller.memory.long_memory import LongMemoryModule
from controller.middle_layer.flyzone_manager import FlyzoneManager
from controller.middle_layer.middle_layer import MiddleLayer
from controller.task import Task
from controller.context_map.graph_manager import GraphManager
from controller.visual_sensing.enviromental_analysis_module import EnvironmentalAnalysisModule

from ..shared_frame import SharedFrame, Frame
from ..yolo_client import YoloClient
from ..yolo_grpc_client import YoloGRPCClient
from ..robot_implementations.tello_wrapper import TelloWrapper
from ..robot_implementations.virtual_robot_wrapper import VirtualRobotWrapper
from ..abs.robot_wrapper import RobotWrapper, RobotType
from ..visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from ..skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from ..utils.general_utils import encode_image, print_t
from ..minispec_interpreter import MiniSpecInterpreter, Statement

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

class LLMController:
    def __init__(self, robot_type, graph_manager: GraphManager, use_http=False, message_queue: Optional[queue.Queue]=None, 
                 user_answer_queue: Optional[queue.Queue]=None, username: str = "Christian"):
        
        # --- 1. Infrastructure Setup ---
        self.message_queue = message_queue
        self.user_answer_queue = user_answer_queue
        self.graph_manager = graph_manager
        
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get() # Expecting cache path as first message

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # --- 2. Memory & User Setup ---
        self.middle_layer = MiddleLayer()
        self.short_memory = ShortMemoryModule()
        self.set_username(username, long_memory_flag=False) 
        self.long_memory_module = LongMemoryModule(username=self.username)
        
        # --- 3. Vision & Perception ---
        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
            
        self.vision = VisionSkillWrapper(self.shared_frame, self.graph_manager)
        self.latest_frame = None
        self.env_analysis_module = EnvironmentalAnalysisModule(middle_layer=self.middle_layer)
        
        # --- 4. Robot Initialization ---
        self.flyzone_manager = FlyzoneManager(middle_layer=self.middle_layer)
        self.drone = self._initialize_robot(robot_type)
        
        # --- 5. Planning & Skills ---
        self.current_task = None
        self.last_task = None
        self.current_plan = None
        self.execution_time = time.time()
        self.planner = LLMPlanner(robot_type, self.current_task, self.latest_frame, self.flyzone_manager, cache_folder=self.cache_folder)
        
        # Register Skills
        self._register_skills()

        # --- 6. State Flags ---
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.images_counter = 0
        self.directions = {0: "north", 1: "north-east", 2: "east", 3: "south-east", 
                           4: "south", 5: "south-west", 6: "west", 7: "north-west"}

    def _initialize_robot(self, robot_type) -> RobotWrapper:
        """Factory method to initialize the correct robot wrapper."""
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                return TelloWrapper(move_enable=True, graph_manager=self.graph_manager)
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from ..robot_implementations.gear_wrapper import GearWrapper
                return GearWrapper()
            case RobotType.CRAZYFLIE:
                print_t("[C] Start Crazyflie drone...")
                from ..robot_implementations.crazyflie_wrapper import CrazyflieWrapper
                return CrazyflieWrapper(move_enable=True)
            case _:
                print_t("[C] Start Virtual drone...")
                return VirtualRobotWrapper(graph_manager=self.graph_manager, move_enable=True)

    def _register_skills(self):
        """Registers all low-level and high-level skills."""
        
        # 1. Initialize Low-Level SkillSet
        self.low_level_skillset = SkillSet(level="low")

        # --- DYNAMIC LOADING FROM JSON ---
        self._load_low_level_skills_from_json()
        
        # 2. Initialize High-Level SkillSet
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        self._load_high_level_skills()

        # 3. Link Planner
        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, 
                          low_level_skillset=self.low_level_skillset, 
                          vision_skill=self.vision)

    def _load_low_level_skills_from_json(self):
        """
        Reads skills from JSON and resolves string handlers (e.g., 'drone.takeoff') 
        to actual Python methods using getattr.
        """
        from controller.utils.constants import LOW_LEVEL_SKILL_CONFIG_PATH

        if not os.path.exists(LOW_LEVEL_SKILL_CONFIG_PATH):
            print_t(f"[Error] Low level skill config not found at {LOW_LEVEL_SKILL_CONFIG_PATH}")
            return

        try:
            with open(LOW_LEVEL_SKILL_CONFIG_PATH, 'r') as f:
                skills_data = json.load(f)
            
            # Type mapping: String in JSON -> Python Type Class
            type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "Optional[str]": Optional[str]
            }

            for skill_def in skills_data:
                try:
                    # 1. Resolve the handler function
                    handler_path = skill_def["handler"]
                    handler_parts = handler_path.split('.')
                    
                    current_obj = self
                    for part in handler_parts:
                        current_obj = getattr(current_obj, part)
                    
                    handler_func = current_obj

                    # 2. Build Arguments
                    skill_args = []
                    for arg in skill_def.get("args", []):
                        arg_type_str = arg["type"]
                        py_type = type_map.get(arg_type_str, str) 
                        skill_args.append(SkillArg(arg["name"], py_type))

                    # 3. Register Skill
                    # Instantiates LowLevelSkillItem with correct keyword args found in skillset.py
                    self.low_level_skillset.add_skill(
                        LowLevelSkillItem(
                            skill_name=skill_def["name"],
                            skill_callable=handler_func,
                            skill_description=skill_def["description"],
                            args=skill_args
                        )
                    )

                except AttributeError as e:
                    print_t(f"[Warning] Could not load skill '{skill_def.get('name')}': Handler '{skill_def.get('handler')}' not found. {e}")
                except Exception as e:
                    print_t(f"[Error] Failed to load skill '{skill_def.get('name', 'Unknown')}': {e}")

        except Exception as e:
            print_t(f"[Error] Critical failure loading low-level skills: {e}")

    def _load_high_level_skills(self):
        """Loads common and user-specific skills from JSON."""
        common_skills = []
        user_skills = []
        
        # Ensure user skill file exists
        if not os.path.exists(self.user_high_level_skill_path):
            open(self.user_high_level_skill_path, "w").close()
            
            # If new user, load common defaults
            with open(HIGH_LEVEL_SKILL_PATH, "r") as f:
                common_skills = json.load(f)
                for skill in common_skills:
                    self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))
        else:
            # Load existing user skills
            with open(self.user_high_level_skill_path, "r") as f:
                try:
                    user_skills = json.load(f)
                    for skill in user_skills:
                        self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))
                except json.JSONDecodeError:
                    print_t("[C] Warning: User skills file empty or corrupted.")

        # Save merged state
        with open(self.user_high_level_skill_path, "w") as f:
            json.dump(common_skills + user_skills, f, indent=4)

    def set_username(self, username, long_memory_flag: bool = True) -> Tuple[None, False]:
        self.username = username
        self.user_high_level_skill_path = os.path.join(SKILL_PATH, self.username, "user_high_level_skills.json")
        if long_memory_flag:
            self.long_memory_module.change_username(username)
        return None, False
    
    def get_username(self) -> str:
        return self.username 

    def save_task_user_feedback(self, user_feedback: str) -> Tuple[None, False]:
        if self.last_task is not None:
            self.last_task.set_user_feedback(user_feedback)
            self.long_memory_module.save_task_summary(self.last_task, 
                                                      high_level_skills=self.high_level_skillset,
                                                      low_level_skills=self.low_level_skillset)
        else:
            self.append_message("\nNo previous task saved.")
        return None, False
    
    def delete_task_user_feedback(self, user_feedback: str) -> Tuple[None, False]:
        self.long_memory_module.delete_task_user_feedback(user_feedback)
        return None, False
    
    def change_evergreen_user_feedback(self, user_request: str) -> Tuple[None, False]:
        self.long_memory_module.change_feedback_prompt(user_request)
        return None, False
    
    def save_shortcut(self, shortcut: str) -> Tuple[None, False]:
        if self.last_task is not None:
            self.long_memory_module.save_shortcut_task(shortcut, self.last_task)
        else:
            self.append_message("\nNo previous task saved.")
        return None, False
    
    def delete_shortcut(self, shortcut: str) -> Tuple[None, False]:
        self.long_memory_module.delete_shortcut_task(shortcut)
        return None, False
    
    def execute_shortcut(self, shortcut: str) -> Tuple[None, False]:
        self.execute_task_description(is_shortcut=True, shortcut=shortcut)
        return None, False
    
    def get_drone(self) -> RobotWrapper:
        return self.drone
    
    def get_flyzone_manager(self) -> FlyzoneManager:
        return self.flyzone_manager
    
    def get_drone_pose(self):
        return self.drone.get_position()

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
        if isinstance(x, float):
            if x > 0.55:
                self.drone.turn_cw(int((x - 0.5) * 70))
            elif x < 0.45:
                self.drone.turn_ccw(int((0.5 - x) * 70))

        self.drone.move_north(110)
        return None, False
    
    def explore_new_region(self, direction: int, distance: int = REGION_THRESHOLD) -> Tuple[None, bool]:
        match direction:
            case 0: self.drone.move_north(distance_cm=distance)
            case 180: self.drone.move_south(distance_cm=distance)
            case -90: self.drone.move_west(distance_cm=distance)
            case 90: self.drone.move_east(distance_cm=distance)
            case _: self.drone.move_direction(direction, distance)
        return None, False

    def _name_region(self, region_name: str) -> Tuple[None, bool]:
        self.graph_manager.name_region(region_name)        

    def create_graph(self, description: Optional[str], image_present: bool = False) -> Tuple[None, bool]:
        image = None
        if image_present:
            image = encode_image(FLYZONE_USER_IMAGE_PATH)
        
        graph = self.graph_manager.request_new_graph(description, image).get("context_graph")
        print(graph)
        with open(GRAPH_TXT_PATH, "w") as f:
            json.dump(graph, f, indent=4)
        self.graph_manager.update_graph_from_file()
        return None, False

    def skill_take_picture(self) -> Tuple[None, bool]:
        time.sleep(0.1)
        if self.images_counter == 0:
            self.env_analysis_module.reset_updated_directions()
            
        direction = self.directions.get(self.images_counter)
        img_path = os.path.join(self.cache_folder, f"{direction}.jpg")
        
        self.env_analysis_module.set_updated_directions(direction)
        self.images_counter = (self.images_counter + 1) % 8
        
        if self.latest_frame is not None:
            Image.fromarray(self.latest_frame).save(img_path)
            print_t(f"[C] Picture saved to {img_path}")
            self.append_message((img_path,))
        else:
            print_t("[C] Error: No frame available to take picture")
            
        return None, False
    
    def skill_explore_direction(self, hint: Optional[str]) -> Tuple[None, bool]:
        (direction, distance, region_name) = self.env_analysis_module.choose_direction(
            self.current_task.get_task_description(), 
            self.cache_folder, 
            self.get_drone_pose(), 
            hint
        )
        self._name_region(region_name)
        valid_directions = ["north", "east", "south", "west", "north-east", "north-west", "south-east", "south-west"]
        if direction in valid_directions:
            print(f"{direction} is a valid direction")
            next_yaw = {
                "north": 0, "north-east": 45, "east": 90, "south-east": 135, 
                "south": 180, "south-west": -135, "west": -90, "north-west": -45
            }[direction]
            print(f"Next yaw {next_yaw}")
            self.drone.move_direction(next_yaw, distance)
        return 0, False
    
    def text_to_speech(self, text: str):
        """Convert text to speech using gTTS and play via system audio (server-side)."""
        try:
            tts = gTTS(text, lang="en")
            speech_path = os.path.join(self.cache_folder, "speech.mp3")
            tts.save(speech_path)
            os.system(f"mpg123 {speech_path}") 
        except Exception as e:
            print_t(f"[C] TTS Error: {e}")

    def skill_log(self, text: str) -> Tuple[None, bool]:
        text = str(text)

        # --- 1. HANDLE UI DISPLAY ---
        # Convert newlines to HTML breaks for the Gradio Chatbot
        display_text = text.replace('\n', '<br>')
        self.append_message(display_text)

        # --- 2. HANDLE AUDIO (CLEANING) ---
        # Create a separate version just for the TTS engine
        audio_text = text

        # Fix A: Replace actual newlines with a pause (comma or space)
        # This prevents words sticking together like "Hello\nWorld" -> "HelloWorld"
        audio_text = audio_text.replace('\n', ', ')
        
        # Fix B: Remove literal "1n" or "\n" string artifacts if they exist
        # If your TTS literally says "One N", this cleans it.
        audio_text = audio_text.replace('\\n', ' ')

        # Fix C: Remove Markdown/Special chars (so it doesn't say "Star Star Hello Star Star")
        # This regex removes special characters but keeps text, numbers, and basic punctuation
        audio_text = re.sub(r'[*_#`]', '', audio_text)

        # Send the CLEAN text to the robot
        # self.text_to_speech(audio_text)

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
        print(f"Skill added: {skill_name}")

        skills = []
        if os.path.exists(self.user_high_level_skill_path):
            with open(self.user_high_level_skill_path, "r") as f:
                try:
                    skills = json.load(f)
                    if not isinstance(skills, list): skills = []
                except:
                    skills = []

        skills = [s for s in skills if s.get("skill_name") != skill_name]

        skills.append({
            "skill_name": skill_name,
            "skill_description": description,
            "definition": minispec_def
        })

        with open(self.user_high_level_skill_path, "w") as f:
            json.dump(skills, f, indent=4)

        Statement.high_level_skillset.update(self.user_high_level_skill_path)
        return True, False
    
    def delete_skill(self, skill_name) -> Tuple[None, bool]:
        if os.path.exists(self.user_high_level_skill_path):
            with open(self.user_high_level_skill_path, "r") as f:
                skills = json.load(f)
            
            skills = [s for s in skills if s.get("skill_name") != skill_name]

            with open(self.user_high_level_skill_path, "w") as f:
                json.dump(skills, f, indent=4)
        return None, False

    def skill_ask_user(self, question: str) -> Tuple[None, bool, bool]:
        self.append_message(f"{question}")
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
            YoloClient.plot_results_oi(image, objects_list)
        return image
    
    def execute_minispec(self, minispec: str, iteration_description: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        ret_val = interpreter.execute([minispec])
        self.current_task.update_execution_history(interpreter.execution_history, iteration_description)
        return ret_val

    def execute_task_description(self, task_description: str = "", img_b64 = None, is_shortcut: bool = False, shortcut: str = ""):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
            
        if is_shortcut:
            assert shortcut != "", "shortcut should be not empty"
            current_task_dict = self.long_memory_module.get_shortcut_task(shortcut=shortcut)
            self.current_task = Task(
                task_description=current_task_dict['task_description'], 
                execution_history=current_task_dict['execution_history'],
                current_plan=current_task_dict['current_plan'],
                user_feedback=current_task_dict['user_feedback'],
                is_new=False
            )
            if not self.current_task:
                self.append_message("Sorry. Given shortcut is not existing")
                return
        else:
            assert task_description != "", "task_description should be not empty"
            self.current_task = Task(task_description)

        # self.append_message('[TASK]: ' + task_description)
        ret_val = None
        
        while True:
            self.current_plan, requires_execution, iteration_description = self.planner.plan(
                self.current_task,
                img_b64=img_b64,
                execution_history=self.current_task.get_execution_history(), 
                context_graph=self.graph_manager.get_graph(), 
                current_position=self.graph_manager.get_drone_pose(), 
                current_region=self.graph_manager.get_current_region(),
                old_interactions_feedbacks=self.long_memory_module.retrieve_old_interactions(self.current_task.get_task_description()),
            )

            if requires_execution:
                self.last_task = self.current_task

            if not self.current_plan:
                continue

            self.current_task.set_current_plan(self.current_plan)
            # self.append_message(f'[Plan]: {self.current_plan}')
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
                break
                
        print("[Task ended]")
        self.append_message(f'\n[Task ended]')
        self.append_message('end')
        self.current_plan = None

    def continue_execution(self): 
        pass # TODO

    def start_robot(self):
        print_t("[C] Connecting to robot...")
        self.drone.connect()
        print_t("[C] Starting robot...")
        self.drone.takeoff()
        print_t("[C] Starting stream...")
        self.drone.start_stream()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        
        while self.controller_active:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            self.planner.update_latest_frame(self.latest_frame)
            
            frame = Frame(
                frame_reader.frame,
                frame_reader.depth if hasattr(frame_reader, 'depth') else None
            )
            
            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
                
            time.sleep(0.10)

        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")