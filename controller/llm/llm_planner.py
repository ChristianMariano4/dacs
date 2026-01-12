import os
from typing import Optional, Sequence, Tuple, List, Any
from PIL import Image

from controller.middle_layer.flyzone_manager import FlyzoneManager
from controller.utils.constants import ROBOT_NAME, USER_EVERGREEN_FEEDBACK_PATH
from controller.task import Task

from ..skillset import SkillSet
from .llm_wrapper import GPT5_MINI, LLMWrapper, GPT5, RequestType
from ..visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from ..utils.general_utils import encode_image, print_t
from ..minispec_interpreter import MiniSpecValueType, evaluate_value
from ..abs.robot_wrapper import CommandResult, RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_PROMPT_PATH = os.path.join(CURRENT_DIR, "../assets/tello/plan/user_plan_prompt.txt")
QUERY_PROMPT_PATH = os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/query/user_query_prompt.txt")
FLYZONE_PATH = os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/flyzone/flyzone.txt")

class LLMPlanner:
    def __init__(self, robot_type: RobotType, current_task: Task, latest_frame, flyzone_manager: FlyzoneManager, cache_folder: str = None):
        self.llm = LLMWrapper()
        self.current_task = current_task
        self.latest_frame = latest_frame
        self.flyzone_manager = flyzone_manager
        self.flyzone = ""
        
        # Handle cache path safely
        self.cache_folder = cache_folder if cache_folder else os.path.join(CURRENT_DIR, "cache")
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # Load prompts
        try:
            with open(QUERY_PROMPT_PATH, "r") as f:
                self.prompt_query = f.read()
            
            with open(PLAN_PROMPT_PATH, "r") as f:
                self.prompt_plan = f.read()
        except FileNotFoundError as e:
            print_t(f"[Error] Could not load prompt files: {e}")
            self.prompt_query = ""
            self.prompt_plan = ""

        # Skillsets initialized via init()
        self.high_level_skillset: Optional[SkillSet] = None
        self.low_level_skillset: Optional[SkillSet] = None
        self.vision_skill: Optional[VisionSkillWrapper] = None

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        """Dependency injection for skills."""
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def update_latest_frame(self, latest_frame):
        self.latest_frame = latest_frame

    def plan(self, task: Task, img_b64: str, context_graph: str,
             objects_list: Optional[str] = None, execution_history: Optional[str] = None,
             old_interactions_feedbacks: Optional[List[str]] = None):
        
        # Format task description
        task_description = task.get_task_description()
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if objects_list is None and self.vision_skill:
            objects_list = self.vision_skill.get_obj_list()

        # Retrieve updated flyzone
        if os.path.exists(FLYZONE_PATH):
            with open(FLYZONE_PATH, "r") as f:
                self.flyzone = f.read()

        # Retrieve updated evergreen_preferences
        evergreen_preferences = ""
        if os.path.exists(USER_EVERGREEN_FEEDBACK_PATH):
            with open(USER_EVERGREEN_FEEDBACK_PATH, "r") as f:
                evergreen_preferences = f.read()

        # Handle Shortcut formatting
        if not task.get_is_new():
            task_description = task.to_prompt()
                
        prompt = self.prompt_plan.format(
            high_level_skills=self.high_level_skillset,
            low_level_skills=self.low_level_skillset,
            old_interactions_feedbacks=old_interactions_feedbacks,
            evergreen_preferences=evergreen_preferences,
            objects_list=objects_list,
            task_description=task_description,
            execution_history=execution_history,
            context_graph=context_graph,
            flyzone=self.flyzone,
        )
        
        print_t(f"[P] Planning request: {task_description}")

        response_json = self.llm.request(prompt, request_type=RequestType.PLAN, image=img_b64)
        response_plan = response_json.get("plan")
        requires_execution = response_json.get("requires_execution", True)

        return response_plan, requires_execution
    
    def query(self, question) -> Tuple[MiniSpecValueType, bool]:
        # Fix: Use isinstance instead of 'is' for type checking
        if isinstance(question, list):
            question = question[0]
            
        objects_list = self.vision_skill.get_obj_list() if self.vision_skill else []
        prompt = self.prompt_query.format(objects_list=objects_list, question=question)
        
        print_t(f"[P] Querying question: {question}")
        
        # Fix: Save to the centralized cache folder instead of hardcoded path
        image_path = os.path.join(self.cache_folder, "query.jpg")
        
        if self.latest_frame is not None:
            Image.fromarray(self.latest_frame).save(image_path)
            image = encode_image(Image.open(image_path))
            
            answer = self.llm.request(
                user_prompt=prompt, 
                image=image, 
                model_name=GPT5_MINI, 
                request_type=RequestType.QUERY
            )["answer"]
            
            return CommandResult(value=answer, replan=False)
        else:
            print_t("[Error] No frame available for probing")
            return CommandResult(value=evaluate_value("I cannot see anything right now."), replan=False)