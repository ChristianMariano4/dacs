import json
import os, ast
from typing import Optional, Sequence

from controller.abs.skill_item import SkillItem
from controller.assets.tello.examples import get_graph_examples
from controller.constants import X_BOUND, Y_BOUND
from controller.task import Task

from .skillset import HighLevelSkillItem, SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4, RequestType
from .visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, robot_type: RobotType, current_task: Task):
        self.llm = LLMWrapper()
        self.current_task = current_task
        self.model_name = GPT4

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe_end_iteration.txt"), "r") as f:
            self.prompt_probe_end_iteration = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/new/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        self.context_graph_examples = get_graph_examples()

    def set_model(self, model_name):
        self.model_name = model_name

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def plan(self, task_description: str, context_graph: str, current_position: Sequence[float], current_region: str, scene_description: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()

        type_folder_name = 'tello'
        # self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        # SkillItem.abbr_dict = {}
        #TODO: fix high level skill updating
        # with open(os.path.join(CURRENT_DIR, f"assets/{type_folder_name}/high_level_skills.json"), "r") as f:
        #     json_data = json.load(f)
        #     for skill in json_data:
        #         if skill['skill_name'] not in SkillItem.abbr_dict.keys():
        #             self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))


        # - top-left: [{x_top_left}, {y_top_left}]
        # - top-right: [{x_top_right}, {y_top_right}]
        # - bottom-right: [{x_bottom_right}, {y_bottom_right}]
        # - bottom-left: [{x_bottom_left}, {y_bottom_left}]
                
        prompt = self.prompt_plan.format(system_skill_description_high=self.high_level_skillset,
                                            system_skill_description_low=self.low_level_skillset,
                                            guides=self.guides,
                                            plan_examples=self.plan_examples,
                                            error_message=error_message,
                                            scene_description=scene_description,
                                            task_description=task_description,
                                            execution_history=execution_history,
                                            context_graph=context_graph,
                                            current_position=current_position,
                                            current_region=current_region,
                                            minispec_syntax=self.minispec_syntax,
                                            x_top=X_BOUND,
                                            x_bottom=-X_BOUND,
                                            y_left=-Y_BOUND,
                                            y_right=Y_BOUND,
                                            )
        
        #print(prompt)
        print_t(f"[P] Planning request: {task_description}")

        response_content = self.llm.request(prompt, model_name=self.model_name, stream=False, request_type=RequestType.SIMPLE)

        # Clean up the content - remove markdown code blocks if present
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()

        if not response_content: # resend the request
            return None, None
        
        parsed = json.loads(response_content)
        plan = parsed.get("plan", None)
        reason = parsed.get("reason", None)
        return plan, reason
    
    def probe(self, question: str) -> MiniSpecValueType:
        objects_list = self.vision_skill.get_obj_list()
        image = self.vision_skill.get_current_image() # returns an image in a format accepted by the LLM (e.g. PIL.Image or file path or bytes)
        image = None # for now image is not passed to LLM because it is not working
        prompt = self.prompt_probe.format(scene_description=objects_list, question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt, image, self.model_name)), False
    
    def probe_end_iteration(self):
        task_description = self.current_task.get_task_description()
        execution_history = self.current_task.get_execution_history()
        achievements_summary = self.current_task.get_last_achievements()
        drone_position = self.current_task.get_drone_position()
        battery_percent = self.current_task.get_battery_percent()
        objects_list = self.current_task.get_objects_list() #TODO: probably this should be handle in a different way
        graph_json = self.current_task.get_graph_json()
        prompt = self.prompt_probe_end_iteration.format(task_description=task_description, execution_history=execution_history, achievements_summary=achievements_summary, drone_position=drone_position, battery_percent=battery_percent, objects_list=objects_list, graph_json=graph_json)
        decision_json = self.llm.request(prompt, self.model_name)
        decision = json.load(decision_json)
        return decision