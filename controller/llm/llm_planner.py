import json
import os, ast
from typing import Optional, Sequence
from PIL import Image

from controller.abs.skill_item import SkillItem
from controller.utils.constants import ROBOT_NAME, X_BOUND, Y_BOUND
from controller.task import Task

from ..skillset import HighLevelSkillItem, SkillSet
from .llm_wrapper import GPT5_MINI, LLMWrapper, GPT3, GPT4, GPT5, RequestType
from ..visual_sensing.vision_skill_wrapper import VisionSkillWrapper
from ..utils.general_utils import encode_image, print_t
from ..minispec_interpreter import MiniSpecValueType, evaluate_value
from ..abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, robot_type: RobotType, current_task: Task, latest_frame):
        self.llm = LLMWrapper()
        self.current_task = current_task

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/plan/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()
    
        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/plan/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/probe/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/plan/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, f"../assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        self.flyzone = ""
        self.latest_frame = latest_frame

        self.current_task_id = 0
    

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def update_latest_frame(self, latest_frame):
        self.latest_frame = latest_frame

    def plan(self, task: Task, context_graph: str, current_position: Sequence[float], current_region: str, objects_list: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None, old_interactions_feedbacks: Optional[list[str]] = None, model_name: Optional[str] = GPT5):
        # by default, the task_description is an action
        if not task.get_task_description().startswith("["):
            task_description = "[A] " + task.get_task_description()

        if objects_list is None:
            objects_list = self.vision_skill.get_obj_list()

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

        # Retrieve updated flyzone
        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/flyzone/flyzone.txt"), "r") as f:
            self.flyzone = f.read()

        with open(os.path.join(CURRENT_DIR, f"../assets/{ROBOT_NAME}/demo.json"), "r") as f:
            self.demo = json.loads(f.read())

        if task.get_is_new(): # task is new, so not through shortcut
            task_description = task.get_task_description()
        else: # task is executed through shortcut, so we pass all the information already available
            task_description = task.to_prompt()
                
        prompt = self.prompt_plan.format(high_level_skills=self.high_level_skillset,
                                            low_level_skills=self.low_level_skillset,
                                            guides=self.guides,
                                            plan_examples=self.plan_examples,
                                            old_interactions_feedbacks = old_interactions_feedbacks,
                                            error_message=error_message,
                                            objects_list=objects_list,
                                            task_description=task_description,
                                            execution_history=execution_history,
                                            context_graph=context_graph,
                                            current_position=current_position,
                                            current_region=current_region,
                                            minispec_syntax=self.minispec_syntax,
                                            flyzone=self.flyzone,
                                            )
        
        #print(prompt)
        print_t(f"[P] Planning request: {task_description}")

        # response_content = self.llm.request(prompt, model_name=model_name, stream=False, request_type=RequestType.SIMPLE)

        # Clean up the content - remove markdown code blocks if present
        # if response_content.startswith("```json"):
        #     response_content = response_content.replace("```json", "").replace("```", "").strip()

        # if not response_content: # resend the request
        #     return None, None
        
        # parsed = json.loads(response_content)
        step = self.demo.get(f"{self.current_task_id}", None)
        task = step.get("task")
        print(f"TASK: {task}")
        plan = step.get("plan")
        print(f"PLAN: {plan}")
        self.current_task_id += 1
        # reason = parsed.get("reason", None)
        # iteration_description = parsed.get("description", None)
        # iteration_description = "Description of the iteration: " + iteration_description
        return plan, "", ""
    
    def probe(self, question) -> MiniSpecValueType:
        if question is list:
            question: str = question[0]
        objects_list = self.vision_skill.get_obj_list()
        image = self.vision_skill # returns an image in a format accepted by the LLM (e.g. PIL.Image or file path or bytes)
        # image = None # for now image is not passed to LLM because it is not working
        prompt = self.prompt_probe.format(objects_list=objects_list, question=question)
        print_t(f"[P] Probing question: {question}")
        self.image_path = "serving/webui/cache/probe.jpg"
        Image.fromarray(self.latest_frame).save(self.image_path)
        image = encode_image(Image.open(self.image_path))
        return evaluate_value(self.llm.request(prompt=prompt, image=image, model_name=GPT5_MINI, request_type=RequestType.SINGLE_IMAGE)), False
    
    # TODO: at the end of the iteration, the llm has to reason if the task has endend, still in progress or can be ended because not feasible
    # def probe_end_iteration(self, model_name: Optional[str] = GPT5):
    #     task_description = self.current_task.get_task_description()
    #     execution_history = self.current_task.get_execution_history()
    #     achievements_summary = self.current_task.get_last_achievements()
    #     drone_position = self.current_task.get_drone_position()
    #     battery_percent = self.current_task.get_battery_percent()
    #     objects_list = self.current_task.get_objects_list() #TODO: probably this should be handle in a different way
    #     graph_json = self.current_task.get_graph_json()
    #     prompt = self.prompt_probe_end_iteration.format(task_description=task_description, execution_history=execution_history, achievements_summary=achievements_summary, drone_position=drone_position, battery_percent=battery_percent, objects_list=objects_list, graph_json=graph_json)
    #     decision_json = self.llm.request(prompt=prompt, model_name=model_name)
    #     decision = json.load(decision_json)
    #     return decision