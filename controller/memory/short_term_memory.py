import os
from controller.utils.constants import ROBOT_NAME
from controller.llm.llm_wrapper import GPT5_NANO, LLMWrapper, RequestType
from controller.task import Task
from controller.utils.general_utils import print_t
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

type_folder_name = 'tello'
TASK_ID_FILE = "task_id.json"
MEMORY_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/memory"

class ShortTermMemory:
    '''
    '''
    def __init__(self):
        self.llm_wrapper = LLMWrapper()
        with open(os.path.join(MEMORY_PATH, "user_short_term_memory_prompt.txt"), "r") as f:
            self.short_term_memory_prompt = f.read()
            
    def generate_interaction_summary(self, task: Task, context_graph, low_level_skills, high_level_skills):
        '''
        Save in memory a summary of the last iteraction of current task, in order to keep task status.
        '''
        
        prompt = self.short_term_memory_prompt.format(last_iteration=task.get_last_iteration(),
                                                 previously_iterations=task.get_execution_plan_summary_prompt()[:-1],
                                                 context_graph= context_graph,
                                                 low_level_skills=low_level_skills,
                                                 high_level_skills=high_level_skills)
        
        # Send the request to gpt5-nano, because we just need to summarize information
        response_content = self.llm_wrapper.request(user_prompt=prompt, 
                                                    request_type=RequestType.SHORT_TERM_MEMORY, 
                                                    model_name=GPT5_NANO)

        # Parse the response
        iteration_summary = response_content.get("iteration_summary")

        # Debug print
        # doc_text = f"""
        # Iteration Summary: {iteration_summary}
        # """
        # print_t(doc_text)
        task.add_last_iteration_summary(iteration_summary)

        return iteration_summary