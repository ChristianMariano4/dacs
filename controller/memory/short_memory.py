import os
from controller.utils.constants import ROBOT_NAME
from controller.llm.llm_wrapper import GPT5_NANO, LLMWrapper, RequestType
from controller.task import Task
from controller.utils.general_utils import print_t


GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
import os

load_dotenv()

type_folder_name = 'tello'
TASK_ID_FILE = "task_id.json"
MEMORY_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/memory"

class ShortMemoryModule:
    '''
    '''
    def __init__(self):
        self.llm_wrapper = LLMWrapper()
        with open(os.path.join(MEMORY_PATH, "user_short_memory_prompt.txt"), "r") as f:
            self.short_memory_prompt = f.read()
            
    def generate_interaction_summary(self, task: Task, context_graph, low_level_skills, high_level_skills):
        '''
        Save in memory a summary of the last iteraction of current task, in order to keep task status.
        '''
        
        prompt = self.short_memory_prompt.format(last_iteration=task.get_last_iteration(),
                                                 previously_iterations=task.get_execution_plan_summary_prompt()[:-1],
                                                 context_graph= context_graph,
                                                 low_level_skills=low_level_skills,
                                                 high_level_skills=high_level_skills)
        
        # Send the request to gpt5-nano, because we just need to summarize information
        response_content = self.llm_wrapper.request(user_prompt=prompt, 
                                                    request_type=RequestType.SHORT_MEMORY, 
                                                    model_name=GPT5_NANO)

        # Parse the response
        # response_parsed = json.loads(response_content)
        # iteration_summary = response_parsed.get("iteration_summary", "No iteration_summary provided")
        # task_summary = response_parsed.get("task_summary", "no task_summary provided")
        iteration_summary = response_content.get("iteration_summary")
        task_summary = response_content.get("task_summary")

        doc_text = f"""
        Iteration Summary: {iteration_summary}
        """
        print_t(doc_text)
        task.add_last_iteration_summary(iteration_summary)
        task.update_task_summary(task_summary)

        return iteration_summary