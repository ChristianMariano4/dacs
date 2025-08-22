from datetime import datetime
from io import BytesIO
import json
from typing import Optional
from PIL import Image
import numpy as np
import base64
from openai import OpenAI
import os
from controller.constants import ROBOT_NAME, X_BOUND, Y_BOUND
from controller.llm.llm_wrapper import LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer
from controller.shared_frame import SharedFrame
from controller.task import Task
from controller.utils import encode_image

from sentence_transformers import SentenceTransformer
import chromadb

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
import os

load_dotenv()

type_folder_name = 'tello'

'''
Can be used to bot get a description of the current scene and to get a more high-level description of the context where the drone is
'''
class LongMemoryModule:
    '''
    LongMemoryModule has the goal of put in long memory (a document saved on disk)
    a summary of the interaction obtained with the user in the last task, together
    user's feedback. In this way we can save in a vectorial db all the feedbacks
    given by the user, and use them after in related tasks.
    Then when a new task arrives, we retrieve from vectorial db the related feedbacks,
    by similarity with the current task.
    '''
    def __init__(self, middle_layer: MiddleLayer=None):
        # Set up your client
        self.client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Or use environment variable OPENAI_API_KEY
        self.middle_layer = middle_layer
        self.llm_wrapper = LLMWrapper()

        with open(f"controller/assets/{ROBOT_NAME}/memory/prompt_memory.txt", "r") as f:
            self.memory_prompt = f.read()

        # vector db section
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.client_chromadb = chromadb.PersistentClient(path="/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/tello/memory")
        self.interactions_collection = self.client_chromadb.create_collection(name="interaction_memory")
        self.task_id = 0

            
    def save_interaction_summary(self, task: Task, conf=0.3):
        
        prompt = self.memory_prompt.format(task_text=task.get_task_description(), 
                                        execution_output=task.get_execution_history(), 
                                        feedback_text=task.get_user_feedback(),
                                        )
        
        # Send the request
        response_content = self.llm_wrapper.request(prompt, request_type=RequestType.SIMPLE)
        response_parsed = json.loads(response_content)

        task_summary = response_parsed.get("task_summary", "No task_summary provided")
        execution_summary = response_parsed.get("execution_summary", "No execution_summary provided")
        feedback_summary = response_parsed.get("feedback_summary", "No feedback_summary provided")
        lessons_learned = response_parsed.get("lessons_learned", "No lessons_learned provided")

        doc_text = f"""
        Task Summary: {task_summary}
        Execution Summary: {execution_summary}
        Feedback Summary: {feedback_summary}
        Lessons Learned: {lessons_learned}
        """

        embedding = self.model.encode(doc_text)
        self.interactions_collection.add(
            documents=[doc_text],
            embeddings=[embedding.tolist()],
            ids=[f"task_{self.task_id}"]
        )
        self.task_id += 1
    
    def retrieve_old_interactions(self, new_task: str):
        '''Function to retrieve old useful feedbacks, based on new task'''
        new_task_embedding = self.model.encode(new_task)

        # Search for similar scenes
        results = self.interactions_collection.query(query_embeddings=[new_task_embedding.tolist()], n_results=3)
        print(results)