from datetime import datetime
from io import BytesIO
import json
from typing import Optional
from PIL import Image
import numpy as np
import base64
import uuid
from openai import OpenAI
import os
from controller.utils.constants import ROBOT_NAME, USER_EVERGREEN_FEEDBACK_PATH, USER_EVERGREEN_FEEDBACK_PROMPT_PATH, USER_MEMORY_PATH, USER_PLAN_PROMPT_PATH
from controller.llm.llm_wrapper import GPT5_NANO, LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer
from controller.shared_frame import SharedFrame
from controller.task import Task
from controller.utils.general_utils import encode_image, print_t
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
import os

load_dotenv()

type_folder_name = 'tello'
TASK_ID_FILE = "task_id.json"
MEMORY_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/memory"
SHORTCUTS_PATH = os.path.join(USER_MEMORY_PATH, "shortcuts.json")

class LongMemoryModule:
    '''
    LongMemoryModule has the goal of put in long memory (a document saved on disk)
    a summary of the interaction obtained with the user in the last task, together
    user's feedback. In this way we can save in a vectorial db all the feedbacks
    given by the user, and use them after in related tasks.
    Then when a new task arrives, we retrieve from vectorial db the related feedbacks,
    by similarity with the current task.
    '''
    def __init__(self, username, middle_layer: MiddleLayer=None, shortcuts_file="controller/assets/tello/memory/shortcuts.json"):
        self.client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.middle_layer = middle_layer # TODO: not yet used 
        self.llm_wrapper = LLMWrapper()
        with open(os.path.join(MEMORY_PATH, "user_memory_prompt.txt"), "r") as f:
            self.memory_prompt = f.read()
        self.username = username # used to retrieve the correct vector db
        
        with open(USER_EVERGREEN_FEEDBACK_PROMPT_PATH, "r") as f:
            self.user_evergreen_feedback_prompt = f.read()

        # Vector db section: retrieve the db of the specified user
        self.openai_client = OpenAI()
        self.client_chromadb = chromadb.PersistentClient(path=os.path.join(MEMORY_PATH, self.username))
        self.interactions_collection = self.client_chromadb.get_or_create_collection(name="interaction_memory")

        # Shortcuts: tasks associated to keywords
        self.user_shortcut_tasks = self._load_shortcuts()

    def change_username(self, username):
        self.username = username
        self.user_shortcut_tasks = self._load_shortcuts()
        self.client_chromadb = chromadb.PersistentClient(path=os.path.join(MEMORY_PATH, self.username))
        self.interactions_collection = self.client_chromadb.get_or_create_collection(name="interaction_memory")
    
    def get_username(self):
        return self.username

    def load_last_task_id(self):
        if os.path.exists(os.path.join(MEMORY_PATH, self.username, TASK_ID_FILE)):
            with open(os.path.join(MEMORY_PATH, self.username, TASK_ID_FILE), "r") as f:
                data = json.load(f)
                return data.get("last_task_id", 0)
        elif not os.path.exists(os.path.join(MEMORY_PATH, self.username)):
            raise RuntimeError(f"Not existing user directory for {self.username}")
        return 0
    
    def save_last_task_id(self, task_id):
        with open(os.path.join(MEMORY_PATH, self.username, TASK_ID_FILE), "w") as f:
            json.dump({"last_task_id": task_id}, f)
    
    def get_next_task_id(self):
        last_id = self.load_last_task_id()
        next_id = last_id + 1
        self.save_last_task_id(next_id)
        return next_id
    
    def change_feedback_prompt(self, user_request: str):
        with open(USER_EVERGREEN_FEEDBACK_PATH, "r") as f:
            current_user_feedback = f.read()
        prompt = self.user_evergreen_feedback_prompt.format(user_request=user_request,
                                                            preferences_summary=current_user_feedback)
        response_content: dict = self.llm_wrapper.request(prompt, RequestType.EVERGREEN_FEEDBACK)
        preferences_summary = response_content.get("preferences_summary", None)

        if preferences_summary == None:
            print_t("[ERROR in LongMemoryModule.change_feedback_prompt()] preferences_summary requested but None returned by LLM")
            return
        
        with open(USER_EVERGREEN_FEEDBACK_PATH, "w") as f:
            f.write(preferences_summary)
            
    def save_task_summary(self, task: Task, high_level_skills, low_level_skills, conf=0.3):
        '''
        Save in memory a summary of the executed task, in order to keep some lessons learned.
        '''

        task_text = task.get_task_description()
        prompt = self.memory_prompt.format(task_text=task_text, 
                                        execution_output=task.get_execution_history(), 
                                        feedback_text=task.get_user_feedback(),
                                        high_level_skills=high_level_skills,
                                        low_level_skills=low_level_skills
                                        )
        
        # Send the request to gpt5-nano, because we just need to summarize information
        response_content: dict = self.llm_wrapper.request(user_prompt=prompt, request_type=RequestType.FEEDBACK, model_name=GPT5_NANO)

        # Parse the response
        task_summary = response_content.get("task_summary", "No task_summary provided")
        execution_summary = response_content.get("execution_summary", "No execution_summary provided")
        feedback_summary = response_content.get("feedback_summary", "No feedback_summary provided")
        lessons_learned = response_content.get("lessons_learned", "No lessons_learned provided")

        doc_text = f"""
        Task: {task_text}
        Task Summary: {task_summary}
        Execution Summary: {execution_summary}
        Feedback Summary: {feedback_summary}
        Lessons Learned: {lessons_learned}
        """

        print(f"Interaction summary saved:\n{doc_text}")

        # Add summary to the vector db embeding only task_text, that will be the search key
        # embedding = self.model.encode(doc_text)
        embedding = self.openai_client.embeddings.create(
            input=task_text,
            model="text-embedding-3-large"
        )
        embedding_vector = embedding.data[0].embedding
        self.interactions_collection.add(
            documents=[doc_text],
            embeddings=[embedding_vector],
            ids=[f"fb_{uuid.uuid4()}"]
        )

    def delete_task_user_feedback(self, user_feedback: str, similarity_threshold: float = 0.3):
        """Delete old feedback similar to the given one."""
        # embedding = self.model.encode(user_feedback)
        embedding = self.openai_client.embeddings.create(
            input=user_feedback,
            model="text-embedding-3-large"
        )
        embedding_vector = embedding.data[0].embedding
        all_docs = self.interactions_collection.get()

        if not all_docs['ids']:
            return {"deleted_count": 0}
        
        results = self.interactions_collection.query(
            query_embeddings=[embedding_vector],
            n_results=len(all_docs["ids"])
        )

        ids_to_delete = self.llm_wrapper.request(request_type=RequestType.RETRIEVE_TASK_FEEDBACK,
                                 variables=(user_feedback, results))

        if ids_to_delete:
            self.interactions_collection.delete(ids=ids_to_delete['relevant_ids'])
            print(f"Deleted {len(ids_to_delete)} similar feedback(s), with ids: {ids_to_delete}")
    
    def retrieve_old_interactions(self, new_task: str, N : int = 10, max_distance: float = 1.5) -> list[str]:
        '''Retrieve N old useful feedbacks, based on new task'''

        # Search for top N similar scenarios
        # new_task_embedding = self.model.encode(new_task)
        new_task_embedding = self.openai_client.embeddings.create(
            input=new_task,
            model="text-embedding-3-large"
        )
        embedding_vector = new_task_embedding.data[0].embedding
        results = self.interactions_collection.query(
            query_embeddings=[embedding_vector], 
            n_results=10
        )

        # Collect results
        retrieved_docs = []
        if results:
            print("\n=== Retrieved Interactions ===")
            for idx, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
                if distance < max_distance:
                    print(f"Result {idx+1}:")
                    print(f"  Distance: {distance:.4f}")
                    print(f"  Document: {doc[:200]}{'...' if len(doc) > 200 else ''}")  # Truncate if too long
                    retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def _load_shortcuts(self):
        if os.path.exists(SHORTCUTS_PATH):
            with open(os.path.join(USER_MEMORY_PATH), "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _write_file_shortcuts(self):
        with open(SHORTCUTS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.user_shortcut_tasks, f, indent=4)

    def save_shortcut_task(self, shortcut: str, task: Task):
        '''
        Save in memory (a dictionary) shortcuts of the user.
        '''
        self.user_shortcut_tasks[shortcut.lower()] = task.to_dict()
        self._write_file_shortcuts()

    def get_shortcut_task(self, shortcut: str) -> dict | None:
        '''
        Return the task correspondent to given keyword
        '''
        return self.user_shortcut_tasks[shortcut.lower()]
    
    def delete_shortcut_task(self, shortcut: str):
        '''
        Delete from memory task mapped by key 'shortcut'.
        '''
        del self.user_shortcut_tasks[shortcut]