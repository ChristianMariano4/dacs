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
from controller.llm.llm_wrapper import GPT5_NANO, LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer
from controller.shared_frame import SharedFrame
from controller.task import Task
from controller.utils import encode_image, print_t
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
        with open(os.path.join(MEMORY_PATH, "prompt_memory.txt"), "r") as f:
            self.memory_prompt = f.read()
        self.username = username # used to retrieve the correct vector db

        # Vector db section: retrieve the db of the specified user
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client_chromadb = chromadb.PersistentClient(path=os.path.join(MEMORY_PATH, self.username))
        self.interactions_collection = self.client_chromadb.get_or_create_collection(name="interaction_memory")

        # Shortcuts: tasks associated to keywords
        self.shortcuts_file = Path(shortcuts_file)
        self.user_shortcut_tasks = self._load_shortcuts() # each username is a key, whose value is another dict of shortcuts of that user

    def change_username(self, username):
        self.username = username
        self.client_chromadb = chromadb.PersistentClient(path=os.path.join(MEMORY_PATH, self.username))
        self.interactions_collection = self.client_chromadb.get_or_create_collection(name="interaction_memory")

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
            
    def save_interaction_summary(self, task: Task, conf=0.3):
        '''
        Save in memory a summary of the executed task, in order to keep some lessons learned.
        '''
        
        prompt = self.memory_prompt.format(task_text=task.get_task_description(), 
                                        execution_output=task.get_execution_history(), 
                                        feedback_text=task.get_user_feedback(),
                                        )
        
        # Send the request to gpt5-nano, because we just need to summarize information
        response_content = self.llm_wrapper.request(prompt, request_type=RequestType.SIMPLE, model_name=GPT5_NANO)

        # Parse the response
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

        print(f"Interaction summary saved:\n{doc_text}")

        # Add summary to the vector db
        embedding = self.model.encode(doc_text)
        self.interactions_collection.add(
            documents=[doc_text],
            embeddings=[embedding.tolist()],
            ids=[f"task_{self.get_next_task_id()}"]
        )
    
    def retrieve_old_interactions(self, new_task: str, N : int = 5) -> list[str]:
        '''Retrieve N old useful feedbacks, based on new task'''

        # Search for top N similar scenarios
        new_task_embedding = self.model.encode(new_task)
        results = self.interactions_collection.query(
            query_embeddings=[new_task_embedding.tolist()], 
            n_results=N
        )

        # Collect results
        retrieved_docs = []
        if results:
            print("\n=== Retrieved Interactions ===")
            for idx, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
                print(f"Result {idx+1}:")
                print(f"  Distance: {distance:.4f}")
                print(f"  Document: {doc[:200]}{'...' if len(doc) > 200 else ''}")  # Truncate if too long
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def _load_shortcuts(self):
        if self.shortcuts_file.exists():
            with open(self.shortcuts_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_shortcuts(self):
        with open(self.shortcuts_file, "w", encoding="utf-8") as f:
            json.dump(self.user_shortcut_tasks, f, indent=4)

    def save_shortcut_task(self, username: str, keywords: str, task: Task):
        '''
        Save in memory (a dictionary) shortcuts of the user.
        '''
        username = username.lower()
        if username not in self.user_shortcut_tasks:
            self.user_shortcut_tasks[username] = {}
        self.user_shortcut_tasks[username][keywords] = task.to_dict()
        self._save_shortcuts()

    def get_shortcut_task(self, username: str, keywords: str) -> dict | None:
        '''
        Return the task correspondent to given keyword
        '''
        username = username.lower()
        if username not in self.user_shortcut_tasks:
            return None
        return self.user_shortcut_tasks[username][keywords]

if __name__ == "__main__":
    # python -m controller.llm.memory.long_memory
    # Minimal test for LongMemoryModule with example-style histories
    
    class MockTask:
        def __init__(self, desc, history, feedback):
            self._desc = desc
            self._history = history
            self._feedback = feedback

        def get_task_description(self):
            return self._desc
        
        def get_execution_history(self):
            return self._history
        
        def get_user_feedback(self):
            return self._feedback

    memory_module = LongMemoryModule()

    # Example-like interaction histories with positive & negative feedbacks
    examples = [
        MockTask(
            "Find an apple in the environment.",
            """Iteration 1: Task - I need an apple. 
Scene: [] 
Response: as("search_object", "Scan for an object and explore a new region if not visible", 
"?s($1)==True{l($1 + ' found');->True};ed();rp();"); search_object(["apple"]);""",
            "Positive: Good reasoning, I like the reusable skill definition."
        ),
        MockTask(
            "Return home to region_0.",
            """Iteration 5: Task - Come back to region_0
Scene: [apple_1 x:0.52 y:0.53 width:0.18 height:0.27]
Response: as("return_home", "Return to the home region (0,0)", 
"gx(0, 0); l('Returned to region_0 at [0,0].')"); return_home();""",
            "Positive: Worked correctly, drone returned to base."
        ),
        MockTask(
            "Move safely to a coordinate inside flyzone.",
            """Iteration 7: Task - Move from region_2 at [150, 150] to [600, 500].
Reason: Direct path exits flyzone, rerouting required.
Response: gx(300, 150); gx(300, 300); gx(600, 300); 
l("Stopped at [600, 300] because the requested [600, 500] is outside the flyzone.");""",
            "Positive: Correctly detected flyzone violation."
        ),
        MockTask(
            "Search for an orange.",
            """Iteration 12: Task - Where can I find an orange?
Response: fruit_hunt(["orange"]);
If already in graph (region_2 at [150, 150]): 
l("orange found in region_2 at [150, 150]");""",
            "Positive: Very efficient reuse of fruit_hunt skill."
        ),
        # --- Negative feedback examples ---
        MockTask(
            "Explore for bananas.",
            """Iteration X: Task - I need a banana.
Response: gx(600, 500); l("Moved outside flyzone.");""",
            "Negative: Unsafe execution, drone went outside flyzone."
        ),
        MockTask(
            "Return to base with invalid command.",
            """Iteration Y: Task - Return to region_0
Response: ed(); rp(); l('Trying to replan endlessly.');""",
            "Negative: The plan was incorrect, drone never attempted to return home."
        ),
        MockTask(
            "Find an apple but ignored scene info.",
            """Iteration Z: Task - I need an apple. 
Scene: [apple_1 x:0.52 y:0.53 width:0.18 height:0.27]
Response: ed(); rp(); l('No apple found, exploring...');""",
            "Negative: Wrong, an apple was already visible but ignored."
        )
    ]

    print("=== Saving interactions ===")
    for task in examples:
        memory_module.save_interaction_summary(task)

    # Query with a new task
    query = "Where can I find a banana?"
    print("\n=== Retrieving interactions for:", query, "===")
    memory_module.retrieve_old_interactions(query)
