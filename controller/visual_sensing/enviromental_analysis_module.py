from datetime import datetime
from datetime import datetime
from io import BytesIO
import json
import json
from typing import Optional
from PIL import Image
import numpy as np
import base64
from openai import OpenAI
import os
from controller.utils.constants import ROBOT_NAME, X_BOUND, Y_BOUND
from controller.llm.llm_wrapper import GPT5_MINI, GPT5_NANO, LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer
from controller.shared_frame import SharedFrame
from controller.utils.general_utils import encode_image

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
class EnvironmentalAnalysisModule:
    def __init__(self, middle_layer: MiddleLayer=None):
        # Set up your client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Or use environment variable OPENAI_API_KEY
        with open(f"controller/assets/{ROBOT_NAME}/direction/user_direction_prompt.txt", "r") as f:
            self.direction_prompt = f.read()

        self.middle_layer = middle_layer
        if self.middle_layer != None:
            self.flyzone = middle_layer.get_flyzone_txt()

        self.llm_wrapper = LLMWrapper()

        # Array of boolean with one element for each of the 8 directions: True if the image of that direction is updated with last scan
        self.updated_directions = {"north": False, "north-east": False, "east": False, "south-east": False, "south": False, "south-west": False, "west": False, "north-west": False}

    def reset_updated_directions(self):
        for dir in self.updated_directions:
            self.updated_directions.update({dir: False})
    
    def set_updated_directions(self, direction: str):
        self.updated_directions[direction] = True
            
    def choose_direction(self, current_task, base_path, current_position, hint: Optional[str]):
        try:
            # Read and encode updated directional images
            images = {}
            for dir in self.updated_directions:
                if self.updated_directions[dir]:
                    images.update({dir: encode_image(os.path.join(base_path, f'{dir}.jpg'))})
                    print(f"Image of direction {dir} appended")

            total_size = sum(len(images.get(dir)) for dir in images)
            print(f"Total base64 size: {total_size / 1024 / 1024:.2f} MB")
            prompt = self.direction_prompt.format(task=current_task, 
                                                  hint=hint, 
                                                  flyzone=self.flyzone,
                                                  current_position=current_position,
                                                  )

            # Make the API call with all appended images
            response_content = self.llm_wrapper.request(prompt, images=images, model_name=GPT5_MINI, request_type=RequestType.EXPLORE_DIRECTION)
            print(f"Raw API response: {response_content}")

            if not response_content:
                print("ERROR: Received empty response from OpenAI API")
                # Return default direction as fallback
                return "north", 150, None
            
            direction = str(response_content.get('direction', None)).lower()
            region_name = response_content.get('region_name', None)
            reason = response_content.get('reason', None)
            distance = response_content.get('distance', None)
                
            print(f"{current_task}: chosen {direction} because {reason}. Press a key to continue\n")
            return direction, distance, region_name
        
        except Exception as e:
            print(f"ERROR in choose_direction: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return "north"  # Safe fallback