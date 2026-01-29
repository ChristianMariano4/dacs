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
from controller.utils.constants import ROBOT_NAME, USE_OLLAMA
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
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        with open(f"controller/assets/{ROBOT_NAME}/direction/user_choose_direction_prompt.txt", "r") as f:
            self.direction_prompt = f.read()
        
        self.middle_layer = middle_layer
        if self.middle_layer != None:
            self.flyzone = middle_layer.get_flyzone_txt()
        
        self.llm_wrapper = LLMWrapper()
        
        # Track which yaw angles have updated images
        # Dictionary: {yaw_degrees: bool}
        self.updated_directions = {}  # Start empty, populate as photos are taken

    def reset_updated_directions(self):
        """Clear all tracked yaw angles"""
        self.updated_directions.clear()

    def set_updated_directions(self, yaw: int):
        """Mark a specific yaw angle as having an updated image"""
        yaw = yaw % 360  # Normalize to [0, 360)
        self.updated_directions[yaw] = True
        print(f"[ENV] Direction {yaw}° marked as updated")

    def choose_direction(self, current_task, base_path, context_graph, execution_history):
        try:
            # Read images by yaw angle
            images = {}
            for yaw, is_updated in self.updated_directions.items():
                if is_updated:
                    img_path = os.path.join(base_path, f'{yaw}.jpg')
                    if os.path.exists(img_path):
                        images[f"{yaw}°"] = encode_image(img_path)  # Key: "0°", "45°", etc.
                        print(f"Image at yaw {yaw}° appended")
                    else:
                        print(f"Warning: Image at {img_path} not found")
            
            if not images:
                print("ERROR: No images available for direction analysis")
                return 0, 150, None  # Default: face forward
            
            total_size = sum(len(img_data) for img_data in images.values())
            print(f"Total base64 size: {total_size / 1024 / 1024:.2f} MB")
            
            # Update prompt to mention yaw angles instead of compass directions
            prompt = self.direction_prompt.format(
                task=current_task, 
                flyzone=self.flyzone,
                context_graph=context_graph,
                execution_history=execution_history
            )
            
            response_content = self.llm_wrapper.request(
                prompt, 
                images=images, 
                request_type=RequestType.CHOOSE_DIRECTION,
                use_ollama=USE_OLLAMA
            )
            
            print(f"Raw API response: {response_content}")
            
            if not response_content:
                print("ERROR: Received empty response from OpenAI API")
                return 0, 150, None
            
            # Expect yaw as integer now, not direction string
            target_yaw = int(response_content.get('yaw', 0))  # Changed from 'direction'
            region_name = response_content.get('region_name', None)
            reason = response_content.get('reason', None)
            distance = int(response_content.get('distance', 150))
            
            print(f"{current_task}: Moving to yaw {target_yaw}° because {reason}")
            
            return target_yaw, distance, region_name
            
        except Exception as e:
            print(f"ERROR in choose_direction: {e}")
            import traceback
            traceback.print_exc()
            return 0, 150, None  # Safe fallback: move forward