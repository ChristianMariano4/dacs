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
from controller.constants import ROBOT_NAME, X_BOUND, Y_BOUND
from controller.llm_wrapper import LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer
from controller.shared_frame import SharedFrame
from controller.utils import encode_image

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
        with open(f"controller/assets/{ROBOT_NAME}/direction/prompt_choose_direction.txt", "r") as f:
            self.direction_prompt = f.read()

        self.middle_layer = middle_layer
        if self.middle_layer != None:
            self.flyzone = middle_layer.getFlyzone()

        self.llm_wrapper = LLMWrapper()
    
    def get_scene_description(self, frame: SharedFrame, conf=0.3):
        image = frame.get_image()
        # image_base64 = self.encode_image(image.resize(self.image_size))
        image_base64 = encode_image(image)
        
        # Prepare the request
        response = self.client.chat.completions.create(
            model=GPT4,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "Describe the image, considering is the current view of a drone and you are its sensing module"}
                    ]
                }
            ],
            max_tokens=700,
        )
        
        scene_description = response.choices[0].message.content
        print(scene_description)
        return scene_description
        
    def choose_direction(self, current_task, base_path, current_position, hint: Optional[str]):
        try:
            # Read and encode all 8 directional images
            north_image = encode_image(Image.open(os.path.join(base_path, 'forward.jpg')))
            east_image = encode_image(Image.open(os.path.join(base_path, 'right.jpg')))
            south_image = encode_image(Image.open(os.path.join(base_path, 'backward.jpg')))
            west_image = encode_image(Image.open(os.path.join(base_path, 'left.jpg')))
            
            # Diagonal direction images
            north_east_image = encode_image(Image.open(os.path.join(base_path, 'north-east.jpg')))
            north_west_image = encode_image(Image.open(os.path.join(base_path, 'north-west.jpg')))
            south_east_image = encode_image(Image.open(os.path.join(base_path, 'south-east.jpg')))
            south_west_image = encode_image(Image.open(os.path.join(base_path, 'south-west.jpg')))

            total_size = sum(len(img) for img in [
                north_image, east_image, south_image, west_image,
                north_east_image, north_west_image, south_east_image, south_west_image
            ])
            print(f"Total base64 size: {total_size / 1024 / 1024:.2f} MB")
            prompt = self.direction_prompt.format(task=current_task, 
                                                  hint=hint, 
                                                  flyzone=self.flyzone,
                                                  current_position=current_position,
                                                  )

            # Make the API call with all 8 images
            images = (north_image, north_east_image, east_image, south_east_image, south_image, south_west_image, west_image, north_west_image)
            response_content = self.llm_wrapper.request(prompt, images=images, request_type=RequestType.EXPLORE_DIRECTION)
            print(f"Raw API response: {response_content}")

            if not response_content:
                print("ERROR: Received empty response from OpenAI API")
                # Return default direction as fallback
                return "north"
                
            # Parse JSON response
            try:
                parsed = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response content: {response_content}")
                # Try to extract direction from malformed response - now includes diagonal directions
                response_lower = response_content.lower()
                
                # Check for diagonal directions first (more specific)
                if "north-east" in response_lower or "northeast" in response_lower:
                    return "north-east"
                elif "north-west" in response_lower or "northwest" in response_lower:
                    return "north-west"
                elif "south-east" in response_lower or "southeast" in response_lower:
                    return "south-east"
                elif "south-west" in response_lower or "southwest" in response_lower:
                    return "south-west"
                # Then check cardinal directions
                elif "north" in response_lower:
                    return "north"
                elif "west" in response_lower:
                    return "west"
                elif "south" in response_lower:
                    return "south"
                elif "west" in response_lower:
                    return "west"
                else:
                    return "north"  # Default fallback

            # Validate parsed response structure
            if not isinstance(parsed, dict):
                print(f"ERROR: Expected dict, got {type(parsed)}")
                return "north"
            
            if "direction" not in parsed:
                print(f"ERROR: 'direction' key not found in response: {parsed}")
                return "north"
            
            direction : str = parsed["direction"]
            direction = direction.lower()
            reason = parsed.get("reason", "No reason provided")
            distance = parsed.get("distance", None)
            region_name = parsed.get("region_name", None)

            

            # Validate direction value - now includes diagonal directions
            valid_directions = ["north", "east", "south", "west", "north-east", "north-west", "south-east", "south-west"]
            if direction not in valid_directions:
                print(f"ERROR: Invalid direction '{direction}', using 'north' as fallback")
                direction = "north"

            input(f"{current_task}: chosen {direction} because {reason}. Press a key to continue\n")
            return direction, distance, region_name
        
        except Exception as e:
            print(f"ERROR in choose_direction: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return "north"  # Safe fallback