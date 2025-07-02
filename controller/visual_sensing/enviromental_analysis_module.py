from datetime import datetime
from io import BytesIO
import json
from PIL import Image
import numpy as np
import base64
from openai import OpenAI
import os
from controller.shared_frame import SharedFrame
from controller.utils import encode_image

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
type_folder_name = 'tello'

'''
Can be used to bot get a description of the current scene and to get a more high-level description of the context where the drone is
'''
class EnvironmentalAnalysisModule:
    def __init__(self):
        # Set up your client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Or use environment variable OPENAI_API_KEY
        with open("/home/christian/Desktop/polimi/prova_finale/codebase/TypeFly/controller/assets/tello/new/prompt_choose_direction.txt", "r") as f:
            self.direction_prompt = f.read()
    
    def get_scene_description(self, frame: SharedFrame, conf=0.3):
        image = frame.get_image()
        # image_base64 = self.encode_image(image.resize(self.image_size))
        image_base64 = encode_image(image)
        
        # Prepare the request
        response = self.client.chat.completions.create(
            model=GPT3,
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
        
    def choose_direction(self, current_task, base_path):
        # Read and encode the images
        forward_image = encode_image(Image.open(os.path.join(base_path, 'forward.jpg')))
        right_image = encode_image(Image.open(os.path.join(base_path, 'right.jpg')))
        backward_image = encode_image(Image.open(os.path.join(base_path, 'backward.jpg')))
        left_image = encode_image(Image.open(os.path.join(base_path, 'left.jpg')))


        prompt = self.direction_prompt.format(task=current_task,
                                            #   forward_image=forward_image,
                                            #   right_image=right_image,
                                            #   backward_image=backward_image,
                                            #   left_image=left_image,
                                              )
        
        # response = self.client.chat.completions.create(
        #     model="gpt-4o",  # Use gpt-4o instead of gpt-4o-vision-preview
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ],
        #     response_format={"type": "json_object"}  # Fixed: changed from text_format to response_format
        # )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + forward_image, "name": "forward.jpg"}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + right_image, "name": "right.jpg"}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + backward_image, "name": "backward.jpg"}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + left_image, "name": "left.jpg"}},
                    ]
                }
            ],
            response_format={"type": "json_object"} 
        )

        # Parse response
        response_content = response.choices[0].message.content
        parsed = json.loads(response_content)
        dir = parsed["direction"]
        reason = parsed["reason"]  # Fixed: removed (response_content) which was incorrect
        
        # Log the chat
        # self._log_direction_chat(current_task, response_content, dir, base_path)

        print(f"{current_task}: chosen {dir} because {reason}")
        return dir
    
    
    def _log_direction_chat(self, current_task, response_content, direction, base_path):
        """Log the direction choice chat to a file"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(base_path, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"direction_chat_{timestamp}.log")
            
            # Prepare log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "task": current_task,
                "prompt": f"The goal is: '{current_task}'\nI've attached 4 directional images. Return one of: forward, right, backward, left, or none. Respond in JSON format with the structure: {'direction':'right'}",
                "images_used": ["forward.jpg", "right.jpg", "backward.jpg", "left.jpg"],
                "raw_response": response_content,
                "chosen_direction": direction,
                "model": "gpt-4o"
            }
            
            # Write to log file
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
            
            print(f"[LOG] Direction chat logged to {log_file}")
            
        except Exception as e:
            print(f"[LOG ERROR] Failed to log direction chat: {e}")