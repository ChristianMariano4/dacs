from io import BytesIO
from PIL import Image
import numpy as np
import base64
from openai import OpenAI
import os
from controller.shared_frame import SharedFrame
from controller.utils import encode_image

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"

'''
Can be used to bot get a description of the current scene and to get a more high-level description of the context where the drone is
'''
class EnvironmentalAnalysisModule:
    def __init__(self):
        # Set up your client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Or use environment variable OPENAI_API_KEY
    
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