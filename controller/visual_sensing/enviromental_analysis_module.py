from io import BytesIO
from PIL import Image
import numpy as np
import base64
from openai import OpenAI
import os
from controller.shared_frame import SharedFrame

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"

class EnvironmentalAnalysisModule:
    def __init__(self):
        # Set up your client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Or use environment variable OPENAI_API_KEY
    
    # Encode the image in base64
    def encode_image(self, image):
        """Convert an image (PIL or numpy) to base64 string"""
        if isinstance(image, np.ndarray):
            # If it's an OpenCV image
            import cv2
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Could not encode numpy image")
            return base64.b64encode(buffer).decode("utf-8")
        elif isinstance(image, Image.Image):
            # If it's a PIL image
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise TypeError("Unsupported image type")
    
    def get_scene_description(self, frame: SharedFrame, question: str, conf=0.3):
        image = frame.get_image()
        # image_base64 = self.encode_image(image.resize(self.image_size))
        image_base64 = self.encode_image(image)
        
        # Prepare the request
        response = self.client.chat.completions.create(
            model=GPT3,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": question}
                    ]
                }
            ],
            max_tokens=700,
        )
        
        scene_description = response.choices[0].message.content
        print(scene_description)
        return scene_description