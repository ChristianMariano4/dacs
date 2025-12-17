from PIL import Image
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import threading
import time

class Frame():
    def __init__(self, image: Image.Image | NDArray[np.uint8]=None, depth: Optional[NDArray[np.int16]]=None):
        if image is None:
            self._image_buffer = np.zeros((352, 640, 3), dtype=np.uint8)
            self._image = Image.fromarray(self._image_buffer)
        if isinstance(image, np.ndarray):
            self._image_buffer = image
            self._image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            self._image = image
            self._image_buffer = np.array(image)
        self._depth = depth
    
    @property
    def image(self) -> Image.Image:
        return self._image
    
    @property
    def depth(self) -> Optional[NDArray[np.int16]]:
        return self._depth
    
    @image.setter
    def image(self, image: Image.Image):
        self._image = image
        self._image_buffer = np.array(image)

    @depth.setter
    def depth(self, depth: Optional[NDArray[np.int16]]):
        self._depth = depth

    @property
    def image_buffer(self) -> NDArray[np.uint8]:
        return self._image_buffer
    
    @image_buffer.setter
    def image_buffer(self, image_buffer: NDArray[np.uint8]):
        self._image_buffer = image_buffer
        self._image = Image.fromarray(image_buffer)

import torch
import cv2
import numpy as np
from PIL import Image

class SharedFrame:
    def __init__(self):
        # Load MiDaS model + transforms
        # self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.midas.to(self.device)
        # self.midas.eval()

        # Transform must match model
        # self.transform = midas_transforms.dpt_transform  

        self.lock = threading.Lock()

    def get_image(self) -> Optional[Image.Image]:
        with self.lock:
            return self.frame.image
    
    def get_yolo_result(self) -> dict:
        with self.lock:
            return self.yolo_result
    
    def get_depth(self) -> Optional[NDArray[np.int16]]:
        with self.lock:
            return self.frame.depth
        

    def set(self, frame: Frame, yolo_result: dict):
        # Convert to RGB
        # img_np = np.array(frame.image)
        # img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_np

        # Apply transform + move to device
        # input_batch = self.transform(img_rgb).to(self.device)

        # with torch.no_grad():
        #     prediction = self.midas(input_batch)
        #     prediction = torch.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=img_rgb.shape[:2],
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze()

        # depth_map = prediction.cpu().numpy().astype(np.int16)

        # # Save both RGB and depth into Frame
        # frame.depth = depth_map

        with self.lock:
            self.frame = frame
            self.timestamp = time.time()
            self.yolo_result = yolo_result
