import base64
from io import BytesIO
from typing import Optional, Union
import datetime
import cv2
import numpy as np
from PIL import Image

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def split_args(s: str) -> list[str]:
    args, cur, depth = [], '', 0
    in_single = in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                args.append(cur.strip())
                cur = ''
                continue
        cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args

def encode_image(image):
    """Convert an image (PIL or numpy) to base64 string"""
    if isinstance(image, np.ndarray):
        # If it's an OpenCV image
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
        raise TypeError(f"Unsupported image type {type(image)}")