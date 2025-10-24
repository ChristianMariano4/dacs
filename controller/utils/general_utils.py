import base64
from io import BytesIO
from typing import Optional, Union
import datetime
import cv2
import numpy as np
from PIL import Image
import csv
import datetime
import subprocess

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

# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def encode_image(image_path, quality=40, max_size=(500, 500)):
    """Compress and convert an image (PIL or numpy) to base64 string."""
    image = Image.open(image_path)
    if isinstance(image, np.ndarray):
        # Resize if needed
        height, width = image.shape[:2]
        if max(height, width) > max(max_size):
            scale = min(max_size[0] / width, max_size[1] / height)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Compress and encode
        success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise ValueError("Could not encode numpy image")
        Image.fromarray(image).save(image_path)
        return base64.b64encode(buffer).decode("utf-8")

    elif isinstance(image, Image.Image):
        # Resize if needed
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Compress and encode
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
        Image.fromarray(image).save(image_path)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    else:
        raise TypeError(f"Unsupported image type {type(image)}")    


# -----------------------------------------------------------------------------
# Movement helpers
# -----------------------------------------------------------------------------
MOVEMENT_MIN = 20     # cm
MOVEMENT_MAX = 500    # cm

def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
    return distance
    
# ───── CSV Utilities ─────

def save_to_csv(timed_waypoints, filename):
    """Write time-stamped waypoints to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['t', 'x', 'y', 'z', 'yaw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(timed_waypoints['x'])):
            writer.writerow({'t': timed_waypoints['t'][i],
                            'x': timed_waypoints['x'][i],
                            'y': timed_waypoints['y'][i],
                            'z': timed_waypoints['z'][i],
                            'yaw': timed_waypoints['yaw'][i],
                            })
            
def import_csv(filename):
    """Load header and data from a CSV file."""
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            data.append([float(value) for value in row])
    return header, data

# ───── Logging with Timestamps ─────

def print_t(*args, **kwargs):
    """Print with timestamp prefix."""
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal):
    """Input with timestamp prefix."""
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def run_command(command):
    """Run a shell command and show output/errors."""
    print("Running command:", command)
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        print("Command output:")
        print(result.stdout)
        if result.stderr:
            print("Command errors:")
            print(result.stderr)
    
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        print("Full command output:")
        print(e.output)