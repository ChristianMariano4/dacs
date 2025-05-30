# Encode the image in base64
import base64
from io import BytesIO


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