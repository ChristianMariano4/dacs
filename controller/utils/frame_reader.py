from controller.robot_implementations.crazyflie_wrapper import adjust_exposure, sharpen_image


class FrameReader:
    """Tiny wrapper that post-processes frames (exposure + sharpen) on access."""
    def __init__(self, fr):
        # Initialize the video capture
        self.fr = fr

    @property
    def frame(self):
        # Read a frame from the video capture
        frame = self.fr.frame
        if frame is None:
            return None
        frame = adjust_exposure(frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)