import json
import os
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

from controller.utils.general_utils import encode_image


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.utils.constants import FLYZONE_USER_IMAGE_PATH, ROBOT_NAME
from controller.llm.llm_wrapper import GPT5, GPT_O4_MINI, LLMWrapper, RequestType
from controller.middle_layer.middle_layer import MiddleLayer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class FlyzoneManager:
    def __init__(self, middle_layer: MiddleLayer):
        ## Square flyzone
        self.middle_layer = middle_layer
        
        self.llm_wrapper = LLMWrapper()
        with open(os.path.join(CURRENT_DIR, f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/flyzone/user_flyzone_prompt.txt"), "r") as f:
            self.prompt_flyzone = f.read()

    def plot_current_flyzone(self):
        plt.clf()  # Clear the current figure before plotting a new flyzone

        for poly in self.middle_layer.get_flyzone_txt():
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.5)

        plt.gca().set_aspect('equal')

        # Add axis labels with units
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")

        plt.savefig("controller/assets/tello/flyzone/flyzone_plot.png")
        print("Saved flyzone plot to 'flyzone_plot.png'")


    def parse_current_flyzone(self):
        """Saves the current flyzone polygons in a dense, scannable format."""
        file_path = "controller/assets/tello/flyzone/flyzone.txt"
        
        with open(file_path, "w") as f:
            for idx, poly in enumerate(self.middle_layer.get_flyzone_txt()):
                # Get exterior coordinates; shapely polygons repeat the first point at the end
                coords = list(poly.exterior.coords)
                
                # Create a dense string: (x,y; x,y; x,y)
                # We round to 1 decimal place to keep it compact but precise
                dense_coords = "; ".join([f"{round(p[0], 1)},{round(p[1], 1)}" for p in coords])
                
                f.write(f"P{idx+1}: [{dense_coords}]\n")
                
        print(f"Saved dense flyzone to '{file_path}'")

    def format_flyzone_for_prompt(self) -> str:
        descriptions = []
        for idx, poly in enumerate(self.middle_layer.get_flyzone_txt()):
            coords = list(poly.exterior.coords)
            points_str = ", ".join([f"({round(x)}, {round(y)})" for x, y in coords])
            descriptions.append(f"Polygon {idx+1}: defined by points {points_str}.")
        return "The allowed flyzone is composed of the following polygonal regions:\n" + "\n".join(descriptions)
    
    def request_new_flyzone(self, instruction: str, image_present: bool, llm_model_name: str = GPT5):
        prompt = self.prompt_flyzone.format(instruction=instruction)
        if image_present:
            image = encode_image(FLYZONE_USER_IMAGE_PATH)
            response_content = self.llm_wrapper.request(user_prompt=prompt, request_type=RequestType.FLYZONE, image=image)
        else:
            response_content = self.llm_wrapper.request(user_prompt=prompt, request_type=RequestType.FLYZONE)
        # if response_content.startswith("```json"):
        #     response_content = response_content.replace("```json", "").replace("```", "").strip()

        print(f"Raw API response: {response_content}")

        if not response_content:
            print("ERROR: Received empty response from OpenAI API")
            # Return default direction as fallback
            return
            
        # Parse JSON response
        # try:
        #     parsed = json.loads(response_content)
        # except json.JSONDecodeError as e:
        #     print(f"JSON parsing error: {e}")
        #     print(f"Response content: {response_content}")
        #     return

        # eval the returned string of polygons into a safe environment
        points_list = [Polygon(coords) for coords in response_content['points_list']]
        print("Request done")
        self.middle_layer.set_flyzone(points_list)
        self.plot_current_flyzone()
        self.parse_current_flyzone()    


if __name__ == '__main__':
    flyzone_manager = FlyzoneManager(middle_layer=MiddleLayer())
    # flyzone_manager.request_new_flyzone(instruction="a simple squares of length 850cm")
    # flyzone_manager.request_new_flyzone(instruction="2 squares of length 500 cm one next the other linked by a corridor of 100 cm")
    flyzone_manager.request_new_flyzone(instruction="2 circles of radius 500 cm one next the other linked by a corridor of 100 cm")
    # flyzone_manager.request_new_flyzone(instruction="L-shaped with height and width 1000cm ")
    flyzone_manager.plot_current_flyzone()
    flyzone_manager.parse_current_flyzone()
    print(flyzone_manager.format_flyzone_for_prompt())