import json
import os
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.constants import ROBOT_NAME
from controller.llm_wrapper import LLMWrapper
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class FlyzoneManager:
    def __init__(self):
        ## Square flyzone
        # self.flyzone_polygons = [Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])]

        ## U-shaped flyzone
        self.flyzone_polygons = [
            # Left vertical leg of the U
            Polygon([
                (0, 0),
                (0, 100),
                (30, 100),
                (30, 0)
            ]),
            
            # Right vertical leg of the U
            Polygon([
                (70, 0),
                (70, 100),
                (100, 100),
                (100, 0)
            ]),
            
            # Top horizontal bar of the U
            Polygon([
                (30, 70),
                (30, 100),
                (70, 100),
                (70, 70)
            ])
        ]
        self.llm_wrapper = LLMWrapper()
        with open(os.path.join(CURRENT_DIR, f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/tello/flyzone/prompt_flyzone.txt"), "r") as f:
            self.prompt_flyzone = f.read()


    def plot_current_flyzone(self):
        for poly in self.flyzone_polygons:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.5)
        plt.gca().set_aspect('equal')
        plt.savefig("controller/assets/tello/flyzone/flyzone_plot.png")
        print("Saved flyzone plot to 'flyzone_plot.png'")

    def parse_current_flyzone(self):
        with open("controller/assets/tello/flyzone/flyzone.txt", "w") as f:
            for idx, poly in enumerate(self.flyzone_polygons):
                coords = list(poly.exterior.coords)
                f.write(f"Flyzone Polygon {idx+1}:\n")
                for point in coords:
                    f.write(f"  - {point}\n")
        print("Saved flyzone text to 'flyzone.txt'")

    def format_flyzone_for_prompt(self) -> str:
        descriptions = []
        for idx, poly in enumerate(self.flyzone_polygons):
            coords = list(poly.exterior.coords)
            points_str = ", ".join([f"({round(x)}, {round(y)})" for x, y in coords])
            descriptions.append(f"Polygon {idx+1}: defined by points {points_str}.")
        return "The allowed flyzone is composed of the following polygonal regions:\n" + "\n".join(descriptions)
    
    def request_new_flyzone(self, instruction: str):
        # Centered at (0, 0), radius = 25 cm
        circle = Point(0, 0).buffer(25, resolution=32)  # resolution controls smoothness
        # resolution=32 creates 128 boundary points (4 × resolution) — higher resolution means smoother approximation.

        # Convert to list of (x, y) tuples
        self.flyzone_polygons = [Polygon(list(circle.exterior.coords))]

        prompt = self.prompt_flyzone.format(instruction=instruction)
        response_content = self.llm_wrapper.request(prompt=prompt)
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()

        print(f"Raw API response: {response_content}")

        if not response_content:
            print("ERROR: Received empty response from OpenAI API")
            # Return default direction as fallback
            return self.flyzone_polygons
            
        # Parse JSON response
        try:
            parsed = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response_content}")
            return self.flyzone_polygons
        


        # eval the returned string of polygons into a safe environment
        polygons_list = [Polygon(coords) for coords in parsed['polygon_list']]
        print("Request done")
        self.flyzone_polygons = polygons_list

    


if __name__ == '__main__':
    flyzone_manager = FlyzoneManager()
    flyzone_manager.request_new_flyzone(instruction="U-shaped flyzone")
    flyzone_manager.plot_current_flyzone()
    flyzone_manager.parse_current_flyzone()
    print(flyzone_manager.format_flyzone_for_prompt())