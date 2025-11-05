import os
from shapely import Polygon

from controller.utils.constants import FLYZONE_TXT_PATH


class MiddleLayer:
    # class used to globally save user preference about interaction
    # TODO: every info is saved on disk to not lose preferences between an interaction and another 
    def __init__(self):
        # default square flyzone
        with open(FLYZONE_TXT_PATH) as f:
            self.flyzone = f.read()
        # self.username = ""

        ## U-shaped flyzone
        # self.flyzone_polygons = [
        #     # Left vertical leg of the U
        #     Polygon([
        #         (0, 0),
        #         (0, 100),
        #         (30, 100),
        #         (30, 0)
        #     ]),
            
        #     # Right vertical leg of the U
        #     Polygon([
        #         (70, 0),
        #         (70, 100),
        #         (100, 100),
        #         (100, 0)
        #     ]),
            
        #     # Top horizontal bar of the U
        #     Polygon([
        #         (30, 70),
        #         (30, 100),
        #         (70, 100),
        #         (70, 70)
        #     ])
        # ]

        ## Cirle flyzone
        # circle = Point(0, 0).buffer(25, resolution=32)  # resolution controls smoothness
        # # resolution=32 creates 128 boundary points (4 × resolution) — higher resolution means smoother approximation.
        # self.flyzone_polygons = [Polygon(list(circle.exterior.coords))]

    @classmethod
    def _parse_flyzone(cls, file_path):
        polygons = []
        coords = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Flyzone Polygon"):
                    if coords:
                        polygons.append(Polygon(coords))
                        coords = []
                elif line.startswith("- ("):
                    # Safer parsing: remove "- (" at start and ")" at end explicitly
                    coord_str = line.replace("- (", "").replace(")", "")
                    x_str, y_str = coord_str.split(",")
                    x, y = float(x_str.strip()), float(y_str.strip())
                    coords.append((x, y))

        if coords:
            polygons.append(Polygon(coords))

        return polygons




    def set_flyzone(self, flyzone):
        self.flyzone = flyzone
        
    def get_flyzone_polygon(self):
        if os.path.exists(FLYZONE_TXT_PATH):
            return self._parse_flyzone(FLYZONE_TXT_PATH)
        return []
    
    def get_flyzone_txt(self):
        return self.flyzone
    
    # def set_username(self, username):
    #     self.username = username

    # def get_username(self):
    #     return self.username
    

