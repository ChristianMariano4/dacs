from shapely import Polygon

from controller.utils.constants import FLYZONE_TXT


class MiddleLayer:
    # class used to globally save user preference about interaction
    # TODO: every info is saved on disk to not lose preferences between an interaction and another 
    def __init__(self):
        # default square flyzone
        self.flyzone = MiddleLayer._parse_flyzone(FLYZONE_TXT)
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
    def _parse_flyzone(file_path):
        coords = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("  - ("):
                    # Extract numbers between parentheses
                    nums = line.strip("- ()").split(",")
                    x, y = map(float, nums)
                    coords.append(x, y)
        return [Polygon(coords)]


    def set_flyzone(self, flyzone):
        self.flyzone = flyzone

    def get_flyzone(self):
        return self.flyzone
    
    # def set_username(self, username):
    #     self.username = username

    # def get_username(self):
    #     return self.username
    

