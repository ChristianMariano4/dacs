from shapely import Polygon


class MiddleLayer:
    # class used to globally save user preference about interaction
    # TODO: every info is saved on disk to not lose preferences between an interaction and another 
    def __init__(self):
        # default square flyzone
        self.flyzone = [Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])] 

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


    def setFlyzone(self, flyzone):
        self.flyzone = flyzone

    def getFlyzone(self):
        return self.flyzone
