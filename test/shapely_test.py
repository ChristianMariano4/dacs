from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


def is_in_flyzone(point: tuple[float, float], flyzone_polygons: list[Polygon]) -> bool:
    pt = Point(point)
    return any(poly.contains(pt) for poly in flyzone_polygons)

def plot_flyzone(polygons):
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.5)
    plt.gca().set_aspect('equal')
    plt.savefig("flyzone_plot.png")
    print("Saved flyzone plot to 'flyzone_plot.png'")

# Create a U-shaped zone from three rectangles
zone1 = Polygon([(0, 0), (0, 30), (10, 30), (10, 0)])
zone2 = Polygon([(20, 0), (20, 30), (30, 30), (30, 0)])
zone3 = Polygon([(10, 20), (10, 30), (20, 30), (20, 20)])

# Reuse the polygons (e.g., from the U-shape example)
flyzone_polygons = [zone1, zone2, zone3]

# Define a test point
test_point = (15, 25)  # inside the top bar of the U

# Check if it's within the flyzone
inside = is_in_flyzone(test_point, flyzone_polygons)
print(f"Point {test_point} is inside the flyzone: {inside}")

flyzone = zone1.union(zone2).union(zone3)

# Check if a point is inside
point = Point(15, 25)
print(flyzone.contains(point))  # True
print(flyzone)  # True


plot_flyzone(flyzone_polygons)