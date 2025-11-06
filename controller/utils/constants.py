X_BOUND: int = 50
Y_BOUND: int = 50
REGION_THRESHOLD: int = 100 # How distant in centimeters are the regions. After that distance, a new region is automatically created

# Used paths
ROBOT_NAME = 'tello'
SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills"
HIGH_LEVEL_SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills/high_level_skills.json"
FLYZONE_TXT_PATH = "controller/assets/tello/flyzone/flyzone.txt"
GRAPH_TXT_PATH = "controller/assets/tello/memory/graph.txt"
FLYZONE_USER_IMAGE_PATH = "controller/assets/tello/temp/user_flyzone.jpg"