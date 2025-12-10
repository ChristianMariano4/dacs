REGION_THRESHOLD: int = 50 # How distant in centimeters are the regions. After that distance, a new region is automatically created


ROBOT_NAME = "tello"
USERNAME = "Christian"
DEBUG = False


# Used paths
SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills"
HIGH_LEVEL_SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills/high_level_skills.json"
FLYZONE_TXT_PATH = "controller/assets/tello/flyzone/flyzone.txt"
GRAPH_TXT_PATH = "controller/assets/tello/memory/graph.txt"
FLYZONE_USER_IMAGE_PATH = "controller/assets/tello/temp/user_flyzone.jpg"
USER_PLAN_PROMPT_PATH = "controller/assets/tello/plan/user_plan_prompt.txt"
USER_EVERGREEN_FEEDBACK_PATH = f"controller/assets/tello/memory/{USERNAME}/evergreen_feedback.txt"
USER_EVERGREEN_FEEDBACK_PROMPT_PATH = "controller/assets/tello/memory/user_evergreen_feedback_prompt.txt"
USER_MEMORY_PATH = f"controller/assets/tello/memory/{USERNAME}"
LOW_LEVEL_SKILL_CONFIG_PATH=f"controller/assets/{ROBOT_NAME}/skills/low_level_skills.json"