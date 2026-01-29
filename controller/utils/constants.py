import os


REGION_THRESHOLD: int = 100 # How distant in centimeters are the regions. After that distance, a new region is automatically created


ROBOT_NAME = "tello"
USERNAME = "Christian"
USE_OLLAMA = True
DEBUG = False
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Used paths
SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills"
HIGH_LEVEL_SKILL_PATH = f"controller/assets/{ROBOT_NAME}/skills/high_level_skills.json"
FLYZONE_TXT_PATH = "controller/assets/tello/flyzone/flyzone.txt"
USER_GRAPH_PROMPT_PATH = "controller/assets/tello/graph/user_graph_prompt.txt"
GRAPH_TXT_PATH = "controller/assets/tello/graph/graph.txt"
FLYZONE_USER_IMAGE_PATH = "controller/assets/tello/temp/user_flyzone.jpg"
USER_PLAN_PROMPT_PATH = "controller/assets/tello/plan/user_plan _prompt.txt"
LOW_LEVEL_SKILL_CONFIG_PATH=f"controller/assets/{ROBOT_NAME}/skills/low_level_skills.json"
SHORT_TERM_MEMORY_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/short_term_memory"
LONG_TERM_MEMORY_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/long_term_memory"
TASK_FEEDBACK_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/long_term_memory/feedback/task_related"
UNIVERSAL_FEEDBACK_PROMPT_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/long_term_memory/feedback/universal/user_universal_feedback_prompt.txt"
CHANGE_UNIVERSAL_FEEDBACK_PROMPT_PATH = f"/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/{ROBOT_NAME}/long_term_memory/feedback/universal/user_update_universal_feedback_prompt.txt"
USER_SHORTCUTS_PATH = f"controller/assets/tello/shortcuts/{USERNAME}/shortcuts.json"
EVALUATION_LOG_PATH = "/home/christo/Desktop/polimi/prova_finale/SmartDrone/controller/assets/evaluation_log.txt"


SYSTEM_PROMPT_PATHS = {
    "PLAN": "controller/assets/tello/plan/system_plan_prompt.txt",
    "QUERY": "controller/assets/tello/query/system_query_prompt.txt",
    "SHORT_TERM_MEMORY": "controller/assets/tello/short_term_memory/system_short_memory_prompt.txt",
    "SAVE_TASK_FEEDBACK": "controller/assets/tello/long_term_memory/feedback/task_related/system_save_task_feedback_prompt.txt",
    "RETRIEVE_TASK_FEEDBACK": "controller/assets/tello/long_term_memory/feedback/task_related/system_retrieve_task_feedback_prompt.txt",
    "UNIVERSAL_FEEDBACK": "controller/assets/tello/long_term_memory/feedback/universal/system_update_universal_feedback_prompt.txt",
    "CHOOSE_DIRECTION": "controller/assets/tello/direction/system_choose_direction_prompt.txt",
    "CREATE_FLYZONE": "controller/assets/tello/flyzone/system_create_flyzone_prompt.txt",
    "CREATE_GRAPH": "controller/assets/tello/graph/system_create_graph_prompt.txt",
}