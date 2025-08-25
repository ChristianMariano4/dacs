from controller.minispec_interpreter import Statement


class Task():
    def __init__(self, task_description):
        self.task_description = task_description
        self.execution_history: list[Statement | str] = []
        self.current_plan = ""
        self.last_achievements = ""
        self.drone_position = None
        self.battery_percent = None
        self.objects_list = []
        self.graph_json = {}
        self.user_feedback = ""

    # def set_task_description(self, task_description: str):
    #     self.task_description = task_description
    
    def get_task_description(self):
        return self.task_description
    
    def update_execution_history(self, *new_statements: Statement | str):
        # convert all to str, then join with space (or another delimiter if you prefer)
        combined = ". ".join(str(stmt) for stmt in new_statements)
        self.execution_history.append(combined)
    
    def get_execution_history(self):
        return self.execution_history
    
    def set_current_plan(self, new_plan):
        self.current_plan = new_plan
    
    def get_current_plan(self):
        return self.current_plan
    
    def set_last_achievements(self, last_achievements):
        self.last_achievements = last_achievements
    
    def get_last_achievements(self):
        return self.last_achievements

    # Getter and Setter for drone_position
    def get_drone_position(self):
        return self.drone_position

    def set_drone_position(self, value):
        self.drone_position = value

    # Getter and Setter for battery_percent
    def get_battery_percent(self):
        return self.battery_percent

    def set_battery_percent(self, value):
        self.battery_percent = value

    # Getter and Setter for objects_list
    def get_objects_list(self):
        return self.objects_list

    def set_objects_list(self, value):
        self.objects_list = value

    # Getter and Setter for graph_json
    def get_graph_json(self):
        return self.graph_json

    def set_graph_json(self, value):
        self.graph_json = value

    def set_user_feedback(self, user_feedback: str):
        self.user_feedback = user_feedback

    def get_user_feedback(self, user_feedback: str) -> str:
        return self.user_feedback