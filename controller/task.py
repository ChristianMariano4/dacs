from controller.minispec_interpreter import Statement


class Task():
    def __init__(self, task_description, is_new: bool = True):
        self.task_description = task_description
        self.execution_history: list[Statement | str] = []
        self.current_plan = ""
        self.last_achievements = ""
        self.drone_position = None
        self.user_feedback = ""
        self.is_new = is_new

    # def set_task_description(self, task_description: str):
    #     self.task_description = task_description
    
    def get_task_description(self) -> str:
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

    def set_user_feedback(self, user_feedback: str):
        self.user_feedback = user_feedback

    def get_user_feedback(self) -> str:
        return self.user_feedback

    def get_is_new(self) -> bool:
        return self.is_new

    def to_dict(self):
        return {
            "task_description": self.task_description,
            "execution_history": [
                str(item) for item in self.execution_history  # convert objects to string
            ],
            "current_plan": self.current_plan,
            "user_feedback": self.user_feedback,
        }

    def to_prompt(self) -> str:
        parts = []

        if self.task_description:
            parts.append(f"The task is: {self.task_description}")

        if self.execution_history:
            history_str = " | ".join(str(item) for item in self.execution_history)
            parts.append(f"This task was already executed before and now the user wants to execute it again, so you can exploit what already previously done. The execution history was: {history_str}")

        if self.user_feedback:
            parts.append(f"The user feedback was: {self.user_feedback}")

        return "\n".join(parts)
    
    @classmethod
    def from_dict(cls, data: dict):
        task = cls(data.get("task_description", ""))
        task.execution_history = data.get("execution_history", [])
        task.current_plan = data.get("current_plan", "")
        task.user_feedback = data.get("user_feedback", "")
        return task