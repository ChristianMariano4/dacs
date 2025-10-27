from typing import Optional, List, Tuple
from controller.minispec_interpreter import MiniSpecReturnValue, Statement

class Task():
    def __init__(
        self, 
        task_description: str, 
        execution_history: Optional[List[List]] = None,
        current_plan: Optional[str] = "", 
        user_feedback: Optional[str] = "", 
        is_new: bool = True
    ):
        self.task_description = task_description
        self.current_plan = current_plan
        # Initialize with empty list if None, avoiding mutable default argument issue
        self.execution_history_list: List[List] = execution_history if execution_history is not None else []
        
        if current_plan != "":
            self.execution_history_list.append([self.current_plan, [], ""])  # [plan, actions, summary]
        
        self.task_summary = ""
        self.last_achievements = ""
        self.drone_position = None
        self.user_feedback = user_feedback
        self.is_new = is_new
    
    def get_task_description(self) -> str:
        return self.task_description
    
    def update_execution_history(self, *new_statements: Statement | str):
        """Add statements to the last iteration's action list"""
        if not self.execution_history_list:
            # If no history exists, create a new entry
            self.execution_history_list.append(["", [], ""])
        
        # Get the last iteration
        last_iteration = self.execution_history_list[-1]
        
        # Ensure it's a list with at least 3 elements
        if not isinstance(last_iteration, list):
            # If it's a string (old format), convert it
            self.execution_history_list[-1] = ["", [], str(last_iteration)]
            last_iteration = self.execution_history_list[-1]
        
        while len(last_iteration) < 3:
            if len(last_iteration) == 0:
                last_iteration.append("")  # plan
            elif len(last_iteration) == 1:
                last_iteration.append([])  # actions
            else:
                last_iteration.append("")  # summary
        
        # Add the new statements to the actions list (index 1)
        for stmt in new_statements:
            last_iteration[1].append(str(stmt))
    
    def get_execution_history(self):
        return self.execution_history_list
    
    def set_current_plan(self, new_plan):
        self.current_plan = new_plan
        self.execution_history_list.append([self.current_plan, [], ""])  # [plan, actions, summary]
    
    def get_current_plan(self):
        return self.current_plan
    
    def set_user_feedback(self, user_feedback: str):
        self.user_feedback = user_feedback
    
    def get_user_feedback(self) -> str:
        return self.user_feedback
    
    def get_is_new(self) -> bool:
        return self.is_new
    
    def get_last_iteration(self) -> List:
        if not self.execution_history_list:
            raise IndexError("No execution history available")
        return self.execution_history_list[-1]
    
    def add_last_iteration_summary(self, summary):
        '''Add natural language summary to last entry in execution history'''
        if not self.execution_history_list:
            raise IndexError("No execution history to update")
        
        last_item = self.execution_history_list[-1]
        
        # Check if last_item is a string (old format) and convert it
        if isinstance(last_item, str):
            self.execution_history_list[-1] = ["", [], last_item]
            last_item = self.execution_history_list[-1]
        
        # Ensure we have at least 3 elements in the last item
        if not isinstance(last_item, list):
            raise TypeError(f"Expected list, got {type(last_item)}")
        
        while len(last_item) < 3:
            if len(last_item) == 0:
                last_item.append("")  # plan
            elif len(last_item) == 1:
                last_item.append([])  # actions
            else:
                last_item.append("")  # summary
        
        # Update the summary (third element, index 2)
        self.execution_history_list[-1][2] = summary
    
    def update_task_summary(self, summary):
        self.task_summary = summary
    
    def to_dict(self):
        return {
            "task_description": self.task_description,
            "execution_history": [
                str(item) for item in self.execution_history_list  # convert objects to string
            ],
            "current_plan": self.current_plan,
            "user_feedback": self.user_feedback,
        }
    
    def to_prompt(self) -> str:
        parts = []
        if self.task_description:
            parts.append(f"The task is: {self.task_description}")
        
        if self.execution_history_list:
            history_str = " | ".join(str(item) for item in self.execution_history_list)
            parts.append(
                f"This task was already executed before and now the user wants to execute it again, "
                f"so you can exploit what was already previously done. The execution history was: {history_str}"
            )
        
        if self.user_feedback:
            parts.append(f"The user feedback was: {self.user_feedback}")
        
        return "\n".join(parts)
    
    @classmethod
    def from_dict(cls, data: dict):
        task = cls(data.get("task_description", ""))
        task.execution_history_list = data.get("execution_history", [])
        task.current_plan = data.get("current_plan", "")
        task.user_feedback = data.get("user_feedback", "")
        return task