import json
import os
import uuid
from typing import Optional, Sequence
import numpy as np

from controller.context_map.graph_handler import GraphHandler
from controller.llm.llm_wrapper import LLMWrapper, RequestType
from controller.task import Task
from controller.utils.constants import GRAPH_TXT_PATH, USER_GRAPH_PROMPT_PATH
from controller.utils.general_utils import print_debug


class GraphManager:
    """
    Keeps an up-to-date scene graph that every other TypeFly
    component can share without ROS.
    """
    def __init__(self, start_coords: tuple[float, float] = (0.0, 0.0)) -> None:
        self.llm_wrapper = LLMWrapper()
        
        # Load prompt safely
        if os.path.exists(USER_GRAPH_PROMPT_PATH):
            with open(USER_GRAPH_PROMPT_PATH, "r") as f:
                self.user_prompt = f.read()
        else:
            print(f"[Warning] Graph prompt not found at {USER_GRAPH_PROMPT_PATH}")
            self.user_prompt = "{description}"

        self.drone_pose = np.zeros(3)
        self.graph_handler = GraphHandler()
        
        # Initialize starting region
        start_region = self.graph_handler.get_closest_region()
        
        # Handle empty graph case
        if start_region is None:
            start_region = "region_0"

        self.graph_handler.update_location(start_region)
        self.current_region = start_region

        # Ensure the start node exists
        if not self.graph_handler.graph.has_node(start_region):
            self.graph_handler.update_with_node_flexible(
                node=start_region,
                edges=[],
                attrs={"coords": list(start_coords), "type": "region"},
            )

        # Set current task, then used to update drone position
        self.current_task: Task = None

    def set_current_task(self, task: Task):
        self.current_task = task

    def update_graph_from_file(self):
        self.graph_handler.update_graph_from_file()
        # Sync region pointer after reload so new detections connect properly.
        if self.graph_handler.current_location:
            self.current_region = self.graph_handler.current_location
        else:
            closest = self.graph_handler.get_closest_region()
            if closest:
                self.graph_handler.update_location(closest)
                self.current_region = closest

    def request_new_graph(self, description: Optional[str], image: Optional[str]) -> dict:
        prompt = self.user_prompt.format(description=description, coordinates=self.drone_pose)
        return self.llm_wrapper.request(prompt, RequestType.NEW_GRAPH, image=image)

    def get_drone_pose(self):
        return self.drone_pose
    
    def get_current_region(self):
        return self.current_region
    
    def get_graph(self):
        """
        Returns the current graph state as a JSON string and persists it to disk.
        """
        self.write_graph_to_file()
        return self.graph_handler.to_json_str()
    

    def get_dense_graph(self):
        """
        Returns the current graph state as an adiency list string representation and persists it to disk.
        """
        self.write_graph_to_file()
        return self.graph_handler.to_adjacency_list_str()
    
    def name_region(self, name: str):
        """Renames the current region if it hasn't been named yet (starts with 'region')."""
        if self.current_region.startswith("region"):
            print(f"Naming region {self.current_region} to {name}")
            self.graph_handler.name_region(name)
            # Note: Logic to update self.current_region might be needed here 
            # if graph_handler.name_region doesn't handle the pointer update implicitly via ID change
    
    # --- Pose
    def update_pose(self, pose: Sequence[float], yaw: float) -> None:
        """
        Call from TelloWrapper after every motion command.
        xy is in *world* centimetres.
        """
        self.drone_pose = np.asarray(pose)[:3]
        self.drone_pose[3] = yaw
        print_debug(f"Drone pose update: {self.drone_pose}")
        
        result = self.graph_handler.ensure_region_for_pose(self.drone_pose[:3])
        if result[1]: # If region changed or created
            self.current_region = result[0]
        
        self.current_task.update_drone_position(self.drone_pose, self.current_region)

    # --- Objects
    def add_object_detection(self, label: str, xy: Sequence[float] = None) -> None:
            # Normalize label (e.g., remove unique IDs like "cup_01" -> "cup")
            # or keep it unique depending on your needs.
            node_id = label.split("_")[0] 
            
            # 1. Avoid duplicates in current region
            # (This relies on the topological check we added to GraphHandler)
            if self.graph_handler.is_node_in_current_region(node_id): 
                return

            # 2. Enforce Topological-Only Objects
            # We explicitly ignore 'xy' here.
            attrs = {"type": "object"}
                
            # 3. Update Graph
            # Connects purely based on current_region (Topology)
            self.graph_handler.update_with_node_flexible(
                node=node_id, 
                edges=[self.current_region], 
                attrs=attrs
            )

    def write_graph_to_file(self):
        """Persist graph to disk for visualization or debugging."""
        try:
            with open(GRAPH_TXT_PATH, "w") as f:
                graph_dict = json.loads(self.graph_handler.to_json_str())
                json.dump(graph_dict, f, indent=2)
                f.write("\n")
        except Exception as e:
            print(f"[Error] Failed to write graph to file: {e}")
                
    # --- Regions
    def add_region(self, region_xy: Sequence[float], region_name: str = None) -> str:
        """
        Adds a new region node discovered e.g. when the drone crosses a
        distance threshold.
        """
        if region_name is None:
            region_name = f"region_{uuid.uuid4().hex[:4]}"
            
        attrs = {"coords": list(map(float, region_xy)), "type": "region"}
        self.graph_handler.update_with_node_flexible(node=region_name, edges=[self.current_region], attrs=attrs)
        
        return region_name
