import json
import math
import os
from typing import Dict, List, Optional, Sequence
from controller.context_map.graph_handler import GraphHandler
import numpy as np
import uuid

from controller.llm.llm_wrapper import LLMWrapper, RequestType


class GraphManager:
    """
    Keeps an up-to-date scene graph that every other TypeFly
    component can share without ROS.
    """
    def __init__(self, llm_controller, start_coords: tuple[float, float] = (0.0, 0.0)) -> None:
        #TODO: put start_region as the closer region to the drone
        self.llmController = llm_controller
        self.llm_wrapper = LLMWrapper()
        with open("controller/assets/tello/memory/user_graph_prompt.txt", "r") as f:
            self.user_prompt = f.read()
        self.drone_pose = np.zeros(3)
        self.graph_handler = GraphHandler()
        start_region = self.graph_handler.get_closest_region()
        self.graph_handler.update_location(start_region)
        self.current_region = start_region

        if not self.graph_handler.graph.has_node(start_region):
            # no neighbours yet – just a stand-alone region
            self.graph_handler.update_with_node_flexible(
                node=start_region,
                edges=[],
                attrs={"coords": list(start_coords), "type": "region"},
            )
    def request_new_graph(self, description: Optional[str], image: Optional[str]) -> dict:
        prompt = self.user_prompt.format(description=description)
        return self.llm_wrapper.request(prompt, RequestType.NEW_GRAPH, image=image)

    def get_drone_pose(self):
        return self.drone_pose
    
    def get_current_region(self):
        return self.current_region
    
    def get_graph(self):
        return self.graph_handler.to_json_str()
    
    def name_region(self, name:str):
        if self.current_region.startswith("region"):
            print("Naming region")
            self.graph_handler.name_region(name)
            # self.current_region = name
    
    # --- Pose
    def update_pose(self, pose: Sequence[float]) -> None:
        """
        Call from TelloWrapper after every motion command.
        xy is in *world* centimetres; keep units consistent.
        """
        self.drone_pose = np.asarray(pose)[:3]
        print(f"[DEBUG] Inside update_pose function. The pose of the drone is: {self.drone_pose}")
        result = self.graph_handler.ensure_region_for_pose(self.drone_pose)
        if result[1]:
            self.current_region = result[0]
        # self.graph_handler.update_location(self.current_region)
        # self.updater.update(location_updates=[self.current_region])

    # --- Objects
    def add_object_detection(self, label: str, xy: Sequence[float] = None) -> None:
        # node_id = f"{label}_{uuid.uuid4().hex[:4]}"
        node_id = label.split("_")[0]
        if self.graph_handler.is_node_in_current_region(label): #o bject already saved in this region
            return
        if xy is not None:
            attrs   = {"coords": list(map(float, xy)), "type": "object"}
            # self.updater.update(new_nodes=[{"name": node_id,
            #                     "type": "object",
            #                     "coords": f"[{xy[0]:.1f}, {xy[1]:.1f}]"}],
            #                     new_connections=[[node_id, self.current_region]])
        else:
            attrs   = {"type": "object"}
            # self.updater.update(new_nodes=[{"name": node_id,
            #                         "type": "object"}])
        self.graph_handler.update_with_node_flexible(node=node_id, edges=[self.current_region], attrs=attrs)
        # prepare LLM-prompt diff

        # ---------------------------------------------
        # Append graph snapshot to file
        # ---------------------------------------------
        # print("Object added")
        log_dir = "controller/assets/tello/memory"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "graph.txt"), "w") as f:
            graph_dict = json.loads(self.graph_handler.to_json_str())  # parse string → dict
            json.dump(graph_dict, f, indent=2)  # pretty JSON, no escapes
            f.write("\n")
                
    # --- Regions
    def add_region(self, region_xy: Sequence[float], region_name: str = None) -> str:
        """
        Adds a new region node discovered e.g. when the drone crosses a
        distance threshold. Returns the region node name so the caller can
        immediately `goto()` it in the graph.
        """
        if region_name is None:
            region_name = f"region_{uuid.uuid4().hex[:4]}"
        attrs  = {"coords": list(map(float, region_xy)), "type": "region"}
        self.graph_handler.update_with_node_flexible(node=region_name, edges=[self.current_region], attrs=attrs)

        # self.updater.update(new_nodes=[{"name": region_name,
        #                                 "type": "region",
        #                                 "coords": f"[{region_xy[0]:.1f}, {region_xy[1]:.1f}]"}],
        #                                 new_connections=[[self.current_region, region_name]])
        return region_name
    
    # --- Prompt
    def flush_prompt_updates(self) -> str:
        """Return and clear the diff string ready to feed the SPINE LLM."""
        return self.updater.form_updates()