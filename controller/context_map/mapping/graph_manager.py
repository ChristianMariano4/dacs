import json
import os
from typing import Dict, List, Sequence
from controller.context_map.mapping.graph_handler import GraphHandler
from controller.context_map.spine_util import UpdatePromptFormer
import numpy as np
import uuid

from controller.llm_controller import LLMController

class GraphManager:
    """
    Keeps an up-to-date scene graph that every other TypeFly
    component can share without ROS.
    """
    def __init__(self, llmController: LLMController, init_graph_json: str | None = None, start_region: str = "region_0", start_coords: tuple[float, float] = (0.0, 0.0)) -> None:
        self.llmController = llmController
        self.graph_handler = GraphHandler(self, init_graph_json or "", init_node=start_region)
        self.updater = UpdatePromptFormer()
        self.current_region = start_region

        if not self.graph_handler.graph.has_node(start_region):
            # no neighbours yet – just a stand-alone region
            self.graph_handler.update_with_node(
                node=start_region,
                edges=[],
                attrs={"coords": list(start_coords), "type": "region"},
            )

    def get_llm_controller(self) -> LLMController:
        return self.llmController
    
    # --- Pose
    def update_pose(self, xy: Sequence[float]) -> None:
        """
        Call from TelloWrapper after every motion command.
        xy is in *world* centimetres; keep units consistent.
        """
        result = self.graph_handler.ensure_region_for_pose(xy)
        if result[1]:
            self.current_region = result[0]
        # self.graph_handler.update_location(self.current_region)
        # self.updater.update(location_updates=[self.current_region])

    # --- Objects
    def add_object_detection(self, label: str, xy: Sequence[float]) -> None:
        node_id = f"{label}_{uuid.uuid4().hex[:4]}"
        attrs   = {"coords": list(map(float, xy)), "type": "object"}
        self.graph_handler.update_with_node(node=node_id, edges=[self.current_region], attrs=attrs)
        # prepare LLM-prompt diff
        self.updater.update(new_nodes=[{"name": node_id,
                                        "type": "object",
                                        "coords": f"[{xy[0]:.1f}, {xy[1]:.1f}]"}],
                                        new_connections=[[node_id, self.current_region]])
        # ---------------------------------------------
        # Append graph snapshot to file
        # ---------------------------------------------
        log_dir = "graph_logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "graph_history.jsonl"), "a") as f:
            json.dump({
                "graph": self.graph_handler.as_json_str,
            }, f)
            f.write("\n")
        
        
    # --- Regions
    def add_region(self, region_xy: Sequence[float]) -> str:
        """
        Adds a new region node discovered e.g. when the drone crosses a
        distance threshold. Returns the region node name so the caller can
        immediately `goto()` it in the graph.
        """
        name = f"region_{uuid.uuid4().hex[:4]}"
        attrs  = {"coords": list(map(float, region_xy)), "type": "region"}
        self.graph_handler.update_with_node(node=name, edges=[self.current_region], attrs=attrs)

        self.updater.update(new_nodes=[{"name": name,
                                        "type": "region",
                                        "coords": f"[{region_xy[0]:.1f}, {region_xy[1]:.1f}]"}],
                                        new_connections=[[self.current_region, name]])
        return name
    
    # --- Prompt
    def flush_prompt_updates(self) -> str:
        """Return and clear the diff string ready to feed the SPINE LLM."""
        return self.updater.form_updates()