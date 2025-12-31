import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

from controller.utils.constants import GRAPH_TXT_PATH

class GraphHandler:
    """
    Manages the NetworkX graph representation of the environment.
    Handles parsing, updating, and querying topological data.
    """
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.current_location: Optional[str] = None
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.as_json_str = "{}"
        
        # Initialize graph state
        self.update_graph_from_file()

    def update_graph_from_file(self):
        """Loads the graph state from the persistent memory file."""
        if not os.path.exists(GRAPH_TXT_PATH):
            # Start empty if no file exists
            self.graph = nx.Graph()
            return

        try:
            with open(GRAPH_TXT_PATH, "r") as f:
                data = json.load(f)
            
            # 1. Parse the new graph structure
            self.graph, self.as_json_str, self.drone_position = parse_graph(data)
            
            # 2. State Validation
            # If we had a region, check if it still exists in the new graph.
            if self.current_location and self.current_location not in self.graph.nodes:
                print(f"[GraphHandler] Warning: Stale location '{self.current_location}' not found in updated graph. Resetting.")
                self.current_location = None

            # 3. Re-binding
            # If location is missing (was null or just reset), find the geometrically closest region
            if not self.current_location and self.graph.nodes:
                self.current_location = self.get_closest_region()
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Error] Failed to load graph from file: {e}")
            self.graph = nx.Graph()

    def get_closest_region(self) -> Optional[str]:
        """Finds the region node geometrically closest to the drone."""
        if not self.graph.nodes:
            return None
            
        x, y, _ = self.drone_position
        min_dist = float('inf')
        curr_region = None
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "region" and "coords" in attrs:
                cx, cy = attrs["coords"]
                curr_dist = math.dist([x, y], [cx, cy])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    curr_region = node
                    
        return curr_region
    
    def ensure_region_for_pose(
        self,
        pose_xyz: Sequence[float],
        threshold: float = 100.0, # cm
        connect_to_nearest: bool = True,
    ) -> Tuple[str, bool]:
        """
        Ensures the current pose belongs to a region.
        If existing region is close, returns it. Otherwise, creates a new one.
        
        Returns: (region_name, created_new_flag)
        """
        pose_xy = np.asarray(pose_xyz)[:3]
        
        # 1. Check existing regions
        region_nodes = [n for n, a in self.graph.nodes(data=True) if a.get("type") == "region"]
        
        best_node = None
        min_dist = float('inf')

        for node in region_nodes:
            coords = self.graph.nodes[node]["coords"]
            dist = np.linalg.norm(np.array(coords) - np.array(pose_xy[:2]))
            if dist < min_dist:
                min_dist = dist
                best_node = node

        # Reuse if close enough
        if best_node and min_dist <= threshold:
            self.current_location = best_node
            return best_node, False

        # 2. Create new region
        # Find first available ID
        i = 0
        while True:
            name = f"region_{i}"
            if name not in self.graph.nodes:
                break
            i += 1
        
        # 3. Add node
        self.update_with_node_flexible(
            name,
            edges=[], 
            attrs={"type": "region", "coords": pose_xy[:2].round(1).tolist()}
        )

        # 4. Connect to topological neighbor (Graph building)
        if connect_to_nearest and best_node:
            # Distance becomes edge weight
            w = float(min_dist)
            self.update_with_edge(
                (name, best_node),
                {"type": "region", "weight": w}
            )

        self.current_location = name
        return name, True

    def name_region(self, name: str):
        """Add a semantic display name to the current region (e.g. 'Kitchen')."""
        if not self.current_location:
            return
        
        if self.current_location not in self.graph.nodes:
            return
        
        if self.graph.nodes[self.current_location].get("type") != "region":
            return

        self.update_node_description(self.current_location, display_name=name)

    def to_json_str(self) -> str:
            """Serializes to JSON. Objects strictly exclude coordinates."""
            graph_dict = {
                "objects": [],
                "regions": [],
                "object_connections": [],
                "region_connections": [],
                "current_position": {}
            }
            
            added_edges = set()
            
            for node, attrs in self.graph.nodes(data=True):
                node_type = attrs.get("type", "object")
                display_name = attrs.get("display_name", node)
                
                if node_type == "region":
                    # Regions act as spatial anchors
                    coords = attrs.get("coords", [0, 0])
                    graph_dict["regions"].append({"name": display_name, "coords": coords})
                else:
                    # Objects are purely topological now
                    graph_dict["objects"].append({"name": display_name})

                # Process edges
                for neighbor in self.graph.neighbors(node):
                    edge_pair = tuple(sorted((node, neighbor)))
                    if edge_pair in added_edges:
                        continue
                    
                    neighbor_attrs = self.graph.nodes[neighbor]
                    n1_name = display_name
                    n2_name = neighbor_attrs.get("display_name", neighbor)
                    
                    # specific key logic can remain, or simplify to generic connections
                    conn_key = f"{node_type}_connections" 
                    graph_dict[conn_key].append(sorted([n1_name, n2_name]))
                    added_edges.add(edge_pair)

            # Update current position
            if self.current_location and self.current_location in self.graph.nodes:
                current_loc_name = self.graph.nodes[self.current_location].get("display_name", self.current_location)
                graph_dict["current_position"] = {
                    "coords": self.drone_position.tolist(), 
                    "region": current_loc_name
                }
            else:
                graph_dict["current_position"] = {
                    "coords": self.drone_position.tolist(),
                    "region": "unknown"
                }

            return json.dumps(graph_dict, indent=2)

    def to_adjacency_list_str(self) -> str:
        """
        Serializes the graph into a dense Adjacency List format.
        Format: Region(x,y): obj1, obj2 | Current: [x,y,z] @ Region
        """
        adj = defaultdict(list)
        region_info = {}
        
        # 1. Map Regions and their coordinates
        for node, attrs in self.graph.nodes(data=True):
            name = attrs.get("display_name", node)
            if attrs.get("type") == "region":
                coords = attrs.get("coords", [0, 0])
                region_info[name] = f"({coords[0]},{coords[1]})"
                
        # 2. Build Adjacency: Group neighbors under their parent region
        for u, v in self.graph.edges():
            u_attr = self.graph.nodes[u]
            v_attr = self.graph.nodes[v]
            u_name = u_attr.get("display_name", u)
            v_name = v_attr.get("display_name", v)
            
            # If one is a region, treat it as the 'key' for the adjacency list
            if u_attr.get("type") == "region":
                adj[u_name].append(v_name)
            elif v_attr.get("type") == "region":
                adj[v_name].append(u_name)
            else:
                # For object-to-object edges (if any)
                adj[u_name].append(v_name)

        # 3. Serialize into a compact string
        lines = []
        for region, neighbors in adj.items():
            coord_str = region_info.get(region, "")
            neighbors_str = ", ".join(neighbors)
            lines.append(f"{region}{coord_str}: {neighbors_str}")

        # 4. Current Position
        curr_reg = "unknown"
        if self.current_location in self.graph.nodes:
            curr_reg = self.graph.nodes[self.current_location].get("display_name", self.current_location)
        
        pos_str = f"POS: {self.drone_position.tolist()} @ {curr_reg}"
        
        return " | ".join(lines) + " | " + pos_str

    def update_location(self, new_location: str) -> bool:
        if new_location not in self.graph.nodes:
            return False
        self.current_location = new_location
        return True

    def update_with_node_flexible(
            self,
            node: str,
            edges: Optional[List[str]] = None,
            attrs: Dict[str, Any] = {},
        ) -> None:
            """
            Adds a node. Handles Metric weighting for Regions and Topological weighting for Objects.
            """
            self.graph.add_node(node, **attrs)

            if edges is None:
                edges = [self.current_location] if self.current_location else []

            # Determine if the new node has spatial data
            has_coords = "coords" in attrs
            node_coords = np.array(attrs["coords"]) if has_coords else None

            for target in edges:
                if target not in self.graph.nodes:
                    continue
                
                target_attrs = self.graph.nodes[target]
                target_has_coords = "coords" in target_attrs
                
                # Calculate Edge Weight
                # 1. If both have coordinates (Region <-> Region), use Euclidean distance
                if has_coords and target_has_coords:
                    target_coords = np.array(target_attrs["coords"])
                    weight = np.linalg.norm(node_coords - target_coords)
                # 2. If object is linked to a region, weight is symbolic (e.g., 1.0 or 0.0)
                else:
                    weight = 1.0

                self.graph.add_edge(
                    node,
                    target,
                    type=attrs.get("type", "object"),
                    weight=weight,
                )
                
    def update_with_edge(self, edge: Tuple[str, str], attrs: Dict[str, Any] = {}):
        self.graph.add_edge(edge[0], edge[1], **attrs)

    def update_node_description(self, node, **attrs) -> None:
        if node in self.graph.nodes:
            self.graph.nodes[node].update(attrs)

    # --- Query Helpers ---

    def is_node_in_current_region(self, node: str) -> bool:
        """Checks if a node is directly connected to the current region."""
        if not self.current_location or node not in self.graph.nodes:
            return False
        if node == self.current_location:
            return True
        return self.current_location in self.graph.neighbors(node)

    def get_path(self, start_node, end_node) -> List[str]:
        try:
            return nx.shortest_path(self.graph, start_node, end_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def lookup_node(self, node: str) -> Tuple[Dict, bool]:
        if node in self.graph.nodes:
            return self.graph.nodes[node], True
        return {}, False

    # --- Parsing Utils ---

def to_float_list(x: Union[str, List]) -> List[Any]:
    """Robustly converts string representation of list to actual list."""
    if isinstance(x, list):
        return x
    # Remove brackets and split
    cleaned = str(x).replace("[", "").replace("]", "")
    parts = [p.strip() for p in cleaned.split(",")]
    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            result.append(p)
    return result

def parse_graph_coord(
    coord_input: Any, origin: np.ndarray, rotation: Optional[Rotation] = None
) -> List[float]:
    """Parses coordinates and applies SE2 transformation."""
    coords = np.array(to_float_list(coord_input))
    
    # Apply transformation
    coords -= origin
    if rotation is not None:
        # Lift to 3D for rotation, then project back
        coords_3d = np.concatenate([coords, np.zeros(1)])
        rotated = rotation.apply(coords_3d)
        coords = rotated[:2]

    return [round(float(coords[0]), 1), round(float(coords[1]), 1)]

def parse_graph(
    data: Dict[str, Any],
    rotation: Optional[Rotation] = None,
    utm_origin: Optional[np.ndarray] = None,
) -> Tuple[nx.Graph, str, np.ndarray]:
    """
    Reconstructs graph. Objects are parsed without coordinates.
    """
    origin = utm_origin if utm_origin is not None else np.array([0, 0])
    G = nx.Graph()
    
    # 1. Add Objects (Topology only)
    for node in data.get("objects", []):
        # distinct from regions: we do not parse/store coords
        G.add_node(node["name"], type="object")

    # 2. Add Regions (Topology + Metric)
    for node in data.get("regions", []):
        coords = parse_graph_coord(node.get("coords", [0,0]), origin=origin, rotation=rotation)
        G.add_node(node["name"], coords=coords, type="region")

    # 3. Add Edges
    for conn_type in ["object_connections", "region_connections"]:
        type_label = "object" if "object" in conn_type else "region"
        for edge in data.get(conn_type, []):
            if len(edge) < 2: continue
            u, v = edge[0], edge[1]
            
            if G.has_node(u) and G.has_node(v):
                # Check if both nodes have coordinates for metric distance
                u_attr, v_attr = G.nodes[u], G.nodes[v]
                
                if "coords" in u_attr and "coords" in v_attr:
                    c1 = np.array(u_attr["coords"])
                    c2 = np.array(v_attr["coords"])
                    w = np.linalg.norm(c1 - c2)
                else:
                    # Topological connection (Object -> Region)
                    w = 1.0
                    
                G.add_edge(u, v, type=type_label, weight=w)

    drone_pos_data = data.get("current_position", {}).get("coords", [0,0,0])
    drone_position = np.array(drone_pos_data)

    return G, json.dumps(data), drone_position