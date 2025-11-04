import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

from controller.llm.llm_wrapper import LLMWrapper, RequestType


# empty graph from which to start
EMPTY_GRAPH = {
    "objects": [],
    "regions": [{"name": "region_0", "coords": [0, 0]}],
    "region_connections": [],
    "object_connections": [],
}


# from string to list
def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return x.replace("[", "").replace("]", "").split(",")


def to_float_list(x: str):
    return [float(y) for y in to_list(x)]


# parse coordinate
def parse_graph_coord(
    coord_as_str: str, origin: np.ndarray, rotation: Optional[Rotation] = None
) -> List[float]:
    """Parsing coordinate from incoming graph.
    Coordinates are assumed to be ENU

    Parameters
    ----------
    coord_as_str : str
        Coordinate as a string `[x, y]`
    origin : np.ndarray
        Origin of coordinate system
    rotation : Optional[Rotation], optional
        Rotation of coordinate system

    Returns
    -------
    List[float]
        coordinates `[x, y]` after SE2 transform defined
        by origin and rotation
    """
    # apply SE2 transform
    coords = np.array(to_float_list(coord_as_str))
    # print(f"--")
    # print(coords)
    coords = apply_transform(coords, origin=origin, rotation=rotation)

    # back to list
    coords = [round(float(coords[0]), 1), round(float(coords[1]), 1)]
    # print(coords)
    # print("--")

    return coords


def apply_transform(
    x: np.ndarray, origin: np.ndarray, rotation: Rotation
) -> np.ndarray:
    """Apply SE2 transform to x

    Note rot is an SO3 transform, so we lift x to R-3
    then map back to R-2, ignoring the third dim.

    Parameters
    ----------
    x : np.ndarray
        A vector in R-2
    origin : np.ndarray
        An origin in R-2
    rot : Rotation
        A rotation in SO3

    Returns
    -------
    np.ndarray
        Transformed vector
    """
    x -= origin
    if rotation != None:
        # print(f"rotation xyz: {rotation.as_euler('xyz')}")
        x = np.concatenate([x, np.zeros(1)])
        x = rotation.apply(x)
        # print(f"transformed x {x}")
        x = x[:2]

    return x

# parse a graph given as input
def parse_graph(
    data: Dict[str, Dict[str, str]],
    custom_data: Optional[Dict[str, Dict[str, str]]] = {},
    rotation: Optional[Rotation] = None,
    utm_origin: Optional[np.ndarray] = None,
    flip_coords=False,
) -> Tuple[nx.Graph, str]:
    """Parse scene graph in `data` into a networkx object.

    Parameters
    ----------
    data : Dict[str, Dict[str, str]]
        graph where keys-values are nodes-attributes
    rotation : Optional[Rotation]
        current rotation of robot

    Returns
    -------
    Tuple[nx.Graph, str]
        Networkx and string of json
    """
    origin = np.array([0, 0])
    """
    if "origin" in data:
        origin = utm_origin
        origin = np.array(to_float_list(data["origin"]))
        # print(f"original origin is: {origin}")
        origin = utm_origin - origin
    """
    if utm_origin is not None:
        origin = utm_origin

    if len(custom_data):
        add_keys = ["regions", "region_connections", "objects", "object_connections"]
        for key in add_keys:
            if key in data and key in custom_data:
                data[key].extend(custom_data[key])

    # print(f"origin: {origin}, rot: {rotation}")

    G = nx.Graph()
    for node in data["objects"]:
        # c = node["coords"]
        # print(f"node: {node}, coords: {c}")
        coords = parse_graph_coord(node["coords"], origin=origin, rotation=rotation)
        if flip_coords:
            raise ValueError()
            # print("flipping coords")
            coords = [coords[0], -coords[1]]
        G.add_node(node["name"], coords=coords, type="object")

    for node in data["regions"]:
        assert "coords" in node, node
        c = node["coords"]
        # print(f"node: {node}, coords: {c}")
        coords = parse_graph_coord(node["coords"], origin=origin, rotation=rotation)

        if flip_coords:
            raise ValueError
            # print("flipping coords")
            coords = [coords[0], -coords[1]]

        G.add_node(node["name"], coords=coords, type="region")

    for edge in data["object_connections"]:
        c1 = G.nodes[edge[0]]["coords"]
        c2 = G.nodes[edge[1]]["coords"]
        # print(f"edge: {edge}, c1, c2: {c1}, {c2}")
        dist = np.linalg.norm(np.array(c1) - np.array(c2))
        G.add_edge(edge[0], edge[1], type="object", weight=dist)

    for edge in data["region_connections"]:
        c1 = G.nodes[edge[0]]["coords"]
        c2 = G.nodes[edge[1]]["coords"]
        # print(f"edge: {edge}, c1, c2: {c1}, {c2}")
        dist = np.linalg.norm(np.array(c1) - np.array(c2))
        G.add_edge(edge[0], edge[1], type="region", weight=dist)
    
    drone_position = data["current_position"]["coords"]

    return G, str(data), drone_position

# class that handle the graph. If given one as input (grapt_path), otherwise crate an empty one
class GraphHandler:
    def __init__(self, graph_path: str) -> None:
        if graph_path == "":
            self.graph = nx.Graph()
            self.as_json_str = "{}"
            self.current_location = "region_0"
            self.drone_position = np.zeros(3)
        else:
            self.current_location = None
            with open(graph_path) as f:
                data = json.load(f)
            self.graph, self.as_json_str, self.drone_position = parse_graph(data)
            self.current_location = self.get_closest_region()


    
    def get_closest_region(self):
        x, y, _ = self.drone_position
        graph_json = json.loads(self.to_json_str())
        min_dist = None
        curr_region = None
        for region in graph_json["regions"]:
            curr_dist = math.dist([x, y], [region["coords"][0], region["coords"][1]])
            if min_dist == None or curr_dist < min_dist:
                min_dist = curr_dist
                curr_region = region
        return curr_region["name"]
    
    def ensure_region_for_pose(
        self,
        pose_xyz: Sequence[float],          # (x, y) in the SAME frame your graph uses
        threshold: float = 100.0,            # cm; tweak to taste TODO: add global threshold
        connect_to_nearest: bool = True,   # make an edge to the closest region
    ) -> Tuple[str, bool]:
        """
        Make sure the point `pose_xy` is assigned to a region.
        Returns (region_name, created_new_flag)

        • If an existing region is within `threshold`, we just return it.
        • Otherwise we:
            1. generate a unique region name,
            2. add the node with type='region' and the given coords,
            3. optionally connect it to the nearest region with a
               weighted 'region' edge,
            4. update self.current_location.
        """
        pose_xy = np.asarray(pose_xyz)[:3]
        pose_xy = np.asarray(pose_xy, dtype=float)

        # --- 1. Gather existing regions -------------------------
        region_nodes, region_locs = self.get_region_nodes_and_locs()
        if region_locs.size:                       # at least one region exists
            dists = np.linalg.norm(region_locs - pose_xy, axis=1)
            min_idx = dists.argmin()
            if dists[min_idx] <= threshold:       # close enough → reuse
                region_name = region_nodes[min_idx]
                self.current_location = region_name
                return region_name, False         # no new region created
            nearest_old = region_nodes[min_idx]
        else:
            nearest_old = None                    # first region ever

        # --- 2. Create a unique name ----------------------------
        #TODO: add a global counter for the new region id
        base = "region"
        i = 1
        while f"{base}_{i}" in self.graph.nodes:
            i += 1
        region_name = f"{base}_{i}"

        # --- 3. Add the new node --------------------------------
        self.update_with_node_flexible(
            region_name,
            edges=[],                               # edges added below (if any)
            attrs={"type": "region", "coords": pose_xy.round(1).tolist()},
        )
        self.update_location(region_name)

        # --- 4. Optionally connect it ---------------------------
        if connect_to_nearest and nearest_old is not None:
            w = float(np.linalg.norm(pose_xy - self.get_node_coords(nearest_old)[0]))
            self.update_with_edge(
                (region_name, nearest_old),
                {"type": "region", "weight": w},
            )

        # --- 5. Book-keeping ------------------------------------
        self.current_location = region_name
        print(f"New region created {region_name}")
        return region_name, True

    def name_region(self, name):
        """Add a display name to the current region."""
        if not self.current_location:
            print("No current location set")
            return
        
        # Verify the node exists and is a region
        if not self.contains_node(self.current_location):
            print(f"Current location {self.current_location} not found")
            return
        
        node_attrs = self.graph.nodes[self.current_location]
        if node_attrs.get("type") != "region":
            print(f"Current location {self.current_location} is not a region")
            return

        # Add or update the display_name attribute
        self.update_node_description(self.current_location, display_name=name)
        print(f"Added display name '{name}' to {self.current_location}")

        # print(self.current_location)
        result = self.lookup_node(self.current_location)
        print(result[0])
        # if str(result[0]["name"]).startswith("region"):
        #     # the region has been already name before
        #     return
        # if result[1]:
        #     result[0]["name"] = name

    def reset(
        self,
        graph_as_json: str,
        current_location: Optional[str] = "",
        rotation: Optional[Rotation] = None,
        utm_origin: Optional[np.ndarray] = None,
        custom_data: Optional[Dict[str, Dict[str, str]]] = {},
        flip_coords=False,
    ) -> bool:
        try:
            data = json.loads(graph_as_json)

            # TODO, logic is obtuse
            # priority is current location -> incoming argument -> value in data

            if self.current_location == None or self.current_location == "":
                self.current_location = current_location

            self.graph, self.as_json_str = parse_graph(
                data,
                rotation=rotation,
                utm_origin=utm_origin,
                custom_data=custom_data,
                flip_coords=flip_coords,
            )
            self.as_json_str = self.to_json_str()
        except Exception as ex:
            print(f"\nexception: {ex}")
            return False
        return True

    def to_json_str(self, extra_data={}) -> str:
        added_edges = set()
        graph_dict = {
            "objects": [],
            "regions": [],
            "object_connections": [],
            "region_connections": [],
            "current_position": {}
        }
        
        for node in self.graph.nodes:
            node_attrs = self.graph.nodes[node]
            node_type = node_attrs["type"]
            
            if node_type == "region":
                name = node_attrs.get("display_name", node)
                coords = node_attrs.get("coords", node)
                # Use display_name if available, otherwise use the node identifier
                graph_dict[f"{node_type}s"].append({"name": name, "coords": coords})
            else:
                # For objects, check for display_name as well
                name = node_attrs.get("display_name", node)
                coords = node_attrs.get("coords", node)
                graph_dict[f"{node_type}s"].append({"name": name, "coords": coords})

            # Handle edges
            for neighbor in self.get_neighbors(node):
                if tuple(sorted((node, neighbor))) not in added_edges:
                    # Get the display names for edge representation
                    node_name = self.graph.nodes[node].get("display_name", node)
                    neighbor_name = self.graph.nodes[neighbor].get("display_name", neighbor)
                    
                    graph_dict[f"{node_type}_connections"].append(
                        sorted([node_name, neighbor_name])
                    )
                    added_edges.add(tuple(sorted((node, neighbor))))

        if self.current_location != None:
            # Use display name for current location too
            current_loc_attrs = self.graph.nodes.get(self.current_location, {})
            current_loc_name = current_loc_attrs.get("display_name", self.current_location)
            graph_dict["current_position"] = {"coords": self.drone_position, "region": current_loc_name}

        graph_dict.update(extra_data)

        return json.dumps(graph_dict, indent=2)

    # update location of the robot
    def update_location(self, new_location: str) -> bool:
        if not self.contains_node(new_location):
            return False
        self.current_location = new_location
        return True

    # return the neighbors of a node, given a specif node type (if given) 
    def get_neighbors_by_type(
        self, node: str, node_type: Optional[str] = ""
    ) -> Dict[str, List[str]]:
        """Get neighbors of `node`

        Parameters
        ----------
        node : str
        type : Optional[str]
            If given, only return neighbors of this type

        Returns
        -------
        Dict[str, List[str]]
            node: attributes
        """
        ret_val = {}
        if node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))

            for neighbor in neighbors:
                if node_type == "":
                    ret_val[neighbor] = self.lookup_node(neighbor)[0]
                elif (
                    node_type != "" and self.lookup_node(neighbor)[0]["type"] == "type"
                ):
                    ret_val[neighbor] = self.lookup_node(neighbor)[0]

            # return {node: self.lookup_node(node)[0] for node in neighbors}
        return ret_val

    def get_edges(self, node: str) -> Dict[str, List[str]]:
        out = {}
        for edge in self.graph.edges(node):
            out[edge] = self.graph.edges[edge]
        return out

    # search for an object in the graph and return where it is
    def lookup_object(
        self, node: str
    ) -> Tuple[Tuple[str, Dict], Tuple[str, Dict], bool]:
        """Check if an object is in the graph. If so, return the object
        attributes and the connecting region, if any

        Parameters
        ----------
        node : str
            _description_

        Returns
        -------
        Tuple[Tuple[str, Dict], Tuple[str, Dict], bool]:
        - object name, attributes
        - region name, attributes
        - true if all information found

        """
        if node in self.graph.nodes and self.graph.nodes[node]["type"] == "object":
            node_attr = self.graph.nodes[node]

            neighbors = list(self.graph.neighbors(node))

            # if found, just region first for now
            if len(neighbors) >= 1:
                # object-region connections are always (object, region) order
                region_attr = self.graph.nodes[neighbors[0]]
                return (
                    (node, node_attr),
                    (neighbors[0], region_attr),
                    True,
                )
            else:
                return (node, node_attr), (None, None), False

        return (None, None), (None, None), False

    def get_neighbors(self, node_name: str) -> List[str]:
        if not self.contains_node(node_name):
            return []
        return list(nx.neighbors(self.graph, node_name))

    def get_path(
        self, start_node, end_node, only_regions: Optional[bool] = False
    ) -> List[str]:
        return nx.shortest_path(self.graph, start_node, end_node)

    def contains_node(self, node: str) -> bool:
        return node in self.graph.nodes
    
    def is_node_in_current_region(self, node: str) -> bool:
        """
        Return ``True`` when *node* belongs to the region stored in
        ``self.current_location``.

        ─ For region nodes, this is true only if the node **is** the current
          region itself.

        ─ For object nodes, this is true when the object has a direct edge
          to the current region (that edge is created automatically when the
          object is added).

        The method falls back to ``False`` if:
          * the graph has no current region yet, or
          * the node does not exist, or
          * the object is not connected to the current region.
        """
        # make sure both the region and the node are valid
        if not self.current_location or not self.contains_node(node):
            return False

        # a region is contained in itself
        if node == self.current_location:
            return True

        # for objects, look for a direct connection to the current region
        return self.current_location in self.get_neighbors(node)

    def path_exists_from_current_loc(self, target: str) -> bool:
        assert self.current_location != None, "current location is unknown"
        return nx.has_path(self.graph, self.current_location, target)

    def lookup_node(self, node: str) -> Tuple[Dict, bool]:
        if self.contains_node(node):
            return self.graph.nodes[node], True
        else:
            return {}, False

    def get_node_coords(self, node: str) -> Tuple[np.ndarray, bool]:
        if self.contains_node(node):
            return self.graph.nodes[node]["coords"], True
        else:
            return (
                np.zeros(
                    2,
                ),
                False,
            )

    def update_node_description(self, node, **attrs) -> None:
        self.graph.nodes[node].update(attrs)

    def get_node_type(self, node: str) -> str:
        node_info, success = self.lookup_node(node)
        if success and "type" in node_info:
            return node_info["type"]
        else:
            return ""
   
    def update_with_node_flexible(
        self,
        node: str,
        edges: Optional[List[str]] = None,
        attrs: Dict[str, Any] = {},
    ) -> None:
        assert "type" in attrs, "Node attribute 'type' is required"

        node_type = attrs["type"]
        has_coords = "coords" in attrs and attrs["coords"] is not None

        if edges is None:
            edges = [self.current_location]

        self.graph.add_node(node, **attrs)
        # print(node)

        for edge in edges:
            # If either node or edge lacks coordinates, set default weight and skip distance calc
            try:
                c1 = self.graph.nodes[node]["coords"]
                c2 = self.graph.nodes[edge]["coords"]
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
            except KeyError:
                dist = 1.0  # default weight if coordinates are missing

            self.graph.add_edge(
                node,
                edge,
                type=self.graph.nodes[node]["type"],
                weight=dist,
            )


    def update_with_edge(self, edge: Tuple[str, str], attrs: Dict[str, Any] = {}):
        self.graph.add_edge(edge[0], edge[1], **attrs)

    def remove_edge(self, start: str, end: str) -> None:
        try:  # TODO hacky should check if edge exists first
            # note edges are bidirectional
            self.graph.remove_edge(start, end)
        except Exception as ex:
            return

    # TODO figure out what we wanna do with this
    def get_region_nodes_and_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Update set of region nodes. Assumes graph will be updated
        during operation.

        Updates
        - region_nodes: array of region names (str)
        - region_node_locs: array of region locations
        """

        region_nodes_locs = []
        region_nodes = []

        for node in self.graph.nodes:
            if self.graph.nodes[node]["type"] == "region":
                node_loc = self.graph.nodes[node]["coords"]
                region_nodes_locs.append(node_loc)
                region_nodes.append(node)

        # TODO finish
        self.region_nodes = np.array(region_nodes)
        self.region_node_locs = np.array(region_nodes_locs)

        return self.region_nodes, self.region_node_locs

    def get_closest_reachable_node(
        self, goal_node: str, current_node: Optional[str] = None
    ) -> str:
        """Get closest reachable node from current_node to goal_node

        Parameters
        ----------
        goal_node : str
        current_node : str

        Returns
        -------
        str
            closest reachable node from current_node to goal_node
        """
        if current_node == None:
            assert self.current_location != None, "current_location must be set"
            current_node = self.current_location

        nodes_reachable_from_curr_loc = list(
            nx.node_connected_component(self.graph, current_node)
        )
        nodes_reachable_from_goal = list(
            nx.node_connected_component(self.graph, goal_node)
        )

        # only consider region nodes
        nodes_reachable_from_curr_loc = [
            n
            for n in nodes_reachable_from_curr_loc
            if self.get_node_type(n) == "region"
        ]
        nodes_reachable_from_goal = [
            n for n in nodes_reachable_from_goal if self.get_node_type(n) == "region"
        ]

        coords_of_nodes_reachable_curr_loc = np.array(
            [self.get_node_coords(n)[0] for n in nodes_reachable_from_curr_loc]
        )

        closest_node_dist = np.inf
        closest_node_id = current_node
        target_node_id = goal_node
        for node_reachable_from_goal in nodes_reachable_from_goal:
            node_dists = np.linalg.norm(
                coords_of_nodes_reachable_curr_loc
                - np.array(self.get_node_coords(node_reachable_from_goal)[0]),
                axis=-1,
            )

            if node_dists.min() < closest_node_dist:
                closest_node_dist = node_dists.min()
                closest_node_id = nodes_reachable_from_curr_loc[node_dists.argmin()]
                target_node_id = node_reachable_from_goal

        # return nodes_reachable_from_curr_loc[closest_node]
        return closest_node_id, target_node_id

    def __str__(self) -> str:
        out = f"Nodes\n---\n"
        for node in self.graph.nodes:
            attrs, _ = self.lookup_node(node)
            out += f"\t{node}: {attrs}"

        object_edges = ""
        region_edges = ""
        for edge in self.graph.edges:
            if "object" in [self.get_node_type(e) for e in edge]:
                object_edges += f"\t[{edge[0]}, {edge[1]}]\n"
            else:
                region_edges += f"\t[{edge[0]}, {edge[1]}]\n"

        out += f"\nObject edges\n---\n"
        out += object_edges + "\n"

        out += f"Region edges:\n---\n"
        out += region_edges
        return out
