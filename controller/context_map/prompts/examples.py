EXAMPLE_GRAPH_1 = {
    "objects": [
        {"name": "house_1"},
        {"name": "house_2"},
        {"name": "grocery_store_1"},
        {"name": "shed_1"},
        {"name": "shed_2"},
    ],
    "regions": [
        {"name": "example_road_1", "coords": [-1, 0]},
        {"name": "example_road_2", "coords": [-2, 0]},
        {"name": "field_11", "coords": [0, 1]},
        {"name": "field_13", "coords": [2, 3]},
    ],
    "object_connections": [
        ["house_1", "example_road_1"],
        ["house_2", "example_road_2"],
        ["shed_1", "field_11"],
        ["shed_2", "field_13"],
    ],
    "region_connections": [
        ["example_road_1", "example_road_2"],
        ["example_road_1", "field_11"],
        ["field_11", "field_13"],
    ],
    "robot_location": "example_road_1",
}


EXAMPLE_1 = [
    {
        "role": "user",
        "content": "task: I need a shovel. Is there one in the scene?"
        + f"Scene graph:{str(EXAMPLE_GRAPH_1)}",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "find a shovel for the user."\
        "relevant_graph": "field_11, field_13, unobserved_node(shovel)", \
        "reasoning": "The graph does not contain any shovels. However, I know that the graph may be incomplete, so I will explore before providing a definitive answer. I will first map, then if needed I will add regions. There are two sheds in the scene, and shovels are often found near sheds. Therefore, for each of the sheds, I will navigate to the nearby region and map.",\
        "plan": "Minispec plan to go to field_11 based on current position and field_11 coordinates; scan the region to search objects; go to field_13 based on current position and field_13 coordinates; scan the region to search objects."\n}',
        
    },
    {
        "role": "user",
        "content": "updates:[no_updates()]",
    },
    {
        "role": "assistant",
        "content": '{\n\
            "primary_goal": "find a shovel for the user."\
            "relevant_graph": "field_13, unobserved_node(shovel)",\
            "reasoning": "After mapping field_11, which is next to shed_1, I did not find a shovel. I also planned on going to example_node_14 to map near shed_2, so I will continue my plan.",\
            "plan": "Minispec plan to go to field_13 based on current position and field_13 coordinates; scan the region to search objects."\n}',
    },
    {
        "role": "user",
        "content": "updates:[add_nodes({ name: shovel_1, type: object}), add_connections([shovel_1, field_13])]",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "find a shovel for the user."\
        "relevant_graph": "shovel_1", \
        "reasoning": "After exploring field_13, which is next to shed_2, I found a nearby shovel, shovel_1, that is connected to field_13. This fulfills the users request.",\
        "plan": "Minispec plan to answer the user that a shovel (shovel_1) is located near shed_2 and connected to field_13."\n}',
    },
    {
        "role": "user",
        "content": "task: How many grocery stores are currently in the scene?",
    },
    {
        "role": "assistant",
        "content": '{ \
        "primary_goal": "count the number of grocery stores in the scene"\
        "relevant_graph": "grocery_store_1",\
        "reasoning": "There is one grocery store in the scene. Although the scene may be incomplete, I am only asked about currently observed grocery stores. Thus the answer is one",\
        "plan": "Minispec plan to answer the user that there is one grocery store in the scene (grocery_store_1)."}',
    },
]


EXAMPLE_GRAPH_2 = {
    "objects": [
        {"name": "office_building_1"},
        {"name": "office_building_2"},
        {"name": "robot_1"},
        {"name": "robot_2"},
        {"name": "robot_3"},
        {"name": "example_truck_1"},
    ],
    "regions": [
        {"name": "example_sidewalk_1", "coords": [180, 51]},
        {"name": "example_road_1", "coords": [190, 50]},
        {"name": "charging_station_1", "coords": [205, 60]},
        {"name": "charging_station_2", "coords": [190, 68]},
        {"name": "example_node_2", "coords": [150, 38]},
        {"name": "example_node_1", "coords": [130, 38]},
    ],
    "object_connections": [
        ["office_building_1", "example_sidewalk_1"],
        ["office_building_2", "example_sidewalk_1"],
        ["robot_1", "charging_station_1"],
        ["robot_2", "charging_station_1"],
        ["robot_3", "charging_station_2"],
        ["example_truck_1", "charging_station_2"],
    ],
    "region_connections": [
        ["example_sidewalk_1", "example_road_1"],
        ["example_sidewalk_1", "charging_station_1"],
        ["example_sidewalk_1", "charging_station_2"],
    ],
    "robot_location": "example_road_1",
}

EXAMPLE_2 = [
    {
        "role": "user",
        "content": "task: How many robots are currently observed, where are they located, and which is the southmost robot?"
        + f"scene graph: {str(EXAMPLE_GRAPH_2)}",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Answer the users questions about robots in the scene. The user is likely referring to robot_1, robot_2, and robot_3"\
        "relevant_graph": "robot_1, robot_2, robot_3", \
        "reasoning": "There are currently three robots in the scene graph, robot_1, robot_2, and robot_2. robot_1 and robot_2 are in charging_station_1, while robot_3 is in charging_station_2. The y coordinate, which corresponds to south / north. robot_1 and robot_2 are in charging_station_1, whose y coordinate is lower than charging_station_2. Thus, robot_1 and robot_2 are equally southmost.",\
        "plan": "Minispec plan to report that robot_1 robot_2 are in charging_station_1 at [205, 60], and robot_3 is in charging_station_2 at [190, 68]; then explain that robot_1 and robot_2 share the lowest y coordinate and are thus equally southmost."\n}',
    },
    {"role": "user", "content": "Feedback: That is correct."},
    {
        "role": "user",
        "content": "task: I cannot reach example_node_2 from my current location, as there is no path. I want to try to find a connection. What node should I go to in order to look for a connection.",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Answer the users questions about the node to go to in order to reach example_node_2 from his current location."\
        "relevant_graph": "example_road_1, example_sidewalk_1, example_node_2", \
        "reasoning": "Based on my current location, example_road_1, I can reach example_sidewalk_1, charging_station_1, and charging_station_2. Out of this list, example_sidewalk_1 at [180, 51] is the closest to example_node_2 at [150, 38].",\
        "plan": "Minispec plan to answer the user that example_sidewalk_1 is the best candidate to explore to reach example_node_2, and suggest going there to try to discover a new connection."\n}',
    },
    {"role": "user", "content": "task: What is around the sidewalk?"},
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find what is around the sidewalk. From scene context, the user is referring to example_sidewalk_1."\
        "relevant_graph": "example_sidewalk_1",\n"reasoning": "The user wants to know what is around the sidewalk. There is only one sidewalk in the scene, example_sidewalk_1, so the user must be referring to that region. Because my intended goal is within a exploration radius of 3 meters, the most efficient way for me to complete that task is to go there and explore the region with a radius of 3 meters.",\
        "plan": "Minispec plan to go to example_sidewalk_1; map the region to record known objects and regions; then explore in a 3 meter radius to detect new regions."\n}',
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: discovered_region_1, type: region coords: [180, 54]}, {name: discovered_region_2, type: region, coords: [180, 48]}, {name: discovered_region_3, type: region, coords: [183, 51]}, {name: discovered_region_4, type: region, coords: [177, 61]}), add_connections([example_sidewalk_1, discovered_region_1], [example_sidewalk_1, discovered_region_2], [example_sidewalk_1, discovered_region_3], [example_sidewalk_1, discovered_region_4]), update_node_attributes({name: discovered_region_4, description=sidewalk}), update_robot_location(discovered_region_4)]",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find what is around the sidewalk. From scene context, the user is referring to example_sidewalk_1."\
        "goal": "example_sidewalk_1, discovered_region_1, discovered_region_2, discovered_region_3, discovered_region_4", \
        "reasoning": "I explore example_sidewalk_1, which means I tried to discover new regions. I then mapped those regions in search of new objects. I did find four new regions. but no new objects. By the description of discovered_region_4, it must be part of a sidewalk.",\
        "plan": "Minispec plan to answer that example_sidewalk_1 has four connected regions around it; mention that discovered_region_4 is also a sidewalk; report that no new objects were discovered."\n}',
    },
]

EXAMPLE_GRAPH_3 = {
    "objects": [],
    "regions": [
        {"name": "example_node_1", "coords": [10, 10]},
        {"name": "example_node_2", "coords": [12, 20]},
        {"name": "example_node_3", "coords": [15, 30]},
        {"name": "example_node_4", "coords": [12, 15]},
        {"name": "example_node_5", "coords": [10, 5]},
        {"name": "example_ground_6", "coords": [-2, -2]},
        {"name": "charging_station_1", "coords": [30, 40]},
    ],
    "object_connections": [],
    "region_connections": [
        ["example_node_2", "example_node_3"],
        ["example_node_1", "example_node_4"],
        ["example_node_4", "example_node_5"],
        ["example_node_5", "example_ground_6"],
    ],
    "robot_location": "example_node_1",
}

EXAMPLE_3 = [
    {
        "role": "user",
        "content": "task: I lost my robot when it was filling its battery. It is red. Can you find it and tell me its status?\n"
        + f"scene graph: {str(EXAMPLE_GRAPH_3)}",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find the users lost robot. The user provided contextual details that will help me find the robot."\
        "relevant_graph": "charging_station_1", \
        "reasoning": "The graph does not contain any robots, but the graph does contain a charging station, charging_station_1. The robot may be charging its battery there. I will determine if the nearby regions are connected to the booth.",\
        "plan": "Minispec plan to go to example_node_3; extend the map toward coordinates [20, 35] to search for a path to charging_station_1; then replan based on new graph state."}',
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: discovered_region_1, type: region, coords: [15, 30]}), add_connections( [example_node_3, discovered_region_1], [discovered_region_1, charging_station_1]), update_robot_location(example_node_3)]",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find the users lost robot. The user provided contextual details that will help me find the robot."\
        "relevant_graph": "charging_station_1", \
        "reasoning": Upon exploring, I was able to add a new region that allows me to navigate to my goal, charging_station_1. I will now map charging_station_1 to see if I can find the users robot",\
        "plan": "Minispec plan to move to charging_station_1 and scan the region to detect any nearby objects such as the lost robot."}',
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: robot_1, type: object}), add_connections([robot_1, charging_station_1]), update_robot_location(charging_station_1)]",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find the users lost robot. The user provided contextual details that will help me find the robot."\
        "relevant_graph": "robot_1", \
        "reasoning": "Upon mapping the charging_station, I found a robot. I will now inspect it.",\
        "plan": "Minispec plan to probe what is the color of robot_1 and whether it is damaged."\n}',
    },
    {
        "role": "user",
        "content": "updates: [update_node_attributes({ name: robot_1, description=this is a red robot that is not damaged})]",
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find the users lost robot. The user provided contextual details that will help me find the robot."\
        "relevant_graph": "robot_1",\
        "reasoning": "I found a robot that is red. The robot is not damaged. This may be the users robot, based on the information I received.",\
        "plan": "Minispec plan to inform the user that a red, undamaged robot was found, which likely matches their lost robot."\n}',
    },
]

EXAMPLE_GRAPH_4 = {
    "objects": [],
    "regions": [],
    "object_connections": [],
    "region_connections": [],
    "robot_location": None,
}

EXAMPLE_4 = [
    {
        "role": "user",
        "content": "task: I need an apple. Is there one in the scene?"
        + f"\nScene graph: {str(EXAMPLE_GRAPH_4)}"
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find an apple for the user.",\n\
        "relevant_graph": "unobserved_node(apple)",\n\
        "reasoning": "The graph is empty, and no apples are currently known. I will begin by exploring a nearby region. Since I have no prior knowledge, I will move in a direction and try to discover regions and objects.",\n\
        "plan": "Minispec plan to explore new region by going forward and scanning the region to discover any objects. Then replan"\n}'
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: discovered_region_1, type: region, coords: [0, 2]}), add_connections([discovered_region_1]), update_robot_location(discovered_region_1)]"
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find an apple for the user.",\n\
        "relevant_graph": "unobserved_node(apple)",\n\
        "reasoning": "After moving to discovered_region_1, I still do not see any apples. I will continue exploring.",\n\
        "plan": "Minispec plan to explore new region by going forward and scan the new region. Then replan"\n}'
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: banana_1, type: object }, { name: orange_1, type: object }, { name: discovered_region_2, type: region, coords: [0, 4] }), add_connections([banana_1, discovered_region_2], [orange_1, discovered_region_2]), update_robot_location(discovered_region_2)]"
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find an apple for the user.",\n\
        "relevant_graph": "banana_1, orange_1, discovered_region_2",\n\
        "reasoning": "While I have not found an apple, I now see two fruits in discovered_region_2. Apples often appear near similar fruit objects, so I will explore adjacent areas.",\n\
        "plan": "Minispec plan to explore new region by going forward and scan the new region. Then replan"\n}'
    },
    {
        "role": "user",
        "content": "updates: [add_nodes({ name: apple_1, type: object }, { name: discovered_region_3, type: region, coords: [0, 6] }), add_connections([apple_1, discovered_region_3]), update_robot_location(discovered_region_3)]"
    },
    {
        "role": "assistant",
        "content": '{\n\
        "primary_goal": "Find an apple for the user.",\n\
        "relevant_graph": "apple_1",\n\
        "reasoning": "I found an apple (apple_1) connected to discovered_region_3, located just beyond a region that contains other fruit. This satisfies the user’s request.",\n\
        "plan": "Minispec plan to answer the user that an apple named apple_1 was found in discovered_region_3."\n}'
    }
]