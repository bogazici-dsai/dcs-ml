# MinigridAssistant/feature_translator.py

def translate_features_for_llm(features: dict) -> dict:
    """
    Converts structured feature values into natural language explanations for LLM.

    Input:
        features: dict of raw environment features (booleans, ints, strings, etc.)

    Output:
        dict with the same keys, but values are human-readable strings
    """
    explanations = {}

    # Grid size
    grid_size = features.get("grid_size")
    if grid_size:
        explanations["grid_size"] = f"The observable grid is {grid_size[0]} rows by {grid_size[1]} columns."
    else:
        explanations["grid_size"] = "The observable grid size is unknown."

    # Key visibility
    explanations["is_key_visible"] = (
        "The key is visible in the agent's view." if features.get("is_key_visible")
        else "The key is not visible in the agent's view."
    )

    # Door visibility
    explanations["is_door_visible"] = (
        "The door is visible in the agent's view." if features.get("is_door_visible")
        else "The door is not visible in the agent's view."
    )

    # Door state
    door_state = features.get("door_state")
    if door_state:
        explanations["door_state"] = f"The door is {door_state}."
    else:
        explanations["door_state"] = "The door is not visible in the agent’s current view, so its state is unknown."
    # Has key
    has_key = features.get("has_key")
    explanations["has_key"] = (
        "The agent has the key." if has_key
        else "The agent does not have the key."
    )

    # Distances
    explanations["dist_to_key"] = f"The distance between key and the agent is {features.get('dist_to_key', '?')}."
    explanations["dist_to_door"] = f"The distance between door and the agent is {features.get('dist_to_door', '?')}."
    explanations["dist_to_goal"] = f"The distance between goal and the agent is {features.get('dist_to_goal', '?')}."
    explanations["dist_to_nearest_object"] = (
        f"The distance between the nearest object and the agent is {features.get('dist_to_nearest_object', '?')}"
    )
    # Vertical distance to goal
    vdist = features.get("vertical_distance_to_goal")
    explanations["vertical_distance_to_goal"] = (
        f"The goal is {vdist} rows away vertically." if vdist is not None
        else "The vertical distance to the goal is unknown."
    )

    # Horizontal distance to goal
    hdist = features.get("horizontal_distance_to_goal")
    explanations["horizontal_distance_to_goal"] = (
        f"The goal is {hdist} columns away horizontally." if hdist is not None
        else "The horizontal distance to the goal is unknown."
    )
    # Vertical distance to key
    vdist_key = features.get("vertical_distance_to_key")
    explanations["vertical_distance_to_key"] = (
        f"The key is {vdist_key} rows away vertically." if vdist_key is not None
        else "The vertical distance to the key is unknown."
    )

    # Horizontal distance to key
    hdist_key = features.get("horizontal_distance_to_key")
    explanations["horizontal_distance_to_key"] = (
        f"The key is {hdist_key} columns away horizontally." if hdist_key is not None
        else "The horizontal distance to the key is unknown."
    )
    # Relative directions
    rel_dir_key = features.get("rel_dir_to_key")
    explanations["rel_dir_to_key"] = (
        f"The key is to the {rel_dir_key} of the agent." if rel_dir_key
        else "The direction to the key is unknown."
    )
    rel_dir_door = features.get("rel_dir_to_door")
    explanations["rel_dir_to_door"] = (
        f"The door is to the {rel_dir_door} of the agent." if rel_dir_door
        else "The direction to the door is unknown."
    )

    # Adjacency
    explanations["is_adjacent_to_key"] = (
        "The agent is adjacent to the key." if features.get("is_adjacent_to_key")
        else "The agent is not adjacent to the key."
    )
    explanations["is_adjacent_to_door"] = (
        "The agent is adjacent to the door." if features.get("is_adjacent_to_door")
        else "The agent is not adjacent to the door."
    )

    # Path availability
    explanations["multiple_paths_open"] = (
        "There are multiple paths the agent can take." if features.get("multiple_paths_open")
        else "The agent is in a narrow or blocked area with limited paths."
    )

    # Facing wall
    is_facing_wall = features.get("facing_wall")
    explanations["facing_wall"] = (
        "The agent is currently facing a wall." if is_facing_wall
        else "The agent is not facing a wall."
    )
    # Facing key
    is_facing_key = features.get("facing_key")
    explanations["facing_key"] = (
        "The agent is currently facing a key." if is_facing_key
        else "The agent is not facing a key."
    )
    #Facing door
    is_facing_door = features.get("facing_door")
    explanations["facing_door"] = (
        "The agent is currently facing a door." if is_facing_door
        else "The agent is not facing a door."
    )

    # Number of visible objects
    num_objects = features.get("num_visible_objects")
    explanations["num_visible_objects"] = (
        f"There are {num_objects} objects visible in the grid."
        if num_objects is not None else "The number of visible objects is unknown."
    )
    # Front Object
    object_ = features.get("front_object")
    explanations["front_object"] = (
        f"The {features.get('front_object')} is in the front of the agent." if object_ else None
    )
    # Positions
    explanations["agent_pos"] = f"The agent is located at grid position {features.get('agent_pos', '?')}."
    explanations["key_pos"] = f"The key is located at grid position {features.get('key_pos', '?')}."
    explanations["door_pos"] = f"The door is located at grid position {features.get('door_pos', '?')}."
    explanations["goal_pos"] = f"The goal is located at grid position {features.get('goal_pos', '?')}."

    return explanations
