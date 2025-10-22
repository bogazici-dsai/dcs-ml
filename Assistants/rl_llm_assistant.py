import re
import numpy as np
from loguru import logger

from Assistants.minigrid_agent_prompt import render_prompt
from Assistants.feature_translator import translate_features_for_llm
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
# HarfangAssistant.py - RL-LLM bridge for HarfangEnv
import json
from typing import Any, Dict, Optional

class MinigridAgent:
    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose

    def extract_features(self, obs: np.ndarray, info: dict) -> dict:

        env = info.get("llm_env", None)
        obj_map = np.fliplr(obs['image'][:, :, 0])
        state_map = np.fliplr(obs['image'][:, :, 2])
        has_key = bool(getattr(env.unwrapped, "carrying", None))

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None


        agent_pos = (3,0)
        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}

        door_state = "unknown"
        if door_pos:
            state_idx = state_map[door_pos]  # an integer like 0, 1, or 2
            door_state = idx_to_state.get(state_idx, "unknown")

        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else None

        dist_to_key = manh(agent_pos, key_pos)
        dist_to_door = manh(agent_pos, door_pos)
        dist_to_goal = manh(agent_pos, goal_pos)

        other_mask = (obj_map > OBJECT_TO_IDX["empty"]) & (obj_map != OBJECT_TO_IDX["agent"]) & (obj_map != OBJECT_TO_IDX["wall"])
        other_locs = [tuple(loc) for loc in np.argwhere(other_mask)]
        dists = [manh(agent_pos, loc) for loc in other_locs]
        valid_dists = [d for d in dists if d is not None]
        dist_to_nearest = min(valid_dists) if valid_dists else None


        is_key_visible = bool(find(OBJECT_TO_IDX["key"]))
        is_door_visible = bool(find(OBJECT_TO_IDX["door"]))

        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        is_adj_key = is_adjacent(agent_pos, key_pos)
        is_adj_door = is_adjacent(agent_pos, door_pos)


        def get_vertical_distance(p, q):
            return abs(p[0] - q[0]) if (p and q) else None

        def get_horizontal_distance(p, q):
            return abs(p[1] - q[1]) if (p and q) else None

        vert_dist_to_goal = get_vertical_distance(agent_pos, goal_pos)
        horiz_dist_to_goal = get_horizontal_distance(agent_pos, goal_pos)
        vertical_distance_to_key = get_vertical_distance(agent_pos, key_pos)
        horizontal_distance_to_key = get_horizontal_distance(agent_pos, key_pos)
        #relative directions
        def rel_dir_agent_frame(agent_pos, q_pos):
            if not (agent_pos and q_pos): return None
            # Global vector
            dy, dx = q_pos[1] - agent_pos[1], q_pos[0] - agent_pos[0]

            if dx == 0:
                if dy < 0:
                    return "down"
                elif dy > 0:
                    return "up"
            elif dx < 0:
                return "left"
            elif dx > 0:
                return "right"

        rel_dir_to_key = rel_dir_agent_frame(agent_pos, key_pos)
        rel_dir_to_door = rel_dir_agent_frame(agent_pos, door_pos)

        grid_size = obs['image'].shape[:2]
        num_visible_objects = int(other_mask.sum())

        counts = {
            name: int((obj_map == idx).sum())
            for name, idx in OBJECT_TO_IDX.items()
            if name not in ("empty", "agent")
        }

        free_mask = (obj_map == OBJECT_TO_IDX["empty"])
        frees = 0
        if agent_pos:
            y, x = agent_pos
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1] and free_mask[ny, nx]:
                    frees += 1
        multiple_paths_open = frees >= 2
        def object_in_front(agent_pos: tuple[int, int],
                                         obj_map: np.ndarray,
                                         ):

            # Compute the neighbor cell one step forward in global coords
            yn = agent_pos[1] + 1
            xn = agent_pos[0]
            # Check facing
            if obj_map[xn, yn] == OBJECT_TO_IDX["key"]:
                return "Key"
            if obj_map[xn, yn] == OBJECT_TO_IDX["door"]:
                return "Door"
            if obj_map[xn, yn] == OBJECT_TO_IDX["wall"]:
                return "Wall"
            if obj_map[xn, yn] == OBJECT_TO_IDX["goal"]:
                return "Goal"
            if obj_map[xn, yn] == OBJECT_TO_IDX["empty"]:
                return "Empty Cell"
        object_in_the_front=object_in_front(agent_pos, obj_map)
        def is_facing_object_agent_frame(agent_pos: tuple[int, int],
                                         obj_map: np.ndarray,
                                         obj_idx: int
                                         ) -> bool:
            """
            Returns True if, in the agent’s own frame, the given object sits
            *one step directly in front* (i.e. at local dir == "up").
            """

            # Compute the neighbor cell one step forward in global coords
            yn = agent_pos[1] + 1
            xn = agent_pos[0]

            # Check facing
            if obj_map[xn, yn] == obj_idx:
                return True
            else:
                return False

        # … inside your feature extractor …
        is_facing_wall = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["wall"])
        # Similarly, you can now do:
        is_facing_key = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["key"])
        is_facing_door = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["door"])
        return {
            "grid_size": grid_size,
            "agent_pos": agent_pos,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "goal_pos": goal_pos,
            "has_key": has_key,
            "door_state": door_state,
            "dist_to_key": dist_to_key,
            "dist_to_door": dist_to_door,
            "dist_to_goal": dist_to_goal,
            "dist_to_nearest_object": dist_to_nearest,
            "is_key_visible": is_key_visible,
            "is_door_visible": is_door_visible,
            "num_visible_objects": num_visible_objects,
            "vertical_distance_to_goal": vert_dist_to_goal,
            "horizontal_distance_to_goal": horiz_dist_to_goal,
            "vertical_distance_to_key": vertical_distance_to_key,
            "horizontal_distance_to_key": horizontal_distance_to_key,
            "rel_dir_to_key": rel_dir_to_key,
            "rel_dir_to_door": rel_dir_to_door,
            "is_adjacent_to_key": is_adj_key,
            "is_adjacent_to_door": is_adj_door,
            "multiple_paths_open": multiple_paths_open,
            "facing_key": is_facing_key,
            "facing_wall": is_facing_wall,
            "facing_door": is_facing_door,
            "front_object": object_in_the_front,
            **counts
        }

    def agent_run(self, sim_step, obs, action, infos, env=None):
        info = infos if isinstance(infos, dict) else infos[0]

        #raw_feats = self.extract_features(obs, info)
        #translated_feats = translate_features_for_llm(raw_feats)

        # prompt = render_prompt(
        #     env_name=info.get("env", "Harfang"),
        #     features=translated_feats,
        #     action=int(action)
        # )
        prompt = None

        try:
            #llm_response = self.llm.invoke(prompt)
            llm_response = "MOCK LLM, Selected action: 4"
            txt = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # Parse "Selected action: X"
            m = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", txt, re.IGNORECASE)
            if not m:
                raise ValueError(f"No action found in LLM output:\n{txt}")

            llm_action = int(m.group(1))
            overridden = (llm_action != int(action))

            # Extract reasoning/explanation
            explanation = txt[m.end():].strip()

            if self.verbose:
                #logger.info(f"WHOLE LLM RESPONSE:{txt}")
                #front_obj = translated_feats.get("front_object", "Unknown")

                if overridden:
                    logger.info(f"[MOCK LLM @ Step {sim_step}] OVERRIDDEN by LLM: PPO {action} → LLM {llm_action}, EXPLANATION: {explanation}")
                else:
                    logger.info(f"[MOCK LLM @ Step {sim_step}] LLM agrees with PPO: {action}, EXPLANATION: {explanation}")
            print()
            return llm_action, overridden

        except Exception as e:
            logger.error(f"[LLM ERROR] {e}; falling back to PPO action")
            return int(action), False

# HARFANG AGENT
class HarfangAgent:
    """
    Assistant that extracts features from HarfangEnv observation,
    queries LLM, and maps response into valid action index.
    """

    ACTIONS = {
        "TRACK": 0,
        "EVADE": 1,
        "CLIMB": 2,
        "FIRE": 3,
        "HOLD": 4
    }

    def __init__(self, llm, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    def extract_features(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from dict-based observation (matches HarfangEnv)."""
        return {
            "altitude": obs.get("altitude"),
            "plane_heading": obs.get("plane_heading"),
            "plane_pitch_att": obs.get("plane_pitch_att"),
            "ally_health": obs.get("ally_health"),
            "oppo_health": obs.get("oppo_health"),
            "locked": obs.get("locked"),
            "target_angle": obs.get("target_angle"),
            "missile1_state": obs.get("missile1_state"),
            "missile_count": obs.get("missile_count"),
            "distance_to_enemy": obs.get("distance_to_enemy"),
            "relative_bearing": obs.get("relative_bearing"),
            "enemy_missile_in_air_count": obs.get("enemy_missile_in_air_count"),
            "ally_unfired_slots": obs.get("ally_unfired_slots"),
            "mwr_signal": obs.get("mwr_signal"),
        }

    def _generate_prompt(self, features: Dict[str, Any]) -> str:
        """Generate closed-form tactical prompt for the LLM."""
        return f"""
You are an air combat assistant for an RL agent.



### Current Situation:
- Altitude: {features['altitude']}
- Heading: {features['plane_heading']}
- Pitch Attitude: {features['plane_pitch_att']}
- Distance to Enemy: {features['distance_to_enemy']}
- Relative Bearing: {features['relative_bearing']}
- Radar Lock: {"YES" if features['locked'] > 0 else "NO"}
- Target Angle: {features['target_angle']}
- Ally Health: {features['ally_health']}
- Enemy Health: {features['oppo_health']}
- Ally Missiles in Air: {features['missile_count']}
- Number of unfired missiles : {features['ally_unfired_slots']} 
- Enemy Missiles in Air: {features['enemy_missile_in_air_count']}
- Missile Threat Signal: {"YES" if features['mwr_signal'] > 0 else "NO"}

# Additional Information about environment:
- FIRE action works if only if Radar Lock is YES

### Available Actions:
- TRACK  (0)
- EVADE  (1)
- CLIMB  (2)
- FIRE   (3)
- HOLD   (4)



### Task:
Select ONLY ONE of the above actions that best fits the tactical situation.
Respond strictly in the following JSON format:

{{
  "action": "<TRACK|EVADE|CLIMB|FIRE|HOLD>",
  "reason": "<short reasoning, max 1 sentence>"
}}
"""

    def query_llm(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Send prompt to LLM and parse response."""
        prompt = self._generate_prompt(features)
        resp = self.llm.invoke(prompt)

        # LLM response may be string or object
        text = resp.content if hasattr(resp, "content") else str(resp)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if self.verbose:
                print("[Assistant] Failed to parse LLM response, fallback HOLD.")
            data = {"action": "HOLD", "reason": "fallback"}

        return data

    def action_mapper(self, llm_output: Dict[str, Any]) -> int:
        """Map LLM action string to action index."""
        action_str = llm_output.get("action", "HOLD").upper()
        return self.ACTIONS.get(action_str, self.ACTIONS["HOLD"])

    def decide(self, obs: Dict[str, Any]) -> int:
        """Main entry: from obs → features → LLM → mapped action index."""
        features = self.extract_features(obs)
        llm_output = self.query_llm(features)
        action_idx = self.action_mapper(llm_output)

        if self.verbose:
            print(f"[Assistant] LLM chose {llm_output.get('action')} → index {action_idx} "
                  f"({llm_output.get('reason')})")

        return action_idx


