import re
import numpy as np
from loguru import logger
from wandb.util import downsample

from TSCAssistant.tsc_agent_prompt import render_prompt
from TSCAssistant.feature_translator import translate_features_for_llm
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC


class TSCAgent:
    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose

    def extract_features(self, obs: np.ndarray, info: dict) -> dict:


        obj_map = np.fliplr(obs['image'][:, :, 0])
        state_map = np.fliplr(obs['image'][:, :, 2])


        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None


        agent_pos = (3,0)
        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        door_state = None
        if door_pos:
            inv_state = {v: k for k, v in STATE_TO_IDX.items()}
            door_state = inv_state.get(int(state_map[door_pos]), "unknown")

        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else None

        dist_to_key = manh(agent_pos, key_pos)
        dist_to_door = manh(agent_pos, door_pos)
        dist_to_goal = manh(agent_pos, goal_pos)

        other_mask = (obj_map > OBJECT_TO_IDX["empty"]) & (obj_map != OBJECT_TO_IDX["agent"])
        other_locs = [tuple(loc) for loc in np.argwhere(other_mask)]
        dists = [manh(agent_pos, loc) for loc in other_locs]
        valid_dists = [d for d in dists if d is not None]
        dist_to_nearest = min(valid_dists) if valid_dists else None

        is_key_visible = key_pos is not None
        is_door_visible = door_pos is not None

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
            **counts
        }

    def agent_run(self, sim_step, obs, action, infos):
        info = infos if isinstance(infos, dict) else infos[0]

        raw_feats = self.extract_features(obs, info)
        translated_feats = translate_features_for_llm(raw_feats)

        prompt = render_prompt(
            env_name=info.get("env", "MiniGrid"),
            features=translated_feats,
            action=int(action)
        )

        try:
            llm_response = self.llm.invoke(prompt)
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
                if overridden:
                    logger.info(f"[LLM @ Step {sim_step}] OVERRIDDEN by LLM: PPO {action} → LLM {llm_action}, EXPLANATION: {explanation}")
                else:
                    logger.info(f"[LLM @ Step {sim_step}] LLM agrees with PPO: {action}, EXPLANATION: {explanation}")
            print()
            return llm_action, overridden

        except Exception as e:
            logger.error(f"[LLM ERROR] {e}; falling back to PPO action")
            return int(action), False
