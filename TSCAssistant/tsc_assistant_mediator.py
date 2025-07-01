import re
import numpy as np
from loguru import logger
from typing import Dict, Tuple, Optional

from TSCAssistant.tsc_agent_prompt import render_prompt
from TSCAssistant.feature_translator import translate_features_for_llm
from TSCAssistant.mediator import Mediator
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class TSCAgentWithMediator:
    """
    Enhanced TSC Agent that uses a Mediator to decide when to query LLM.
    Now includes rich spatial features from tsc_assistant_updated.
    """

    def __init__(self,
                 llm,
                 obs_shape: tuple = (7, 7, 3),
                 device: str = "cpu",
                 verbose: bool = True,
                 train_mediator: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.train_mediator = train_mediator

        # Initialize mediator
        self.mediator = Mediator(
            obs_shape=obs_shape,
            device=device,
            verbose=verbose
        )

        # Track LLM interactions
        self.current_plan = None
        self.interaction_count = 0
        self.override_count = 0

    def extract_features(self, obs: np.ndarray, info: dict) -> dict:
        """
        ENHANCED feature extraction - combines original mediator features
        with rich spatial features from tsc_assistant_updated.
        """

        # Get environment for carrying state (from updated version)
        env = info.get("llm_env", None)

        # Use coordinate system from updated version
        obj_map = np.fliplr(obs['image'][:, :, 0])
        state_map = np.fliplr(obs['image'][:, :, 2])

        # Agent carrying state (from updated version)
        has_key = bool(getattr(env.unwrapped, "carrying", None)) if env else False

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # Agent position (use updated version coordinates)
        agent_pos = (3, 0)

        # Find objects
        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        # Door state (keep original mediator logic but enhance it)
        door_state = None
        if door_pos:
            inv_state = {v: k for k, v in STATE_TO_IDX.items()}
            door_state = inv_state.get(int(state_map[door_pos]), "unknown")

        # Distance calculations
        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else None

        dist_to_key = manh(agent_pos, key_pos)
        dist_to_door = manh(agent_pos, door_pos)
        dist_to_goal = manh(agent_pos, goal_pos)

        # ADD: Enhanced spatial features from updated version
        def get_vertical_distance(p, q):
            return abs(p[0] - q[0]) if (p and q) else None

        def get_horizontal_distance(p, q):
            return abs(p[1] - q[1]) if (p and q) else None

        vert_dist_to_goal = get_vertical_distance(agent_pos, goal_pos)
        horiz_dist_to_goal = get_horizontal_distance(agent_pos, goal_pos)
        vertical_distance_to_key = get_vertical_distance(agent_pos, key_pos)
        horizontal_distance_to_key = get_horizontal_distance(agent_pos, key_pos)

        # ADD: Relative directions from updated version
        def rel_dir_agent_frame(agent_pos, q_pos):
            if not (agent_pos and q_pos):
                return None
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

        # ADD: Enhanced environmental analysis from updated version
        other_mask = (obj_map > OBJECT_TO_IDX["empty"]) & (obj_map != OBJECT_TO_IDX["agent"]) & (
                    obj_map != OBJECT_TO_IDX["wall"])
        other_locs = [tuple(loc) for loc in np.argwhere(other_mask)]
        dists = [manh(agent_pos, loc) for loc in other_locs]
        valid_dists = [d for d in dists if d is not None]
        dist_to_nearest_object = min(valid_dists) if valid_dists else None

        grid_size = obs['image'].shape[:2]
        num_visible_objects = int(other_mask.sum())

        # ADD: Object counts from updated version
        counts = {
            name: int((obj_map == idx).sum())
            for name, idx in OBJECT_TO_IDX.items()
            if name not in ("empty", "agent")
        }

        # ADD: Path analysis from updated version
        free_mask = (obj_map == OBJECT_TO_IDX["empty"])
        frees = 0
        if agent_pos:
            y, x = agent_pos
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1] and free_mask[ny, nx]:
                    frees += 1
        multiple_paths_open = frees >= 2

        # ADD: Enhanced object detection from updated version
        def object_in_front(agent_pos: tuple, obj_map: np.ndarray):
            try:
                # Compute the neighbor cell one step forward in global coords
                yn = agent_pos[1] + 1
                xn = agent_pos[0]
                # Check facing
                if 0 <= xn < obj_map.shape[0] and 0 <= yn < obj_map.shape[1]:
                    if obj_map[xn, yn] == OBJECT_TO_IDX["key"]:
                        return "key"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["door"]:
                        return "door"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["wall"]:
                        return "wall"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["goal"]:
                        return "goal"
                    else:
                        return "empty"
                return "out_of_bounds"
            except (IndexError, TypeError):
                return "error"

        front_object = object_in_front(agent_pos, obj_map)

        # Adjacency (keep original)
        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        is_adj_key = is_adjacent(agent_pos, key_pos)
        is_adj_door = is_adjacent(agent_pos, door_pos)

        # ENHANCED: Facing detection from updated version
        def is_facing_object_agent_frame(agent_pos: tuple, obj_map: np.ndarray, obj_idx: int) -> bool:
            try:
                yn = agent_pos[1] + 1
                xn = agent_pos[0]

                if 0 <= xn < obj_map.shape[0] and 0 <= yn < obj_map.shape[1]:
                    return obj_map[xn, yn] == obj_idx
                return False
            except (IndexError, TypeError):
                return False

        is_facing_key = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["key"])
        is_facing_door = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["door"])
        is_facing_wall = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["wall"])

        # Return ENHANCED feature dictionary (keeping all original + adding new ones)
        return {
            # Original mediator features (unchanged)
            "grid_size": grid_size,
            "agent_pos": agent_pos,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "goal_pos": goal_pos,
            "door_state": door_state,
            "dist_to_key": dist_to_key,
            "dist_to_door": dist_to_door,
            "dist_to_goal": dist_to_goal,
            "is_key_visible": key_pos is not None,
            "is_door_visible": door_pos is not None,
            "is_adjacent_to_key": is_adj_key,
            "is_adjacent_to_door": is_adj_door,
            "facing_key": is_facing_key,
            "facing_wall": is_facing_wall,
            "facing_door": is_facing_door,

            # ADDED: Enhanced features from updated version
            "has_key": has_key,
            "dist_to_nearest_object": dist_to_nearest_object,
            "num_visible_objects": num_visible_objects,
            "vertical_distance_to_goal": vert_dist_to_goal,
            "horizontal_distance_to_goal": horiz_dist_to_goal,
            "vertical_distance_to_key": vertical_distance_to_key,
            "horizontal_distance_to_key": horizontal_distance_to_key,
            "rel_dir_to_key": rel_dir_to_key,
            "rel_dir_to_door": rel_dir_to_door,
            "multiple_paths_open": multiple_paths_open,
            "front_object": front_object,
            **counts  # Add object counts
        }

    def agent_run(self,
                  sim_step: int,
                  obs: Dict,
                  rl_action: int,
                  infos: Dict,
                  reward: Optional[float] = None,
                  use_learned_asking: bool = True) -> Tuple[int, bool, Dict]:
        """Main decision loop: RL primary, mediator decides when to interrupt with LLM."""

        info = infos if isinstance(infos, dict) else infos[0]

        # Extract ENHANCED features
        raw_feats = self.extract_features(obs, info)

        # Mediator decides whether to interrupt RL (unchanged)
        should_interrupt, interrupt_confidence = self.mediator.should_ask_llm(
            obs=obs,
            ppo_action=rl_action,
            use_learned_policy=use_learned_asking
        )

        interaction_info = {
            'asked_llm': should_interrupt,
            'ask_probability': interrupt_confidence,
            'llm_plan_changed': False,
            'interaction_count': self.interaction_count,
            'override_count': self.override_count
        }

        if should_interrupt:
            # LLM interrupts and provides guidance
            llm_action, plan_changed = self._query_llm(raw_feats, rl_action, info)
            final_action = llm_action
            was_interrupted = True
            interaction_info['llm_plan_changed'] = plan_changed
            self.interaction_count += 1

            if plan_changed:
                self.override_count += 1

        else:
            # Use RL action directly
            final_action = rl_action
            was_interrupted = False

        # Train mediator (unchanged)
        if self.train_mediator and reward is not None:
            self.mediator.train_asking_policy(
                obs=obs,
                action=rl_action,
                reward=reward,
                next_obs=obs,
                asked_llm=should_interrupt,
                llm_plan_changed=interaction_info['llm_plan_changed']
            )

        # Update mediator state
        self.mediator.update_state(obs)

        # Logging
        if self.verbose:
            self._log_decision_correct(sim_step, rl_action, final_action, should_interrupt,
                                       interrupt_confidence, was_interrupted, interaction_info['llm_plan_changed'])

        return final_action, was_interrupted, interaction_info

    def _query_llm(self, features: Dict, ppo_action: int, info: Dict) -> Tuple[int, bool]:
        """ENHANCED query LLM with intelligent forbidden action handling using rich features."""

        # ENHANCED: Smart override using new spatial features
        if ppo_action in [4, 6]:
            # Use enhanced features for smarter decisions
            is_adjacent_to_key = features.get('is_adjacent_to_key', False)
            is_adjacent_to_door = features.get('is_adjacent_to_door', False)
            key_visible = features.get('is_key_visible', False)
            door_visible = features.get('is_door_visible', False)

            # NEW: Use relative directions for better navigation
            rel_dir_to_key = features.get('rel_dir_to_key')
            front_object = features.get('front_object')

            logger.warning(f"PPO suggested forbidden action {ppo_action}")

            # ENHANCED OVERRIDE LOGIC using new features:
            if is_adjacent_to_key:
                logger.info("→ Adjacent to key, overriding to PICKUP (3)")
                return 3, True
            elif is_adjacent_to_door:
                logger.info("→ Adjacent to door, overriding to TOGGLE (5)")
                return 5, True
            elif front_object == "key":
                logger.info("→ Key directly in front, moving FORWARD (2)")
                return 2, True
            elif front_object == "door":
                logger.info("→ Door directly in front, moving FORWARD (2)")
                return 2, True
            elif key_visible and rel_dir_to_key:
                # Use relative direction for smarter navigation
                if rel_dir_to_key == "left":
                    logger.info("→ Key visible left, turning LEFT (0)")
                    return 0, True
                elif rel_dir_to_key == "right":
                    logger.info("→ Key visible right, turning RIGHT (1)")
                    return 1, True
                else:
                    logger.info("→ Key visible, moving FORWARD (2)")
                    return 2, True
            elif features.get('multiple_paths_open', False):
                logger.info("→ Multiple paths available, turning LEFT (0) to explore")
                return 0, True
            else:
                logger.info("→ Default exploration, turning LEFT (0)")
                return 0, True

        try:
            # DEBUG: Show enhanced features (optional - can be removed for production)
            if self.verbose:
                logger.info(f"ENHANCED FEATURES:")
                logger.info(f"  Agent pos: {features.get('agent_pos')}")
                logger.info(f"  Key pos: {features.get('key_pos')}")
                logger.info(f"  Rel dir to key: {features.get('rel_dir_to_key')}")
                logger.info(f"  Front object: {features.get('front_object')}")
                logger.info(f"  Has key: {features.get('has_key')}")

            # Translate to natural language (unchanged)
            translated_feats = translate_features_for_llm(features)

            # Generate prompt (unchanged)
            prompt = render_prompt(
                env_name=info.get("env", "MiniGrid"),
                features=translated_feats,
                action=int(ppo_action)
            )

            # Get LLM response (unchanged)
            llm_response = self.llm.invoke(prompt)
            txt = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # Parse action (unchanged)
            m = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", txt, re.IGNORECASE)
            if not m:
                logger.error(f"No action found in: {txt}")
                return 0, True  # Default to turn left

            llm_action = int(m.group(1))
            logger.info(f"🤖 LLM CHOSE: {llm_action}")

            # Safety check - prevent LLM from choosing forbidden actions too
            if llm_action in [4, 6]:
                logger.warning(f"LLM chose forbidden {llm_action}, forcing TURN LEFT (0)")
                llm_action = 0

            plan_changed = (llm_action != ppo_action)
            return llm_action, plan_changed

        except Exception as e:
            logger.error(f"LLM error: {e}, using safe fallback")
            return 0, True  # Turn left as safe fallback

    def _log_decision_correct(self, sim_step: int, rl_action: int, final_action: int,
                              should_interrupt: bool, interrupt_confidence: float,
                              was_interrupted: bool, llm_changed_plan: bool):
        """Log decisions. (unchanged)"""

        if should_interrupt:
            if llm_changed_plan:
                logger.info(f"[Step {sim_step}] 🛑 LLM INTERRUPTED & OVERRIDE: "
                            f"RL {rl_action} → LLM {final_action} "
                            f"(interrupt_confidence={interrupt_confidence:.2f})")
            else:
                logger.info(f"[Step {sim_step}] 🛑 LLM INTERRUPTED BUT AGREED: "
                            f"RL {rl_action} confirmed by LLM "
                            f"(interrupt_confidence={interrupt_confidence:.2f})")
        else:
            logger.info(f"[Step {sim_step}] ✅ RL CONTINUES UNINTERRUPTED: "
                        f"Action {final_action} "
                        f"(interrupt_confidence={interrupt_confidence:.2f})")

    def get_mediator_stats(self) -> Dict:
        """Get mediator statistics. (unchanged)"""
        base_stats = self.mediator.get_statistics()
        base_stats.update({
            'total_interactions': self.interaction_count,
            'total_overrides': self.override_count,
            'override_rate': self.override_count / max(self.interaction_count, 1),
            'interaction_efficiency': self.override_count / max(self.interaction_count, 1)
        })
        return base_stats

    def save_mediator(self, path: str):
        """Save the trained mediator. (unchanged)"""
        self.mediator.save_asking_policy(path)

    def load_mediator(self, path: str):
        """Load a pre-trained mediator. (unchanged)"""
        self.mediator.load_asking_policy(path)