import re
import numpy as np
from loguru import logger
from typing import Dict, Tuple, Optional
from collections import deque

from TSCAssistant.tsc_agent_prompt import render_prompt
from TSCAssistant.feature_translator import translate_features_for_llm
from TSCAssistant.mediator import Mediator
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class TSCAgentWithMediator:
    """
    Simple TSC Agent with basic mediator integration.
    Karmaşık learning phases kaldırıldı - sadece basit exploration-exploitation.
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

        # Initialize simple mediator
        self.mediator = Mediator(
            obs_shape=obs_shape,
            device=device,
            verbose=verbose,
            learning_rate=1e-4
        )

        # Track LLM interactions
        self.current_plan = None
        self.interaction_count = 0
        self.override_count = 0

        # Performance tracking
        self.recent_overrides = deque(maxlen=50)
        self.recent_successes = deque(maxlen=50)

        # Loop detection (Basit)
        self.consecutive_same_llm_decision = 0
        self.last_llm_action = None
        self.llm_failure_count = 0

    def extract_features(self, obs: np.ndarray, info: dict) -> dict:
        """
        Feature extraction - AYNI (mevcut sistemi bozmuyoruz)
        """
        # Get environment for carrying state
        env = info.get("llm_env", None)

        # Use coordinate system from updated version
        obj_map = np.fliplr(obs['image'][:, :, 0])
        state_map = np.fliplr(obs['image'][:, :, 2])

        # Agent carrying state
        has_key = bool(getattr(env.unwrapped, "carrying", None)) if env else False

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # Agent position
        agent_pos = (3, 0)

        # Find objects
        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        # Door state
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

        # Enhanced spatial features
        def get_vertical_distance(p, q):
            return abs(p[0] - q[0]) if (p and q) else None

        def get_horizontal_distance(p, q):
            return abs(p[1] - q[1]) if (p and q) else None

        vert_dist_to_goal = get_vertical_distance(agent_pos, goal_pos)
        horiz_dist_to_goal = get_horizontal_distance(agent_pos, goal_pos)
        vertical_distance_to_key = get_vertical_distance(agent_pos, key_pos)
        horizontal_distance_to_key = get_horizontal_distance(agent_pos, key_pos)

        # Relative directions
        def rel_dir_agent_frame(agent_pos, q_pos):
            if not (agent_pos and q_pos):
                return None
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

        # Environmental analysis
        other_mask = (obj_map > OBJECT_TO_IDX["empty"]) & (obj_map != OBJECT_TO_IDX["agent"]) & (
                obj_map != OBJECT_TO_IDX["wall"])
        other_locs = [tuple(loc) for loc in np.argwhere(other_mask)]
        dists = [manh(agent_pos, loc) for loc in other_locs]
        valid_dists = [d for d in dists if d is not None]
        dist_to_nearest_object = min(valid_dists) if valid_dists else None

        grid_size = obs['image'].shape[:2]
        num_visible_objects = int(other_mask.sum())

        # Object counts
        counts = {
            name: int((obj_map == idx).sum())
            for name, idx in OBJECT_TO_IDX.items()
            if name not in ("empty", "agent")
        }

        # Path analysis
        free_mask = (obj_map == OBJECT_TO_IDX["empty"])
        frees = 0
        if agent_pos:
            y, x = agent_pos
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1] and free_mask[ny, nx]:
                    frees += 1
        multiple_paths_open = frees >= 2

        # Enhanced object detection
        def object_in_front(agent_pos: tuple, obj_map: np.ndarray):
            try:
                yn = agent_pos[1] + 1
                xn = agent_pos[0]
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

        # Adjacency
        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        is_adj_key = is_adjacent(agent_pos, key_pos)
        is_adj_door = is_adjacent(agent_pos, door_pos)

        # Facing detection
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

        # Return enhanced feature dictionary
        return {
            # Original features
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

            # Enhanced features
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
            **counts
        }

    def agent_run(self,
                  sim_step: int,
                  obs: Dict,
                  rl_action: int,
                  infos: Dict,
                  reward: Optional[float] = None,
                  use_learned_asking: bool = True) -> Tuple[int, bool, Dict]:
        """Basit decision logic - karmaşık learning phases yok."""

        info = infos if isinstance(infos, dict) else infos[0]

        # Extract features
        raw_feats = self.extract_features(obs, info)

        # Mediator decides whether to interrupt RL
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
            llm_action, plan_changed = self._query_llm_simple(raw_feats, rl_action, info)

            # Simple loop detection
            if llm_action == self.last_llm_action:
                self.consecutive_same_llm_decision += 1
                if self.consecutive_same_llm_decision > 5:
                    if self.verbose:
                        logger.warning(f"🔄 LLM loop detected - using RL action")
                    llm_action = rl_action
                    plan_changed = False
                    self.llm_failure_count += 1
            else:
                self.consecutive_same_llm_decision = 0

            self.last_llm_action = llm_action

            final_action = llm_action
            was_interrupted = True
            interaction_info['llm_plan_changed'] = plan_changed
            self.interaction_count += 1

            if plan_changed:
                self.override_count += 1
                self.recent_overrides.append(1)
            else:
                self.recent_overrides.append(0)

        else:
            # Use RL action directly
            final_action = rl_action
            was_interrupted = False
            self.consecutive_same_llm_decision = 0

        # Update mediator state
        self.mediator.update_state(obs)

        # Simple logging
        if self.verbose:
            phase = "EXPLORATION" if self.mediator.current_episode < self.mediator.exploration_episodes else "EXPLOITATION"
            if was_interrupted:
                if plan_changed:
                    logger.info(f"[Step {sim_step}] [{phase}] 🛑 LLM OVERRIDE: "
                                f"RL {rl_action} → LLM {final_action}")
                else:
                    logger.info(f"[Step {sim_step}] [{phase}] 🛑 LLM AGREED: "
                                f"RL {rl_action} confirmed")
            else:
                logger.info(f"[Step {sim_step}] [{phase}] ✅ RL CONTINUES: "
                            f"Action {final_action}")

        return final_action, was_interrupted, interaction_info

    def _query_llm_simple(self, features: Dict, ppo_action: int, info: Dict) -> Tuple[int, bool]:
        """Basit LLM querying - karmaşık context yok."""

        # Handle forbidden actions
        if ppo_action in [4, 6]:
            return self._handle_forbidden_action(features, ppo_action)

        try:
            # Create simple context
            context = self._create_simple_context(features, ppo_action)

            # Translate to natural language
            translated_feats = translate_features_for_llm(features)
            translated_feats.update(context)

            # Generate prompt
            prompt = render_prompt(
                env_name=info.get("env", "MiniGrid"),
                features=translated_feats,
                action=int(ppo_action)
            )

            # Get LLM response
            try:
                llm_response = self.llm.invoke(prompt)
                txt = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            except Exception as llm_error:
                logger.error(f"LLM invocation failed: {llm_error}")
                return self._fallback_action(features, ppo_action)

            # Parse action
            m = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", txt, re.IGNORECASE)
            if not m:
                logger.error(f"No action found in LLM response: {txt[:200]}...")
                return self._fallback_action(features, ppo_action)

            llm_action = int(m.group(1))

            # Validate LLM action
            if not self._is_valid_action(llm_action, features):
                logger.warning(f"LLM chose invalid action {llm_action}, using fallback")
                return self._fallback_action(features, ppo_action)

            logger.info(f"🤖 LLM CHOSE: {llm_action} (was {ppo_action})")

            plan_changed = (llm_action != ppo_action)
            return llm_action, plan_changed

        except Exception as e:
            logger.error(f"LLM error: {e}, using fallback")
            self.llm_failure_count += 1
            return self._fallback_action(features, ppo_action)

    def _create_simple_context(self, features: Dict, ppo_action: int) -> Dict:
        """Basit context creation - karmaşık learning awareness yok."""

        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        door_visible = features.get('is_door_visible', False)

        if has_key and door_visible:
            current_objective = "Find and unlock the door"
        elif key_visible and not has_key:
            current_objective = "Navigate to and pick up the key"
        elif not key_visible and not has_key:
            current_objective = "Explore to find the key"
        else:
            current_objective = "Continue current plan"

        return {
            "current_objective": current_objective,
            "ppo_confidence": "high" if ppo_action in [0, 1, 2] else "low",
            "performance_note": "Make efficient decisions."
        }

    def _handle_forbidden_action(self, features: Dict, ppo_action: int) -> Tuple[int, bool]:
        """Forbidden action handling - AYNI"""

        logger.warning(f"PPO suggested forbidden action {ppo_action}")

        # Get current state
        has_key = features.get('has_key', False)
        is_adjacent_to_key = features.get('is_adjacent_to_key', False)
        is_adjacent_to_door = features.get('is_adjacent_to_door', False)
        facing_key = features.get('facing_key', False)
        facing_door = features.get('facing_door', False)
        front_object = features.get('front_object', 'empty')
        rel_dir_to_key = features.get('rel_dir_to_key')

        # Context-aware override decisions
        if is_adjacent_to_key and facing_key and not has_key:
            logger.info("→ Perfect key pickup situation, using PICKUP (3)")
            return 3, True
        elif is_adjacent_to_door and facing_door and has_key:
            logger.info("→ Perfect door toggle situation, using TOGGLE (5)")
            return 5, True
        elif front_object == "key" and not has_key:
            logger.info("→ Key directly in front, moving FORWARD (2)")
            return 2, True
        elif front_object == "door" and has_key:
            logger.info("→ Door directly in front with key, moving FORWARD (2)")
            return 2, True
        elif rel_dir_to_key and not has_key:
            if rel_dir_to_key == "left":
                logger.info("→ Key to the left, turning LEFT (0)")
                return 0, True
            elif rel_dir_to_key == "right":
                logger.info("→ Key to the right, turning RIGHT (1)")
                return 1, True
            else:
                logger.info("→ Key ahead, moving FORWARD (2)")
                return 2, True
        else:
            logger.info("→ Default exploration, turning LEFT (0)")
            return 0, True

    def _is_valid_action(self, action: int, features: Dict) -> bool:
        """Validate if LLM action makes sense - AYNI"""

        # Always forbid actions 4 and 6
        if action in [4, 6]:
            return False

        # Validate pickup action
        if action == 3:
            has_key = features.get('has_key', False)
            is_adjacent_to_key = features.get('is_adjacent_to_key', False)
            if has_key or not is_adjacent_to_key:
                return False

        # Validate toggle action
        if action == 5:
            has_key = features.get('has_key', False)
            is_adjacent_to_door = features.get('is_adjacent_to_door', False)
            if not has_key or not is_adjacent_to_door:
                return False

        return True

    def _fallback_action(self, features: Dict, original_action: int) -> Tuple[int, bool]:
        """Basit fallback action selection"""

        # If original action was valid, use it
        if self._is_valid_action(original_action, features):
            return original_action, False

        # Choose safe action based on state
        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        rel_dir_to_key = features.get('rel_dir_to_key')

        if not has_key and key_visible and rel_dir_to_key:
            if rel_dir_to_key == "left":
                return 0, True  # Turn left
            elif rel_dir_to_key == "right":
                return 1, True  # Turn right
            else:
                return 2, True  # Move forward
        else:
            return 0, True  # Safe default: turn left

    def get_mediator_stats(self) -> Dict:
        """Basit mediator statistics"""
        base_stats = self.mediator.get_statistics()

        # Add performance metrics
        recent_override_rate = np.mean(self.recent_overrides) if self.recent_overrides else 0

        base_stats.update({
            'total_interactions': self.interaction_count,
            'total_overrides': self.override_count,
            'override_rate': self.override_count / max(self.interaction_count, 1),
            'recent_override_rate': recent_override_rate,
            'interaction_efficiency': self.override_count / max(self.interaction_count, 1),
            'llm_failure_count': self.llm_failure_count,
            'consecutive_same_llm_decision': self.consecutive_same_llm_decision
        })
        return base_stats

    def update_performance(self, success: bool):
        """Update performance tracking"""
        self.recent_successes.append(1 if success else 0)

    def save_mediator(self, path: str):
        """Save the trained mediator"""
        self.mediator.save_asking_policy(path)

    def load_mediator(self, path: str):
        """Load a pre-trained mediator"""
        self.mediator.load_asking_policy(path)