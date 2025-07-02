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
    FIXED: Enhanced TSC Agent with better mediator integration and ADVANCED LOOP DETECTION.
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

        # Initialize mediator with FIXED parameters
        self.mediator = Mediator(
            obs_shape=obs_shape,
            device=device,
            verbose=verbose,
            learning_rate=1e-4  # FIXED: Lower learning rate
        )

        # Track LLM interactions
        self.current_plan = None
        self.interaction_count = 0
        self.override_count = 0

        # FIXED: Add performance tracking
        self.recent_overrides = deque(maxlen=20)
        self.recent_successes = deque(maxlen=20)

        # NEW: Enhanced loop detection at TSC level
        self.consecutive_same_llm_decision = 0
        self.last_llm_action = None
        self.last_rl_action = None
        self.same_situation_count = 0
        self.last_situation_hash = None

        # NEW: Emergency circuit breaker
        self.emergency_rl_mode = 0  # Counter for emergency RL-only mode
        self.llm_failure_count = 0  # Track LLM consecutive failures

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

    def _get_situation_hash(self, features: Dict, rl_action: int) -> str:
        """NEW: Create a hash of the current situation for loop detection"""
        # Create a simplified situation signature
        key_elements = [
            features.get('agent_pos'),
            features.get('key_pos'),
            features.get('door_pos'),
            features.get('has_key', False),
            rl_action
        ]
        return str(hash(tuple(str(x) for x in key_elements)))

    def agent_run(self,
                  sim_step: int,
                  obs: Dict,
                  rl_action: int,
                  infos: Dict,
                  reward: Optional[float] = None,
                  use_learned_asking: bool = True) -> Tuple[int, bool, Dict]:
        """FIXED: Better decision logic with ADVANCED LOOP DETECTION."""

        info = infos if isinstance(infos, dict) else infos[0]

        # Extract ENHANCED features
        raw_feats = self.extract_features(obs, info)

        # NEW: Situation-based loop detection
        current_situation = self._get_situation_hash(raw_feats, rl_action)
        if current_situation == self.last_situation_hash:
            self.same_situation_count += 1
        else:
            self.same_situation_count = 0
        self.last_situation_hash = current_situation

        # NEW: Emergency circuit breaker - if same situation too long, force RL mode
        if self.same_situation_count > 15:
            if self.verbose:
                logger.warning(
                    f"🚨 EMERGENCY: Same situation {self.same_situation_count} times - forcing RL mode for 20 steps")
            self.emergency_rl_mode = 20
            self.same_situation_count = 0

        # NEW: Check emergency RL mode
        if self.emergency_rl_mode > 0:
            self.emergency_rl_mode -= 1
            interaction_info = {
                'asked_llm': False,
                'ask_probability': 0.01,
                'llm_plan_changed': False,
                'interaction_count': self.interaction_count,
                'override_count': self.override_count,
                'emergency_mode': True
            }
            if self.verbose and self.emergency_rl_mode % 5 == 0:
                logger.info(f"🚨 EMERGENCY RL MODE: {self.emergency_rl_mode} steps remaining")
            return rl_action, False, interaction_info

        # FIXED: Add performance-based asking adjustment
        recent_performance = self._get_recent_performance()

        # Adjust asking probability based on recent performance
        if recent_performance < 0.3 and len(self.recent_successes) > 10:
            # If performing poorly, be more conservative with asking
            asking_threshold_adjustment = 0.2
        elif recent_performance > 0.8:
            # If performing well, can be more selective
            asking_threshold_adjustment = -0.1
        else:
            asking_threshold_adjustment = 0.0

        # Mediator decides whether to interrupt RL
        should_interrupt, interrupt_confidence = self.mediator.should_ask_llm(
            obs=obs,
            ppo_action=rl_action,
            use_learned_policy=use_learned_asking
        )

        # FIXED: Apply performance-based adjustment
        if asking_threshold_adjustment != 0.0:
            if should_interrupt and interrupt_confidence < (0.6 + asking_threshold_adjustment):
                should_interrupt = False
                if self.verbose:
                    logger.info(
                        f"Performance-based asking suppression: {interrupt_confidence:.3f} < {0.6 + asking_threshold_adjustment:.3f}")

        interaction_info = {
            'asked_llm': should_interrupt,
            'ask_probability': interrupt_confidence,
            'llm_plan_changed': False,
            'interaction_count': self.interaction_count,
            'override_count': self.override_count,
            'recent_performance': recent_performance,  # ADDED
            'asking_adjustment': asking_threshold_adjustment,  # ADDED
            'emergency_mode': False,
            'same_situation_count': self.same_situation_count  # NEW
        }

        if should_interrupt:
            # LLM interrupts and provides guidance
            llm_action, plan_changed = self._query_llm(raw_feats, rl_action, info)

            # NEW: Enhanced loop detection for LLM decisions
            if llm_action == self.last_llm_action and rl_action == self.last_rl_action:
                self.consecutive_same_llm_decision += 1
                if self.consecutive_same_llm_decision > 8:
                    if self.verbose:
                        logger.warning(
                            f"🔄 LLM LOOP DETECTED: Same decision {self.consecutive_same_llm_decision} times - forcing RL action")
                    llm_action = rl_action  # Force RL action to break loop
                    plan_changed = False
                    self.llm_failure_count += 1
            else:
                self.consecutive_same_llm_decision = 0
                self.llm_failure_count = 0

            self.last_llm_action = llm_action
            self.last_rl_action = rl_action

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
            # Reset consecutive counter when not asking
            self.consecutive_same_llm_decision = 0

        # Update mediator state
        self.mediator.update_state(obs)

        # FIXED: Better logging with context
        if self.verbose:
            self._log_decision_enhanced(sim_step, rl_action, final_action, should_interrupt,
                                        interrupt_confidence, was_interrupted,
                                        interaction_info['llm_plan_changed'], raw_feats)

        return final_action, was_interrupted, interaction_info

    def _query_llm(self, features: Dict, ppo_action: int, info: Dict) -> Tuple[int, bool]:
        """FIXED: Enhanced LLM querying with better context and LOOP DETECTION."""

        # FIXED: Smarter forbidden action handling
        if ppo_action in [4, 6]:
            return self._handle_forbidden_action(features, ppo_action)

        try:
            # FIXED: Enhanced feature context for LLM
            enhanced_context = self._create_enhanced_context(features, ppo_action)

            # NEW: Add loop detection context
            if self.consecutive_same_llm_decision > 3:
                enhanced_context[
                    "loop_warning"] = f"This override has been repeated {self.consecutive_same_llm_decision} times. Consider if RL action might be correct."

            if self.same_situation_count > 5:
                enhanced_context[
                    "situation_warning"] = f"Agent has been in similar situation for {self.same_situation_count} steps. Consider different approach."

            # Translate to natural language
            translated_feats = translate_features_for_llm(features)
            translated_feats.update(enhanced_context)  # Add enhanced context

            # Generate prompt
            prompt = render_prompt(
                env_name=info.get("env", "MiniGrid"),
                features=translated_feats,
                action=int(ppo_action)
            )

            # Get LLM response with timeout protection
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

            # FIXED: Validate LLM action
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

    def _handle_forbidden_action(self, features: Dict, ppo_action: int) -> Tuple[int, bool]:
        """FIXED: Smarter forbidden action handling with context awareness."""

        logger.warning(f"PPO suggested forbidden action {ppo_action}")

        # Get current state
        has_key = features.get('has_key', False)
        is_adjacent_to_key = features.get('is_adjacent_to_key', False)
        is_adjacent_to_door = features.get('is_adjacent_to_door', False)
        facing_key = features.get('facing_key', False)
        facing_door = features.get('facing_door', False)
        front_object = features.get('front_object', 'empty')
        rel_dir_to_key = features.get('rel_dir_to_key')

        # FIXED: Context-aware override decisions
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
            # Smart navigation based on relative direction
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

    def _get_recent_performance(self) -> float:
        """Get recent performance metric."""
        if len(self.recent_successes) == 0:
            return 0.5  # Neutral
        return np.mean(self.recent_successes)

    def _log_decision_enhanced(self, sim_step: int, rl_action: int, final_action: int,
                               should_interrupt: bool, interrupt_confidence: float,
                               was_interrupted: bool, llm_changed_plan: bool, features: Dict):
        """FIXED: Enhanced logging with more context."""

        # Get current state for context
        has_key = features.get('has_key', False)
        key_pos = features.get('key_pos')
        door_pos = features.get('door_pos')
        agent_pos = features.get('agent_pos')

        state_str = f"Agent@{agent_pos}, Key@{key_pos}, Door@{door_pos}, HasKey={has_key}"

        if should_interrupt:
            if llm_changed_plan:
                logger.info(f"[Step {sim_step}] 🛑 LLM OVERRIDE: "
                            f"RL {rl_action} → LLM {final_action} "
                            f"(conf={interrupt_confidence:.2f}) | {state_str}")
            else:
                logger.info(f"[Step {sim_step}] 🛑 LLM AGREED: "
                            f"RL {rl_action} confirmed "
                            f"(conf={interrupt_confidence:.2f}) | {state_str}")
        else:
            logger.info(f"[Step {sim_step}] ✅ RL CONTINUES: "
                        f"Action {final_action} "
                        f"(conf={interrupt_confidence:.2f}) | {state_str}")

    def get_mediator_stats(self) -> Dict:
        """FIXED: Enhanced mediator statistics."""
        base_stats = self.mediator.get_statistics()

        # Add performance metrics
        recent_override_rate = np.mean(self.recent_overrides) if self.recent_overrides else 0
        recent_performance = self._get_recent_performance()

        base_stats.update({
            'total_interactions': self.interaction_count,
            'total_overrides': self.override_count,
            'override_rate': self.override_count / max(self.interaction_count, 1),
            'recent_override_rate': recent_override_rate,
            'recent_performance': recent_performance,
            'interaction_efficiency': self.override_count / max(self.interaction_count, 1),
            'performance_trend': 'improving' if recent_performance > 0.6 else 'declining' if recent_performance < 0.4 else 'stable',
            # NEW: Loop detection stats
            'emergency_rl_mode': self.emergency_rl_mode,
            'consecutive_same_llm_decision': self.consecutive_same_llm_decision,
            'same_situation_count': self.same_situation_count,
            'llm_failure_count': self.llm_failure_count
        })
        return base_stats

    def update_performance(self, success: bool):
        """Update recent performance tracking."""
        self.recent_successes.append(1 if success else 0)

    def save_mediator(self, path: str):
        """Save the trained mediator."""
        self.mediator.save_asking_policy(path)

    def load_mediator(self, path: str):
        """Load a pre-trained mediator."""
        self.mediator.load_asking_policy(path)

    def _create_enhanced_context(self, features: Dict, ppo_action: int) -> Dict:
        """FIXED: Create enhanced context for better LLM decisions."""

        context = {}

        # Add decision confidence
        context["ppo_confidence"] = "high" if ppo_action in [0, 1, 2] else "low"

        # Add state assessment
        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        door_visible = features.get('is_door_visible', False)

        if has_key and door_visible:
            context["current_objective"] = "Find and unlock the door"
        elif key_visible and not has_key:
            context["current_objective"] = "Navigate to and pick up the key"
        elif not key_visible and not has_key:
            context["current_objective"] = "Explore to find the key"
        else:
            context["current_objective"] = "Continue current plan"

        # Add recent performance context
        recent_perf = self._get_recent_performance()
        if recent_perf < 0.3:
            context["performance_note"] = "Agent has been struggling recently, prefer safer actions"
        elif recent_perf > 0.8:
            context["performance_note"] = "Agent performing well, current strategy is working"
        else:
            context["performance_note"] = "Agent is learning, make balanced decisions"

        # NEW: Add loop detection context
        if self.llm_failure_count > 3:
            context[
                "failure_warning"] = f"LLM has failed {self.llm_failure_count} times recently. Consider simpler actions."

        return context

    def _is_valid_action(self, action: int, features: Dict) -> bool:
        """FIXED: Validate if LLM action makes sense in current context."""

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
        """FIXED: Smart fallback action selection with loop awareness."""

        # NEW: If we've been in fallback too many times, try original action
        if self.llm_failure_count > 5 and self._is_valid_action(original_action, features):
            logger.info("→ Multiple LLM failures, trying original RL action")
            return original_action, False

        # If original action was valid, use it
        if self._is_valid_action(original_action, features):
            return original_action, False

        # Otherwise, choose safe action based on state
        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        rel_dir_to_key = features.get('rel_dir_to_key')

        if not has_key and key_visible and rel_dir_to_key:
            if rel_dir_to_key == "left":
                return 0, True  # Turn left
            elif rel_dir_to_key == "right":
                return 1, True  # Turn right