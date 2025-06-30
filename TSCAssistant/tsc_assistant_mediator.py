# TSCAssistant/tsc_assistant_with_mediator.py - CLEAN DEBUG VERSION

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
        """Extract features - NO WARNINGS, this is correct MiniGrid behavior."""

        obj_map = obs['image'][:, :, 0]
        state_map = obs['image'][:, :, 2]

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # In MiniGrid partial observation, agent is ALWAYS at center
        height, width = obj_map.shape
        agent_pos = (height // 2, width // 2)  # (3, 3) for 7x7 grid

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

        # Adjacency
        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        is_adj_key = is_adjacent(agent_pos, key_pos)
        is_adj_door = is_adjacent(agent_pos, door_pos)

        # Facing detection
        def is_facing_object(agent_pos: tuple, obj_map: np.ndarray, obj_idx: int) -> bool:
            y, x = agent_pos
            front_y, front_x = y - 1, x  # Check cell in front (up direction)

            if 0 <= front_y < obj_map.shape[0] and 0 <= front_x < obj_map.shape[1]:
                return obj_map[front_y, front_x] == obj_idx
            return False

        is_facing_key = is_facing_object(agent_pos, obj_map, OBJECT_TO_IDX["key"])
        is_facing_door = is_facing_object(agent_pos, obj_map, OBJECT_TO_IDX["door"])
        is_facing_wall = is_facing_object(agent_pos, obj_map, OBJECT_TO_IDX["wall"])

        return {
            "grid_size": obs['image'].shape[:2],
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
            "facing_direction_compass": "north",
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

        # Train mediator
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
        """Query LLM with intelligent forbidden action handling."""

        # SMART override for forbidden actions based on context
        if ppo_action in [4, 6]:
            # Analyze the situation to choose the RIGHT override action
            is_adjacent_to_key = features.get('is_adjacent_to_key', False)
            is_adjacent_to_door = features.get('is_adjacent_to_door', False)
            key_visible = features.get('is_key_visible', False)
            door_visible = features.get('is_door_visible', False)

            logger.warning(f"PPO suggested forbidden action {ppo_action}")

            # SMART OVERRIDE LOGIC:
            if is_adjacent_to_key:
                logger.info("→ Adjacent to key, overriding to PICKUP (3)")
                return 3, True
            elif is_adjacent_to_door:
                logger.info("→ Adjacent to door, overriding to TOGGLE (5)")
                return 5, True
            elif key_visible:
                # Key is visible but not adjacent - navigate toward it
                key_pos = features.get('key_pos')
                agent_pos = features.get('agent_pos')
                if key_pos and agent_pos:
                    # Simple navigation logic
                    dy = key_pos[0] - agent_pos[0]
                    dx = key_pos[1] - agent_pos[1]

                    if abs(dy) > abs(dx):
                        action = 2 if dy > 0 else 2  # Move forward (need to face right direction first)
                    else:
                        action = 2 if dx > 0 else 2  # Move forward

                    logger.info(f"→ Key visible at {key_pos}, navigating with action {action}")
                    return action, True
                else:
                    logger.info("→ Key visible but position unclear, turning LEFT (0)")
                    return 0, True
            elif door_visible:
                logger.info("→ Door visible, moving FORWARD (2)")
                return 2, True
            else:
                # Nothing specific visible, explore by turning
                logger.info("→ Nothing specific visible, TURNING LEFT (0) to explore")
                return 0, True

        try:
            # DEBUG: Show what features we extracted
            logger.info(f"FEATURES:")
            logger.info(f"  Raw agent_pos: {features.get('agent_pos')}")
            logger.info(f"  Raw key_pos: {features.get('key_pos')}")
            logger.info(f"  is_adjacent_to_key: {features.get('is_adjacent_to_key')}")
            logger.info(f"  is_key_visible: {features.get('is_key_visible')}")
            logger.info(f"  dist_to_key: {features.get('dist_to_key')}")
            logger.info(f"  facing_key: {features.get('facing_key')}")

            # Translate to natural language
            translated_feats = translate_features_for_llm(features)

            # Show translated features
            logger.info(f"TRANSLATED FOR LLM:")
            logger.info(f"  Agent pos: {translated_feats.get('agent_pos', 'Missing')}")
            logger.info(f"  Key pos: {translated_feats.get('key_pos', 'Missing')}")
            logger.info(f"  Adjacent to key: {translated_feats.get('is_adjacent_to_key', 'Missing')}")

            # Generate prompt
            prompt = render_prompt(
                env_name=info.get("env", "MiniGrid"),
                features=translated_feats,
                action=int(ppo_action)
            )

            # Show first part of prompt
            logger.info(f"📝 PROMPT START: {prompt[:200]}...")

            # Get LLM response
            llm_response = self.llm.invoke(prompt)
            txt = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # Parse action
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
        """Log decisions."""

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
        """Get mediator statistics."""
        base_stats = self.mediator.get_statistics()
        base_stats.update({
            'total_interactions': self.interaction_count,
            'total_overrides': self.override_count,
            'override_rate': self.override_count / max(self.interaction_count, 1),
            'interaction_efficiency': self.override_count / max(self.interaction_count, 1)
        })
        return base_stats

    def save_mediator(self, path: str):
        """Save the trained mediator."""
        self.mediator.save_asking_policy(path)

    def load_mediator(self, path: str):
        """Load a pre-trained mediator."""
        self.mediator.load_asking_policy(path)