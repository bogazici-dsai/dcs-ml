import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class AskingPolicy(nn.Module):
    """FIXED: Enhanced asking policy with better initialization and regularization"""

    def __init__(self, obs_shape: tuple, hidden_dim: int = 64):
        super().__init__()

        obs_size = np.prod(obs_shape)
        input_dim = obs_size * 3  # current, previous, difference

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Ask or Not Ask
        )

        # FIXED: Better initialization - start more balanced
        with torch.no_grad():
            # Start with slight bias towards not asking, but not extreme
            self.network[-1].bias[0] = -0.1  # Ask bias (slightly negative)
            self.network[-1].bias[1] = 0.1  # Not ask bias (slightly positive)

            # Xavier initialization for better gradient flow
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, current_obs, previous_obs):
        current_flat = current_obs.flatten()
        previous_flat = previous_obs.flatten()
        diff_flat = current_flat - previous_flat
        combined_input = torch.cat([current_flat, previous_flat, diff_flat])
        return self.network(combined_input)


class Mediator:
    """
    COMPLETE FIXED: Enhanced mediator with LOOP DETECTION and aggressive penalty for unnecessary interrupts.
    """

    def __init__(self,
                 obs_shape: tuple,
                 learning_rate: float = 1e-4,
                 device: str = "cpu",
                 verbose: bool = True):
        self.obs_shape = obs_shape
        self.device = device
        self.verbose = verbose

        # Initialize asking policy network
        self.asking_policy = AskingPolicy(obs_shape).to(device)
        self.optimizer = torch.optim.Adam(self.asking_policy.parameters(), lr=learning_rate)

        # FIXED: ALL state tracking attributes
        self.previous_obs = None
        self.last_llm_plan = None
        self.steps_since_last_ask = 0

        # Loop detection
        self.recent_actions = deque(maxlen=10)
        self.recent_positions = deque(maxlen=5)

        # NEW: Enhanced loop detection for LLM overrides
        self.recent_llm_overrides = deque(maxlen=20)  # Track LLM override patterns
        self.recent_situations = deque(maxlen=10)  # Track situation states
        self.override_count_same_situation = 0  # Count overrides in same situation
        self.forced_rl_mode_steps = 0  # Force RL mode counter
        self.last_agent_pos = None  # Track position changes
        self.position_stuck_count = 0  # Count steps without position change

        # FIXED: ALL training progress tracking attributes (ESSENTIAL!)
        self.ask_history = []  # ESSENTIAL - tracks ask decisions
        self.reward_history = []  # ESSENTIAL - tracks rewards
        self.loss_history = []  # ESSENTIAL - tracks NN loss
        self.training_phase = "exploration"

        # FIXED: Aggressive penalty parameters for efficiency
        self.lambda_penalty = 0.02  # Start higher for unnecessary interrupts
        self.max_lambda = 0.2  # Higher max penalty
        self.agreement_penalty = 0.1  # NEW: Specific penalty for asking but agreeing
        self.gamma = 0.99  # Discount factor
        self.recent_loss = 0.0  # For WandB tracking

        # FIXED: Reward smoothing attributes
        self.reward_buffer = deque(maxlen=100)
        self.baseline_reward = 0.0

        # FIXED: Interrupt tracking for efficiency
        self.recent_interrupts = deque(maxlen=50)
        self.recent_agreements = deque(maxlen=50)
        self.interrupt_efficiency_threshold = 0.3  # Minimum efficiency required

    def should_ask_llm(self,
                       obs: Dict,
                       ppo_action: int,
                       use_learned_policy: bool = True,
                       force_exploration: bool = False) -> Tuple[bool, float]:
        """
        FIXED: More conservative asking with ENHANCED LOOP DETECTION
        """

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # NEW: Check if we're in forced RL mode (loop detected)
        if self.forced_rl_mode_steps > 0:
            self.forced_rl_mode_steps -= 1
            if self.verbose:
                logger.warning(f"🚫 FORCED RL MODE: {self.forced_rl_mode_steps} steps remaining")
            return False, 0.05

        # NEW: Enhanced loop detection
        if self._detect_and_handle_loops(obs, action):
            return False, 0.05

        # PHASE 1: Even more conservative exploration
        if len(self.ask_history) < 300:
            return self._ultra_conservative_exploration(obs, action)

        # PHASE 2: Efficiency-aware learned policy
        return self._efficiency_aware_asking(obs, action, use_learned_policy)

    def _detect_and_handle_loops(self, obs: Dict, ppo_action: int) -> bool:
        """
        NEW: Advanced loop detection to prevent infinite override cycles
        """
        # Track current situation
        current_situation = self._get_situation_signature(obs, ppo_action)
        self.recent_situations.append(current_situation)

        # Track agent position
        agent_pos = self._extract_features(obs).get('agent_pos')
        if agent_pos == self.last_agent_pos:
            self.position_stuck_count += 1
        else:
            self.position_stuck_count = 0
        self.last_agent_pos = agent_pos

        # DETECTION 1: Same situation repeated too many times
        if len(self.recent_situations) >= 5:
            recent_situ = list(self.recent_situations)[-5:]
            if len(set(recent_situ)) == 1:  # Exact same situation 5 times
                if self.verbose:
                    logger.warning("🔄 LOOP DETECTED: Same situation 5 times - entering forced RL mode")
                self.forced_rl_mode_steps = 10
                return True

        # DETECTION 2: LLM override oscillation
        if len(self.recent_llm_overrides) >= 6:
            recent_overrides = list(self.recent_llm_overrides)[-6:]
            unique_overrides = len(set(recent_overrides))
            if unique_overrides <= 2:  # Oscillating between 2 actions
                if self.verbose:
                    logger.warning("🔄 LOOP DETECTED: LLM override oscillation - entering forced RL mode")
                self.forced_rl_mode_steps = 8
                return True

        # DETECTION 3: Position stuck with frequent asking
        if self.position_stuck_count >= 10 and len(self.ask_history) >= 10:
            recent_ask_rate = np.mean(self.ask_history[-10:])
            if recent_ask_rate > 0.7:  # High asking rate while stuck
                if self.verbose:
                    logger.warning("🔄 LOOP DETECTED: Stuck position with high asking rate - entering forced RL mode")
                self.forced_rl_mode_steps = 15
                return True

        # DETECTION 4: Repetitive override pattern
        if len(self.recent_llm_overrides) >= 8:
            # Check for ABAB pattern or AAA pattern
            pattern = list(self.recent_llm_overrides)[-8:]
            if (pattern[0] == pattern[2] == pattern[4] == pattern[6] and
                    pattern[1] == pattern[3] == pattern[5] == pattern[7]):
                if self.verbose:
                    logger.warning("🔄 LOOP DETECTED: ABAB override pattern - entering forced RL mode")
                self.forced_rl_mode_steps = 12
                return True

        return False

    def _get_situation_signature(self, obs: Dict, ppo_action: int) -> str:
        """
        NEW: Create a signature for the current situation to detect loops
        """
        features = self._extract_features(obs)

        # Create a compact situation signature
        signature = f"pos:{features.get('agent_pos')}_key:{features.get('key_pos')}_door:{features.get('door_pos')}_haskey:{features.get('has_key', False)}_action:{ppo_action}"

        return signature

    def _ultra_conservative_exploration(self, obs: Dict, action: int) -> Tuple[bool, float]:
        """FIXED: Ultra conservative exploration to prevent wasteful interrupts"""

        # ALWAYS ask if action is clearly problematic
        if self._is_problematic_action(obs, action):
            return True, 0.95

        # CRITICAL situations only
        if self._is_critical_situation(obs, action):
            return True, 0.9

        # VERY rare random exploration (reduced to 3% due to loop issues)
        if self.steps_since_last_ask >= 20 and np.random.random() < 0.03:
            return True, 0.5

        return False, 0.2

    def _efficiency_aware_asking(self, obs: Dict, ppo_action: int, use_learned_policy: bool) -> Tuple[bool, float]:
        """FIXED: Efficiency-aware asking that considers recent performance"""

        # Calculate recent efficiency
        if len(self.recent_interrupts) > 10:
            # FIXED: Convert deque to list for slicing
            recent_interrupt_rate = np.mean(list(self.recent_interrupts)[-20:])
            recent_agreement_rate = np.mean(list(self.recent_agreements)[-20:])
            efficiency = 1 - (recent_agreement_rate / max(recent_interrupt_rate, 0.01))
        else:
            efficiency = 0.5  # Neutral

        # SAFETY: Always override learned policy for critical situations
        if self._is_critical_situation(obs, ppo_action):
            return True, 0.9

        if not use_learned_policy:
            return self._heuristic_asking_decision(obs, ppo_action)

        # Use neural network with efficiency adjustment
        if self.previous_obs is None:
            return True, 1.0

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        with torch.no_grad():
            logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
            probabilities = torch.softmax(logits, dim=0)
            ask_prob = probabilities[0].item()

            # FIXED: Efficiency-adjusted threshold with loop consideration
            base_threshold = 0.6
            if efficiency < 0.2:
                # Very low efficiency: be much more selective
                threshold = 0.85
            elif efficiency < 0.4:
                # Low efficiency: be more selective
                threshold = 0.75
            else:
                # Good efficiency: normal threshold
                threshold = base_threshold

            # NEW: Additional threshold adjustment for loop prevention
            if self.position_stuck_count > 5:
                threshold += 0.1  # Make asking harder when stuck

            if len(self.recent_llm_overrides) > 5:
                recent_override_rate = len([x for x in self.recent_llm_overrides if x == ppo_action]) / len(
                    self.recent_llm_overrides)
                if recent_override_rate > 0.6:  # Same action overridden frequently
                    threshold += 0.2  # Much harder to ask

            should_ask = ask_prob > threshold

        self.steps_since_last_ask += 1

        if should_ask:
            self.steps_since_last_ask = 0

        return should_ask, ask_prob

    def _is_problematic_action(self, obs: Dict, ppo_action: int) -> bool:
        """Detect problematic actions that need LLM intervention"""

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Invalid actions
        if action in [4, 6]:  # Drop, Done (forbidden)
            return True

        # NEW: Don't ask if we're in a detected loop situation
        if self.forced_rl_mode_steps > 0:
            return False

        # Action loops
        self.recent_actions.append(action)
        if len(self.recent_actions) >= 5:
            # Same action repeated 5+ times
            if len(set(list(self.recent_actions)[-5:])) == 1:
                return True

            # Oscillating between 2 actions
            last_4 = list(self.recent_actions)[-4:]
            if len(set(last_4)) == 2 and last_4[0] == last_4[2] and last_4[1] == last_4[3]:
                return True

        return False

    def _is_critical_situation(self, obs: Dict, ppo_action: int) -> bool:
        """Detect critical situations that always need LLM"""

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # NEW: Don't override if we're in forced RL mode
        if self.forced_rl_mode_steps > 0:
            return False

        # Check for problematic actions first
        if self._is_problematic_action(obs, action):
            return True

        # Extract features for situation analysis
        features = self._extract_features(obs)

        # Critical situations (but respect forced RL mode)
        if features.get('is_adjacent_to_key') and action != 3:
            return True

        if features.get('facing_wall') and action == 2:
            return True

        if features.get('is_adjacent_to_door') and action != 5:
            return True

        # Check if trying to go through closed door
        if features.get('facing_door') and not features.get('door_is_open', False) and action == 2:
            return True

        # Been too long without asking (but not if stuck)
        if self.steps_since_last_ask > 15 and self.position_stuck_count < 5:
            return True

        return False

    def _heuristic_asking_decision(self, obs: Dict, ppo_action: int) -> Tuple[bool, float]:
        """Enhanced heuristic asking policy"""

        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        if self._is_critical_situation(obs, action):
            return True, 0.9

        if self._significant_obs_change(obs):
            return True, 0.7

        # Reduced periodic asking
        if self.steps_since_last_ask >= 18:
            return True, 0.6

        return False, 0.3

    def _significant_obs_change(self, obs: Dict) -> bool:
        """Enhanced observation change detection"""
        if self.previous_obs is None:
            return True

        current_features = self._extract_features(obs)
        previous_features = self._extract_features(self.previous_obs)

        # Important feature changes
        important_changes = [
            current_features.get('is_key_visible') != previous_features.get('is_key_visible'),
            current_features.get('is_door_visible') != previous_features.get('is_door_visible'),
            current_features.get('is_adjacent_to_key') != previous_features.get('is_adjacent_to_key'),
            current_features.get('is_adjacent_to_door') != previous_features.get('is_adjacent_to_door'),
            abs(current_features.get('dist_to_key', 999) - previous_features.get('dist_to_key', 999)) > 2,
        ]

        return any(important_changes)

    def train_asking_policy(self,
                            obs: Dict,
                            action: int,
                            reward: float,
                            next_obs: Dict,
                            asked_llm: bool,
                            llm_plan_changed: bool):
        """
        FIXED: Aggressive penalty for asking but agreeing (unnecessary interrupts) with loop detection
        """

        if self.previous_obs is None:
            self.previous_obs = obs
            return

        # NEW: Track LLM overrides for loop detection
        if asked_llm and llm_plan_changed:
            self.recent_llm_overrides.append(action)

        # FIXED: Compute reward with aggressive agreement penalty
        mediator_reward = self._compute_llm4rl_reward(
            task_reward=reward,
            asked_llm=asked_llm,
            llm_plan_changed=llm_plan_changed
        )

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        # Get asking policy prediction
        logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
        probabilities = torch.softmax(logits, dim=0)
        ask_prob = probabilities[0]
        not_ask_prob = probabilities[1]

        # Policy gradient loss (REINFORCE)
        if asked_llm:
            policy_loss = -torch.log(ask_prob + 1e-8) * mediator_reward
        else:
            policy_loss = -torch.log(not_ask_prob + 1e-8) * mediator_reward

        # Add entropy bonus and L2 regularization
        entropy = -(ask_prob * torch.log(ask_prob + 1e-8) +
                    not_ask_prob * torch.log(not_ask_prob + 1e-8))
        entropy_bonus = 0.02 * entropy

        # L2 regularization
        l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.asking_policy.parameters())

        total_loss = policy_loss - entropy_bonus + l2_reg

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(self.asking_policy.parameters(), 0.2)
        self.optimizer.step()

        # ESSENTIAL: Track statistics
        self.ask_history.append(asked_llm)
        self.reward_history.append(mediator_reward)
        self.loss_history.append(total_loss.item())
        self.recent_loss = total_loss.item()

        # Update penalties
        self._update_lambda_penalty()

        # Phase transition
        if len(self.ask_history) >= 300 and self.training_phase == "exploration":
            recent_performance = np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(
                self.reward_history)
            if recent_performance > -0.1:
                self.training_phase = "exploitation"
                if self.verbose:
                    logger.info("Mediator transitioning to exploitation phase")

        # Progress logging
        if self.verbose and len(self.ask_history) % 30 == 0:
            recent_ask_rate = np.mean(self.ask_history[-30:])
            recent_reward = np.mean(self.reward_history[-30:])
            recent_loss = np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else 0

            # NEW: Loop detection status
            loop_status = "FORCE_RL" if self.forced_rl_mode_steps > 0 else "NORMAL"

            logger.info(f"Mediator Stats: Ask Rate={recent_ask_rate:.3f}, "
                        f"Avg Reward={recent_reward:.3f}, Loss={recent_loss:.3f}, "
                        f"Phase={self.training_phase}, λ={self.lambda_penalty:.3f}, "
                        f"Loop_Status={loop_status}, Stuck_Count={self.position_stuck_count}")

    def _compute_llm4rl_reward(self, task_reward: float, asked_llm: bool, llm_plan_changed: bool) -> float:
        """
        FIXED: Much more aggressive penalty for unnecessary interrupts (asking but agreeing)
        """

        # Update baseline (moving average)
        self.reward_buffer.append(task_reward)
        if len(self.reward_buffer) > 10:
            self.baseline_reward = np.mean(list(self.reward_buffer)[-50:])

        # Base reward (advantage)
        advantage = task_reward - self.baseline_reward
        base_reward = advantage

        # Track interrupt patterns
        self.recent_interrupts.append(asked_llm)
        self.recent_agreements.append(asked_llm and not llm_plan_changed)

        # Calculate interrupt efficiency
        recent_interrupt_rate = np.mean(self.recent_interrupts) if self.recent_interrupts else 0
        recent_agreement_rate = np.mean(self.recent_agreements) if self.recent_agreements else 0
        interrupt_efficiency = 1 - (recent_agreement_rate / max(recent_interrupt_rate, 0.01))

        # FIXED: Much more aggressive penalty system
        penalty = 0.0
        bonus = 0.0

        if asked_llm:
            if llm_plan_changed:
                # GOOD: Asked and got different plan
                if task_reward > 0:
                    bonus = 0.1  # Good intervention bonus
                elif task_reward <= 0:
                    penalty = 0.05  # Small penalty for bad intervention
            else:
                # BAD: Asked but LLM agreed (WASTE OF RESOURCES!)
                base_penalty = self.agreement_penalty

                # ESCALATING PENALTY: More you agree, higher the penalty
                if recent_agreement_rate > 0.3:  # If >30% agreements
                    escalation_factor = 2.0 + (recent_agreement_rate - 0.3) * 5
                    penalty = base_penalty * escalation_factor
                else:
                    penalty = base_penalty

                # EFFICIENCY PENALTY: Low efficiency = higher penalty
                if interrupt_efficiency < self.interrupt_efficiency_threshold:
                    efficiency_penalty = (self.interrupt_efficiency_threshold - interrupt_efficiency) * 0.5
                    penalty += efficiency_penalty

                # NEW: Loop penalty - if we're in a detected loop situation
                if self.position_stuck_count > 8:
                    penalty += 0.15  # Heavy penalty for asking while stuck

                # Cap penalty to prevent explosion
                penalty = min(penalty, 0.6)

        else:
            # Didn't ask
            if task_reward <= -0.1:
                # Should have asked in clearly bad situation (but not if forced RL mode)
                if self.forced_rl_mode_steps == 0:
                    penalty = 0.02
            elif task_reward > 0.5:
                # Good: didn't interrupt successful action
                bonus = 0.02

        total_reward = base_reward + bonus - penalty

        # Aggressive clipping for stability
        total_reward = np.clip(total_reward, -0.8, 0.8)

        # Debug logging for tuning
        if self.verbose and len(self.ask_history) % 20 == 0:
            logger.info(f"Reward Debug: base={base_reward:.3f}, bonus={bonus:.3f}, penalty={penalty:.3f}, "
                        f"interrupt_eff={interrupt_efficiency:.3f}, agreement_rate={recent_agreement_rate:.3f}, "
                        f"stuck_count={self.position_stuck_count}")

        return total_reward

    def _update_lambda_penalty(self):
        """FIXED: Dynamic penalty based on interrupt efficiency (deque safe)"""
        progress = min(len(self.ask_history) / 1000.0, 1.0)

        # Base progression: 0.02 → 0.1
        base_lambda = 0.02 + progress * 0.08

        # ADAPTIVE: Increase penalty if efficiency is low
        if len(self.recent_interrupts) > 20:
            # FIXED: Convert deque to list for slicing
            recent_interrupt_rate = np.mean(list(self.recent_interrupts)[-20:])
            recent_agreement_rate = np.mean(list(self.recent_agreements)[-20:])

            if recent_interrupt_rate > 0.01:
                efficiency = 1 - (recent_agreement_rate / recent_interrupt_rate)

                # If efficiency < 30%, increase penalty aggressively
                if efficiency < 0.3:
                    efficiency_multiplier = 2.0 + (0.3 - efficiency) * 3
                    self.lambda_penalty = min(base_lambda * efficiency_multiplier, self.max_lambda)
                else:
                    self.lambda_penalty = base_lambda
            else:
                self.lambda_penalty = base_lambda
        else:
            self.lambda_penalty = base_lambda

        # Update agreement penalty based on recent performance
        if len(self.recent_agreements) > 30:
            # FIXED: Convert deque to list for slicing
            recent_agreement_rate = np.mean(list(self.recent_agreements)[-30:])
            if recent_agreement_rate > 0.4:  # If >40% agreements
                self.agreement_penalty = min(0.2, 0.1 + (recent_agreement_rate - 0.4) * 0.5)
            else:
                self.agreement_penalty = 0.1

    def _obs_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation dictionary to tensor."""
        image = obs['image']
        return torch.FloatTensor(image).to(self.device)

    def _extract_features(self, obs: Dict) -> Dict:
        """Extract features from observation for decision making."""
        obj_map = obs['image'][:, :, 0]
        state_map = obs['image'][:, :, 2]

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # In MiniGrid partial observation, agent is ALWAYS at center
        height, width = obj_map.shape
        agent_pos = (height // 2, width // 2)

        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        # Door state analysis
        door_state = None
        door_is_open = False
        if door_pos:
            inv_state = {v: k for k, v in STATE_TO_IDX.items()}
            door_state_val = int(state_map[door_pos])
            door_state = inv_state.get(door_state_val, "unknown")
            door_is_open = (door_state == "open")

        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else float('inf')

        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        def is_facing_object(agent_pos, obj_pos):
            if not (agent_pos and obj_pos):
                return False
            return (obj_pos[0] == agent_pos[0] - 1 and obj_pos[1] == agent_pos[1])

        # Track position for loop detection
        self.recent_positions.append(agent_pos)

        features = {
            'agent_pos': agent_pos,
            'key_pos': key_pos,
            'door_pos': door_pos,
            'goal_pos': goal_pos,
            'door_state': door_state,
            'door_is_open': door_is_open,
            'dist_to_key': manh(agent_pos, key_pos),
            'dist_to_door': manh(agent_pos, door_pos),
            'dist_to_goal': manh(agent_pos, goal_pos),
            'is_key_visible': key_pos is not None,
            'is_door_visible': door_pos is not None,
            'is_adjacent_to_key': is_adjacent(agent_pos, key_pos),
            'is_adjacent_to_door': is_adjacent(agent_pos, door_pos),
            'facing_key': is_facing_object(agent_pos, key_pos),
            'facing_door': is_facing_object(agent_pos, door_pos),
            'facing_wall': False,
        }

        return features

    def update_state(self, obs: Dict):
        """Update mediator's internal state."""
        self.previous_obs = obs.copy() if obs else None

    def get_statistics(self) -> Dict:
        """FIXED: Enhanced statistics with efficiency metrics and loop detection info"""
        if not self.ask_history:
            return {
                'total_steps': 0,
                'ask_rate': 0.0,
                'avg_reward': 0.0,
                'recent_ask_rate': 0.0,
                'recent_avg_reward': 0.0,
                'recent_loss': 0.0,
                'training_phase': self.training_phase,
                'lambda_penalty': self.lambda_penalty,
                'agreement_penalty': self.agreement_penalty,
                'baseline_reward': self.baseline_reward,
                'interrupt_efficiency': 1.0,
                'recent_interrupt_rate': 0.0,
                'recent_agreement_rate': 0.0,
                'efficiency_threshold': self.interrupt_efficiency_threshold,
                'successful_episodes': 0,
                'forced_rl_mode_steps': self.forced_rl_mode_steps,
                'position_stuck_count': self.position_stuck_count,
                'recent_override_count': len(self.recent_llm_overrides)
            }

        # Calculate efficiency metrics
        # FIXED: Convert deque to list for slicing
        recent_interrupt_rate = np.mean(list(self.recent_interrupts)[-50:]) if len(
            self.recent_interrupts) >= 50 else np.mean(list(self.recent_interrupts)) if self.recent_interrupts else 0
        recent_agreement_rate = np.mean(list(self.recent_agreements)[-50:]) if len(
            self.recent_agreements) >= 50 else np.mean(list(self.recent_agreements)) if self.recent_agreements else 0

        efficiency = 1 - (
                recent_agreement_rate / max(recent_interrupt_rate, 0.01)) if recent_interrupt_rate > 0 else 1.0

        return {
            'total_steps': len(self.ask_history),
            'ask_rate': np.mean(self.ask_history),
            'avg_reward': np.mean(self.reward_history),
            'recent_ask_rate': np.mean(self.ask_history[-100:]) if len(self.ask_history) >= 100 else np.mean(
                self.ask_history),
            'recent_avg_reward': np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                self.reward_history),
            'recent_loss': np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else 0,
            'training_phase': self.training_phase,
            'lambda_penalty': self.lambda_penalty,
            'agreement_penalty': self.agreement_penalty,
            'baseline_reward': self.baseline_reward,
            'interrupt_efficiency': efficiency,
            'recent_interrupt_rate': recent_interrupt_rate,
            'recent_agreement_rate': recent_agreement_rate,
            'efficiency_threshold': self.interrupt_efficiency_threshold,
            'successful_episodes': sum(1 for r in self.reward_history[-100:] if r > 0) if len(
                self.reward_history) >= 100 else sum(1 for r in self.reward_history if r > 0),
            # NEW: Loop detection statistics
            'forced_rl_mode_steps': self.forced_rl_mode_steps,
            'position_stuck_count': self.position_stuck_count,
            'recent_override_count': len(self.recent_llm_overrides),
            'loop_detection_active': self.forced_rl_mode_steps > 0
        }

    def save_asking_policy(self, path: str):
        """Save the learned asking policy."""
        torch.save({
            'model_state_dict': self.asking_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ask_history': self.ask_history,
            'reward_history': self.reward_history,
            'loss_history': self.loss_history,
            'training_phase': self.training_phase,
            'lambda_penalty': self.lambda_penalty,
            'agreement_penalty': self.agreement_penalty,
            'baseline_reward': self.baseline_reward,
            'recent_interrupts': list(self.recent_interrupts),
            'recent_agreements': list(self.recent_agreements),
            # NEW: Save loop detection state
            'recent_llm_overrides': list(self.recent_llm_overrides),
            'forced_rl_mode_steps': self.forced_rl_mode_steps,
            'position_stuck_count': self.position_stuck_count,
        }, path)

    def load_asking_policy(self, path: str):
        """Load a pre-trained asking policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.asking_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ask_history = checkpoint.get('ask_history', [])
        self.reward_history = checkpoint.get('reward_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        self.training_phase = checkpoint.get('training_phase', 'exploration')
        self.lambda_penalty = checkpoint.get('lambda_penalty', 0.02)
        self.agreement_penalty = checkpoint.get('agreement_penalty', 0.1)
        self.baseline_reward = checkpoint.get('baseline_reward', 0.0)
        self.recent_interrupts = deque(checkpoint.get('recent_interrupts', []), maxlen=50)
        self.recent_agreements = deque(checkpoint.get('recent_agreements', []), maxlen=50)
        # NEW: Load loop detection state
        self.recent_llm_overrides = deque(checkpoint.get('recent_llm_overrides', []), maxlen=20)
        self.forced_rl_mode_steps = checkpoint.get('forced_rl_mode_steps', 0)
        self.position_stuck_count = checkpoint.get('position_stuck_count', 0)