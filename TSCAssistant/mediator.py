import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX


class AskingPolicy(nn.Module):
    """Enhanced asking policy with better initialization"""

    def __init__(self, obs_shape: tuple, hidden_dim: int = 64):
        super().__init__()

        obs_size = np.prod(obs_shape)
        input_dim = obs_size * 3  # current, previous, difference

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Ask or Not Ask
        )

        # Initialize to be more likely to ask initially
        with torch.no_grad():
            self.network[-1].bias[0] = 1.0  # Bias toward asking
            self.network[-1].bias[1] = -1.0  # Bias against not asking

    def forward(self, current_obs, previous_obs):
        current_flat = current_obs.flatten()
        previous_flat = previous_obs.flatten()
        diff_flat = current_flat - previous_flat
        combined_input = torch.cat([current_flat, previous_flat, diff_flat])
        return self.network(combined_input)


class Mediator:
    """
    Enhanced mediator with loop detection and better asking strategy.
    """

    def __init__(self,
                 obs_shape: tuple,
                 learning_rate: float = 3e-4,
                 device: str = "cpu",
                 verbose: bool = True):
        self.obs_shape = obs_shape
        self.device = device
        self.verbose = verbose

        # Initialize asking policy network
        self.asking_policy = AskingPolicy(obs_shape).to(device)
        self.optimizer = torch.optim.Adam(self.asking_policy.parameters(), lr=learning_rate)

        # Enhanced state tracking
        self.previous_obs = None
        self.last_llm_plan = None
        self.steps_since_last_ask = 0

        # Loop detection
        self.recent_actions = deque(maxlen=10)  # Track last 10 actions
        self.recent_positions = deque(maxlen=5)  # Track last 5 positions

        # Training progress tracking
        self.ask_history = []
        self.reward_history = []
        self.training_phase = "exploration"  # exploration -> exploitation

    def should_ask_llm(self,
                       obs: Dict,
                       ppo_action: int,
                       use_learned_policy: bool = True,
                       force_exploration: bool = False) -> Tuple[bool, float]:
        """
        Enhanced decision making with loop detection and progressive learning.
        """

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # PHASE 1: Force exploration in early training
        if len(self.ask_history) < 1000 or force_exploration:
            return self._exploration_asking(obs, action)

        # PHASE 2: Use learned policy with safety nets
        return self._learned_asking(obs, action, use_learned_policy)

    def _exploration_asking(self, obs: Dict, ppo_action: int) -> Tuple[bool, float]:
        """Aggressive asking during exploration phase"""

        # ALWAYS ask if action seems problematic
        if self._is_problematic_action(obs, ppo_action):
            return True, 0.95

        # Ask every 3-5 steps during exploration
        if self.steps_since_last_ask >= 4:
            return True, 0.8

        # Ask if significant change detected
        if self._significant_obs_change(obs):
            return True, 0.7

        return False, 0.3

    def _learned_asking(self, obs: Dict, ppo_action: int, use_learned_policy: bool) -> Tuple[bool, float]:
        """Use learned policy with safety overrides"""

        # SAFETY: Always override learned policy for critical situations
        if self._is_critical_situation(obs, ppo_action):
            return True, 0.9

        if not use_learned_policy:
            return self._heuristic_asking_decision(obs, ppo_action)

        # Use neural network policy
        if self.previous_obs is None:
            return True, 1.0

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        with torch.no_grad():
            logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
            probabilities = torch.softmax(logits, dim=0)
            ask_prob = probabilities[0].item()

            # Apply safety threshold - be more aggressive than 0.5
            should_ask = ask_prob > 0.4  # Lower threshold = more asking

        self.steps_since_last_ask += 1

        if should_ask:
            self.steps_since_last_ask = 0

        return should_ask, ask_prob

    def _is_problematic_action(self, obs: Dict, ppo_action: int) -> bool:
        """Detect problematic actions that need LLM intervention"""

        # Ensure action is integer (fix numpy array issue)
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Invalid actions
        if action in [4, 6]:  # Drop, Done (forbidden)
            return True

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

        # Check for problematic actions first
        if self._is_problematic_action(obs, action):
            return True

        # Extract features for situation analysis
        features = self._extract_features(obs)

        # Critical situations
        if features.get('is_adjacent_to_key') and action != 3:
            return True  # Should pick up key but didn't

        if features.get('facing_wall') and action == 2:
            return True  # About to walk into wall

        if features.get('is_adjacent_to_door') and action != 5:
            return True  # Should toggle door but didn't

        # Been too long without asking
        if self.steps_since_last_ask > 15:
            return True

        return False

    def _heuristic_asking_decision(self, obs: Dict, ppo_action: int) -> Tuple[bool, float]:
        """Enhanced heuristic asking policy"""

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        if self._is_critical_situation(obs, action):
            return True, 0.9

        if self._significant_obs_change(obs):
            return True, 0.7

        # Periodic asking to prevent getting stuck
        if self.steps_since_last_ask >= 8:
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
        """Enhanced training with progressive penalty"""

        if self.previous_obs is None:
            self.previous_obs = obs
            return

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        # Get asking policy prediction
        logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
        ask_prob = torch.softmax(logits, dim=0)[0]

        # Progressive penalty system
        penalty = 0.0
        if asked_llm and not llm_plan_changed:
            # Start with small penalty, increase as training progresses
            base_penalty = 0.1 if len(self.ask_history) < 500 else 0.5
            penalty = base_penalty

        # Bonus for successful episodes
        bonus = 0.0
        if reward > 0:  # Successful episode
            bonus = 0.2

        total_reward = reward + bonus - penalty

        # Enhanced loss with exploration bonus
        if asked_llm:
            loss = -torch.log(ask_prob + 1e-8) * total_reward
        else:
            loss = -torch.log(1 - ask_prob + 1e-8) * total_reward

        # Add exploration bonus in early training
        if len(self.ask_history) < 500:
            exploration_bonus = 0.1 * ask_prob  # Encourage asking
            loss = loss - exploration_bonus

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.asking_policy.parameters(), 1.0)
        self.optimizer.step()

        # Track statistics
        self.ask_history.append(asked_llm)
        self.reward_history.append(total_reward)

        # Update training phase
        if len(self.ask_history) >= 500 and self.training_phase == "exploration":
            self.training_phase = "exploitation"
            if self.verbose:
                logger.info("Mediator transitioning from exploration to exploitation phase")

        if self.verbose and len(self.ask_history) % 100 == 0:
            recent_ask_rate = np.mean(self.ask_history[-100:])
            recent_reward = np.mean(self.reward_history[-100:])
            logger.info(f"Mediator Stats: Ask Rate={recent_ask_rate:.2f}, "
                        f"Avg Reward={recent_reward:.2f}, Phase={self.training_phase}")

    def _obs_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation dictionary to tensor."""
        image = obs['image']
        return torch.FloatTensor(image).to(self.device)

    def _extract_features(self, obs: Dict) -> Dict:
        """Extract features from observation for decision making."""
        obj_map = obs['image'][:, :, 0]

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # FIXED: In MiniGrid partial observation, agent is ALWAYS at center
        # The observation IS the agent's view, so agent position is implicit
        height, width = obj_map.shape
        agent_pos = (height // 2, width // 2)  # Always (3, 3) in 7x7 grid

        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else float('inf')

        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        def is_facing_object(agent_pos, obj_pos):
            if not (agent_pos and obj_pos):
                return False
            # In MiniGrid, agent faces "up" by default (negative row direction)
            # Check if object is directly in front (one cell up)
            return (obj_pos[0] == agent_pos[0] - 1 and obj_pos[1] == agent_pos[1])

        # Track position for loop detection
        self.recent_positions.append(agent_pos)

        features = {
            'agent_pos': agent_pos,
            'key_pos': key_pos,
            'door_pos': door_pos,
            'goal_pos': goal_pos,
            'dist_to_key': manh(agent_pos, key_pos),
            'dist_to_door': manh(agent_pos, door_pos),
            'dist_to_goal': manh(agent_pos, goal_pos),
            'is_key_visible': key_pos is not None,
            'is_door_visible': door_pos is not None,
            'is_adjacent_to_key': is_adjacent(agent_pos, key_pos),
            'is_adjacent_to_door': is_adjacent(agent_pos, door_pos),
            'facing_key': is_facing_object(agent_pos, key_pos),
            'facing_door': is_facing_object(agent_pos, door_pos),
            'facing_wall': False,  # Would need direction info to determine accurately
        }

        return features

    def update_state(self, obs: Dict):
        """Update mediator's internal state."""
        self.previous_obs = obs.copy() if obs else None

    def get_statistics(self) -> Dict:
        """Get comprehensive training statistics."""
        if not self.ask_history:
            return {}

        return {
            'total_steps': len(self.ask_history),
            'ask_rate': np.mean(self.ask_history),
            'avg_reward': np.mean(self.reward_history),
            'recent_ask_rate': np.mean(self.ask_history[-100:]) if len(self.ask_history) >= 100 else np.mean(
                self.ask_history),
            'recent_avg_reward': np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                self.reward_history),
            'training_phase': self.training_phase,
            'successful_episodes': sum(1 for r in self.reward_history[-100:] if r > 0) if len(
                self.reward_history) >= 100 else sum(1 for r in self.reward_history if r > 0)
        }

    def save_asking_policy(self, path: str):
        """Save the learned asking policy."""
        torch.save({
            'model_state_dict': self.asking_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ask_history': self.ask_history,
            'reward_history': self.reward_history,
            'training_phase': self.training_phase,
        }, path)

    def load_asking_policy(self, path: str):
        """Load a pre-trained asking policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.asking_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ask_history = checkpoint.get('ask_history', [])
        self.reward_history = checkpoint.get('reward_history', [])
        self.training_phase = checkpoint.get('training_phase', 'exploration')