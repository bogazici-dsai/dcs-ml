import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class AskingPolicy(nn.Module):
    """Basit asking policy network"""

    def __init__(self, obs_shape: tuple, hidden_dim: int = 64):
        super().__init__()

        obs_size = np.prod(obs_shape)
        input_dim = obs_size * 3  # current, previous, difference

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Ask or Not Ask
        )

        # Balanced initialization
        with torch.no_grad():
            self.network[-1].bias[0] = 0.0  # Ask bias
            self.network[-1].bias[1] = 0.0  # Not ask bias

    def forward(self, current_obs, previous_obs):
        current_flat = current_obs.flatten()
        previous_flat = previous_obs.flatten()
        diff_flat = current_flat - previous_flat
        combined_input = torch.cat([current_flat, previous_flat, diff_flat])
        return self.network(combined_input)


class Mediator:
    """
    Basit mediator - exploration-exploitation balance ile
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

        # EXPLORATION-EXPLOITATION (Daha agresif)
        self.exploration_episodes = 60  # Daha uzun exploration
        self.current_episode = 0
        self.epsilon = 0.6  # 60% random exploration (was 30%)
        self.min_epsilon = 0.15  # Daha yüksek minimum (was 5%)
        self.epsilon_decay = 0.985  # Daha yavaş decay (was 0.98)

        # State tracking
        self.previous_obs = None
        self.steps_since_last_ask = 0

        # Performance tracking (Basit)
        self.ask_history = deque(maxlen=100)
        self.success_history = deque(maxlen=50)
        self.episode_rewards = deque(maxlen=20)

        # Training metrics
        self.total_asks = 0
        self.good_asks = 0  # Plan değişti
        self.bad_asks = 0  # Plan değişmedi

        # Loop detection (Basit)
        self.recent_actions = deque(maxlen=10)
        self.recent_positions = deque(maxlen=5)
        self.forced_rl_mode_steps = 0

    def should_ask_llm(self,
                       obs: Dict,
                       ppo_action: int,
                       use_learned_policy: bool = True) -> Tuple[bool, float]:
        """
        Basit asking decision - exploration-exploitation balance ile
        """

        # İlk observation
        if self.previous_obs is None:
            self.previous_obs = obs
            return True, 1.0

        # Basit loop detection
        if self.forced_rl_mode_steps > 0:
            self.forced_rl_mode_steps -= 1
            return False, 0.1

        # Critical situations always ask
        if self._is_critical_situation(obs, ppo_action):
            return True, 0.9

        # EXPLORATION PHASE
        if self.current_episode < self.exploration_episodes:
            # Random exploration ile network'ü kombine et
            if np.random.random() < self.epsilon:
                # Random ask
                should_ask = np.random.random() < 0.4
                return should_ask, 0.5
            else:
                # Network'ten sor ama liberal ol
                should_ask, confidence = self._network_decision(obs)
                return should_ask, confidence * 0.8

        # EXPLOITATION PHASE
        else:
            # Network'e güven ama biraz exploration devam et
            should_ask, confidence = self._network_decision(obs)

            # Küçük epsilon ile exploration
            if np.random.random() < self.min_epsilon:
                should_ask = not should_ask

            return should_ask, confidence

    def _network_decision(self, obs: Dict) -> Tuple[bool, float]:
        """Network'ten asking decision al - Daha liberal"""
        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        with torch.no_grad():
            logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
            probabilities = torch.softmax(logits, dim=0)
            ask_prob = probabilities[0].item()

            # Daha liberal threshold - exploration'da daha kolay "ask" der
            threshold = 0.45 if self.current_episode < self.exploration_episodes else 0.55
            should_ask = ask_prob > threshold

        return should_ask, ask_prob

    def _is_critical_situation(self, obs: Dict, ppo_action: int) -> bool:
        """Critical situations that always need LLM"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Invalid actions
        if action in [4, 6]:
            return True

        # Action loops
        self.recent_actions.append(action)
        if len(self.recent_actions) >= 5:
            if len(set(list(self.recent_actions)[-5:])) == 1:
                self.forced_rl_mode_steps = 3
                return True

        # Long time without asking
        if self.steps_since_last_ask > 20:
            return True

        return False

    def train_asking_policy(self,
                            obs: Dict,
                            action: int,
                            reward: float,
                            next_obs: Dict,
                            asked_llm: bool,
                            llm_plan_changed: bool):
        """
        Exploration-friendly training step
        """
        if self.previous_obs is None:
            self.previous_obs = obs
            return

        # Exploration-friendly reward hesapla
        if asked_llm:
            self.total_asks += 1
            if llm_plan_changed:
                self.good_asks += 1
                # Exploration'da daha büyük bonus
                bonus = 0.15 if self.current_episode < self.exploration_episodes else 0.1
                training_reward = reward + bonus
            else:
                self.bad_asks += 1
                # Exploration'da daha küçük penalty
                penalty = 0.05 if self.current_episode < self.exploration_episodes else 0.1
                training_reward = reward - penalty
        else:
            training_reward = reward

        # Network training
        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
        probabilities = torch.softmax(logits, dim=0)

        # Policy gradient
        if asked_llm:
            ask_prob = probabilities[0]
            loss = -torch.log(ask_prob + 1e-8) * training_reward
        else:
            not_ask_prob = probabilities[1]
            loss = -torch.log(not_ask_prob + 1e-8) * training_reward

        # Exploration-friendly entropy bonus
        entropy_weight = 0.05 if self.current_episode < self.exploration_episodes else 0.02
        entropy = -(probabilities[0] * torch.log(probabilities[0] + 1e-8) +
                    probabilities[1] * torch.log(probabilities[1] + 1e-8))
        loss = loss - entropy_weight * entropy  # Encourage exploration

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.asking_policy.parameters(), 0.5)
        self.optimizer.step()

        # Update tracking
        self.ask_history.append(asked_llm)
        self.episode_rewards.append(reward)
        self.steps_since_last_ask += 1

        if asked_llm:
            self.steps_since_last_ask = 0

        # Yavaş epsilon decay - daha uzun exploration
        if self.current_episode < self.exploration_episodes:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update state
        self.previous_obs = obs

    def update_state(self, obs: Dict):
        """Update mediator's internal state"""
        self.previous_obs = obs.copy() if obs else None

    def episode_end(self, success: bool):
        """Episode bittiğinde çağır"""
        self.current_episode += 1
        self.success_history.append(success)

        # Log every 10 episodes
        if self.current_episode % 10 == 0:
            self._log_progress()

    def _log_progress(self):
        """Basit logging"""
        if len(self.ask_history) > 0:
            recent_ask_rate = np.mean(list(self.ask_history)[-50:])
            recent_success_rate = np.mean(list(self.success_history)[-10:])

            phase = "EXPLORATION" if self.current_episode < self.exploration_episodes else "EXPLOITATION"

            if self.verbose:
                logger.info(f"Episode {self.current_episode} [{phase}] | "
                            f"Ask Rate: {recent_ask_rate:.2f} | "
                            f"Success Rate: {recent_success_rate:.2f} | "
                            f"Good Asks: {self.good_asks}/{self.total_asks} | "
                            f"Epsilon: {self.epsilon:.3f}")

    def get_statistics(self) -> Dict:
        """Basit stats"""
        if self.total_asks > 0:
            efficiency = self.good_asks / self.total_asks
        else:
            efficiency = 0.0

        return {
            'episode': self.current_episode,
            'phase': 'exploration' if self.current_episode < self.exploration_episodes else 'exploitation',
            'epsilon': self.epsilon,
            'ask_rate': np.mean(list(self.ask_history)[-50:]) if self.ask_history else 0,
            'success_rate': np.mean(list(self.success_history)[-10:]) if self.success_history else 0,
            'asking_efficiency': efficiency,
            'total_asks': self.total_asks,
            'good_asks': self.good_asks,
            'bad_asks': self.bad_asks,
            'recent_ask_rate': np.mean(list(self.ask_history)[-20:]) if len(self.ask_history) >= 20 else 0,
            'steps_since_last_ask': self.steps_since_last_ask
        }

    def save_asking_policy(self, path: str):
        """Model'i kaydet"""
        torch.save({
            'model_state_dict': self.asking_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.current_episode,
            'epsilon': self.epsilon,
            'total_asks': self.total_asks,
            'good_asks': self.good_asks,
            'bad_asks': self.bad_asks,
            'ask_history': list(self.ask_history),
            'success_history': list(self.success_history)
        }, path)

    def load_asking_policy(self, path: str):
        """Model'i yükle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.asking_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_episode = checkpoint.get('episode', 0)
        self.epsilon = checkpoint.get('epsilon', 0.3)
        self.total_asks = checkpoint.get('total_asks', 0)
        self.good_asks = checkpoint.get('good_asks', 0)
        self.bad_asks = checkpoint.get('bad_asks', 0)
        self.ask_history = deque(checkpoint.get('ask_history', []), maxlen=100)
        self.success_history = deque(checkpoint.get('success_history', []), maxlen=50)

    def _obs_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation dictionary to tensor"""
        image = obs['image']
        return torch.FloatTensor(image).to(self.device)

    def _extract_features(self, obs: Dict) -> Dict:
        """Extract features from observation - basit version"""
        obj_map = obs['image'][:, :, 0]
        state_map = obs['image'][:, :, 2]

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # Agent position (center in partial observation)
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

        return {
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
            'has_key': False,  # Will be overridden by TSC agent
        }