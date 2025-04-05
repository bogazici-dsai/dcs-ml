'''
@Author: WANG Maonan
@Author: PangAoyu
@Date: 2023-09-05 11:27:05
@Description: TSC Environment Wrapper — prepares state info for LLM
@LastEditTime: 2023-09-15 20:30:41
'''

import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict, List

from tshub.utils.nested_dict_conversion import create_nested_defaultdict, defaultdict2dict
from TSCEnvironment.wrapper_utils import (
    convert_state_to_static_information,
    predict_queue_length,
    OccupancyList
)


class TSCEnvWrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        # Static Intersection Info
        self.movement_ids = None
        self.phase_num = None
        self.llm_static_information = None  # Contains signal structure & geometry

        # Dynamic Info
        self.state = None
        self.last_state = None
        self.occupancy = OccupancyList()

    def transform_occ_data(self, occ: List[float]) -> Dict[str, float]:
        """Map average occupancy values to movement IDs with percentage strings"""
        return {
            movement_id: f"{value * 100:.2f}%"
            for movement_id, value in zip(self.movement_ids, occ)
            if 'r' not in movement_id
        }

    def state_wrapper(self, state: Dict[str, Any]) -> List[float]:
        """Extract occupancy list from state"""
        return state['last_step_occupancy']

    def reset(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        state = self.env.reset()
        self.phase_num = len(state['phase2movements'])
        self.movement_ids = state['movement_ids']
        self.llm_static_information = convert_state_to_static_information(state)

        occ = self.state_wrapper(state)
        self.state = self.transform_occ_data(occ)
        return self.state, state  # Return full state too if needed later

    def step(self, action: Any) -> Tuple[Dict[str, str], bool, Dict[str, Any]]:
        can_perform_action = False
        while not can_perform_action:
            states, rewards, truncated, dones, infos = super().step(action)
            occ = self.state_wrapper(states)
            self.occupancy.add_element(occ)
            can_perform_action = states['can_perform_action']

        avg_occ = self.occupancy.calculate_average()
        self.last_state = self.state
        self.state = self.transform_occ_data(avg_occ)

        return self.state, dones, infos

    def close(self) -> None:
        return super().close()

    # --------------- LLM-friendly Utils -----------------

    def get_available_actions(self) -> List[int]:
        """List of available phase indices (action space)"""
        return list(range(self.phase_num))

    def get_current_occupancy(self) -> Dict[str, str]:
        return self.state

    def get_previous_occupancy(self) -> Dict[str, str]:
        return self.last_state

    def predict_future_scene(self, phase_index: int) -> Dict[str, Any]:
        """Simulate queue length changes after applying specific phase"""
        try:
            phase_index = int(phase_index)
        except:
            raise ValueError(f"`phase_index` must be an integer. Got: {phase_index}")

        predict_state = create_nested_defaultdict()
        for tls_id, tls_info in self.state.items():
            tls_phases = tls_info['phase_queue_lengths']
            for idx, (phase_name, queue_info) in enumerate(tls_phases.items()):
                is_green = (idx == phase_index)
                predicted = predict_queue_length(queue_info, is_green=is_green)
                predict_state[tls_id][phase_name] = predicted

        return defaultdict2dict(predict_state)