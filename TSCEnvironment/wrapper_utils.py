'''
@Author: WANG Maonan
@Date: 2023-09-05 15:26:11
@Description: Processed state features & utilities for LLM
@LastEditTime: 2023-09-15 20:15:02
'''
import numpy as np
from typing import List, Dict, Any
from tshub.utils.nested_dict_conversion import create_nested_defaultdict, defaultdict2dict


class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element: List[float]) -> None:
        """Append a new occupancy vector"""
        if isinstance(element, list) and all(isinstance(e, float) for e in element):
            self.elements.append(element)
        else:
            raise TypeError("Element must be a list of floats.")

    def clear_elements(self) -> None:
        """Clear the buffer"""
        self.elements = []

    def calculate_average(self) -> List[float]:
        """Compute average occupancy over stored values"""
        if not self.elements:
            return []
        arr = np.array(self.elements, dtype=np.float32)
        averages = np.mean(arr, axis=0)
        self.clear_elements()
        return averages


def calculate_queue_lengths(
    movement_ids: List[str],
    jam_length_meters: List[float],
    phase2movements: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute total, average, and max queue length per phase.
    """
    phase_queue_lengths = {
        phase: {'total_length': 0.0, 'count': 0, 'max_length': 0.0, 'average_length': 0.0}
        for phase in phase2movements
    }

    for phase, movements in phase2movements.items():
        for movement in movements:
            movement_key = '_'.join(movement.split('--'))
            if movement_key not in movement_ids:
                continue
            idx = movement_ids.index(movement_key)
            length = jam_length_meters[idx]
            data = phase_queue_lengths[phase]
            data['total_length'] += length
            data['count'] += 1
            data['max_length'] = max(data['max_length'], length)

    for phase, data in phase_queue_lengths.items():
        if data['count'] > 0:
            data['average_length'] = data['total_length'] / data['count']

    return phase_queue_lengths


def predict_queue_length(
    queue_info: Dict[str, float],
    is_green: bool = False,
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Predict queue lengths based on Poisson arrivals and departures.
    """
    predict_queue_info = {}
    leaving_lambda = 4
    arrival_lambda_map = {'max_length': 3, 'average_length': 2}

    for key, base_length in queue_info.items():
        if key not in arrival_lambda_map:
            continue
        arrival_lambda = arrival_lambda_map[key]

        deltas = []
        for _ in range(num_samples):
            arrival = np.random.poisson(arrival_lambda)
            leaving = np.random.poisson(leaving_lambda) if is_green else 0
            delta = arrival - leaving
            deltas.append(delta)

        predicted = max(base_length + (np.mean(deltas) * 6), 0)
        predict_queue_info[key] = predicted

    return predict_queue_info


def convert_state_to_static_information(input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert raw state into structured static intersection data for LLM input.
    Includes movement info (direction, lanes) and phase info (movement IDs).
    """
    output_data = {
        "movement_infos": {},
        "phase_infos": {}
    }

    for movement_id, direction in input_data.get("movement_directions", {}).items():
        if direction == "l":
            direction_text = "Left Turn"
        elif direction == "s":
            direction_text = "Through"
        else:
            continue
        num_lanes = input_data["movement_lane_numbers"].get(movement_id, 0)
        output_data["movement_infos"][movement_id] = {
            "direction": direction_text,
            "number_of_lanes": num_lanes
        }

    for phase, movements in input_data.get("phase2movements", {}).items():
        phase_key = f"Phase {phase}"
        cleaned_movements = ["_".join(m.split('--')) for m in movements]
        output_data["phase_infos"][phase_key] = {"movements": cleaned_movements}

    return output_data