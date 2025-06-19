import numpy as np
from typing import List
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from TSCAssistant.tsc_agent_prompt import (
    SYSTEM_MESSAGE_SUFFIX,
    SYSTEM_MESSAGE_PREFIX,
    HUMAN_MESSAGE,
    FORMAT_INSTRUCTIONS,
    TRAFFIC_RULES,
    DECISION_CAUTIONS,
    HANDLE_PARSING_ERROR
)


"""
MINIGRID DOORKEY
Description
This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square. This environment is difficult, because of the sparse reward, to solve using classical RL algorithms. It is useful to experiment with curiosity or curriculum learning.

Observation Encoding

    Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)

    OBJECT_TO_IDX and COLOR_TO_IDX mapping can be found in minigrid/core/constants.py

    STATE refers to the door state with 0=open, 1=closed and 2=locked


Action Space
0 left Turn left

1 right Turn right

2 forward Move forward

3 pickup Pick up an object

4 drop Unused

5 toggle Toggle/activate an object

6 done Unused



Rewards
A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.

Termination

The episode ends if any one of the following conditions is met:

    The agent reaches the goal.

    Timeout (see max_steps).


"""


class TSCAgent:
    def __init__(self, llm: RunnableLambda, verbose: bool = True, state: float = []) -> None:
        self.tls_id = 'J1'
        self.llm = llm
        self.tools = []
        self.state = state

        self.first_prompt = ChatPromptTemplate.from_template(
            'You can ONLY use one of the following actions: \n action:0 action:1 action:2 action:3'
        )

        self.second_prompt = ChatPromptTemplate.from_template(
            "The action is {Action}, Your explanation was `{Occupancy}` \n To check decision safety: "
        )

        self.phase2movements = {
            "Phase 0": ["E0_s", "-E1_s"],
            "Phase 1": ["E0_l", "-E1_l"],
            "Phase 2": ["-E3_s", "-E2_s"],
            "Phase 3": ["-E3_l", "-E2_l"],
        }

        self.movement_ids = ["E0_s", "-E1_s", "-E1_l", "E0_l", "-E3_s", "-E2_s", "-E3_l", "-E2_l"]

    def get_phase(self):
        return np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1]
        ])

    # TODO: extract useful observation features from the "obs" (7x7x3) return of the Minigrid environment
    def get_grid_size(self, obs):
        grid_size = None
        return grid_size
    def get_goal_object_type(self, obs):
        goal_object_type = None
        return goal_object_type
    def get_vertical_distance_to_goal_object(self, obs):
        distance = None
        return distance
    def get_horizontal_distance_to_goal_object(self, obs):
        distance = None
        return distance


    def get_occupancy(self, states):
        phase_list = self.get_phase()
        occupancy = states[:, :, 1]
        occupancy_list = np.zeros(phase_list.shape[0])
        for i in range(phase_list.shape[0]):
            occupancy_list[i] = (occupancy * phase_list[i]).sum()
        occupancy_list = occupancy_list / 2
        return occupancy_list

    def get_rescue_movement_ids(self, last_step_vehicle_id_list, movement_ids):
        rescue_movement_ids = []
        for vehicle_ids, movement_id in zip(last_step_vehicle_id_list, movement_ids):
            for vehicle_id in vehicle_ids:
                if 'rescue' in vehicle_id:
                    rescue_movement_ids.append(movement_id)
        return rescue_movement_ids

    def agent_run(self, sim_step: float, action: int = 0, obs: float = [], infos: list = {}):
        # TODO: discard TSC functionalities, when the Minigrid is done
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        occupancy = self.get_occupancy(obs)
        Action = action[0]
        step_time = int(infos[0]['step_time'])
        Occupancy = infos[0]['movement_occ']
        jam_length_meters = infos[0]['jam_length_meters']
        movement_ids = infos[0]['movement_ids']
        last_step_vehicle_id_list = infos[0]['last_step_vehicle_id_list']
        information_missing = infos[0]['information_missing']
        missing_id = infos[0]['missing_id']
        rescue_movement_ids = self.get_rescue_movement_ids(last_step_vehicle_id_list, movement_ids)

        # TODO: check if these getted informations are sane
        grid_size = self.get_grid_size(obs)
        goal_object_type = self.get_goal_object_type(obs)
        v_distance = self.get_vertical_distance_to_goal_object(obs)
        h_distance = self.get_horizontal_distance_to_goal_object(obs)

        # TODO: check if this prompt makes sense for our Minigrid Simple Doorkey task, especially state and action definitions
        review_template = """
        decision: Simple Doorkey environment decision-making judgment — whether the Action is reasonable in the current state.
        explanations: Your explanation about your decision, described your suggestions to the sane next decision. 
        final_action: ONLY the number of Action you suggestion, 0, 1, 2, 3, 4, 5, 6.

        Format the output as JSON with the following keys:  
        decision
        explanations
        final_action

        observation: {observation}
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=review_template)

        decision = ResponseSchema(
            name="decision",
            description="Judgment whether the RL Agent's Action is reasonable in the current state"
        )
        explanations = ResponseSchema(
            name="explanations",
            description="Detailed explanation about your decision"
        )
        final_action = ResponseSchema(
            name="final_action",
            description="Final action number: 0, 1, 2, 3, 4, 5, 6"
        )

        output_parser = StructuredOutputParser.from_response_schemas(
            [decision, explanations, final_action]
        )
        format_instructions = output_parser.get_format_instructions()

        # TODO: adapt getted Minigrid features into this string, check if its sane
        observation = (f"""
        
        This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square. This environment is difficult, because of the sparse reward, to solve using classical RL algorithms. It is useful to experiment with curiosity or curriculum learning.
        You, the 'agent in 5x5 MiniGrid DoorKey environment', are now trying to the reach the green goal by picking up the key and open the door.
        The step time is: {step_time}
        The decision RL Agent made is Action: {Action}
        Grid size: {grid_size}
        Goal object type: {goal_object_type}
        Vertical distance to the goal object: {v_distance}
        Horizontal distance to the goal object: {h_distance}
        Info loss: {information_missing}, Missing ID: {missing_id}

        {DECISION_CAUTIONS}
        """)

        messages = prompt.format_messages(
            observation=observation,
            format_instructions=format_instructions
        )

        print(messages[0].content)
        logger.info('RL:' + messages[0].content)

        r = self.llm.invoke(messages)
        output_dict = output_parser.parse(r.content)
        print(r.content)
        logger.info('RL:' + r.content)

        final_action = int(output_dict.get('final_action'))
        logger.info('RL Final Action: ' + str(final_action))
        print('-' * 10)
        return final_action