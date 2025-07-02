PROMPT_TEMPLATE = """You are an expert controller for a reinforcement learning agent in MiniGrid-DoorKey-6x6.

CURRENT SITUATION:
Agent Position: {agent_pos} 
Key Position: {key_pos} | Distance: {dist_to_key} | Visible: {is_key_visible}
Door Position: {door_pos} | State: {door_state} | Distance: {dist_to_door} | Visible: {is_door_visible}
Goal Position: {goal_pos} | Distance: {dist_to_goal}

AGENT STATUS:
- Carrying key: {has_key}
- Current objective: {current_objective}

SPATIAL CONTEXT:
- Key direction: {rel_dir_to_key}
- Door direction: {rel_dir_to_door}
- Object in front: {front_object}
- Facing wall: {facing_wall}

PPO AGENT SUGGESTS: Action {action} (Confidence: {ppo_confidence})

AVAILABLE ACTIONS:
0: Turn left    1: Turn right    2: Move forward    3: Pick up key    5: Toggle door
(Actions 4 and 6 are FORBIDDEN - never use them)

CRITICAL DECISION RULES (MUST FOLLOW EXACTLY):
1. PICKUP RULES:
   - Use Action 3 ONLY when: adjacent_to_key=True AND facing_key=True AND has_key=False
   - NEVER pick up if already carrying key

2. TOGGLE RULES: 
   - Use Action 5 ONLY when: adjacent_to_door=True AND has_key=True AND door is locked
   - Don't toggle if door is already open

3. NAVIGATION RULES:
   - If need key: navigate toward key position
   - If have key: navigate toward door position
   - Don't move forward into walls

4. EFFICIENCY RULES:
   - Agree with PPO unless it clearly violates above rules
   - Only override when PPO action is demonstrably wrong or forbidden

CONTEXT NOTES:
{performance_note}

ANALYSIS STEPS:
1. What is the current objective based on agent status?
2. Is the PPO action appropriate for this objective?
3. Does the PPO action violate any critical rules?
4. Should I agree or override?

RESPONSE FORMAT (MUST START WITH THIS):
Selected action: <number>

Then explain your reasoning briefly.
"""


def render_prompt(env_name: str, features: dict, action: int) -> str:
    """
    Enhanced prompt rendering with better default values and validation.
    """

    # Ensure all required fields have default values
    prompt_features = {
        'env_name': env_name,
        'action': action,
        'agent_pos': features.get('agent_pos', 'unknown'),
        'key_pos': features.get('key_pos', 'unknown'),
        'door_pos': features.get('door_pos', 'unknown'),
        'goal_pos': features.get('goal_pos', 'unknown'),
        'dist_to_key': features.get('dist_to_key', 'unknown'),
        'dist_to_door': features.get('dist_to_door', 'unknown'),
        'dist_to_goal': features.get('dist_to_goal', 'unknown'),
        'is_key_visible': features.get('is_key_visible', 'unknown'),
        'is_door_visible': features.get('is_door_visible', 'unknown'),
        'door_state': features.get('door_state', 'unknown'),
        'has_key': features.get('has_key', 'unknown'),
        'rel_dir_to_key': features.get('rel_dir_to_key', 'unknown'),
        'rel_dir_to_door': features.get('rel_dir_to_door', 'unknown'),
        'front_object': features.get('front_object', 'unknown'),
        'facing_wall': features.get('facing_wall', 'unknown'),
        'current_objective': features.get('current_objective', 'Explore and complete task'),
        'ppo_confidence': features.get('ppo_confidence', 'medium'),
        'performance_note': features.get('performance_note', 'Agent is learning, make safe decisions.'),
    }

    return PROMPT_TEMPLATE.format(**prompt_features)