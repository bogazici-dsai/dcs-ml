# MinigridAssistant/minigrid_agent_prompt.py


PROMPT_TEMPLATE = """You are controlling a reinforcement learning agent in the MiniGrid-DoorKey-6x6 environment. 
The agent must pick up the key to unlock the door and reach the green goal square.

Your task is to decide whether to agree with the PPO agent's suggested action or override it.

✔️ Agree if the PPO action is safe and logical ( turning toward objects, toggling reachable door).  
❌ Override if the action is unsafe or illogical (e.g., moving into a wall, picking up a key that isn’t directly in front, using unnecessary actions).

---

🔍 **Current Observations**

Agent Position: {agent_pos}  
Key → Pos: {key_pos} | Dist: {dist_to_key} | Visible: {is_key_visible}  
Door → Pos: {door_pos} | State: {door_state} | Dist: {dist_to_door} | Visible: {is_door_visible}  
Goal → Pos: {goal_pos} | Dist: {dist_to_goal}  
Front Object: {front_object} | Facing Wall: {facing_wall}

📊 **Agent Status**  
- Carrying key: {has_key}  
- Relative to Key: {rel_dir_to_key} | Distance V/H: {vertical_distance_to_key} / {horizontal_distance_to_key}  
- Relative to Door: {rel_dir_to_door} | Distance V/H: {vertical_distance_to_goal} / {horizontal_distance_to_goal}  

📌 **Adjacency**  
- Adjacent to Key: {is_adjacent_to_key}  
- Adjacent to Door: {is_adjacent_to_door}  

---

🤖 **PPO Suggests:** Action {action}  
🎮 **Available Actions:**  
0: Turn left (to turn direction to left)
1: Turn right (to direction to right)
2: Move forward (to move forward to front cell)
3: Pick up key
5: Toggle door  
⚠️ Actions 4 and 6 are forbidden

---

📏 **Strict Rules for Decision**

1. 🔑 **KEY PICKUP (Action 3)**  
   - Use **only** if ALL of the following are true:  
     • The agent **is NOT already carrying the key**  
     • The **key is the object directly in front** of the agent (`front_object = key`)  
     • The **distance to the key is exactly 1**  
   - If **any** of these conditions are false, override Action 3

2. 🚪 **TOGGLE DOOR (Action 5)**  
   - Use **only** if **all** of the following are true:  
     • The **door is directly in front** of the agent (`front_object = door`)  
     • The door is **locked**  
     • The agent **has the key** (`has_key = True`)  
     • The **key is not visible** (no need to pick it up)  
   - If **any** of these are false, override Action 5

3. 🚷 MOVE FORWARD (Action 2):
    - ✅ Allowed only if the object directly in front is an empty cell (i.e., front_object = empty cell).
    - Allowed only if unless it is necessary. IMPORTANT!!!!
    - ❌ Do NOT move forward if the front object is a wall, or key.

4. 🎯 **GOAL IS VISIBLE**  
   - If the goal is visible:  
     • **Avoid Action 3 (Pick up key)** — key is no longer needed  
     • **Avoid Action 5 (Toggle door)** — door should already be unlocked

5. ❌ **FORBIDDEN ACTIONS**  
   - Never use Actions 4 or 6 under any circumstances

6. ✅ **DEFAULT TO PPO**  
   - If the suggested action does **not** violate any of the above rules, you must agree with it

---

🧠 **How to Respond (STRICT FORMAT)**

Your reply **must start with this exact line** (first line only):

Selected action: <number>

✅ Example: `Selected action: 2`

Then, give a **very short and clear explanation** for your choice.

⚠️ If you override PPO’s action, explain **which rule** was violated and why your action is better.  
⚠️ If you choose a different action than PPO, **do NOT claim to "agree"** with PPO — this is a contradiction and will be rejected.

If you follow all instructions correctly, your decision will be accepted and executed.
"""

# def render_prompt(env_name: str, features: dict, action: int) -> str:
#     return PROMPT_TEMPLATE.format(env_name=env_name, action=action, **features)
def render_prompt(env_name: str, action: int) -> str:
    return PROMPT_TEMPLATE.format(env_name=env_name, action=action)

