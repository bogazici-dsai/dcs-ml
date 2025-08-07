# HarfangEnv_GYM.py
from ctypes.macholib.dyld import dyld_env

import numpy as np
from numpy.array_api import uint8
#from numpy.distutils.system_info import default_x11_lib_dirs

import hirl.environments.dogfight_client as df
from hirl.environments.constants import *
import gym
import os
import inspect
import random
import math
from numpy.array_api import uint8

class HarfangEnv():
    def __init__(self):
        self.done = False
        self.loc_diff = 0
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]),
                                           dtype=np.float64)
        self.Plane_ID_oppo = "ennemy_2"  # Opponent aircrafts name
        self.Plane_ID_ally = "ally_1"  # our aircrafts name
        self.Aircraft_Loc = None
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.reward = 0
        self.Plane_Irtifa = 0
        self.now_missile_state = False # 导弹此时是否发射（本次step是否发射了导弹）

        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally) # 导弹列表
        self.missile1_id = self.missile[0] # 导弹1
        self.oppo_health = 0.2 # 敌机血量
        self.target_angle = None
        self.success = 0 # stepsuccess
        self.episode_success = False
        self.fire_success = False
        self.missile_handler = MissileHandler()

    def reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Oppo_target_locked = False
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.success = 0
        self.done = False
        self._reset_machine()
        self._reset_missile() # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally # 当前时刻状态
        self.missile_handler.reset()

        return state_ally

    def random_reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.n_Oppo_target_locked = False
        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.success = 0
        self.done = False
        self._random_reset_machine()
        self._reset_missile() # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally

        return state_ally

    def _random_reset_machine(self):
        df.reset_machine("ally_1") # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2 #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def step(self, action):
        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.success

    def step_test(self, action):
        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success

    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        self._get_loc_diff()  # get location difference information for reward

        # 距离惩罚：帮助追击
        self.reward -= (0.0001 * self.loc_diff) # 0.4

        # 目标角惩罚：帮助锁敌
        self.reward -= self.target_angle * 10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4

        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        # 开火奖励：帮助开火
        if self.now_missile_state == True: # 如果此step导弹发射
            self.reward -= 8
            if self.missile1_state == True and self.Ally_target_locked == False: # 且导弹存在、不锁敌
                # self.reward -= 100 # 4、4
                self.success = -1
                print('failed to fire')
            elif self.missile1_state == True and self.Ally_target_locked == True: # 且导弹存在且锁敌
                # self.reward += 8 # 100、4
                print('successful to fire')
                self.success = 1
                self.fire_success = True
            else:
                # self.reward -= 10
                self.reward -= 0

        # 坠落奖励（最终奖励）
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600 # 无、无
            print('enemy have fallen')

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        # self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0) #
            self.now_missile_state = True # 此时导弹发射
            # Get the plane missiles list
            missiles = df.get_machine_missiles_list(self.Plane_ID_ally)
            print("[DEBUG] Missiles list:", missiles)
            if len(missiles) <= 1:
                print("[ERROR] Missile slot boş, index:", 1, "Missiles:", missiles)

            missile_id = missiles[1]
            missile_state_new = df.get_missile_state(missile_id)
            print("[DEBUG] missile_state_new:", missile_state_new)

            print("fire")
        else:
            self.now_missile_state = False
        
        df.update_scene()

    def _get_termination(self):
        # if self.loc_diff < 200:
        #     self.done = True
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0: # 敌机血量低于0则结束
            self.done = True
            self.episode_success = True
        # if self.now_missile_state == True:
        #     self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1") # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2 #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self): #
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally) # 重新装填导弹

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) + ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) + ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** (1 / 2)

    def _get_observation(self): # 注意get的是n_state
        # Plane States
        Plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                     Plane_state["position"][1] / NormStates["Plane_position"],
                     Plane_state["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [Plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"], # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
                       Plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"], # 航向角，0 -> pai -> -pai -> 0
                       Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]] # 横滚角，顺时针：0 -> -pai -> pai -> 0
        # print("=================== Plane euler agles degree: ",math.degrees(Plane_state["Euler_angles"][0])*(-1))
        # print("=================== Plane Euler: ", Plane_Euler)
        # Plane_Heading = Plane_state["heading"] / NormStates["Plane_heading"] # 航向角，0 -> 360
        Plane_Heading = Plane_state["heading"]  # 航向角，0 -> 360
        Plane_Pitch_Att = Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"] # 俯仰：俯0 -> -90，仰0 -> 90
        Plane_Roll_Att = Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"] # 横滚角，顺时针：0 -> -90 ->0 -> 90 -> 0

        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]


        # Opponent States
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"], # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
                       Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"], # 航向角，0 -> pai -> -pai -> 0
                       Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]] # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        self.Plane_Irtifa = Plane_state["position"][1]
        self.Aircraft_Loc = Plane_state["position"]
        self.Oppo_Loc = Oppo_state["position"]

        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = Plane_state["target_locked"]
        if self.n_Ally_target_locked == True: #
            locked = 1
        else:
            locked = -1

        # Opponent Locked
        self.Oppo_target_locked = self.n_Oppo_target_locked
        self.n_Oppo_target_locked = Oppo_state["target_locked"]
        if self.n_Oppo_target_locked:  #
            oppo_locked = 1
        else:
            oppo_locked = -1

        # target_angle = Plane_state['target_angle'] / 180
        target_angle = Plane_state['target_angle']
        self.target_angle = target_angle

        # dx, dz, dy
        # Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], Plane_Pos[1] - Oppo_Pos[1], Plane_Pos[2] - Oppo_Pos[2]]
        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0], Oppo_Pos[1] - Plane_Pos[1], Oppo_Pos[2] - Plane_Pos[2]]

        self.oppo_health = df.get_health(self.Plane_ID_oppo)

        oppo_hea = self.oppo_health['health_level'] # 敌机初始血量为20

        # if self.now_missile_state == True:
        #     if_fire = 1
        # else:
        #     if_fire = -1

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)

        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] # 更新导弹1是否存在

        # update
        self.missile_handler.update()
        # vectorize
        missile_state = self.missile_handler.get_state()

        if self.n_missile1_state == True:

            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
                                #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用

        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)

        States = np.concatenate((Pos_Diff, Plane_Euler, target_angle, locked, missile1_state, Oppo_Euler, oppo_hea, Plane_Pos, Oppo_Pos, Plane_Heading,Plane_state), axis=None)
        # print("States:", States)
        # States = np.concatenate(States, missile_state)

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array([plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2]])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array([plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2]])

    def save_parameters_to_txt(self, log_dir):
        # os.makedirs(log_dir)
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)


        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, 'w', encoding= 'utf-8') as file:
            file.write(source_code1)
            file.write(' ')
            file.write(source_code2)
            file.write(' ')
            file.write(source_code3)

    # for expert data

    def get_loc_diff(self, state):
        loc_diff = ((((state[0]) * 10000) ** 2) + (((state[1]) * 10000) ** 2) + (((state[2]) * 10000) ** 2)) ** (1 / 2)
        return loc_diff

    def get_reward(self, state, action, n_state):
        reward = 0
        step_success = 0
        loc_diff = self.get_loc_diff(n_state)  # get location difference information for reward

        # 距离惩罚：帮助追击
        reward -= (0.0001 * loc_diff)

        # 目标角惩罚：帮助锁敌
        reward -= (n_state[6]) * 10

        # 开火奖励：帮助开火
        if action[-1] > 0: # 如果导弹发射
            reward -= 8
            if state[8] > 0 and state[7] < 0: # 且导弹存在、不锁敌
                # reward -= 100 # 4
                step_success = -1
            elif state[8] > 0 and state[7] > 0: # 且导弹存在、锁敌
                # reward += 8 # 100
                step_success = 1
            else:
                # reward -= 10 # 1
                reward -= 0

        if n_state[-1] < 0.1:
            reward += 600

        return reward, step_success

    def get_termination(self, state):
        done = False
        if state[-1] <= 0.1: # 敌机血量低于0则结束
            done = True
        return done

class MissileHandler():
    def __init__(self):
        pass


class HarfangSerpentineEnv(HarfangEnv):
    def __init__(self):
        super(HarfangSerpentineEnv, self).__init__()

    def set_ennemy_yaw(self):
        self.serpentine_step += 1

        if self.serpentine_step % self.duration == 0:
            self.serpentine_step = 0
            # 切换偏航方向（正负交替）
            self.oppo_yaw = 0.1 * (-1 if self.oppo_yaw > 0 else 1)
            self.duration = 500 # 300

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(self.oppo_yaw))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()

        # self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 1) #


            self.now_missile_state = True # 此时导弹发射
            # print("fire")
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2 # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2 # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.oppo_yaw = 0
        self.serpentine_step = 0
        # 初始偏航角
        self.oppo_yaw = -0.1
        self.duration = 250 # 150

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2) # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_oppo)

class HarfangCircularEnv(HarfangEnv):
    def __init__(self):
        super(HarfangCircularEnv, self).__init__()

    def set_ennemy_yaw(self):
        self.circular_step += 1

        if self.circular_step < 100:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.02))
            df.set_plane_roll(self.Plane_ID_oppo, float(0.84))
        else:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.01))
        df.set_plane_roll(self.Plane_ID_oppo, float(0.28))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        # if self.circular_step < 1000:
        #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.02))
        #     df.set_plane_roll(self.Plane_ID_ally, float(0.81))
        # else:
        #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.01))
        # df.set_plane_roll(self.Plane_ID_ally, float(0.27))
        # df.set_plane_yaw(self.Plane_ID_ally, float(0))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()

        if float(action_ally[3] > 0): # 大于0发射导弹
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True # 此时导弹发射
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.circular_step = 0

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2) # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)

class HarfangSerpentineInfiniteEnv(HarfangSerpentineEnv):
    def __init__(self):
        super(HarfangSerpentineInfiniteEnv, self).__init__()
        self.infinite_total_success = 0
        self.infinite_total_fire = 0
        self.infinite_total_step = 0

    def step_test(self, action):
        self.infinite_total_step += 1
        if self.infinite_total_step % 60 == 0:
            df.rearm_machine(self.Plane_ID_ally)
        # df.set_health("ennemy_2", 1) # 恢复敌机健康值

        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success

    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        self._get_loc_diff()  # get location difference information for reward

        # 距离惩罚：帮助追击
        self.reward -= (0.0001 * (self.loc_diff)) # 0.4

        # 目标角惩罚：帮助锁敌
        self.reward -= (self.target_angle)*10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4

        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        # 开火奖励：帮助开火
        if self.now_missile_state == True: # 如果此step导弹发射
            self.reward -= 8
            if self.missile1_state == True and self.Ally_target_locked == False: # 且导弹存在、不锁敌
                # self.reward -= 100 # 4、4
                self.infinite_total_fire += 1
                self.success = -1
                print('failed to fire')
            elif self.missile1_state == True and self.Ally_target_locked == True: # 且导弹存在且锁敌
                # self.reward += 8 # 100、4
                print('successful to fire')
                self.infinite_total_fire += 1
                self.infinite_total_success += 1
                self.success = 1
                self.fire_success = True
            else:
                # self.reward -= 10
                self.reward -= 0

        # 坠落奖励（最终奖励）
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600 # 无、无
            print('enemy have fallen')
class HarfangSmoothZigzagEnemyEnv(HarfangEnv):
    def __init__(self):
        super().__init__()
        self.zigzag_t = 0

    def set_ennemy_smooth_zigzag(self):
        self.zigzag_t += 1
        # Sinüzoidal heading ve roll hareketi
        roll = 0.7 * np.sin(self.zigzag_t / 12)
        yaw = 0.12 * np.cos(self.zigzag_t / 12)
        df.set_plane_pitch(self.Plane_ID_oppo, 0.03)
        df.set_plane_roll(self.Plane_ID_oppo, roll)
        df.set_plane_yaw(self.Plane_ID_oppo, yaw)

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        self.set_ennemy_smooth_zigzag()
        if float(action_ally[3]) > 0:
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False
        df.update_scene()

    def _reset_machine(self):
        super()._reset_machine()
        self.zigzag_t = 0

    def _random_reset_machine(self):
        super()._random_reset_machine()
        self.zigzag_t = 0


class HarfangDoctrineEnemyEnv(HarfangEnv):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.tactic = "neutral"
        self.last_dist = None

    def select_tactic(self, agent_pos, oppo_pos, agent_vel, oppo_vel):
        # Mesafe, irtifa ve yaklaşma hızına göre taktik seç
        dx = oppo_pos[0] - agent_pos[0]
        dy = oppo_pos[1] - agent_pos[1]
        dz = oppo_pos[2] - agent_pos[2]
        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        alt_diff = oppo_pos[1] - agent_pos[1]
        closing_speed = np.dot(agent_vel - oppo_vel, np.array([dx, dy, dz]) / (dist + 1e-5))

        # Yakınsan "break turn" veya "barrel roll", ortadaysan "yo-yo", uzaksan enerji koru
        if dist < 1000:
            if abs(alt_diff) < 300:
                self.tactic = "break_turn"
            else:
                self.tactic = "barrel_roll"
        elif dist < 2000:
            self.tactic = "yo_yo"
        else:
            self.tactic = "energy_management"
        # Random bazen evasive (kaçıngan) manevra da seçebilir.
        if np.random.rand() < 0.07:
            self.tactic = "notch"

    def doctrine_maneuver(self, agent_state, oppo_state):
        # Pozisyon ve hızdan komutlar üret
        agent_pos = agent_state["position"]
        oppo_pos = oppo_state["position"]
        agent_vel = np.array(agent_state["linear_speed"])
        oppo_vel = np.array(oppo_state["linear_speed"])
        self.select_tactic(agent_pos, oppo_pos, agent_vel, oppo_vel)
        t = self.step_count

        if self.tactic == "break_turn":
            # Ani heading ve roll değişimi: Savunma dönüşü
            roll = 0.95 * np.sign(np.random.randn())
            yaw = 0.23 * np.sign(np.random.randn())
            pitch = -0.18
        elif self.tactic == "barrel_roll":
            roll = 0.80 * np.sin(t / 7)
            yaw = 0.10 * np.cos(t / 7)
            pitch = 0.16 * np.sin(t / 4)
        elif self.tactic == "yo_yo":
            # High yo-yo: Hız kaybetmemek için tırman ve heading değiştir
            roll = 0.7 * np.sin(t / 12)
            yaw = 0.18 * np.cos(t / 18)
            pitch = 0.10 + 0.10 * np.sin(t / 10)
        elif self.tactic == "energy_management":
            # Sabit yükseklikte hızını koru, arada hafif roll-yaw
            roll = 0.25 * np.sin(t / 18)
            yaw = 0.08 * np.sin(t / 9)
            pitch = 0.04
        elif self.tactic == "notch":
            # Radar’dan ve füze kilidinden kaçmak için: Yaw büyük, roll az
            roll = 0.09
            yaw = 0.34 * np.sign(np.random.randn())
            pitch = -0.03
        else:
            roll = yaw = pitch = 0.0
        return pitch, roll, yaw

    def set_ennemy_doctrine(self):
        self.step_count += 1
        agent_state = df.get_plane_state(self.Plane_ID_ally)
        oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        pitch, roll, yaw = self.doctrine_maneuver(agent_state, oppo_state)
        df.set_plane_pitch(self.Plane_ID_oppo, pitch)
        df.set_plane_roll(self.Plane_ID_oppo, roll)
        df.set_plane_yaw(self.Plane_ID_oppo, yaw)

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        self.set_ennemy_doctrine()
        if float(action_ally[3]) > 0:
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False
        df.update_scene()

    def _reset_machine(self):
        super()._reset_machine()
        self.step_count = 0
        self.tactic = "neutral"
        self.last_dist = None

    def _random_reset_machine(self):
        super()._random_reset_machine()
        self.step_count = 0
        self.tactic = "neutral"
        self.last_dist = None


class HarfangTacticalEnv(HarfangEnv):
    """
    Gerçekçi hava savaşı doktrinleri uygulayan akıllı opponent ile dogfight ortamı.

    Uygulanan Taktikler:
    - BVR (Beyond Visual Range) taktikleri
    - Energy Management (Enerji Yönetimi)
    - Defensive/Offensive manevralar
    - Multi-phase combat logic
    - Threat assessment
    """

    def __init__(self):
        super(HarfangTacticalEnv, self).__init__()

        # Taktiksel durum değişkenleri
        self.combat_phase = "BVR"  # BVR, MERGE, DOGFIGHT, DEFENSIVE, OFFENSIVE
        self.threat_level = 0.0  # 0-1 arası tehdit seviyesi
        self.energy_state = "HIGH"  # HIGH, MEDIUM, LOW
        self.last_maneuver = "NONE"
        self.maneuver_timer = 0
        self.evasion_timer = 0
        self.offensive_timer = 0

        # Pozisyon ve geometri analizi
        self.range_to_target = 0
        self.aspect_angle = 0  # Hedefin burun açısı
        self.antenna_train_angle = 0  # Radar anteni açısı
        self.closure_rate = 0  # Yaklaşma hızı

        # Taktiksel parametreler
        self.BVR_RANGE = 8000  # BVR savaş mesafesi
        self.MERGE_RANGE = 2000  # Birleşme mesafesi
        self.DEFENSIVE_RANGE = 1000  # Savunma mesafesi
        self.MIN_ALTITUDE = 1500  # Minimum güvenli irtifa
        self.MAX_ALTITUDE = 8000  # Maksimum irtifa

        # Performans limitleri (gerçekçi F-16 değerleri)
        self.MAX_G = 9.0
        self.CORNER_SPEED = 250  # Köşe hızı (optimal dönüş)
        self.MACH_LIMIT = 350  # Hız limiti

    def _apply_action(self, action_ally):
        """Müttefik uçak aksiyonunu uygula ve düşman AI taktiklerini çalıştır"""

        # Müttefik uçak kontrolü
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        # Füze ateşleme
        if float(action_ally[3] > 0):
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False

        # Düşman AI taktiksel karar verme
        self._execute_tactical_ai()

        df.update_scene()

    def _execute_tactical_ai(self):
        """Ana taktiksel AI karar verme motoru"""

        # Durum analizi
        self._analyze_tactical_situation()

        # Savaş fazını belirle
        self._determine_combat_phase()

        # Taktiksel aksiyonu seç ve uygula
        if self.combat_phase == "BVR":
            self._execute_bvr_tactics()
        elif self.combat_phase == "MERGE":
            self._execute_merge_tactics()
        elif self.combat_phase == "DEFENSIVE":
            self._execute_defensive_tactics()
        elif self.combat_phase == "OFFENSIVE":
            self._execute_offensive_tactics()
        else:  # DOGFIGHT
            self._execute_dogfight_tactics()

    def _analyze_tactical_situation(self):
        """Taktiksel durumu analiz et"""

        # Mevcut pozisyonları al
        ally_pos = np.array(self.Aircraft_Loc)
        enemy_pos = np.array(self.Oppo_Loc)

        # Mesafe ve geometri hesapla
        self.range_to_target = np.linalg.norm(ally_pos - enemy_pos)

        # Aspect angle hesapla (düşmanın burun açısı)
        ally_state = df.get_plane_state(self.Plane_ID_ally)
        enemy_state = df.get_plane_state(self.Plane_ID_oppo)

        # Relatif pozisyon vektörü
        rel_pos = ally_pos - enemy_pos

        # Düşman uçağın heading vektörü
        enemy_heading_rad = math.radians(enemy_state["heading"])
        enemy_forward = np.array([math.sin(enemy_heading_rad), 0, math.cos(enemy_heading_rad)])

        # Aspect angle (0-180 derece)
        dot_product = np.dot(rel_pos, enemy_forward)
        self.aspect_angle = math.degrees(math.acos(np.clip(dot_product /
                                                           (np.linalg.norm(rel_pos) * np.linalg.norm(enemy_forward)),
                                                           -1.0, 1.0)))

        # Tehdit seviyesi hesapla
        self._calculate_threat_level()

        # Enerji durumu analizi
        self._analyze_energy_state()

    def _calculate_threat_level(self):
        """Tehdit seviyesini hesapla (0-1 arası)"""

        threat = 0.0

        # Mesafe faktörü (yakın = tehlikeli)
        if self.range_to_target < 1500:
            threat += 0.4
        elif self.range_to_target < 3000:
            threat += 0.2

        # Aspect angle faktörü (nose-on saldırı tehlikeli)
        if self.aspect_angle < 45:
            threat += 0.3
        elif self.aspect_angle < 90:
            threat += 0.1

        # Müttefik füze durumu
        if self.Ally_target_locked:
            threat += 0.3

        self.threat_level = min(1.0, threat)

    def _analyze_energy_state(self):
        """Enerji durumunu analiz et (altitude + speed)"""

        enemy_state = df.get_plane_state(self.Plane_ID_oppo)
        altitude = enemy_state["position"][1]

        # Basit enerji durumu sınıflandırması
        if altitude > 5000:
            self.energy_state = "HIGH"
        elif altitude > 3000:
            self.energy_state = "MEDIUM"
        else:
            self.energy_state = "LOW"

    def _determine_combat_phase(self):
        """Savaş fazını belirle"""

        if self.range_to_target > self.BVR_RANGE:
            self.combat_phase = "BVR"
        elif self.range_to_target > self.MERGE_RANGE:
            if self.threat_level > 0.6:
                self.combat_phase = "DEFENSIVE"
            else:
                self.combat_phase = "MERGE"
        elif self.threat_level > 0.7:
            self.combat_phase = "DEFENSIVE"
        elif self.aspect_angle < 60 and self.range_to_target < 1500:
            self.combat_phase = "OFFENSIVE"
        else:
            self.combat_phase = "DOGFIGHT"

    def _execute_bvr_tactics(self):
        """BVR (Menzil Ötesi) taktikleri"""

        # Yüksek irtifada kalmaya çalış (enerji avantajı)
        if self.Oppo_Loc[1] < 6000:
            self._climb_maneuver()

        # Müttefike doğru yönelmek yerine açısal yaklaşım
        self._notch_maneuver()  # 90 derece açıyla yaklaş

        # Füze menzili içindeyse ateş et
        if self.range_to_target < 6000 and self.n_Oppo_target_locked:
            self._fire_missile_if_available()

    def _execute_merge_tactics(self):
        """Birleşme fazı taktikleri"""

        if self.aspect_angle < 90:
            # Head-on yaklaşım - son anda yan kaçış
            if self.range_to_target < 1000:
                self._defensive_split()
            else:
                self._maintain_course()
        else:
            # Yan/arkadan yaklaşım - saldırı pozisyonu al
            self._get_attack_position()

    def _execute_defensive_tactics(self):
        """Savunma taktikleri"""

        self.evasion_timer += 1

        if self.evasion_timer < 100:
            self._barrel_roll()  # Varil dönüşü
        elif self.evasion_timer < 200:
            self._defensive_spiral()  # Savunma spirali
        elif self.evasion_timer < 300:
            self._notch_maneuver()  # Notch manevrası
        else:
            self._split_s()  # Split-S manevrası
            self.evasion_timer = 0

    def _execute_offensive_tactics(self):
        """Saldırı taktikleri"""

        self.offensive_timer += 1

        if self.aspect_angle > 120:  # Arkadan yaklaşım
            self._pursuit_curve()  # Takip eğrisi
        else:
            self._lead_pursuit()  # Öncül takip

        # Ateş pozisyonundaysa füze at
        if (self.range_to_target < 2000 and
                self.aspect_angle > 150 and
                self.n_Oppo_target_locked):
            self._fire_missile_if_available()

    def _execute_dogfight_tactics(self):
        """Yakın mesafe köpek dövüşü taktikleri"""

        self.maneuver_timer += 1

        # Enerji durumuna göre taktik seç
        if self.energy_state == "HIGH":
            if self.maneuver_timer % 200 < 100:
                self._high_yo_yo()  # Yüksek yo-yo
            else:
                self._immelman_turn()  # Immelman dönüşü
        elif self.energy_state == "MEDIUM":
            if self.maneuver_timer % 150 < 75:
                self._barrel_roll()
            else:
                self._scissors()  # Makas manevrası
        else:  # LOW energy
            self._defensive_spiral()  # Enerji kazanmaya çalış

    # === MANEUVER IMPLEMENTATIONS ===

    def _climb_maneuver(self):
        """Tırmanış manevrası"""
        df.set_plane_pitch(self.Plane_ID_oppo, -0.3)
        df.set_plane_roll(self.Plane_ID_oppo, 0.0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.0)

    def _notch_maneuver(self):
        """Notch manevrası (90 derece açıyla uçuş)"""
        df.set_plane_pitch(self.Plane_ID_oppo, 0.0)
        df.set_plane_roll(self.Plane_ID_oppo, 0.0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.4)

    def _barrel_roll(self):
        """Varil dönüşü manevrası"""
        df.set_plane_pitch(self.Plane_ID_oppo, 0.1)
        df.set_plane_roll(self.Plane_ID_oppo, 0.8)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.0)

    def _defensive_spiral(self):
        """Savunma spirali"""
        df.set_plane_pitch(self.Plane_ID_oppo, -0.2)
        df.set_plane_roll(self.Plane_ID_oppo, 0.6)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.2)

    def _split_s(self):
        """Split-S manevrası"""
        df.set_plane_pitch(self.Plane_ID_oppo, 0.5)
        df.set_plane_roll(self.Plane_ID_oppo, 1.0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.0)

    def _high_yo_yo(self):
        """Yüksek yo-yo manevrası"""
        df.set_plane_pitch(self.Plane_ID_oppo, -0.4)
        df.set_plane_roll(self.Plane_ID_oppo, 0.3)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.1)

    def _immelman_turn(self):
        """Immelman dönüşü"""
        df.set_plane_pitch(self.Plane_ID_oppo, -0.6)
        df.set_plane_roll(self.Plane_ID_oppo, 0.0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.0)

    def _scissors(self):
        """Makas manevrası"""
        turn_direction = 1 if (self.maneuver_timer // 50) % 2 == 0 else -1
        df.set_plane_pitch(self.Plane_ID_oppo, 0.0)
        df.set_plane_roll(self.Plane_ID_oppo, 0.7 * turn_direction)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.3 * turn_direction)

    def _pursuit_curve(self):
        """Takip eğrisi (pure pursuit)"""
        # Hedefin mevcut pozisyonuna doğru yönel
        df.set_plane_pitch(self.Plane_ID_oppo, 0.1)
        df.set_plane_roll(self.Plane_ID_oppo, 0.2)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.1)

    def _lead_pursuit(self):
        """Öncül takip (lead pursuit)"""
        # Hedefin gideceği noktaya yönel
        df.set_plane_pitch(self.Plane_ID_oppo, 0.0)
        df.set_plane_roll(self.Plane_ID_oppo, 0.4)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.2)

    def _defensive_split(self):
        """Savunma ayrımı"""
        df.set_plane_pitch(self.Plane_ID_oppo, 0.2)
        df.set_plane_roll(self.Plane_ID_oppo, 0.9)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.4)

    def _maintain_course(self):
        """Rotayı koru"""
        df.set_plane_pitch(self.Plane_ID_oppo, 0.0)
        df.set_plane_roll(self.Plane_ID_oppo, 0.0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.0)

    def _get_attack_position(self):
        """Saldırı pozisyonu al"""
        df.set_plane_pitch(self.Plane_ID_oppo, -0.1)
        df.set_plane_roll(self.Plane_ID_oppo, 0.3)
        df.set_plane_yaw(self.Plane_ID_oppo, 0.2)

    def _fire_missile_if_available(self):
        """Müsait olduğunda füze ateşle"""
        df.fire_missile(self.Plane_ID_oppo, 0)


    def _reset_machine(self):
        """Makineleri sıfırla - taktiksel AI için optimize edilmiş başlangıç"""

        # Temel reset
        super()._reset_machine()

        # Taktiksel değişkenleri sıfırla
        self.combat_phase = "BVR"
        self.threat_level = 0.0
        self.energy_state = "HIGH"
        self.last_maneuver = "NONE"
        self.maneuver_timer = 0
        self.evasion_timer = 0
        self.offensive_timer = 0

        # Düşman uçağı daha güçlü yap (denk rakip)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)  # Daha yüksek thrust
        df.set_plane_linear_speed(self.Plane_ID_oppo, 280)  # Daha yüksek hız
        df.set_health(self.Plane_ID_oppo, 0.5)  # Daha fazla sağlık

    def _random_reset_machine(self):
        """Rastgele pozisyonla sıfırla"""

        # Taktiksel değişkenleri sıfırla
        self.combat_phase = "BVR"
        self.threat_level = 0.0
        self.energy_state = "HIGH"
        self.maneuver_timer = 0
        self.evasion_timer = 0
        self.offensive_timer = 0

        # Uçakları sıfırla
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")

        # Düşman daha güçlü ve rastgele pozisyonda
        enemy_x = random.randint(-200, 200)
        enemy_y = random.randint(4000, 6000)  # Yüksek başlangıç (enerji avantajı)
        enemy_z = random.randint(-200, 200)

        ally_x = random.randint(-100, 100)
        ally_y = random.randint(3000, 4000)
        ally_z = random.randint(-4200, -3800)

        df.reset_machine_matrix(self.Plane_ID_oppo, enemy_x, enemy_y, enemy_z, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, ally_x, ally_y, ally_z, 0, 0, 0)

        # Denk performans parametreleri
        df.set_plane_thrust(self.Plane_ID_ally, 1.0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.85)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 285)
        df.set_health(self.Plane_ID_oppo, 0.5)
        self.oppo_health = 0.5

        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def get_tactical_info(self):
        """Taktiksel bilgileri döndür (debug/analiz için)"""
        return {
            'combat_phase': self.combat_phase,
            'threat_level': self.threat_level,
            'energy_state': self.energy_state,
            'range_to_target': self.range_to_target,
            'aspect_angle': self.aspect_angle,
            'last_maneuver': self.last_maneuver
        }



