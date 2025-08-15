# HarfangEnv_GYM.py
#from ctypes.macholib.dyld import dyld_env

import numpy as np
from numpy.array_api import uint8
#from numpy.distutils.system_info import default_x11_lib_dirs

import harfang_env.environments.dogfight_client as df
from harfang_env.environments.constants import *
import gym
import os
import inspect
import random
import math
from numpy.array_api import uint8
import re
import time

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
        self.ally_health = 1
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
        self.missile_handler.refresh_missiles()
        state_ally = self._get_observation()  # get observations
        state_oppo = self._get_enemy_observation()
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile
        df.set_target_id(self.Plane_ID_oppo, self.Plane_ID_ally)  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally # 当前时刻状态
        self.oppo_state = state_oppo


        return state_ally, state_oppo

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
        self.missile_handler.refresh_missiles()
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

    def step(self, action, oppo_action):
        self._apply_action(action, oppo_action)  # apply neural networks output
        self.missile_handler.refresh_missiles()
        n_state = self._get_observation()  # 执行动作后的状态
        n_state_oppo = self._get_enemy_observation()
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self.oppo_state = n_state_oppo
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done,n_state_oppo, {}, self.success

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

    def _apply_action(self, action_ally, action_enemy):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0) #
            self.now_missile_state = True # 此时导弹发射
            # print("fire")
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
        if self.ally_health['health_level'] < 1.0:
            self.done = True
        # if self.now_missile_state == True:
        #     self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1") # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        df.set_health("ally_1", 1)
        self.oppo_health = 0.2 #
        self.ally_health = 1
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
        self.ally_health = df.get_health(self.Plane_ID_ally)
        ally_hea = self.ally_health['health_level']
        oppo_hea = self.oppo_health['health_level'] # 敌机初始血量为20

        # if self.now_missile_state == True:
        #     if_fire = 1
        # else:
        #     if_fire = -1

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)

        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] # 更新导弹1是否存在

        # update missiles
        self.missile_handler.refresh_missiles()
        # vectorize missiles
        missile_vec = self.get_enemy_missile_vector()



        if self.n_missile1_state == True:

            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
                                #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用

        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)

        States = np.concatenate((Pos_Diff, Plane_Euler, target_angle, locked, missile1_state, Oppo_Euler, oppo_hea, Plane_Pos, Oppo_Pos, Plane_Heading, ally_hea, Plane_Pitch_Att), axis=None)
        # print("States:", States)

        # missile_vec = state[22]
        States = np.concatenate((States, missile_vec))

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def _get_enemy_observation(self):
        # Enemy POV: enemy'nin gözünden ally'nin göreceli durumu
        Plane_state = df.get_plane_state(self.Plane_ID_oppo)  # enemy (oppo) kendisi
        Oppo_state = df.get_plane_state(self.Plane_ID_ally)  # ally (bizim uçak)

        # Pozisyonlar (normalize)
        Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                     Plane_state["position"][1] / NormStates["Plane_position"],
                     Plane_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]

        # Euler açıları (normalize)
        Plane_Euler = [Plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       Plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        # Heading/pitch/roll (normalize)
        # Plane_Heading = Plane_state["heading"]  # already degrees
        # Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Plane_Heading = Plane_state["heading"]  # enemy'nin heading'i (degree)
        Plane_Pitch_Att = Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Plane_Roll_Att = Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        # Enemy kontrol seviyeleri (opsiyonel)
        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]

        # Target lock durumları
        n_Oppo_target_locked = Plane_state["target_locked"]
        locked = 1 if n_Oppo_target_locked else -1

        n_Ally_target_locked = Oppo_state["target_locked"]
        oppo_locked = 1 if n_Ally_target_locked else -1

        # target_angle: enemy'nin ally'e göre açısı
        target_angle = Plane_state["target_angle"]

        # Pos_Diff = [Ally - Enemy] (enemy'nin gözünden, ally'nin göreli pozisyonu)
        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0],
                    Oppo_Pos[1] - Plane_Pos[1],
                    Oppo_Pos[2] - Plane_Pos[2]]

        # ally'nun enemy'ye attığı füzelerle ilgili vektörler
        missile_vec = self.get_ally_missile_vector()

        # Enemy'nin kendisinin füzesi var mı?
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_oppo)
        missile1_state = 1 if Missile_state["missiles_slots"][0] else -1

        # Enemy'nin (oppo) sağlığı (health)
        oppo_health = df.get_health(self.Plane_ID_oppo)  # bu fonksiyon kendisi için çağrıldı
        oppo_hea = oppo_health['health_level']

        # Ally'nin pozisyonu ve Euler açıları
        # (isteğe bağlı: RL input formatında istersen, çıkarabilirsin veya bırakarak direkt simetri sağlarsın)

        # STATES BİRLEŞTİRME (aynı sırayla, kendi observation’un gibi)
        States = np.concatenate((
            Pos_Diff,  # 0-2: Ally'nin enemy'ye göre konumu (dx, dy, dz)
            Plane_Euler,  # 3-5: Enemy'nin kendi Euler açıları
            [target_angle],  # 6  : Enemy'nin ally'e göre açısı
            [locked],  # 7  : Enemy target lock
            [missile1_state],  # 8  : Enemy missile state
            Oppo_Euler,  # 9-11: Ally'nin Euler açıları
            [oppo_hea],  # 12 : Enemy'nin health'i
            Plane_Pos,  # 13-15: Enemy'nin kendi pozisyonu
            Oppo_Pos,  # 16-18: Ally'nin pozisyonu
            [Plane_Heading],  # 19 : Enemy'nin heading'i
        ), axis=None)
        # missile_vec = state[20]
        States = np.concatenate((States, missile_vec))
        return States

    def get_enemy_missile_vector(self):
        """
        Returns a list of dicts for all current enemy missiles:
        [
          {
            'missile_id': ...,
            'position': [x, y, z],
            'heading': ...,
            # optionally add more (velocity, life_time, etc)
          }, ...
        ]
        """
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.enemy_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                        # add more fields if you want
                    })
            except Exception:
                pass  # or log
        return missile_info_list

    def get_ally_missile_vector(self):
        """
        Returns a list of dicts for all current ally missiles:
        [
          {
            'missile_id': ...,
            'position': [x, y, z],
            'heading': ...,
            # optionally add more (velocity, life_time, etc)
          }, ...
        ]
        """
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.ally_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                        # add more fields if you want
                    })
            except Exception:
                pass  # or log
        return missile_info_list

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

# []
class MissileHandler:
    MISSILE_TYPES = ["AIM_SL", "Meteor", "Karaoke", "Mica", "S400", "Sidewinder"]

    def __init__(self):
        # Takip edilen missilelar (sahnedeki objeler), slotlara göre gruplanmış olarak tutulabilir
        self.ally_id = "ally_1"
        self.enemy_id = "ennemy_2"
        self.ally_slots = []
        self.enemy_slots = []
        self.ally_missiles = set()
        self.enemy_missiles = set()
        self.refresh_missiles()

    def refresh_missiles(self):
        """ Slot ve missile listelerini günceller, wreck olanları temizler. """
        # Ally
        self.ally_slots = df.get_machine_missiles_list(self.ally_id)
        self.ally_slot_states = df.get_missiles_device_slots_state(self.ally_id).get("missiles_slots", [])
        self.ally_missiles = self._get_current_missiles("ally")

        # Enemy
        self.enemy_slots = df.get_machine_missiles_list(self.enemy_id)
        self.enemy_slot_states = df.get_missiles_device_slots_state(self.enemy_id).get("missiles_slots", [])
        self.enemy_missiles = self._get_current_missiles("ennemy")

        # Aktif missile objelerinden wreck olanları filtrele
        self.ally_missiles = set([mid for mid in self.ally_missiles if not self.is_missile_wreck(mid)])
        self.enemy_missiles = set([mid for mid in self.enemy_missiles if not self.is_missile_wreck(mid)])

        # print(f"current missiles: {self.enemy_missiles}")

    def _get_current_missiles(self, side_str):
        """ Tüm sahnedeki missile ID'lerinden sadece ally/enemy olanları ve tipte olanları döndürür. """
        missiles = df.get_missiles_list()
        filtered = []
        for mid in missiles:
            if any(t in mid for t in self.MISSILE_TYPES) and mid.startswith(side_str):
                filtered.append(mid)
        return set(filtered)

    @staticmethod
    def slotid_to_missileid(slot_id):
        # ally_1AIM_SL0 -> ally_1-AIM_SL-0
        m = re.match(r"(\w+?)_?(AIM_SL|Meteor|Karaoke|Mica|S400|Sidewinder)(\d+)", slot_id)
        if m:
            prefix, mtype, num = m.groups()
            return f"{prefix}-{mtype}-{num}"
        return slot_id.replace("_", "-")

    @staticmethod
    def missileid_to_slotid(missile_id):
        # ally_1-AIM_SL-0 -> ally_1AIM_SL0
        parts = missile_id.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}{parts[1]}{parts[2]}"
        return missile_id.replace("-", "")

    def missile_slot_status(self, plane_id):
        """ Hangi slotta hangi missile var ve wreck mi? """
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        results = []
        for idx, slot_id in enumerate(slots):
            present = slot_state[idx] if idx < len(slot_state) else False
            missile_id_guess = self.slotid_to_missileid(slot_id)
            missile_state = None
            wreck = None
            pos = None
            if present:
                try:
                    missile_state = df.get_missile_state(missile_id_guess)
                    wreck = missile_state.get("wreck", None)
                    pos = missile_state.get("position", None)
                except Exception:
                    wreck = None
            results.append({
                "slot_idx": idx,
                "slot_id": slot_id,
                "missile_id": missile_id_guess,
                "slot_active": present,
                "wreck": wreck,
                "position": pos
            })
        return results

    def is_missile_wreck(self, missile_id):
        """ Missile wreck mi? """
        try:
            missile_state = df.get_missile_state(missile_id)
            return missile_state.get("wreck", False)
        except Exception:
            return True

    def track_all_missiles(self, side="ally", steps=10, print_missing=True):
        """ Aktif missileların pozisyonunu N adım boyunca yaz. """
        self.refresh_missiles()
        missile_ids = list(self.ally_missiles) if side == "ally" else list(self.enemy_missiles)
        if not missile_ids:
            print(f"[{side.upper()}] Aktif missile yok.")
            return
        for mid in missile_ids:
            for i in range(steps):
                try:
                    state = df.get_missile_state(mid)
                    pos = state.get("position", None)
                    wreck = state.get("wreck", None)
                    if pos is not None and wreck is not True:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
                    else:
                        if print_missing:
                            print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (wreck/position yok)")
                        break
                except Exception:
                    if print_missing:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (missile state alınamadı)")
                    break


    def missile_summary(self):
        """ Anlık özet: kaç aktif missile, wreck slot var? """
        self.refresh_missiles()
        print(f"ALLY missiles: {sorted(self.ally_missiles)}")
        print(f"ENEMY missiles: {sorted(self.enemy_missiles)}")
        print("Ally slot state:", self.missile_slot_status(self.ally_id))
        print("Enemy slot state:", self.missile_slot_status(self.enemy_id))



class SimpleEnemy(HarfangEnv):
    """
    Minimal air combat simulation for testing evade behavior.
    The enemy aircraft fires a missile unconditionally if within 3000 meters.
    No maneuvering or tactical behavior is implemented.
    """

    def __init__(self):
        super(SimpleEnemy, self).__init__()
        self.has_fired = False  # Ensures one-time fire per threshold breach

    def _apply_action(self, action_ally, action_enemy):
        """Apply ally actions and run minimal enemy logic."""
        # Apply ally control inputs
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        ally_missile_vec = self.get_ally_missile_vector()
        # ally_1 ile başlayan missile'lar
        ally_missile_list = [
            m for m in ally_missile_vec
            if m['missile_id'].startswith("ally_1")
        ]

        # Ateş edilmemişler (pozisyon 0,0,0) → sadece missile_id
        ally_unfired_missiles = [
            m['missile_id'] for m in ally_missile_list
            if m['position'] == [0.0, 0.0, 0.0]
        ]

        # Ateş edilmişler (pozisyonu 0,0,0 olmayanlar) → sadece missile_id
        ally_fired_missiles = [
            m['missile_id'] for m in ally_missile_list
            if m['position'] != [0.0, 0.0, 0.0]
        ]
        # [track:0 , evade:1, climb:2, fire:3]
        # Meteor atışı
        # if float(action_ally[3]) == 0:  # Meteor
        #     meteor_slots = [
        #         int(m.split('-')[-1])
        #         for m in ally_unfired_missiles
        #         if "Meteor" in m
        #     ]
        #     if meteor_slots:  # Ateş edilmemiş Meteor varsa
        #         slot = min(meteor_slots)
        #         df.fire_missile(self.Plane_ID_ally, slot)
        #     else:
        #         print("[INFO] Ateşlenecek Meteor yok.")

        # AIM_SL atışı
        # if float(action_ally[3]) == 1:  # AIM_SL
        #     aim_sl_slots = [
        #         int(m.split('-')[-1])
        #         for m in ally_unfired_missiles
        #         if "AIM_SL" in m
        #     ]
        #     if aim_sl_slots:  # Ateş edilmemiş AIM_SL varsa
        #         slot = min(aim_sl_slots)
        #         df.fire_missile(self.Plane_ID_ally, slot)
        #     else:
        #         print("[INFO] Ateşlenecek AIM_SL yok.")

        oppo_missile_vec = self.get_enemy_missile_vector()
        # ennemy_2 ile başlayan missile'lar
        oppo_missile_list = [
            m for m in ally_missile_vec
            if m['missile_id'].startswith("ennemy_2")
        ]

        # Ateş edilmemişler (pozisyon 0,0,0) → sadece missile_id
        oppo_unfired_missiles = [
            m['missile_id'] for m in ally_missile_list
            if m['position'] == [0.0, 0.0, 0.0]
        ]

        # Ateş edilmişler (pozisyonu 0,0,0 olmayanlar) → sadece missile_id
        oppo_fired_missiles = [
            m['missile_id'] for m in ally_missile_list
            if m['position'] != [0.0, 0.0, 0.0]
        ]

        # Enemy Ateşleme mantığı: en küçük slot numarasındaki füzeyi seç
        if float(action_enemy[3]) > 0 and oppo_unfired_missiles:
            # missile_id formatı: "ennemy_2-<type>-<slot>"
            oppo_slots = [
                int(m.split('-')[-1]) for m in oppo_unfired_missiles
            ]
            slot = min(oppo_slots)
            df.fire_missile(self.Plane_ID_oppo, slot)
            print(" === enemy fired missile! ===")


        # Finalize step
        df.update_scene()

    def _reset_machine(self):
        """Reset simulation to a standard test configuration."""
        super()._reset_machine()
        self.has_fired = False

        # Position aircraft in test formation
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3000, -4000, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 5000, 0, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1.0)
        df.set_plane_thrust(self.Plane_ID_oppo, 1.0)

        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 300)

        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def get_debug_info(self):
        """Return distance and missile firing status for debug/analysis."""
        ally_pos = np.array(self.Aircraft_Loc)
        enemy_pos = np.array(self.Oppo_Loc)
        distance = np.linalg.norm(ally_pos - enemy_pos)
        return {
            'distance_to_ally': distance,
            'missile_fired': self.has_fired
        }



