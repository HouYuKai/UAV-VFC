# -*- encoding: utf-8 -*-
'''
@File    :   uav.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import math
import numpy as np
from .FogNodeBase import FogNodeBase
class UAV(FogNodeBase):
    def __init__(self, uid, height, start_position, start_direction, velocity, cpu, reputation_score = 100, cheat_possibility = 0, init_revenue = 0,power_capacity = 1000):
        super().__init__(uid, start_position, reputation_score, cpu, cheat_possibility)
        self.height = height
        self.direction = start_direction
        self.velocity = velocity
        self.power_capacity = power_capacity # 当前剩余的电量KMh
        self.type_name = 'UAV'
        self.total_revenues = init_revenue
        self.flied_distance = 0
        self.activated = True
        
    def update_direction(self, direction):
        self.direction = direction # 2pi

    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_position(self, time_step):
        # if self.power_capacity > 0:
        org_pos = self.position.copy()
        self.position = [self.position[0] + np.cos(self.direction) * self.velocity * time_step, self.position[1] + np.sin(self.direction) * self.velocity * time_step]
        self.flied_distance += np.sqrt((self.position[0] - org_pos[0])**2 + (self.position[1] - org_pos[1])**2)
        self.power_capacity -= self.velocity**2 / 1000 + 1 # 简化能量模型
        return True
        # else:
        #     return False


    
