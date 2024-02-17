# -*- encoding: utf-8 -*-
'''
@File    :   vehicle.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''




import numpy as np

from .FogNodeBase import FogNodeBase

class Vehicle(FogNodeBase):
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, id, start_position, start_direction, velocity, cpu, serving = True, task_lambda = 0, init_revenue = 0, init_score = 0, cheat_possibility = 0):
        super().__init__(id, start_position, init_score, cpu, cheat_possibility)
        self.direction = start_direction
        self.velocity = velocity
        self.task_lambda = task_lambda
        self.computing_res_alloc = [] # 分配计算资源策略,每个step更新（task_queue中的任务，分配的资源x）
        self.serving = serving 
        self.assigned_to = -1
        self.total_revenues = init_revenue # 钱包，记录的是自身有多少的奖励
        self.type_name = 'Vehicle'
        self.neighbor_vehicles = []
        self.reward = 0
        self.pCPU = 0
        self.served_last_period = False

    def update_position(self, position):
        # assert self.is_running
        self.position = position
    
    def update_velocity(self, velocity):
        # assert self.is_running
        self.velocity = velocity

    def update_direction(self, direction):
        # assert self.is_running
        self.direction = direction # 360 degree
    
    def update_time(self, time):
        # assert self.is_running
        self.time = time
        self.time = round(self.time, 1)