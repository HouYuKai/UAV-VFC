# -*- encoding: utf-8 -*-
'''
@File    :   visualization_tkinter.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''

from xmlrpc.client import Error
from uvfogsim.algorithms.KM_Area_Module import KM_Area_Module
from uvfogsim.vehicle_manager import VehicleManager
import curses
import threading
import traci
import subprocess
from PIL import ImageGrab
import sys
import math
from uvfogsim.environment import Environment
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import time
from uvfogsim.tkinter_utils import *
from uvfogsim.algorithms import set_seed
import math
# from uvfogsim.algorithms.Cluster_Algorithm_Module import Cluster_Algorithm_Module
from CNN_Position_MARL_Algorithm import Pos_CNN_MAPPO
from uvfogsim.utils.my_generator import random_colors_generator
from uvfogsim.utils.randomTrips import RandomEdgeGenerator
from masac_config_parser import parse_arguments_for_MASAC, method, model_maddpg, model_mappo, model_KM, model_pos_PPO, model_area_pos_PPO
class DRLEnvironmentWrapper():
    def __init__(self, traci_connection, args):
        self.args = args
        self.traci_connection = traci_connection
        self.environment = None
        self.algorithm_module = None
        self.sumocfg_file = args.sumocfg_path
        self.osm_file_path = args.osm_path
        self.net_file_path = args.net_path
        self.step_per_episode = args.max_steps
        self.iteration_episodes = args.n_episode
        self.n_iteration = args.n_iter
        self.old_vehicle_position_dict = {}
        self.time_step = None
        self.bbox = None
        self.location_bound = None
        self.cur_episode = 0
        self.cur_iter = 0
        self.cur_step = 0
        self.calculate_bbox()

        self.n_veh = args.n_veh
        self.map_data = None
        self.simulation_delay = 0
        
    def calculate_bbox(self):
        # 从net.xml文件中读取地图的bbox,通过parse_location_info函数
        conv_boundary, orig_boundary, proj_params, netOffset = parse_location_info(self.net_file_path)
        orig_boundary = tuple(map(float, orig_boundary.split(',')))
        conv_boundary = tuple(map(float, conv_boundary.split(',')))
        min_x = orig_boundary[0]
        min_y = orig_boundary[1]
        max_x = orig_boundary[2]
        max_y = orig_boundary[3] 
        self.proj_params = proj_params
        self.netOffset = netOffset
        self.bbox = min_x, min_y, max_x, max_y
        self.location_bound = conv_boundary
    def update_vehicle_positions(self, vehicle_ids):
        row_data_dict = {}
        for vehicle_id in vehicle_ids:
            x, y = self.traci_connection.vehicle.getPosition(vehicle_id)
            row_data = {}
            row_data['id'] = int(vehicle_id)
            row_data['x'] = x
            row_data['y'] = y
            row_data['angle'] = self.traci_connection.vehicle.getAngle(vehicle_id)
            row_data['speed'] = self.traci_connection.vehicle.getSpeed(vehicle_id)
            row_data['speedFactor'] = self.traci_connection.vehicle.getSpeedFactor(vehicle_id)
            row_data_dict[int(vehicle_id)] = row_data
        return row_data_dict

        
    def run(self):
        sumo_cmd = ["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", self.sumocfg_file]
        sumo_process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
        self.traci_connection.init(8813, host='127.0.0.1', numRetries=10)
        self.time_step = self.traci_connection.simulation.getDeltaT()
        print("仿真过程中决策的时隙等于SUMO仿真的时隙长度: ", self.time_step)
        env = Environment(args = self.args,draw_it = False, n_UAV = 4, time_step = self.time_step, TTI = self.args.TTI_length)
        env.initialize(self.location_bound)
        if method in model_pos_PPO:
            self.algorithm_module = Pos_CNN_MAPPO(env, args)
        elif method in model_KM:
            self.algorithm_module = KM_Area_Module(env, args)
        else:
            raise Error("method error")
        self.environment = env
        manager = VehicleManager(self.n_veh, self.net_file_path) # 指定200辆车
        cnt = 0
        self.traci_connection.simulationStep()
        isGCN_MARL = method in model_maddpg + model_mappo+model_pos_PPO + model_area_pos_PPO
        self.isGCN_MARL = isGCN_MARL
        for iteration in range(self.n_iteration):
            self.cur_iter = iteration
            cnt = 0
            env.initialize(self.location_bound)
            # manager.turn_off_traffic_lights(self.traci_connection)
            while cnt <= self.iteration_episodes * self.step_per_episode:
                self.cur_step = cnt % self.step_per_episode
                self.cur_episode = cnt // self.step_per_episode
                step_start_time = time.time()
                # 0. 每一个episode开始之前，调整初始化，以及车辆数量或最大服务车辆数量
                if cnt % (self.step_per_episode) == 0:
                    env.initialize(self.location_bound)
                    num_flag = np.random.randint(0, 5)  - 2 #  -2, -1, 0, 1, 2
                    env.max_serving_vehicles = 60 + num_flag * 10
                    self.algorithm_module.reset_state()
                    # manager.set_vehicle_number(self.n_veh)
                    # if cnt % (5 * self.step_per_episode) == 0:
                    manager.clear_all_vehicles(self.traci_connection)
                # 1 每个time_step进行，控制区域内车辆数量
                manager.manage_vehicles(self.traci_connection)
                self.traci_connection.simulationStep() 
                vehicle_ids = self.traci_connection.vehicle.getIDList()
                # 1.1 控制车辆在canvas显示的位置
                row_data_dict = self.update_vehicle_positions(vehicle_ids)
                # 1.2 更新车辆在仿真内的位置
                removed_vehicle_ids = env.renew_veh_positions(vehicle_ids, row_data_dict)
                env.FastFading_Module()  
                # 1.2.1 更新UAV和车辆之间的相对方向
                env.update_UAV_Veh_direct_M()
                # 1.2.2 执行agent所有的动作，包含UAV的移动和接入决策
                if isGCN_MARL:
                    self.algorithm_module.take_agent_action(act_area=self.cur_step%1==0) 
                # 1.3 算法得到UAV的方向和速度
                uav_directions, uav_speeds = self.algorithm_module.act_mobility(env)
                # 1.4 更新UAV的位置,flags表示是否到达边界
                flags = env.renew_uav_positions(uav_directions, uav_speeds)
                # 2 根据位移信息，更新通信信道状态
                env.FastFading_Module()  
                # 3 任务生成
                env.Task_Generation_Module()
                # 4 通过算法获取卸载决策（每个time_step进行，一次性卸载当前step内所有的to_offload_tasks）
                task_path_dict_list = self.algorithm_module.act_offloading(env)
                TTI_flag = False
                while not TTI_flag:
                    # 5 执行任务卸载（每个TTI进行）
                    task_path_dict_list = env.Offload_Tasks(task_path_dict_list) # 只offload task.start_time == cur_time的任务
                    # 每一个TTI都需要执行RB和计算的资源分配
                    activated_offloading_tasks_with_RB_Nos = self.algorithm_module.act_RB_allocation(env)
                    env.Communication_RB_Allocation(activated_offloading_tasks_with_RB_Nos)
                    env.Compute_Rate()
                    env.Execute_Communicate()
                    cpu_allocation_for_fog_nodes = self.algorithm_module.act_CPU_allocation(env)
                    env.Execute_Compute(cpu_allocation_for_fog_nodes)
                    # 11 更新环境状态
                    TTI_flag = env.Time_Update_Module()
                self.algorithm_module.calculate_reward()
                if isGCN_MARL:
                    self.algorithm_module.store_experience()
                    if method in model_mappo+model_pos_PPO+model_area_pos_PPO:
                        self.algorithm_module.update_agents()
                    else:
                        self.algorithm_module.update_agents()
                else:
                    self.algorithm_module.print_result(env)
                # 12 检查超时的任务（包括计算和验算，以及to_pay）
                env.Check_To_X_Tasks()
                self.simulation_delay = time.time() - step_start_time
                cnt += 1
            self.traci_connection.close()
            time.sleep(5) # 等待一秒，确保sumo进程关闭
            self.traci_connection.start(sumo_cmd)
            manager.reset()
import torch
if __name__ == "__main__":
    torch.set_num_threads(6)
    args = parse_arguments_for_MASAC()
    set_seed(args.random_seed)
    app = DRLEnvironmentWrapper(traci, args)
    try:
        app.run()
    # 抓捕keyboard interrupt 和 runtime exception
    except (KeyboardInterrupt) as e:
        print('KeyboardInterrupt, stopping...')
        if app.isGCN_MARL:
            app.algorithm_module.save_agents(terminated=True)
        print("Exiting program.")