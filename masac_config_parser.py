# config_parser.py

import argparse

model_maddpg = ['MADDPG_1', 'MADDPG_2']
model_mappo = ['MAPPO', 'MAPPO2', 'MAPPO3', 'MAPPO4']
model_KM = ['KMeans', 'BaseKMeans', 'KM2']
model_pos_PPO = ['Pos_MAPPO','Pos_MAPPO2','Pos_MAPPO3','Pos_MAPPO4','Pos_MAPPO_test']
model_area_pos_PPO = ['AreaPos_MAPPO']
method = 'Pos_MAPPO3'
print('method: ', method)
assert method in model_maddpg + model_mappo + model_KM + model_pos_PPO+model_area_pos_PPO
def parse_arguments_for_MASAC():
    parser = argparse.ArgumentParser(description='示例程序')
    # 添加参数
    # 1. sumo文件所在地址的参数 
    parser.add_argument('--img_path', type=str, help='存储icon的文件夹路径', default= "/home/weizhiwei/data/uav_compute/python_240102/icon")
    parser.add_argument('--sumocfg_path', type=str, help='sumocfg文件路径', default= "/home/weizhiwei/data/uav_compute/sumo_berlin/map.sumocfg")
    parser.add_argument('--osm_path', type=str, help='osm文件路径', default= "/home/weizhiwei/data/uav_compute/sumo_berlin/map.osm")
    parser.add_argument('--net_path', type=str, help='net文件路径', default= "/home/weizhiwei/data/uav_compute/sumo_berlin/map.net.xml")
    parser.add_argument('--tensorboard_writer_file', type=str, help='tensorboard_writer_file', default= f"/home/weizhiwei/data/uav_compute/python_240102/{method}")
    parser.add_argument('--saved_path', type=str, help='saved_path', default= f"/home/weizhiwei/data/uav_compute/python_240102/{method}_models")

    # 2. 仿真环境的参数 
    parser.add_argument('--random_seed', type=int, help='随机种子', default=42)
    parser.add_argument('--n_iter', type=int, help='迭代次数', default=100)
    parser.add_argument('--n_episode', type=int, help='每次迭代的episode数，等价于sumo重启的次数', default=30)
    parser.add_argument('--max_steps', type=int, help='每个episode的最大步数', default=200)
    parser.add_argument('--n_veh', type=int, help='车辆数', default=120)
    parser.add_argument('--n_serving_veh', type=int, help='服务车辆数', default=60)
    parser.add_argument('--TTI_length', type=float, help='TTI的长度 (s)', default=0.05)
    parser.add_argument('--fading_threshold', type=float, help='fading_threshold', default=130)
    parser.add_argument('--UAV_communication_range', type=float, help='UAV_communication_range', default=300)
    parser.add_argument('--RSU_communication_range', type=float, help='RSU_communication_range', default=500)
    parser.add_argument('--v_neighbor_Veh', type=int, help='v_neighbor_Veh', default=20)
    parser.add_argument('--V2V_RB', type=int, help='V2V_band', default=6)
    parser.add_argument('--V2U_RB', type=int, help='V2U_band', default=6)
    parser.add_argument('--V2I_RB', type=int, help='V2I_band', default=8)
    #load
    parser.add_argument('--load_terminated', type=bool, default=False)
    parser.add_argument('--load_without_training', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--fre_to_draw', type=int, default=50)
    parser.add_argument('--fre_to_save', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=600 if method in model_mappo+model_pos_PPO+model_area_pos_PPO else 512)
    parser.add_argument('--start_to_train', type=int, default=600 if method in model_mappo+model_pos_PPO+model_area_pos_PPO else 512)
    parser.add_argument('--update_every', type=int, default=200 if method in model_mappo+model_pos_PPO+model_area_pos_PPO else 200)


    # 3. MADRL
    parser.add_argument('--graph_hidden',type = int, default=128)
    parser.add_argument('--n_hidden',type = int, default=512)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--memory_size', type=int, default=600 if method in model_mappo+model_pos_PPO+model_area_pos_PPO else 10000)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--omega', type=float, default=0.) # 个体奖励和协作奖励的比例，0代表只有个体奖励，1代表只有协作奖励

    # MAPPO
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--agreement_iter', type=int, default=1) # 代表在take action之前进行agreement的次数，1代表不进行agreement

    # PosMAPPO
    parser.add_argument('--grid_width', type=int, default=50) # 代表网格的宽度
    parser.add_argument('--grid_num_x', type=int, default=30) # 代表x网格的数量
    parser.add_argument('--grid_num_y', type=int, default=20) # 代表y网格的数量
    parser.add_argument('--K_history', type=int, default=1) # 代表UAV获取的历史信息的数量
    parser.add_argument('--ava_selected', type=int, default=4) # 代表可以选择的区域数量，从中选择一个区域进行探索
    
    # print parser中的参数
    args = parser.parse_args()
    print(args)
    # 解析命令行参数
    return args
