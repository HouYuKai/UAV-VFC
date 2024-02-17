from cProfile import label
from cmath import log
from collections import deque
from this import d
from uvfogsim.algorithms.Base_Algorithm_Module import Base_Algorithm_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from uvfogsim.algorithms.WHO_algorithm import WHO
from uvfogsim.uav import UAV
from uvfogsim.bs import BS
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt
import copy

def Node_Matrix_to_Graph(node_features, adjacency_matrix):
    # node_features: [n_nodes, state_dim]
    # adjacency_matrix: [n_nodes, n_nodes]
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.cpu().detach().numpy()
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()
    edge_idx = np.nonzero(adjacency_matrix)
    edge_index = np.array(edge_idx)
    # 判断edge_index是不是2行，如果不是，需要转置
    if edge_index.shape[0] != 2:
        edge_index = edge_index.transpose(0, 1)
        
    edge_attr = np.array(adjacency_matrix[edge_index[0,:], edge_index[1,:]])
    graph = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(), edge_attr=torch.tensor(edge_attr, dtype=torch.float32))
    return graph


class COMA_Actor(nn.Module):
    def __init__(self, n_hiddens, actor_lr, device, args):
        super(COMA_Actor, self).__init__()
        self.args = args
        self.actor_lr = actor_lr
        self.device = device
        self.n_hiddens = n_hiddens
        self.grid_num_x = args.grid_num_x
        self.grid_num_y = args.grid_num_y
        self.n_agent = 6
        
        self.map_encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Resblock(8, 8),
            Resblock(8, 8),
            nn.Flatten(),
            nn.Linear(8*self.grid_num_x * self.grid_num_y, self.n_hiddens)
        ).to(self.device)
        self.gru_encoder = nn.GRU(self.n_hiddens, self.n_hiddens, batch_first=True).to(self.device)
        # 输出的是mean和std
        self.output = nn.Sequential(
            nn.Linear(self.n_hiddens+1+self.n_agent + 5*self.n_agent + 50, self.n_hiddens // 2), # 输入agent_id以及各个agent的权重
            nn.Tanh(),
            nn.Linear(self.n_hiddens // 2, 2) 
        ).to(self.device)
        self.output_optimizer = torch.optim.Adam(self.output.parameters(), lr=self.actor_lr, eps=1e-5)
        self.map_encoder_optimizer = torch.optim.Adam(self.map_encoder.parameters(), lr=self.actor_lr, eps=1e-5)
        self.gru_encoder_optimizer = torch.optim.Adam(self.gru_encoder.parameters(), lr=self.actor_lr, eps=1e-5)
        self.epsilon = 0.8

    def opt_zero_grad(self):
        self.output_optimizer.zero_grad()
        self.map_encoder_optimizer.zero_grad()
        self.gru_encoder_optimizer.zero_grad()
    def opt_step(self):
        nn.utils.clip_grad_norm_(self.output.parameters(), 0.5)
        self.output_optimizer.step()
        nn.utils.clip_grad_norm_(self.map_encoder.parameters(), 0.5)
        self.map_encoder_optimizer.step()
        nn.utils.clip_grad_norm_(self.gru_encoder.parameters(), 0.5)
        self.gru_encoder_optimizer.step()
        self.epsilon = max(0.05, self.epsilon * 0.95)

    def forward(self, map_history, agent_id, all_agent_weights, map_encoding_history, all_agent_weight_histroy, agent_position_vector):
        # map_history: [batch_size, K_history * 2, grid_num, grid_num]
        # agent_id: [batch_size]
        batch_size = map_history.shape[0]
        map_history = map_history.reshape((batch_size, -1, self.grid_num_x, self.grid_num_y))
        agent_position_vector = agent_position_vector.reshape((batch_size, -1))
        all_agent_weight_histroy = all_agent_weight_histroy.reshape((batch_size, -1))
        cur_map_hidden = self.map_encoder(map_history)
        # gru_encoder
        map_encoding_history = map_encoding_history.reshape((batch_size, -1, self.n_hiddens))
        # map_hidden_history作为h0，[num_layers * num_directions, batch_size, n_hiddens]，需要转置
        map_encoding_history = map_encoding_history.transpose(0, 1)
        cur_map_hidden = cur_map_hidden.unsqueeze(1) # [batch_size, length=1, n_hiddens]
        _, map_hidden = self.gru_encoder(cur_map_hidden, map_encoding_history) # map_hidden是h_n，[num_layers * num_directions, batch_size, n_hiddens]
        map_hidden = map_hidden.squeeze(0) # [batch_size, n_hiddens]
        action_input = torch.cat([map_hidden, agent_id, all_agent_weights, all_agent_weight_histroy, agent_position_vector], dim=-1)
        output = self.output(action_input)
        output = output.reshape((batch_size, 2)) # mean, log_std
        # 放缩到[0,1]
        alpha = F.softplus(output[:,0]) + 1.0
        beta = F.softplus(output[:,1]) + 1.0
        return alpha, beta, map_hidden.detach()
    
    def take_action(self, map_history, agent_id, all_agent_weights, action_mask, map_encoding_history, all_agent_weight_histroy, agent_position_vector):
        map_history = map_history.unsqueeze(0) # [1, 4, 30, 20]
        map_encoding_history = map_encoding_history.unsqueeze(0) # [1, 1, 256]
        agent_id = agent_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
        all_agent_weight_histroy = all_agent_weight_histroy.unsqueeze(0) # [1, 6, 5]
        all_agent_weights = all_agent_weights.unsqueeze(0) # [batch_size, n_agent]
        agent_position_vector = agent_position_vector.unsqueeze(0) # [1, 50]
        # 由于问题规模，这里mean不需要进行scale，直接使用即可；log_std需要scale 10倍，再进行clip,进行exp
        alpha, beta, map_hidden = self.forward(map_history, agent_id, all_agent_weights, map_encoding_history, all_agent_weight_histroy, agent_position_vector) # [3]
        beta_dist = torch.distributions.Beta(alpha, beta) 
        weight_value = beta_dist.sample() # range [0,1] 
        log_prob = beta_dist.log_prob(weight_value)
        weight_value = weight_value * 4 # range [0,4]
        # 求sqrt
        # weight_value = torch.sqrt(weight_value)
        return weight_value, log_prob, map_hidden

    def save_agent(self, path, id):
        torch.save(self.output.state_dict(), os.path.join(path, '_output_' + str(id) + '.pth'))
        torch.save(self.map_encoder.state_dict(), os.path.join(path, '_act_map_encoder_' + str(id) + '.pth'))
        torch.save(self.gru_encoder.state_dict(), os.path.join(path, '_act_gru_encoder_' + str(id) + '.pth'))

    def load_agent(self, path, id):
        if os.path.exists(os.path.join(path, '_output_' + str(id) + '.pth')) and not self.args.load_terminated:
            self.output.load_state_dict(torch.load(os.path.join(path, '_output_' + str(id) + '.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, '_act_map_encoder_' + str(id) + '.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, '_act_gru_encoder_' + str(id) + '.pth')))
        else:
            self.output.load_state_dict(torch.load(os.path.join(path, 'terminated', '_output_' + str(id) + '.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', '_act_map_encoder_' + str(id) + '.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', '_act_gru_encoder_' + str(id) + '.pth')))
        # self.output.eval()
        # self.map_encoder.eval()
        # self.gru_encoder.eval()
class Resblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu((self.conv1(x)))
        out = out + residual
        out = self.relu(out)
        return out
class PosPPO_Actor(nn.Module):
    def __init__(self, n_hiddens, actor_lr, device, args):
        super(PosPPO_Actor, self).__init__()
        self.n_hiddens = n_hiddens
        self.args = args
        self.output_dim = 5 # 5, 表示方向
        self.actor_lr = actor_lr
        self.device = device
        self.grid_num_x = args.grid_num_x
        self.grid_num_y = args.grid_num_y
        # 先输入一个CNN，对地图（batch_size, 2, self.grid_num, self.grid_num）输入，输出一个map_dim维度的向量
        # 网络架构为conv2d->relu->3*resblock->relu->flatten
        self.map_encoder = nn.Sequential( # 输入[4, 30, 20]
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # 输出[8, 30, 20]
            nn.ReLU(), # 输出[8, 30, 20]
            Resblock(8, 8), # 输出[8, 30, 20]
            Resblock(8, 8),
            nn.Flatten(), # 输出[8*30*20]
            nn.Linear(8*self.grid_num_x * self.grid_num_y, self.n_hiddens)
        ).to(self.device)
        self.gru_encoder = nn.GRU(self.n_hiddens, self.n_hiddens, batch_first=True).to(self.device)
        self.actor = nn.Sequential(
            nn.Linear(50 + self.n_hiddens, self.n_hiddens // 2),
            nn.ReLU(),
            nn.Linear(self.n_hiddens // 2, self.output_dim)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.map_encoder_optimizer = torch.optim.Adam(self.map_encoder.parameters(), lr=self.actor_lr, eps=1e-5)
        self.gru_encoder_optimizer = torch.optim.Adam(self.gru_encoder.parameters(), lr=self.actor_lr, eps=1e-5)
        self.epsilon = 0.8
    def forward(self, agent_map_for_CNN, cur_UAV_pos, uav_mobility_mask, map_encoding_history):
        # agent_map_for_CNN: [batch_size, 1, grid_num, grid_num]
        # cur_UAV_pos: [batch_size, n_UAV, 2]
        # uav_mobility_mask: [batch_size, n_UAV, output_dim]
        batch_size = agent_map_for_CNN.shape[0]
        cur_map_hidden = self.map_encoder(agent_map_for_CNN)
        # gru_encoder
        map_encoding_history = map_encoding_history.reshape((batch_size, -1, self.n_hiddens))
        # map_hidden_history作为h0，[num_layers * num_directions, batch_size, n_hiddens]，需要转置
        map_encoding_history = map_encoding_history.transpose(0, 1)
        cur_map_hidden = cur_map_hidden.unsqueeze(1) # [batch_size, length=1, n_hiddens]
        _, map_hidden = self.gru_encoder(cur_map_hidden, map_encoding_history) # map_hidden是h_n，[num_layers * num_directions, batch_size, n_hiddens]
        map_hidden = map_hidden.squeeze(0) # [batch_size, n_hiddens]
        action_input = torch.concat([map_hidden, cur_UAV_pos], dim=-1)
        output = self.actor(action_input)
        output = output.reshape((batch_size, self.output_dim))
        # uav_mobility_mask是{0,1}的，需要把0的地方,在output置为-1e5
        output = torch.where(uav_mobility_mask == 0, torch.tensor(-1e5).to(self.device), output)
        return output, map_hidden.detach()

    def save_agent(self, path, agent_id):
        torch.save(self.actor.state_dict(), os.path.join(path, f'_uav_actor_{agent_id}.pth'))
        torch.save(self.map_encoder.state_dict(), os.path.join(path, f'_map_encoder_{agent_id}.pth'))
        torch.save(self.gru_encoder.state_dict(), os.path.join(path, f'_gru_encoder_{agent_id}.pth'))
    def load_agent(self, path, agent_id):
        if os.path.exists(os.path.join(path, '_uav_actor.pth')) and not self.args.load_terminated:
            self.actor.load_state_dict(torch.load(os.path.join(path, f'_uav_actor_{agent_id}.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, f'_map_encoder_{agent_id}.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, f'_gru_encoder_{agent_id}.pth')))
        else:
            self.actor.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_uav_actor_{agent_id}.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_map_encoder_{agent_id}.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_gru_encoder_{agent_id}.pth')))
        # self.actor.eval()
        # self.map_encoder.eval()
        # self.gru_encoder.eval()
    def opt_zero_grad(self):
        self.actor_optimizer.zero_grad()
        self.map_encoder_optimizer.zero_grad()
        self.gru_encoder_optimizer.zero_grad()
    def opt_step(self):
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        nn.utils.clip_grad_norm_(self.map_encoder.parameters(), 0.5)
        self.map_encoder_optimizer.step()
        nn.utils.clip_grad_norm_(self.gru_encoder.parameters(), 0.5)
        self.gru_encoder_optimizer.step()
        self.epsilon = max(0.05, self.epsilon * 0.95)
    def take_action(self, agent_map_for_CNN, cur_UAV_pos, uav_mobility_mask, map_hidden_history):
        # cur_UAV_pos (tensor): [2]，先乘以20，再取整，然后x,y分别映射到[30],[20]的独热向量
        cur_UAV_pos = cur_UAV_pos * 20
        cur_UAV_pos = cur_UAV_pos.long()
        # 限制范围
        cur_UAV_pos[0] = torch.clamp(cur_UAV_pos[0], 0, 29)
        cur_UAV_pos[1] = torch.clamp(cur_UAV_pos[1], 0, 19)
        cur_UAV_pos_x_one_hot = torch.zeros(30).to(self.device)
        cur_UAV_pos_y_one_hot = torch.zeros(20).to(self.device)
        cur_UAV_pos_x_one_hot[cur_UAV_pos[0]] = 1
        cur_UAV_pos_y_one_hot[cur_UAV_pos[1]] = 1
        cur_UAV_pos = torch.cat([cur_UAV_pos_x_one_hot, cur_UAV_pos_y_one_hot], dim=-1)
        # 1. unsqueeze
        agent_map_for_CNN = agent_map_for_CNN.unsqueeze(0)
        cur_UAV_pos = cur_UAV_pos.unsqueeze(0)
        map_hidden_history = map_hidden_history.unsqueeze(0)
        uav_mobility_mask = uav_mobility_mask.unsqueeze(0)
        output, map_hidden = self.forward(agent_map_for_CNN, cur_UAV_pos, uav_mobility_mask, map_hidden_history) # [n_UAV, 5]
        output_value = output.cpu().detach().numpy().squeeze(0) # [5]
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            # 概率需要把uav_mobility_mask中为0的地方设置为0
            np_p = uav_mobility_mask.cpu().detach().numpy().squeeze(0)
            np_p = np_p / np.sum(np_p)
            output_action = np.random.choice(5, p=np_p)
        else:
            output_action = np.argmax(output_value)
        return output_action, map_hidden

class COMA_critic(nn.Module):
    def __init__(self, n_hiddens, critic_lr, device, args):
        super(COMA_critic, self).__init__()
        self.n_hiddens = n_hiddens
        self.args = args
        self.output_dim = 5 # 5, 表示方向
        self.critic_lr = critic_lr
        self.device = device
        self.grid_num_x = args.grid_num_x
        self.grid_num_y = args.grid_num_y
        # 先输入一个CNN，对地图（batch_size, 2, self.grid_num, self.grid_num）输入，输出一个map_dim维度的向量
        
        self.map_encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Resblock(8, 8),
            Resblock(8, 8),
            nn.Flatten(),
            nn.Linear(8*self.grid_num_x * self.grid_num_y, self.n_hiddens)
        ).to(self.device)
        self.gru_encoder = nn.GRU(self.n_hiddens, self.n_hiddens, batch_first=True).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(1 + self.n_hiddens + 30 + 6 * 50, self.n_hiddens // 2),
            nn.Tanh(),
            nn.Linear(self.n_hiddens // 2, 1) # 输出一个值,作为value
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
        self.map_encoder_optimizer = torch.optim.Adam(self.map_encoder.parameters(), lr=self.critic_lr, eps=1e-5)
        self.gru_encoder_optimizer = torch.optim.Adam(self.gru_encoder.parameters(), lr=self.critic_lr, eps=1e-5)

    def forward(self, map_history, agent_id, map_encoding_history, all_agent_weights, all_agent_positions):
        # map_history: [batch_size, K_history * 2, grid_num, grid_num]
        # agent_id: [batch_size]
        batch_size = map_history.shape[0]
        map_history = map_history.reshape((batch_size, -1, self.grid_num_x, self.grid_num_y))
        all_agent_positions = all_agent_positions.reshape((batch_size, -1))
        cur_map_hidden = self.map_encoder(map_history)
        # gru_encoder
        map_encoding_history = map_encoding_history.reshape((batch_size, -1, self.n_hiddens))
        # map_hidden_history作为h0，[num_layers * num_directions, batch_size, n_hiddens]，需要转置
        map_encoding_history = map_encoding_history.transpose(0, 1)
        cur_map_hidden = cur_map_hidden.unsqueeze(1) # [batch_size, length=1, n_hiddens]
        _, map_hidden = self.gru_encoder(cur_map_hidden, map_encoding_history) # map_hidden是h_n，[num_layers * num_directions, batch_size, n_hiddens]
        map_hidden = map_hidden.squeeze(0)
        action_input = torch.cat([map_hidden, agent_id, all_agent_weights, all_agent_positions], dim=-1)
        output = self.critic(action_input)
        return output

    def save_agent(self, path, agent_id):
        torch.save(self.critic.state_dict(), os.path.join(path, '_critic_' + str(agent_id) + '.pth'))
        torch.save(self.map_encoder.state_dict(), os.path.join(path, '_critic_map_encoder_' + str(agent_id) + '.pth'))
        torch.save(self.gru_encoder.state_dict(), os.path.join(path, '_critic_gru_encoder_' + str(agent_id) + '.pth'))

    def load_agent(self, path, agent_id):
        if os.path.exists(os.path.join(path, f'_critic_{agent_id}.pth')) and not self.args.load_terminated:
            self.critic.load_state_dict(torch.load(os.path.join(path, f'_critic_{agent_id}.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, f'_critic_map_encoder_{agent_id}.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, f'_critic_gru_encoder_{agent_id}.pth')))
        else:
            self.critic.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_critic_{agent_id}.pth')))
            self.map_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_critic_map_encoder_{agent_id}.pth')))
            self.gru_encoder.load_state_dict(torch.load(os.path.join(path, 'terminated', f'_critic_gru_encoder_{agent_id}.pth')))
        # self.critic.eval()
        # self.map_encoder.eval()
        # self.gru_encoder.eval()

    def opt_zero_grad(self):
        self.critic_optimizer.zero_grad()
        self.map_encoder_optimizer.zero_grad()
        self.gru_encoder_optimizer.zero_grad()

    def opt_step(self):
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        nn.utils.clip_grad_norm_(self.map_encoder.parameters(), 0.5)
        self.map_encoder_optimizer.step()
        nn.utils.clip_grad_norm_(self.gru_encoder.parameters(), 0.5)
        self.gru_encoder_optimizer.step()

from torch_geometric.data import Data, Batch
class PosReplayBuffer():
    # 存储last_map_history, cur_map, last_pos, cur_pos, reward, action_target_grid, action_log_probs
    def __init__(self, memory_size, k_history):
        self.memory_size = memory_size
        self.cur_pointer = 0 
        self.last_map_history = []
        self.cur_map = []
        self.agent_rewards = []
        self.agent_weights = []
        self.agent_log_probs = []
        self.recent_rewards = deque([0],maxlen=200)
        self.weight_actions = []
        self.agent_weight_masks = []
        self.last_map_encoding_history = []
        self.cur_map_encoding_history = []
        self.last_agent_weight_history_for_train = []
        self.cur_agent_weight_history_for_train = []
        self.last_agent_positions = []
        self.cur_agent_positions = []
    def avg_rewards(self):
        return np.mean(self.recent_rewards)
        
    def add_experience(self, last_map_history, cur_map, agent_weights, agent_log_probs, agent_rewards, agent_weight_actions, agent_weight_masks, last_map_encoding_history, cur_map_encoding_history, last_agent_weight_history_for_train, cur_agent_weight_history_for_train, last_agent_positions, cur_agent_positions):
        self.recent_rewards.append(torch.sum(agent_rewards).detach().cpu().numpy())
        if len(self.last_map_history)>=self.memory_size:
            self.last_map_history[self.cur_pointer] = last_map_history
            self.cur_map[self.cur_pointer] = cur_map
            self.agent_weights[self.cur_pointer] = agent_weights
            self.agent_log_probs[self.cur_pointer] = agent_log_probs
            self.agent_rewards[self.cur_pointer] = agent_rewards
            self.weight_actions[self.cur_pointer] = agent_weight_actions
            self.agent_weight_masks[self.cur_pointer] = agent_weight_masks
            self.last_map_encoding_history[self.cur_pointer] = last_map_encoding_history
            self.cur_map_encoding_history[self.cur_pointer] = cur_map_encoding_history
            self.last_agent_weight_history_for_train[self.cur_pointer] = last_agent_weight_history_for_train
            self.cur_agent_weight_history_for_train[self.cur_pointer] = cur_agent_weight_history_for_train
            self.last_agent_positions[self.cur_pointer] = last_agent_positions
            self.cur_agent_positions[self.cur_pointer] = cur_agent_positions
            self.cur_pointer = (self.cur_pointer + 1) % self.memory_size
        else:
            self.last_map_history.append(last_map_history)
            self.cur_map.append(cur_map)
            self.agent_weights.append(agent_weights)
            self.agent_log_probs.append(agent_log_probs)
            self.agent_rewards.append(agent_rewards)
            self.weight_actions.append(agent_weight_actions)
            self.agent_weight_masks.append(agent_weight_masks)
            self.last_map_encoding_history.append(last_map_encoding_history)
            self.cur_map_encoding_history.append(cur_map_encoding_history)
            self.last_agent_weight_history_for_train.append(last_agent_weight_history_for_train)
            self.cur_agent_weight_history_for_train.append(cur_agent_weight_history_for_train)
            self.last_agent_positions.append(last_agent_positions)
            self.cur_agent_positions.append(cur_agent_positions)

    @property
    def size(self):
        return len(self.last_map_history)
    
    def sample(self, batch_size, shuffle=False):
        if shuffle:
            indices = np.random.choice(len(self.last_map_history), batch_size)
        else:
            indices = np.arange(batch_size)
        batch_last_map_history = [self.last_map_history[idx] for idx in indices]
        batch_cur_map = [self.cur_map[idx] for idx in indices]
        batch_agent_weights = [self.agent_weights[idx] for idx in indices]
        batch_agent_log_probs = [self.agent_log_probs[idx] for idx in indices]
        batch_agent_rewards = [self.agent_rewards[idx] for idx in indices]
        batch_weight_actions = [self.weight_actions[idx] for idx in indices]
        batch_agent_weight_masks = [self.agent_weight_masks[idx] for idx in indices]
        batch_last_map_encoding_history = [self.last_map_encoding_history[idx] for idx in indices]
        batch_cur_map_encoding_history = [self.cur_map_encoding_history[idx] for idx in indices]
        batch_last_agent_weight_history_for_train = [self.last_agent_weight_history_for_train[idx] for idx in indices]
        batch_cur_agent_weight_history_for_train = [self.cur_agent_weight_history_for_train[idx] for idx in indices]
        batch_cur_agent_positions = [self.cur_agent_positions[idx] for idx in indices]
        batch_last_agent_positions = [self.last_agent_positions[idx] for idx in indices]

        # 转换为tensor
        batch_last_map_history = torch.stack(batch_last_map_history, dim=0)
        batch_cur_map = torch.stack(batch_cur_map, dim=0)
        batch_agent_weights = torch.stack(batch_agent_weights, dim=0)
        batch_agent_log_probs = torch.stack(batch_agent_log_probs, dim=0)
        batch_agent_rewards = torch.stack(batch_agent_rewards, dim=0)
        batch_weight_actions = torch.stack(batch_weight_actions, dim=0)
        batch_agent_weight_masks = torch.stack(batch_agent_weight_masks, dim=0)
        batch_last_map_encoding_history = torch.stack(batch_last_map_encoding_history, dim=0)
        batch_cur_map_encoding_history = torch.stack(batch_cur_map_encoding_history, dim=0)
        batch_last_agent_weight_history_for_train = torch.stack(batch_last_agent_weight_history_for_train, dim=0)
        batch_cur_agent_weight_history_for_train = torch.stack(batch_cur_agent_weight_history_for_train, dim=0)
        batch_cur_agent_positions = torch.stack(batch_cur_agent_positions, dim=0)
        batch_last_agent_positions = torch.stack(batch_last_agent_positions, dim=0)
        return batch_last_map_history, batch_cur_map, batch_agent_weights, batch_agent_log_probs, batch_agent_rewards, batch_weight_actions, batch_agent_weight_masks, batch_last_map_encoding_history, batch_cur_map_encoding_history, batch_last_agent_weight_history_for_train, batch_cur_agent_weight_history_for_train, batch_last_agent_positions, batch_cur_agent_positions

class ReplayBuffer(): # 专门给UAV进行飞行的存储经验，是off-line的DQN
    def __init__(self, memory_size, n_agent):
        self.memory_size = memory_size
        self.cur_pointer = 0 
        # last_map_for_CNN, cur_map_for_CNN, last_pos, cur_pos, last_uav_mobility_mask, cur_uav_mobility_mask, uav_directions, rewards
        self.last_map_for_CNNs = []
        self.cur_map_for_CNNs = []
        self.last_poses = []
        self.cur_poses = []
        self.last_uav_mobility_masks = []
        self.cur_uav_mobility_masks = []
        self.uav_directions = []
        self.rewards = []
        self.map_encoding_history = []
        self.recent_rewards = deque([0],maxlen=200)

    def avg_rewards(self):
        return np.mean(self.recent_rewards)

    def add_experience(self, last_map_for_CNN, cur_map_for_CNN, last_pos, cur_pos, last_uav_mobility_mask, cur_uav_mobility_mask, uav_direction, reward, map_encoding_history):
        self.recent_rewards.append(reward.detach().cpu().numpy()[0])
        if len(self.last_map_for_CNNs)>=self.memory_size:
            self.last_map_for_CNNs[self.cur_pointer] = last_map_for_CNN
            self.cur_map_for_CNNs[self.cur_pointer] = cur_map_for_CNN
            self.last_poses[self.cur_pointer] = last_pos
            self.cur_poses[self.cur_pointer] = cur_pos
            self.last_uav_mobility_masks[self.cur_pointer] = last_uav_mobility_mask
            self.cur_uav_mobility_masks[self.cur_pointer] = cur_uav_mobility_mask
            self.uav_directions[self.cur_pointer] = uav_direction
            self.rewards[self.cur_pointer] = reward
            self.map_encoding_history[self.cur_pointer] = map_encoding_history
            self.cur_pointer = (self.cur_pointer + 1) % self.memory_size
        else:
            self.last_map_for_CNNs.append(last_map_for_CNN)
            self.cur_map_for_CNNs.append(cur_map_for_CNN)
            self.last_poses.append(last_pos)
            self.cur_poses.append(cur_pos)
            self.last_uav_mobility_masks.append(last_uav_mobility_mask)
            self.cur_uav_mobility_masks.append(cur_uav_mobility_mask)
            self.uav_directions.append(uav_direction)
            self.rewards.append(reward)
            self.map_encoding_history.append(map_encoding_history)

    @property
    def size(self):
        return len(self.last_map_for_CNNs)
    
    def sample(self, batch_size, shuffle=False):
        if shuffle:
            indices = np.random.choice(len(self.last_map_for_CNNs), batch_size)
        else:
            indices = np.arange(batch_size)
        batch_last_map_for_CNN = [self.last_map_for_CNNs[idx] for idx in indices]
        batch_cur_map_for_CNN = [self.cur_map_for_CNNs[idx] for idx in indices]
        batch_last_pos = [self.last_poses[idx] for idx in indices]
        batch_cur_pos = [self.cur_poses[idx] for idx in indices]
        batch_last_uav_mobility_mask = [self.last_uav_mobility_masks[idx] for idx in indices]
        batch_cur_uav_mobility_mask = [self.cur_uav_mobility_masks[idx] for idx in indices]
        batch_uav_direction = [self.uav_directions[idx] for idx in indices]
        batch_reward = [self.rewards[idx] for idx in indices]
        batch_map_encoding_history = [self.map_encoding_history[idx] for idx in indices]
        # 转换为tensor
        batch_last_map_for_CNN = torch.stack(batch_last_map_for_CNN, dim=0)
        batch_cur_map_for_CNN = torch.stack(batch_cur_map_for_CNN, dim=0)
        batch_last_pos = torch.stack(batch_last_pos, dim=0)
        batch_cur_pos = torch.stack(batch_cur_pos, dim=0)
        batch_last_uav_mobility_mask = torch.stack(batch_last_uav_mobility_mask, dim=0)
        batch_cur_uav_mobility_mask = torch.stack(batch_cur_uav_mobility_mask, dim=0)
        batch_uav_direction = torch.stack(batch_uav_direction, dim=0)
        batch_reward = torch.stack(batch_reward, dim=0)
        batch_map_encoding_history = torch.stack(batch_map_encoding_history, dim=0)
        return batch_last_map_for_CNN, batch_cur_map_for_CNN, batch_last_pos, batch_cur_pos, batch_last_uav_mobility_mask, batch_cur_uav_mobility_mask, batch_uav_direction, batch_reward, batch_map_encoding_history
        
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

class Pos_CNN_MAPPO(Base_Algorithm_Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.env = env
        self.graph_hidden = args.graph_hidden
        self.n_hiddens = args.n_hidden
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.eps = args.eps
        self.device = args.device
        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.n_agent = env.n_UAV + env.n_BS
        self.state_dim = 9 + 4 + self.n_agent + 2 + 1 # state dim of each node, take graph as input
        self.gcn_hidden_output = self.graph_hidden * 2
        # 需要注意的是，地面的RSU也是agent，但是只有一个动作，就是是否接入地面车辆的graph
        self.actions_dim = [2, 9, 2] # 第一个是graph上节点的动作(0-1标识是否接入)，表示是否接入给该agent，第二个是运动的方向（8+1），第三个是无人机的速度(连续值)
        self.coma_agents = [COMA_Actor(self.n_hiddens, self.actor_lr, self.device, self.args) for _ in range(self.n_agent)]
        self.coma_critics = [COMA_critic(self.n_hiddens, self.critic_lr, self.device, self.args) for _ in range(self.n_agent)]
        self.grid_width = args.grid_width # 正方形网格，height就不需要了
        self.grid_num_x = int((env.max_range_x - env.min_range_x) // self.grid_width)
        self.grid_num_y = int((env.max_range_y - env.min_range_y) // self.grid_width)
        self.map_history = deque(maxlen=args.K_history)
        self.map_encoding_history = deque([np.zeros(self.args.n_hidden)], maxlen=1)
        # agent_map用于每个agent在调整区域的时候使用
        self.agent_map_encoding_history = [deque([np.zeros(self.args.n_hidden)], maxlen=self.args.K_history) for _ in range(self.n_agent)]
        self.last_agent_map_encoding_history = [deque([np.zeros(self.args.n_hidden)], maxlen=self.args.K_history) for _ in range(self.n_agent)]
        self.map_history_dim = 900 + self.n_agent * (1 + 2 + 1)
        
        # 过往若干时隙内，每个grid内部的车辆数量，用channel表示，使用CNN进行处理。输出N个无人机的坐标，即网格的横轴和纵轴
        self.UAV_position_actors = [PosPPO_Actor(self.n_hiddens, self.actor_lr * 0.5, self.device, self.args) for _ in range(self.n_agent)]
        self.target_UAV_position_actors = [PosPPO_Actor(self.n_hiddens, self.actor_lr * 0.5, self.device, self.args) for _ in range(self.n_agent)]
        self.last_state = None
        self.replay_buffers = [ReplayBuffer(2000, self.n_agent) for _ in range(self.n_agent)]
        self.pos_replay_buffer = PosReplayBuffer(args.memory_size // 1 - 2, self.n_agent)
        self.action_cnt = 0
        self.update_cnt = 0
        self.last_cnt_succ_num = 0
        self.critic_loss = deque([0], maxlen=100)
        self.actor_losses = [deque([0], maxlen=100) for _ in range(self.n_agent)]
        self.UAV_actor_losses = deque([0], maxlen=100)
        self.UAV_critic_losses = deque([0], maxlen=100)
        self.entropies = [deque([0], maxlen=100) for _ in range(self.n_agent)]
        self.served_within_range = 0
        self.total_served = 0
        self.served_as_nth_veh = np.zeros(self.args.v_neighbor_Veh+1)
        self.avg_iteration = []
        self.UAV_target_positions = np.zeros((self.n_agent, 2))
        self.to_offload_task_ids = []
        from torch.utils.tensorboard import SummaryWriter
        import os
        import shutil
        writer_path = f"{args.tensorboard_writer_file}"
        if os.path.exists(writer_path):
            shutil.rmtree(writer_path)
        self.writer = SummaryWriter(writer_path)
        self.best_total_reward = 0
        self.save_best = False
        if self.args.load_model:
            self.load_agents()
        for agent_id in range(self.n_agent):
            soft_update(self.target_UAV_position_actors[agent_id], self.UAV_position_actors[agent_id], 1.0)

    def load_agents(self):
        saved_path = self.args.saved_path
        for agent_id, agent in enumerate(self.coma_agents):
            self.UAV_position_actors[agent_id].load_agent(saved_path, agent_id)
            agent.load_agent(saved_path, agent_id)
            self.coma_critics[agent_id].load_agent(saved_path, agent_id)
        print('Agents loaded successfully!')

    def reset_state(self):
        self.last_state = None
        self.served_within_range = 0
        self.served_as_nth_veh = np.zeros(self.args.v_neighbor_Veh+1)
        self.total_served = 0
        self.last_cnt_succ_num = 0
        self.avg_iteration = []
        self.map_history = deque(maxlen=self.args.K_history)
        self.map_encoding_history = deque([np.zeros(self.args.n_hidden)], maxlen=1)
        self.agent_weight_history = [deque([0 for _ in range(5)], maxlen=5) for _ in range(self.n_agent)]
        self.agent_map_reward = np.zeros((self.n_agent))
        
        self.agent_map_encoding_history = [deque([np.zeros(self.args.n_hidden)], maxlen=self.args.K_history) for _ in range(self.n_agent)]
        self.last_agent_map_encoding_history = [deque([np.zeros(self.args.n_hidden)], maxlen=self.args.K_history) for _ in range(self.n_agent)]
        self.area_agent_rewards = np.zeros((self.n_agent))
        self.last_area_agent_rewards = np.zeros((self.n_agent))
        self.scale_map_reward = np.ones((self.n_agent))
        self.last_map_history = None
        
        self.to_offload_task_ids = []
        self.agent_weights = np.zeros((self.n_agent)) # 存储每个agent的权重, weighted voronoi diagram
        self.update_can_access_map_by_weight()
    def update_can_access_map_by_weight(self):
        # 根据每个agent的位置，按照权重为0来更新can_access_map
        self.can_access_map = -np.ones((self.n_agent, self.grid_num_x, self.grid_num_y))
        x_range = [self.env.min_range_x, self.env.max_range_x]
        y_range = [self.env.min_range_y, self.env.max_range_y]
        for agent_id in range(self.n_agent):
            if agent_id < self.env.n_UAV:
                agent = self.env.UAVs[agent_id]
            else:
                agent = self.env.BSs[agent_id - self.env.n_UAV]
            agent_position = agent.position
            agent_position_grid_x = int((agent_position[0] - x_range[0]) // self.grid_width)
            agent_position_grid_y = int((agent_position[1] - y_range[0]) // self.grid_width)
            agent_position_grid_x = max(0, min(self.grid_num_x-1, agent_position_grid_x))
            agent_position_grid_y = max(0, min(self.grid_num_y-1, agent_position_grid_y))
        # 遍历网格，根据距离+权重，判断是否属于对应的agent
        for x in range(self.grid_num_x):
            for y in range(self.grid_num_y):
                min_distance = 1e9
                assigned_agent_id = -1
                for agent_id in range(self.n_agent):
                    agent_position = self.env.area_centers[agent_id] # 中心的点位
                    distance = np.sqrt((agent_position[0] - (x* self.grid_width+x_range[0])) ** 2 + (agent_position[1] - (y* self.grid_width+y_range[0])) ** 2) - self.agent_weights[agent_id] * self.grid_width
                    if distance < min_distance:
                        min_distance = distance
                        assigned_agent_id = agent_id
                agent = self.env.UAVs[assigned_agent_id] if assigned_agent_id < self.env.n_UAV else self.env.BSs[assigned_agent_id - self.env.n_UAV]
                comm_range = self.args.UAV_communication_range if assigned_agent_id < self.env.n_UAV else self.args.RSU_communication_range
                dist2agent = np.sqrt((agent.position[0] - (x* self.grid_width+x_range[0])) ** 2 + (agent.position[1] - (y* self.grid_width+y_range[0])) ** 2)
                if dist2agent <= comm_range:
                    self.can_access_map[assigned_agent_id, x, y] = 1 # 可以访问
                else:
                    self.can_access_map[assigned_agent_id, x, y] = 0 # 不能访问，但是在范围内
        return self.can_access_map
            

    def act_pay_and_punish(self):
        pass
    
    
    def get_sum_inverse_sumRcpu(self, V2V_Rate, tv_list, sv_list, provided_CPU, calculate_for_all_veh = False):
        avg_task_ddl = self.env.avg_task_ddl
        avg_task_data = self.env.avg_task_data / avg_task_ddl
        avg_task_cpu = self.env.avg_task_cpu / avg_task_ddl
        sumRcpu_X = {}
        candidate_sv4tv = {}
        total_lambda = 0
        for tv in tv_list:
            total_lambda += tv.task_lambda
        # V2V_Rate = V2V_Rate / max(1, total_lambda)
        for tv in tv_list:
            candidate_sv = sv_list.copy()
            trans_data_list = []
            tot_trans_time = 0
            tot_cpu = 0
            for sv in candidate_sv:
                trans_data_list.append((sv, V2V_Rate[tv.cur_index][sv.cur_index]))
                tot_trans_time += V2V_Rate[tv.cur_index][sv.cur_index]
                tot_cpu += sv.CPU_frequency
            tot_cpu /= len(tv_list)
            tot_trans_time /= max(1, len(candidate_sv)) # avg throughout
            avg_comp_time = avg_task_cpu * tv.task_lambda / max(1, tot_cpu)
            avg_trans_time = avg_task_data * tv.task_lambda / max(1, tot_trans_time)
            avg_trans_time_thr = avg_trans_time / max(0.1, avg_trans_time + avg_comp_time)
            # 降序排列
            trans_data_list.sort(key=lambda x:x[1], reverse=True)
            threshold_num = 1
            # 判断trans_data_list>=avg_task_data*0.5*tv.task_lambda的最小的N为threshold_num
            for i in range(len(sv_list)):
                # if threshold_num >= 20:
                #     break
                if trans_data_list[i][1] >= avg_task_data * avg_trans_time_thr * tv.task_lambda:
                    threshold_num = i
                else:
                    break
            # threshold_num = min(10, threshold_num)
            candidate_sv = [trans_data_list[i][0] for i in range(min(threshold_num, len(sv_list)))]
            trans_data_list = np.array([trans_data_list[i][1] for i in range(min(threshold_num, len(sv_list)))])
            candidate_sv4tv[tv] = (candidate_sv, [], [])
            trans_time_list = []
            # trans_time = avg_task_data * tv.task_lambda / max(1, np.sum(trans_data_list))
            if len(candidate_sv) == 0: # 说明没有找到合适的N
                continue
            trans_time_list = (avg_task_data * tv.task_lambda / len(candidate_sv) / trans_data_list)
            alloc_task_portion = tv.task_lambda / len(candidate_sv)
            Rcpu_list = []
            for svid, sv in enumerate(candidate_sv):
                R_cpu = alloc_task_portion * avg_task_cpu
                Rcpu_list.append(R_cpu)
                # R_cpu是每一个tv对每一个sv请求的cpu资源，求和
                if sumRcpu_X.get(sv) is None:
                    sumRcpu_X[sv] = R_cpu
                else:
                    sumRcpu_X[sv] += R_cpu
            candidate_sv4tv[tv] = (candidate_sv, Rcpu_list, trans_time_list)
        sum_inverse_sumRcpu = 0
        sum_tv_reward = 0
        for tv in tv_list:
            if calculate_for_all_veh:
                tv.pCPU = 0
            candidate_sv_list, Rcpu_list, trans_time_list = candidate_sv4tv[tv]
            tmp_inverse_sum = 0
            for idx, sv in enumerate(candidate_sv_list):
                tmp_pCPU = (sv.CPU_frequency+provided_CPU) / max(0.1, sumRcpu_X[sv]) * Rcpu_list[idx] # tv可以被分配到的资源
                tmp_comp_time = (avg_task_cpu * tv.task_lambda) / len(candidate_sv_list) / max(0.1, tmp_pCPU)
                tmp_time = tmp_comp_time + trans_time_list[idx]
                tmp_utility = 1 + np.log(1 + 1 - tmp_time) if tmp_time < 1 else np.exp(5 * (1 - tmp_time))
                # tmp_utility = min(1, 1/tmp_time)
                tmp_utility = tmp_utility * tv.task_lambda / len(candidate_sv_list)
                tmp_inverse_sum += tmp_utility
                # tmp_inverse_sum += tmp_pCPU
            # tmp_total_time = (avg_task_cpu * tv.task_lambda) / max(0.1, tmp_inverse_sum) + trans_time
            # tmp_utility = 1 + np.log(1 + 1 - tmp_total_time) if tmp_total_time < 1 else np.exp(1 - tmp_total_time)
            # tmp_utility = min(1, 1/tmp_total_time)
            # tmp_utility *= tv.task_lambda
            if calculate_for_all_veh:
                # tv.pCPU -= (np.sum(Rcpu_list)) 
                tv.pCPU = tmp_inverse_sum
            sum_tv_reward += tv.pCPU 
            sum_inverse_sumRcpu += tmp_inverse_sum - tv.pCPU
        return sum_inverse_sumRcpu, sum_tv_reward
    def calculate_reward(self):
        '''根据env以及内部存储的task信息，对比传输的数据量和计算的数据量，计算reward，存储在reward_memory中'''
        reward_type = 1
        all_rewards = np.zeros(self.n_agent)
        total_all_reward = 0 # 用于计算整个系统的奖励，指导UAV轨迹
        if reward_type == 1:
            succeed_tasks = self.env.succeed_tasks # dict
            assigned_to_cnt = np.zeros(self.n_agent)
            for task_id in self.to_offload_task_ids:
                if succeed_tasks.get(task_id) is None:
                    continue
                task = succeed_tasks[task_id]
                gen_veh = task.g_veh
                assigned_to = gen_veh.assigned_to
                if assigned_to == -1:
                    continue
                agent = self.env.UAVs[assigned_to] if assigned_to < self.env.n_UAV else self.env.BSs[assigned_to-self.env.n_UAV]
                comm_range = self.args.UAV_communication_range if assigned_to < self.env.n_UAV else self.args.RSU_communication_range
                if self.env.distance(agent.position, gen_veh.position) > comm_range:
                    gen_veh.assigned_to = -1
                    continue
                assigned_to_cnt[assigned_to] += 1
                all_rewards[assigned_to] += task.get_task_utility()
            total_all_reward = np.sum(all_rewards)
            # all_rewards = all_rewards / (assigned_to_cnt + 1e-5) # 按照n_tv进行平均
        elif reward_type == 2:
            assigned_TV_list = [[] for _ in range(self.n_agent)]
            assigned_SV_list = [[] for _ in range(self.n_agent)]
            covered_TV_list = [[] for _ in range(self.n_agent)]
            covered_SV_list = [[] for _ in range(self.n_agent)]
            vehicle_list = self.env.vehicle_by_index
            total_task_per_second = 0
            for cur_idx, vehicle in enumerate(vehicle_list):
                vehicle.cur_index = cur_idx
                agent_id = vehicle.assigned_to
                if agent_id != -1: 
                    agent = self.env.UAVs[agent_id] if agent_id < self.env.n_UAV else self.env.BSs[agent_id-self.env.n_UAV]
                    comm_range = self.args.UAV_communication_range if agent_id < self.env.n_UAV else self.args.RSU_communication_range
                    if self.env.distance(agent.position, vehicle.position) > comm_range:
                        vehicle.assigned_to = -1
                        continue
                    if not vehicle.serving:
                        assigned_TV_list[agent_id].append(vehicle)
                        total_task_per_second += vehicle.task_lambda
                    else:
                        assigned_SV_list[agent_id].append(vehicle)
                # 然后判断covered,根据self.can_access_map
                
                veh_gridx = int((vehicle.position[0] - self.env.min_range_x) // self.grid_width)
                veh_gridy = int((vehicle.position[1] - self.env.min_range_y) // self.grid_width)
                veh_gridx = max(0, min(veh_gridx, self.grid_num_x-1))
                veh_gridy = max(0, min(veh_gridy, self.grid_num_y-1))
                for agent_id in range(self.n_agent):
                    if self.can_access_map[agent_id, veh_gridx, veh_gridy] == 1:
                        if not vehicle.serving:
                            covered_TV_list[agent_id].append(vehicle)
                        else:
                            covered_SV_list[agent_id].append(vehicle)
            # 先假设完全没有RSU和UAV，完全通过veh自己的决策，他们作为离散设备形成的coalition，计算reward
            all_sv_list = []
            for sv_list in assigned_SV_list:
                all_sv_list.extend(sv_list)
            all_tv_list = []
            for tv_list in assigned_TV_list:
                all_tv_list.extend(tv_list)
            org_v2v = self.get_V2VRateWithoutBand(self.env).copy() * self.args.V2V_RB / self.env.n_RB
            org_v2u = self.get_V2URateWithoutBand(self.env).copy() * (self.args.V2U_RB) / self.env.n_RB
            org_u2v = self.get_U2VRateWithoutBand(self.env).copy() * (self.args.V2U_RB) / self.env.n_RB
            org_v2i = self.get_V2IRateWithoutBand(self.env).copy() * (self.args.V2I_RB) / self.env.n_RB
            org_i2v = self.get_I2VRateWithoutBand(self.env).copy() * (self.args.V2I_RB) / self.env.n_RB
            assumed_band = self.env.bandwidth #/ max(1, total_task_per_second)
            # all_after_sum_inverse_sumRcpu, all_sum_sv_reward = self.get_sum_inverse_sumRcpu(org_v2v * assumed_band, all_tv_list, all_sv_list, provided_CPU = 0, calculate_for_all_veh=True)
            for agent_id in range(self.n_agent):
                if agent_id < self.env.n_UAV:
                    V2XRate = org_v2u.copy() * assumed_band
                    X2VRate = org_u2v.copy() * assumed_band
                    X_device = self.env.UAVs[agent_id]
                else:
                    V2XRate = org_v2i.copy() * assumed_band
                    X2VRate = org_i2v.copy() * assumed_band
                    X_device = self.env.BSs[agent_id-self.env.n_UAV]
                # # 获取V2V_fading和V2X_fading
                V2VRate = org_v2v.copy() * assumed_band
                tv_list = assigned_TV_list[agent_id]
                sv_list = assigned_SV_list[agent_id]
                # tv_list = covered_TV_list[agent_id]
                # sv_list = covered_SV_list[agent_id]
                sv_list.append(X_device)
                X_device.cur_index = len(sv_list)
                X_device.reward = 0
                if len(tv_list) == 0:
                    continue
                before_sum_inverse_sumRcpu = 0
                i = agent_id if agent_id < self.env.n_UAV else agent_id - self.env.n_UAV
                for tv in tv_list:
                    vidx = tv.cur_index
                    intRate = V2XRate[vidx, i] * X2VRate[i, :] / (V2XRate[vidx, i] + X2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                # V2VRate 在 第二个维度 加上一个X_device
                V2VRate = np.concatenate((V2VRate, np.zeros((V2VRate.shape[0], 1))), axis=1)
                # 这个X_device的V2VRate是所有的tv对它的V2XRate
                V2VRate[:, -1] = V2XRate[:, i]
                after_sum_inverse_sumRcpu, sum_sv_reward = self.get_sum_inverse_sumRcpu(V2VRate, tv_list, sv_list, provided_CPU = 0, calculate_for_all_veh=False)
                total_all_reward += after_sum_inverse_sumRcpu + sum_sv_reward
                all_rewards[agent_id] = after_sum_inverse_sumRcpu + sum_sv_reward
        all_rewards = all_rewards * 1000 / self.env.total_task_lambda
        self.last_cnt_succ_num = self.env.get_succeed_task_cnt()
        uav_to_center_dist = np.zeros((self.env.n_UAV))
        for agent_id in range(self.env.n_UAV):
            uav_to_center_dist[agent_id] = self.env.distance(self.env.UAVs[agent_id].position, self.env.area_centers[agent_id])
        
        # mu = np.mean(all_rewards)
        # std = np.std(all_rewards + 1e-3)
        # reward_shared = - 0.5 * np.log(std + 1e-3) + 2 * np.log(mu + 1e-3)
        # reward_agent = - np.log((all_rewards - mu)**2 + 1e-3)
        # area_agent_rewards = reward_agent + reward_shared
        # self.uav_rewards = area_agent_rewards[:self.env.n_UAV]
        self.uav_rewards = all_rewards[:self.env.n_UAV]
        # self.uav_rewards -= uav_to_center_dist * 0.05
        for agent_id in range(self.n_agent):
            self.area_agent_rewards[agent_id] += all_rewards[agent_id] #* (assigned_to_cnt + 1e-5)

        
    def update_agents(self):
        # 需要update两次agent，一个是从pos_replay_buffer中采样更新MAPPO，一个是从replay_buffer中采样更新DQN
        # 1. 先更新DQN
        batch_size = 5 #DQN的batch_size调大
        if self.replay_buffers[0].size >= batch_size and self.action_cnt % 2 == 0: # 每个step都更新
            for agent_id in range(self.env.n_UAV): # 只更新无人机的DQN
                batch_last_map_for_CNN, batch_cur_map_for_CNN, batch_last_pos, batch_cur_pos, batch_last_uav_mobility_mask, batch_cur_uav_mobility_mask, batch_uav_direction, batch_reward, batch_map_encoding_history = self.replay_buffers[agent_id].sample(batch_size, shuffle=True)
                # 把batch_cur_pos和batch_last_pos乘以20，取整，然后把他们各自的x,y坐标分别映射到[30]和[20]的one-hot向量
                batch_last_pos = batch_last_pos * 20
                batch_cur_pos = batch_cur_pos * 20
                batch_last_pos = batch_last_pos.long()
                batch_cur_pos = batch_cur_pos.long()
                # 限制范围
                batch_last_pos[:,0] = torch.clamp(batch_last_pos[:,0], 0, 29)
                batch_last_pos[:,1] = torch.clamp(batch_last_pos[:,1], 0, 19)
                batch_cur_pos[:,0] = torch.clamp(batch_cur_pos[:,0], 0, 29)
                batch_cur_pos[:,1] = torch.clamp(batch_cur_pos[:,1], 0, 19)
                batch_last_pos_x_one_hot = torch.zeros((batch_size, 30)).to(self.device)
                batch_last_pos_y_one_hot = torch.zeros((batch_size, 20)).to(self.device)
                batch_cur_pos_x_one_hot = torch.zeros((batch_size, 30)).to(self.device)
                batch_cur_pos_y_one_hot = torch.zeros((batch_size, 20)).to(self.device)
                batch_last_pos_x_one_hot.scatter_(1, batch_last_pos[:,0].unsqueeze(-1), 1)
                batch_last_pos_y_one_hot.scatter_(1, batch_last_pos[:,1].unsqueeze(-1), 1)
                batch_cur_pos_x_one_hot.scatter_(1, batch_cur_pos[:,0].unsqueeze(-1), 1)
                batch_cur_pos_y_one_hot.scatter_(1, batch_cur_pos[:,1].unsqueeze(-1), 1)
                #最后把batch_last_pos_x_one_hot和batch_last_pos_y_one_hot拼接起来，作为batch_last_pos
                batch_last_pos = torch.cat((batch_last_pos_x_one_hot, batch_last_pos_y_one_hot), dim=-1) # [batch_size, 50]
                batch_cur_pos = torch.cat((batch_cur_pos_x_one_hot, batch_cur_pos_y_one_hot), dim=-1) # [batch_size, 50]
                batch_reward = batch_reward.view(-1, 1)
                # 根据batch_last，计算各个动作的q value，然后gather direction
                q_values, batch_cur_map_encoding = self.UAV_position_actors[agent_id](batch_last_map_for_CNN, batch_last_pos, batch_last_uav_mobility_mask, batch_map_encoding_history) # [batch_size, 5]
                # 把batch_cur_map_encoding作为batch_next_map_encoding_history
                batch_next_map_encoding_history = batch_cur_map_encoding.unsqueeze(1).detach().clone()
                cert_q_values = q_values.gather(1, batch_uav_direction.long()) # [batch_size]
                max_next_q_values, _ = self.target_UAV_position_actors[agent_id](batch_cur_map_for_CNN, batch_cur_pos, batch_cur_uav_mobility_mask, batch_next_map_encoding_history) # [batch_size]
                max_next_q_values = max_next_q_values.max(1)[0].detach()
                target_q_values = batch_reward + self.gamma * max_next_q_values.unsqueeze(-1) # [batch_size, 1]
                dqn_loss = F.mse_loss(cert_q_values, target_q_values)
                self.UAV_position_actors[agent_id].opt_zero_grad()
                dqn_loss.backward()
                self.UAV_position_actors[agent_id].opt_step()
                self.UAV_actor_losses.append(dqn_loss.item())
                soft_update(self.target_UAV_position_actors[agent_id], self.UAV_position_actors[agent_id], self.args.tau)

        # 2. 更新MAPPO
        episode_num = max(1, int(self.args.batch_size // 200))
        batch_size = self.args.batch_size // 1 - 2 * episode_num
        if self.action_cnt >= self.args.start_to_train and self.action_cnt % self.args.update_every == 0:
            self.log_to_tensorboard()
            self.update_cnt += 1
            if self.args.save_model and self.update_cnt > 1 and self.update_cnt % self.args.fre_to_save == 0:
                self.save_agents()
                if self.save_best:
                    self.save_agents("best")
                    self.save_best = False
                
            # 从memory中采样
            batch_last_map_history, batch_cur_map, batch_agent_weights, batch_agent_log_probs, batch_agent_rewards, batch_weight_actions, batch_agent_weight_masks, batch_last_map_encoding_history, batch_cur_map_encoding_history, batch_last_agent_weights, batch_cur_agent_weights, batch_cur_agent_positions, batch_last_agent_positions = self.pos_replay_buffer.sample(batch_size, shuffle=False) # 是否都在device, 是否都是detach
            # batch_agent_weights = batch_agent_weights ** 2 / 15 # 使得agent_weights在[0,1]之间
            batch_agent_weights = batch_agent_weights / 4
            # 在[0,1]
            batch_agent_weights = torch.clamp(batch_agent_weights, 0, 1) 
            batch_last_agent_positions = batch_last_agent_positions * 20
            batch_cur_agent_positions = batch_cur_agent_positions * 20
            batch_last_pos = batch_last_agent_positions.long()
            batch_cur_pos = batch_cur_agent_positions.long()
            # 限制范围
            batch_last_pos[:,:,0] = torch.clamp(batch_last_pos[:,:,0], 0, 29)
            batch_last_pos[:,:,1] = torch.clamp(batch_last_pos[:,:,1], 0, 19)
            batch_cur_pos[:,:,0] = torch.clamp(batch_cur_pos[:,:,0], 0, 29)
            batch_cur_pos[:,:,1] = torch.clamp(batch_cur_pos[:,:,1], 0, 19)
            batch_last_pos_x_one_hot = torch.zeros((batch_size, self.n_agent, 30)).to(self.device)
            batch_last_pos_y_one_hot = torch.zeros((batch_size, self.n_agent, 20)).to(self.device)
            batch_cur_pos_x_one_hot = torch.zeros((batch_size, self.n_agent, 30)).to(self.device)
            batch_cur_pos_y_one_hot = torch.zeros((batch_size, self.n_agent, 20)).to(self.device)
            batch_last_pos_x_one_hot.scatter_(2, batch_last_pos[:,:,0].unsqueeze(-1), 1)
            batch_last_pos_y_one_hot.scatter_(2, batch_last_pos[:,:,1].unsqueeze(-1), 1)
            batch_cur_pos_x_one_hot.scatter_(2, batch_cur_pos[:,:,0].unsqueeze(-1), 1)
            batch_cur_pos_y_one_hot.scatter_(2, batch_cur_pos[:,:,1].unsqueeze(-1), 1)
            #最后把batch_last_pos_x_one_hot和batch_last_pos_y_one_hot拼接起来，作为batch_last_pos
            batch_last_pos = torch.cat((batch_last_pos_x_one_hot, batch_last_pos_y_one_hot), dim=-1) # [batch_size, 50]
            batch_cur_pos = torch.cat((batch_cur_pos_x_one_hot, batch_cur_pos_y_one_hot), dim=-1) # [batch_size, 50]
            for agent_id, agent in enumerate(self.coma_agents):
                # -----------------------------------------------------
                agent = self.coma_agents[agent_id]
                critic = self.coma_critics[agent_id]
                # -----------------------------------------------------
                # 重复batch_size次数
                tensor_agent_id = torch.tensor([agent_id]).to(self.device)
                batch_agent_id = tensor_agent_id.repeat(batch_size, 1).view(batch_size, 1)
                # 1.1 计算target_value
                value = critic(batch_cur_map.detach().clone(), batch_agent_id.detach().clone(), batch_cur_map_encoding_history[:,agent_id,:,:].detach().clone(), batch_cur_agent_weights[:,agent_id,:].detach().clone(), batch_cur_pos.detach().clone())
                target_value = batch_agent_rewards[:, agent_id].unsqueeze(-1) + self.gamma * value
                td_delta = target_value - value
                td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
                advantage_list = []  # 存放每个时序的优势函数值
                batch_per_epi = int(self.args.batch_size) // 1 - 2 * episode_num
                for episode in range(episode_num):
                    tmp_advantage_list = []
                    advantage = 0  # 累计一个序列上的优势函数
                    for delta in td_delta[episode*batch_per_epi:(episode+1)*batch_per_epi][::-1]:
                        advantage = self.gamma * self.lmbda * advantage + delta
                        tmp_advantage_list.append(advantage)
                    tmp_advantage_list.reverse()
                    advantage_list.extend(tmp_advantage_list)
                advantage_list = np.array(advantage_list)
                advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                # 计算log_prob
                alpha, beta, map_encoding = agent(batch_last_map_history.detach().clone(), batch_agent_id.detach().clone(), batch_agent_weights.detach().clone(), batch_last_map_encoding_history[:,agent_id,:,:].detach().clone(), batch_last_agent_weights[:,agent_id,:,:].detach().clone(), batch_last_pos[:,agent_id].detach().clone())
                # 通过mean和log_std求出action的log_prob
                dist = torch.distributions.Beta(alpha, beta)
                log_probs = dist.log_prob(batch_agent_weights[:, agent_id].detach().clone()).unsqueeze(-1)
                ratio = torch.exp(log_probs - batch_agent_log_probs[:, agent_id, :].clone().detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
                entropy = dist.entropy().mean()
                actor_loss = torch.mean(-torch.min(surr1, surr2)) - 0.01 * torch.mean(entropy)

                agent.opt_zero_grad()
                actor_loss.backward()
                agent.opt_step()
                critic_loss = F.mse_loss(value, target_value.detach().float())
                # 1.3 更新critic
                critic.opt_zero_grad()
                critic_loss.backward()
                critic.opt_step()

                # 记录loss
                self.actor_losses[agent_id].append(actor_loss.item())
                self.entropies[agent_id].append(torch.mean(entropy).item())
            self.critic_loss.append(critic_loss.item())


    def save_agents(self, add_str=None, terminated = False):
        saved_path = self.args.saved_path
        if terminated:
            print('Agents terminated! Saving to terminated folder...')
            saved_path = os.path.join(saved_path, 'terminated')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
        if add_str is not None:
            saved_path = os.path.join(saved_path, add_str)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        for agent_id, agent in enumerate(self.coma_agents):
            self.UAV_position_actors[agent_id].save_agent(saved_path, agent_id)
            agent.save_agent(saved_path, agent_id)
            self.coma_critics[agent_id].save_agent(saved_path, agent_id)
        
        print(f'Agents saved to {saved_path} successfully!')

    def draw_figure_to_tensorboard(self):
        env = self.env
        # 从env获取vehicle_by_index，里面各个vehicle的位置，根据vehicle.serving判断是任务车辆还是服务车辆
        # 从env获取UAVs和BSs的位置
        # 使用matplotlib画图，保存到tensorboard
        vehicle_by_index = env.vehicle_by_index
        uavs = env.UAVs
        bss = env.BSs
        # 画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Env Time: {self.env.cur_time}')
        # Create dummy plots for the legend
        ax.scatter([], [], c='r', marker='o', label='serving')
        ax.scatter([], [], c='b', marker='o', label='task')
        ax.scatter([], [], c='g', marker='^', label='uav')
        ax.scatter([], [], c='y', marker='^', label='bs')

        # Plot the points
        for vehicle in vehicle_by_index:
            if vehicle.serving:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='r', marker='o')
            else:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='b', marker='o')
        for bs in bss:
            ax.scatter(bs.position[0], bs.position[1], c='y', marker='^')
        for uav in uavs:
            ax.scatter(uav.position[0], uav.position[1], c='g', marker='^')

        ax.legend()
        self.writer.add_figure('system_fig', fig, global_step=self.update_cnt)
        plt.close()
        # 再把self.can_access_map画出来,把每个agent_id对应的数值按照不同的颜色点出来即可
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Env Time: {self.env.cur_time}')
        # Create dummy plots for the legend
        for agent_id in range(self.n_agent):
            # self.can_access_map[agent_id, x, y] != -1,用一个颜色scatter出来所有的点
            point_list = np.where(self.can_access_map[agent_id] == 1)
            point_list = np.array(point_list)
            x_list = point_list[0] * self.grid_width + self.env.min_range_x
            y_list = point_list[1] * self.grid_width + self.env.min_range_y
            ax.scatter(x_list, y_list, label=f'agent{agent_id}')
        ax.legend()
        self.writer.add_figure('can_access_map', fig, global_step=self.update_cnt)

        # 绘制神经网络的参数分布，使用add_histogram
        for agent_id, agent in enumerate(self.coma_agents):
            for name, param in agent.named_parameters():
                self.writer.add_histogram(f'agent{agent_id}_{name}', param.clone().cpu().data.numpy(), global_step=self.update_cnt)
            for name, param in self.coma_critics[agent_id].named_parameters():
                self.writer.add_histogram(f'critic{agent_id}_{name}', param.clone().cpu().data.numpy(), global_step=self.update_cnt)
            for name, param in self.UAV_position_actors[agent_id].named_parameters():
                self.writer.add_histogram(f'UAV_position_actor{agent_id}_{name}', param.clone().cpu().data.numpy(), global_step=self.update_cnt)
            break
        
        


    def log_to_tensorboard(self):
        # 从env获取
        succ_task_cnt = self.env.get_succeed_task_cnt()
        failed_task_cnt = self.env.get_failed_task_cnt()
        tot_task_cnt = succ_task_cnt + failed_task_cnt
        avg_task_latency = self.env.get_avg_task_latency()
        avg_task_ratio = succ_task_cnt / max(1, tot_task_cnt)
        if self.update_cnt > 0:# and self.update_cnt % 3 == 0:
            endings = '\n'
            print("Step {}, critic_loss: {:.3f}, actor_loss: {:.3f}, reward: {:.3f}, ratio: {:.3f}, latency: {:.3f}, served_within_range_ratio: {:.3f}".format(
                self.action_cnt, 
                np.mean(self.critic_loss), 
                np.mean([np.mean(loss) for loss in self.actor_losses]), 
                np.mean(self.replay_buffers[0].avg_rewards()), 
                avg_task_ratio, 
                avg_task_latency, 
                self.served_within_range / max(1, self.total_served)
                # np.mean(self.avg_iteration)
            ), end=endings)
            self.writer.add_scalar('Loss/critic_loss', np.mean(self.critic_loss), self.update_cnt)
            for agent_id, loss in enumerate(self.actor_losses):
                self.writer.add_scalar(f'Loss/agent{agent_id}_actor_loss', np.mean(loss), self.update_cnt)
                self.writer.add_scalar(f'Loss/agent{agent_id}_entropy', np.mean(self.entropies[agent_id]), self.update_cnt)
                if agent_id < self.env.n_UAV:
                    self.writer.add_scalar(f'Metric/total_reward_{agent_id}', (self.replay_buffers[agent_id].avg_rewards()), self.update_cnt)
            self.writer.add_scalar('Loss/UAV_actor_loss', np.mean(self.UAV_actor_losses), self.update_cnt * self.args.update_every)
            # self.writer.add_scalar('Loss/UAV_critic_loss', np.mean(self.UAV_critic_losses), self.update_cnt * self.args.update_every)
            self.writer.add_scalar('Metric/UAV_reward', (self.pos_replay_buffer.avg_rewards()), self.update_cnt * self.args.update_every)
            self.writer.add_scalar('Metric/ratio', avg_task_ratio, self.update_cnt)
            self.writer.add_scalar('Metric/latency', avg_task_latency, self.update_cnt)
            # self.writer.add_scalar('Metric/avg_iteration', np.mean(self.avg_iteration), self.update_cnt)
        best_total_reward = self.pos_replay_buffer.avg_rewards()
        if best_total_reward > self.best_total_reward:
            self.best_total_reward = best_total_reward
            self.save_best = True
        if self.update_cnt % self.args.fre_to_draw == 0:
            self.draw_figure_to_tensorboard()
    def get_map_grid_vector(self, grid_width, grid_num_x, grid_num_y, grid_x_range = [500, 2000], grid_y_range = [250, 1250]):
        # return grid_num * grid_num, 每个grid存储的信息是预计获得的reward
        serve_vehicle_by_grid = [[[] for _ in range(grid_num_y)] for _ in range(grid_num_x)]
        task_vehicle_by_grid = [[[] for _ in range(grid_num_y)] for _ in range(grid_num_x)]
        all_sv_list = []
        all_tv_list = []
        sv_num_in_agent = np.zeros(self.n_agent)
        assigned_tv_list = [[] for _ in range(self.n_agent)]
        assigned_sv_list = [[] for _ in range(self.n_agent)]
        
        for idx, vehicle in enumerate(self.env.vehicle_by_index):
            vehicle.cur_index = idx
            grid_x = (int((vehicle.position[0] - grid_x_range[0]) // grid_width))
            grid_y = (int((vehicle.position[1] - grid_y_range[0]) // grid_width))
            grid_x = max(0, min(grid_x, grid_num_x-1))
            grid_y = max(0, min(grid_y, grid_num_y-1))
            assigned_agent = np.argmax(self.can_access_map[:, grid_x, grid_y])
            if not(assigned_agent != -1 and self.can_access_map[assigned_agent, grid_x, grid_y]==1):
                vehicle.assigned_to = -1
                continue
            if vehicle.serving:
                sv_num_in_agent[assigned_agent] += 1
                serve_vehicle_by_grid[grid_x][grid_y].append(vehicle)
                all_sv_list.append(vehicle)
                assigned_sv_list[assigned_agent].append(vehicle)
            else:
                vehicle.pCPU = 0 # 先把pCPU清零
                task_vehicle_by_grid[grid_x][grid_y].append(vehicle)
                all_tv_list.append(vehicle)
                assigned_tv_list[assigned_agent].append(vehicle)
        # 2. 遍历，修改uav 0的位置到每个grid的中心
        veh_num_map = np.zeros((grid_num_x, grid_num_y))
        cpu_num_map = np.zeros((grid_num_x, grid_num_y))
        map_grid_vector_asUAV = np.zeros((grid_num_x, grid_num_y))
        X_device = self.env.UAVs[0]
        org_v2v = self.get_V2VRateWithoutBand(self.env)
        org_i2v = self.get_I2VRateWithoutBand(self.env)
        org_v2i = self.get_V2IRateWithoutBand(self.env)
        org_v2v_as_u2u = self.get_V2VAsU2URateWithoutBand(self.env)
        V2V_band =  self.args.V2V_RB * self.env.bandwidth / self.env.n_RB
        V2I_band =  self.args.V2I_RB * self.env.bandwidth / self.env.n_RB
        V2VRate = org_v2v.copy() * V2V_band
        V2IRate = org_v2i.copy() * V2I_band
        I2VRate = org_i2v.copy() * V2I_band
        V2VaURate = org_v2v_as_u2u.copy() * V2I_band
        for i in range(self.env.n_BS):
            X_device = self.env.BSs[i]
            for tvidx, tv in enumerate(all_tv_list):
                if self.env.distance(tv.position, X_device.position) <= self.args.RSU_communication_range:
                    vidx = tv.cur_index
                    intRate = V2IRate[vidx, i] * I2VRate[i, :] / (V2IRate[vidx, i] + I2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    # V2VaU
                    # idxs = intRate[:] > V2VaURate[vidx, :]
                    # V2VaURate[vidx, idxs] = intRate[idxs]
        for agent_id in range(self.n_agent):
            addedCPU = self.env.uav_cpu / max(1, sv_num_in_agent[agent_id])
            after_sum_inverse_sumRcpu, sum_sv_reward = self.get_sum_inverse_sumRcpu(V2VaURate, assigned_tv_list[agent_id], assigned_sv_list[agent_id], provided_CPU = addedCPU, calculate_for_all_veh=True)
        self.agent_map_reward = np.zeros((self.n_agent))
        for grid_x in range(grid_num_x):
            for grid_y in range(grid_num_y):
                tv_cnt = 0
                for tv in task_vehicle_by_grid[grid_x][grid_y]:
                    map_grid_vector_asUAV[grid_x, grid_y] += tv.pCPU
                    veh_num_map[grid_x, grid_y] += tv.task_lambda / 10 #/ max(1, self.env.n_TV)
                    tv_cnt += 1
                for sv in serve_vehicle_by_grid[grid_x][grid_y]:
                    cpu_num_map[grid_x, grid_y] += sv.CPU_frequency / 10 #/ max(1, self.env.n_SV)
                assigned_agent = np.argmax(self.can_access_map[:, grid_x, grid_y])
                if assigned_agent != -1 and self.can_access_map[assigned_agent, grid_x, grid_y]==1:
                    self.agent_map_reward[assigned_agent] += map_grid_vector_asUAV[grid_x, grid_y]
        # map_grid_vector /= max(1, np.max(map_grid_vector)) # 归一化
        return map_grid_vector_asUAV.reshape((grid_num_x * grid_num_y)), veh_num_map.reshape((grid_num_x * grid_num_y)), cpu_num_map.reshape((grid_num_x * grid_num_y))

    def take_agent_action(self, act_area = False):
        # 先获取所有agent的hidden
        self.total_comp_time = 0
        self.action_cnt += 1
        cur_UAV_pos = self.env.get_uav_positions()
        cur_UAV_pos = torch.tensor(cur_UAV_pos, dtype=torch.float32).to(self.device) / 1000
        agent_map_for_CNN, uav_mobility_mask, global_map_for_area = self.resolve_map_for_CNN()
        uav_directions = np.zeros((self.env.n_UAV))
        np_uav_mobility_mask = uav_mobility_mask.cpu().detach().numpy()
        map_encoding_history = np.concatenate(self.map_encoding_history, axis=0).reshape((1, -1))
        map_encoding_history = torch.tensor(map_encoding_history, dtype=torch.float32).to(self.device)
        self.last_map_encoding_history = map_encoding_history.clone() # 这个是用来给replay buffer存储的
        cur_map_encoding = None # 共用map_encoder的前提下，共用map_encoding
        for i in range(self.env.n_UAV):
            direction_action, cur_map_hiddden = self.UAV_position_actors[i].take_action(agent_map_for_CNN[i, :, :,:], cur_UAV_pos[i,:], uav_mobility_mask[i,:], map_encoding_history.clone())
            if cur_map_encoding is None:
                cur_map_encoding = cur_map_hiddden.squeeze(0)
            # 如果uav_mobility_mask前4个都是0，那么就必须赶回对应的中心，即self.env.area_centers[i]
            if np.sum(np_uav_mobility_mask[i, :4]) == 0 and False:
                target_position = self.env.area_centers[i]
                dx = 0
                if target_position[0] - cur_UAV_pos[i, 0] > 0:
                    dx = 1
                elif target_position[0] - cur_UAV_pos[i, 0] < 0:
                    dx = -1
                dy = 0
                if target_position[1] - cur_UAV_pos[i, 1] > 0:
                    dy = 1
                elif target_position[1] - cur_UAV_pos[i, 1] < 0:
                    dy = -1
                if dx == 0 and dy == 1:
                    direction_action = 0
                elif dx == 1 and dy == 0:
                    direction_action = 1
                elif dx == 0 and dy == -1:
                    direction_action = 2
                elif dx == -1 and dy == 0:
                    direction_action = 3
                uav_mobility_mask[i, direction_action] = 1

            uav_directions[i] = direction_action
        self.map_encoding_history.append(cur_map_encoding.cpu().detach().numpy())
        # 根据target_grids计算目标位置
        for uav_id in range(self.env.n_UAV):
            direction_ = uav_directions[uav_id]
            # 0代表上，即dx=0,dy=1，1代表右，即dx=1,dy=0，2代表下，即dx=0,dy=-1，3代表左，即dx=-1,dy=0, 4代表不动
            if direction_ == 0:
                dx = 0
                dy = 1
            elif direction_ == 1:
                dx = 1
                dy = 0
            elif direction_ == 2:
                dx = 0
                dy = -1
            elif direction_ == 3:
                dx = -1
                dy = 0
            else:
                dx = 0
                dy = 0
            target_position = self.env.UAVs[uav_id].position + np.array([dx, dy]) * self.grid_width
            
            self.UAV_target_positions[uav_id, :] = target_position
        self.cur_position_state = (agent_map_for_CNN.detach(), cur_UAV_pos.detach(), uav_mobility_mask.detach())
        # 保存到map_history中，代表全局信息的变化
        self.uav_directions = uav_directions
        # 先处理direction_action,从self.UAV_target_positions中获取,根据每个UAV当前的位置和目标位置，计算方向.如果进入到目标区域,就停止
        self.direction_action = np.zeros((self.n_agent))
        for agent_id in range(self.n_agent):
            if agent_id < self.env.n_UAV:
                uav = self.env.UAVs[agent_id]
                target_position = self.UAV_target_positions[agent_id,:]
                # direction = self.env.get_direction(uav.position, target_position)
                direction = (np.arctan2(target_position[1] - uav.position[1], target_position[0] - uav.position[0]) + 2 * np.pi) % ( 2* np.pi)
                # 判断距离是否小于self.grid_width
                if self.env.distance(uav.position, target_position) >= 0:
                    self.direction_action[agent_id] = direction
        # 分配vehicle的assigned_to，按照最近距离分配，同时需要保证在通信范围内
        for vidx, vehicle in enumerate(self.env.vehicle_by_index):
            vehicle.assigned_to = -1
            veh_gridx = int((vehicle.position[0] - self.env.min_range_x) // self.grid_width)
            veh_gridy = int((vehicle.position[1] - self.env.min_range_y) // self.grid_width)
            veh_gridx = max(0, min(veh_gridx, self.grid_num_x-1))
            veh_gridy = max(0, min(veh_gridy, self.grid_num_y-1))
            agent_id = np.argmax(self.can_access_map[:, veh_gridx, veh_gridy])
            if not(agent_id != -1 and self.can_access_map[agent_id, veh_gridx, veh_gridy]==1):
                vehicle.assigned_to = -1
                continue
            agent = self.env.UAVs[agent_id] if agent_id < self.env.n_UAV else self.env.BSs[agent_id - self.env.n_UAV]
            comm_range = self.args.UAV_communication_range if agent_id < self.env.n_UAV else self.args.RSU_communication_range
            if self.env.distance(vehicle.position, agent.position) <= comm_range:
                vehicle.assigned_to = agent_id
        self.map_history.append(global_map_for_area) # numpy.array
        if act_area and len(self.map_history) > 0: # 对区域进行划分
            agent_positions = np.zeros((self.n_agent, 2))
            agent_positions[:self.env.n_UAV, :] = self.env.get_uav_positions()
            agent_positions[self.env.n_UAV:, :] = self.env.get_bs_positions()
            agent_positions /= 1000
            target_agent_positions = np.zeros((self.n_agent, 2))
            target_agent_positions[:self.env.n_UAV, :] = self.UAV_target_positions[:self.env.n_UAV, :]
            target_agent_positions[self.env.n_UAV:, :] = self.env.get_bs_positions()
            target_agent_positions /= 1000
            map_history = np.concatenate(self.map_history, axis=0)
            map_history = torch.tensor(map_history, dtype=torch.float32).to(self.device) 
            agent_weights = []
            log_probs = []
            cur_all_weight = torch.from_numpy(self.agent_weights).to(self.device).float()
            agent_weight_actions = []
            agent_weight_masks = []
        
            last_agent_map_encoding_history = np.concatenate(self.last_agent_map_encoding_history, axis=0).reshape((self.n_agent, 1, -1)) # [n_agent, 1, n_hidden]
            last_agent_map_encoding_history = torch.tensor(last_agent_map_encoding_history, dtype=torch.float32).to(self.device)
            agent_weight_history_for_train = []
            for agent_id, agent in enumerate(self.coma_agents):
                # -----------------------------------------------------------
                agent = self.coma_agents[agent_id]
                # -----------------------------------------------------------
                weight_action_mask = np.ones(3) # 0代表不动，1代表加，2代表减
                # 判断当前agent的weight，如果>=5，则weight_action_mask[1] = 0，如果<=-5，则weight_action_mask[2] = 0
                # if self.agent_weights[agent_id] >= 2:
                #     weight_action_mask[1] = 0
                # if self.agent_weights[agent_id] <= 1:
                #     weight_action_mask[2] = 0
                # 把self.agent_weight_history封装为tensor
                agent_weight_history = np.concatenate(self.agent_weight_history, axis=0).reshape((self.n_agent, 5))
                agent_weight_history = torch.tensor(agent_weight_history, dtype=torch.float32).to(self.device)
                agent_weight_history_for_train.append(agent_weight_history)
                weight_action_mask = torch.from_numpy(weight_action_mask).to(self.device).float()
                agent_weight_masks.append(weight_action_mask)
                cur_pos = (agent_positions[agent_id, :] * 20).astype(np.int)
                cur_pos[0] = np.clip(cur_pos[0], 0, 29)
                cur_pos[1] = np.clip(cur_pos[1], 0, 19)
                agent_position_vector_x = torch.zeros(30)
                agent_position_vector_y = torch.zeros(20)
                agent_position_vector_x[cur_pos[0]] = 1
                agent_position_vector_y[cur_pos[1]] = 1
                agent_position_vector = torch.cat((agent_position_vector_x, agent_position_vector_y), dim=-1).to(self.device)
                weight_value, log_prob, map_hidden = agent.take_action(map_history, torch.tensor(agent_id, dtype=torch.float32).to(self.device), cur_all_weight, weight_action_mask, last_agent_map_encoding_history[agent_id, :, :], agent_weight_history, agent_position_vector)
                self.agent_map_encoding_history[agent_id].append(map_hidden.cpu().detach().numpy())
                log_probs.append(log_prob)
                agent_weights.append(weight_value.detach().cpu().numpy()[0])
                self.agent_weight_history[agent_id].append(weight_value.detach().cpu().numpy()[0])
            cur_agent_weight_history = np.concatenate(self.agent_weight_history, axis=0).reshape((-1))
            cur_agent_weight_history = torch.tensor(cur_agent_weight_history, dtype=torch.float32).to(self.device)
            # cur_agent_weight_history_for_train应该是cur_agent_weight_history重复n_agent次
            cur_agent_weight_history_for_train = cur_agent_weight_history.repeat(self.n_agent).view(self.n_agent, -1)
            agent_weights = np.array(agent_weights)
            agent_weight_actions = np.array(agent_weight_actions)
            last_agent_weight_history_for_train = torch.stack(agent_weight_history_for_train).to(self.device) # [n_agent, n_agent*5]
            if self.last_map_history is not None:
                # # 1. 设置reward为所有agent的reward之和
                self.area_agent_rewards = self.area_agent_rewards * (1 - 0.63) + 0.63 * self.last_area_agent_rewards # 平滑
                # self.last_area_agent_rewards = self.area_agent_rewards.copy()
                # self.area_agent_rewards = np.ones_like(self.area_agent_rewards) * np.sum(self.area_agent_rewards) 
                # 2. 根据self.agent_map_reward，计算均值mu和方差std，然后用 mu-log(std+epsilon)作为shared reward, 使用-log((reward-mu)^2+epsilon)作为agent reward
                mu = np.mean(self.area_agent_rewards)
                std = np.std(self.area_agent_rewards + 1e-3)
                reward_shared = - np.log(std + 1e-3) + 2 * np.log(mu + 1e-3)
                reward_agent = - np.log((self.area_agent_rewards - mu)**2 + 1e-3)
                self.area_agent_rewards = reward_agent + reward_shared

                agent_map_encoding_history = np.concatenate(self.agent_map_encoding_history, axis=0).reshape((self.n_agent, 1, -1)) # [n_agent, 1, n_hidden]
                agent_map_encoding_history = torch.tensor(agent_map_encoding_history, dtype=torch.float32).to(self.device)
                self.pos_replay_buffer.add_experience(self.last_map_history, map_history, torch.from_numpy(agent_weights).to(self.device).float(), torch.stack(log_probs).to(self.device), torch.from_numpy(self.area_agent_rewards).to(self.device).float(), torch.from_numpy(agent_weight_actions).to(self.device).float(), torch.stack(agent_weight_masks).to(self.device), last_agent_map_encoding_history, agent_map_encoding_history, last_agent_weight_history_for_train, cur_agent_weight_history_for_train, torch.from_numpy(agent_positions).to(self.device).float(), torch.from_numpy(target_agent_positions).to(self.device).float())
            self.area_agent_rewards = np.zeros((self.n_agent)) # 之前积累的奖励清零
            self.last_map_history = map_history.clone().detach() # 作为下一次的last_map_history
            self.last_agent_map_encoding_history = copy.deepcopy(self.agent_map_encoding_history)
            self.agent_weights = agent_weights
            self.update_can_access_map_by_weight()
    
    def resolve_map_for_CNN(self):
        # 返回每一个agent应该获取的输入，即一个2 channel的map，第一层是global map，即仅V2V下区域奖赏；第二层是0,1,-1,对于每一个agent而言，是否划分给当前agent（1），当前agent所在位置（0），不可抵达区域（-1）的map。第二层的结果根据加权Voronoi图来划分，从self.can_access_map中获取
        x_range = [self.env.min_range_x, self.env.max_range_x]
        y_range = [self.env.min_range_y, self.env.max_range_y]
        global_map, veh_num_map, cpu_num_map = self.get_map_grid_vector(self.grid_width, self.grid_num_x, self.grid_num_y, grid_x_range=x_range, grid_y_range=y_range)
        can_access_map = copy.deepcopy(self.can_access_map) # 复制当前的np.array，分别是
        uav_mobility_mask = np.zeros((self.env.n_UAV, 5)) # 0代表上，即dx=0,dy=1，1代表右，即dx=1,dy=0，2代表下，即dx=0,dy=-1，3代表左，即dx=-1,dy=0, 4代表不动
        # 遍历网格，根据加权数值判断应该属于哪个agent
        agent_position = []
        for agent_id in range(self.n_agent):
            # can_access_map[agent_id][:,:] == 1 的地方，就是global_map_for_area[:,:] = agent_id 的地方
            if agent_id < self.env.n_UAV:
                agent = self.env.UAVs[agent_id]
            else:
                agent = self.env.BSs[agent_id - self.env.n_UAV]
            agent_position.append(agent.position)
            # 以grid来划分，然后在can_access_map中记录为0
            agent_grid_x = max(0, min(int((agent.position[0] - self.env.min_range_x) // self.grid_width), self.grid_num_x-1))
            agent_grid_y = max(0, min(int((agent.position[1] - self.env.min_range_y) // self.grid_width), self.grid_num_y-1))
            # # 在can_access_map中，以grid_x, grid_y为中心，comm_range // self.grid_width为半径，且can_access_map==1的地方，都置为-2
            # comm_range = self.args.UAV_communication_range if agent_id < self.env.n_UAV else self.args.RSU_communication_range
            # comm_range = int(comm_range // self.grid_width)
            # covered_grids = np.zeros_like(can_access_map[agent_id])
            # for i in range(agent_grid_x-comm_range, agent_grid_x+comm_range+1):
            #     for j in range(agent_grid_y-comm_range, agent_grid_y+comm_range+1):
            #         if i >= 0 and i < self.grid_num_x and j >= 0 and j < self.grid_num_y:
            #             if (i - agent_grid_x)**2 + (j - agent_grid_y)**2 <= comm_range**2 and can_access_map[agent_id][i, j] == 1:
            #                 covered_grids[i, j] = 1
            # can_access_map[agent_id][covered_grids == 1] = -3
            can_access_map[agent_id][agent_grid_x, agent_grid_y] = -2

            # 更新uav_mobility_mask
            if agent_id < self.env.n_UAV:
                if agent_grid_y + 1 < self.grid_num_y:
                    can_access_0 = can_access_map[agent_id][agent_grid_x, agent_grid_y+1]
                    if can_access_0 != -1:
                        uav_mobility_mask[agent_id, 0] = 1
                if agent_grid_x + 1 < self.grid_num_x:
                    can_access_1 = can_access_map[agent_id][agent_grid_x+1, agent_grid_y]
                    if can_access_1 != -1:
                        uav_mobility_mask[agent_id, 1] = 1
                if agent_grid_y - 1 >= 0:
                    can_access_2 = can_access_map[agent_id][agent_grid_x, agent_grid_y-1]
                    if can_access_2 != -1:
                        uav_mobility_mask[agent_id, 2] = 1
                if agent_grid_x - 1 >= 0:
                    can_access_3 = can_access_map[agent_id][agent_grid_x-1, agent_grid_y]
                    if can_access_3 != -1:
                        uav_mobility_mask[agent_id, 3] = 1
                uav_mobility_mask[agent_id, 4] = 1
        returned_map = -np.ones((self.n_agent, 4, self.grid_num_x, self.grid_num_y)) # 分别代表上下左右不动5种情况下，global_map中网格的数值，最后一个格子是veh_num_map
        tmp_global_map = copy.deepcopy(global_map.reshape((self.grid_num_x, self.grid_num_y)))
        for agent_id in range(self.n_agent):
            returned_map[agent_id, 0, :, :] = tmp_global_map.copy()
            returned_map[agent_id, 1, :, :] = copy.deepcopy(veh_num_map.reshape((self.grid_num_x, self.grid_num_y)))
            returned_map[agent_id, 2, :, :] = copy.deepcopy(cpu_num_map.reshape((self.grid_num_x, self.grid_num_y)))
            returned_map[agent_id, 3, can_access_map[agent_id][:,:] == 0] = -1 # 不可通信
            returned_map[agent_id, 3, can_access_map[agent_id][:,:] == -2] = 1 # 当前位置
            returned_map[agent_id, 3, can_access_map[agent_id][:,:] == 1] = 1 # 可以通信
            # returned_map[agent_id, :, can_access_map[agent_id][:,:] == -1] = 0 # 把不可达的地方置为0，不给他看
            # 把每个agent的returned_map归一化，防止有0
            self.scale_map_reward[agent_id] = max(1, np.max(returned_map[agent_id, 0, :, :]))
        uav_mobility_mask = torch.tensor(uav_mobility_mask, dtype=torch.float32).to(self.device)
        global_map_for_area = -np.ones((4, self.grid_num_x, self.grid_num_y)) 
        global_map_for_area[0, :, :] = copy.deepcopy(global_map.reshape((self.grid_num_x, self.grid_num_y)))
        global_map_for_area[1, :, :] = copy.deepcopy(veh_num_map.reshape((self.grid_num_x, self.grid_num_y)))
        global_map_for_area[2, :, :] = copy.deepcopy(cpu_num_map.reshape((self.grid_num_x, self.grid_num_y)))
        for agent_id in range(self.n_agent):
            global_map_for_area[3, can_access_map[agent_id][:,:] == 0] = - agent_id - 1 # 代表当前agent的通信范围之外
            global_map_for_area[3, can_access_map[agent_id][:,:] == -2] = agent_id # 代表当前agent的位置
            global_map_for_area[3, can_access_map[agent_id][:,:] == 1] = agent_id # 代表当前agent的通信内
        returned_map = torch.tensor(returned_map, dtype=torch.float32).to(self.device)
        return returned_map, uav_mobility_mask, global_map_for_area
    def store_experience(self, store_area = False):
        cur_UAV_pos = self.env.get_uav_positions()
        cur_UAV_pos = torch.tensor(cur_UAV_pos, dtype=torch.float32).to(self.device) / 1000
        agent_map_for_CNN, uav_mobility_mask, _ = self.resolve_map_for_CNN()
        uav_rewards = self.uav_rewards
        # 把每个UAV的结果都存储到replay_buffer中，包括reward，由于DQN是离线的，所以可以混合使用memory
        for uav_id in range(self.env.n_UAV):
            self.replay_buffers[uav_id].add_experience(self.cur_position_state[0][uav_id,:,:,:], agent_map_for_CNN[uav_id, :, :,:].detach(), self.cur_position_state[1][uav_id,:], cur_UAV_pos[uav_id, :], self.cur_position_state[2][uav_id, :], uav_mobility_mask[uav_id, :], self.uav_directions[uav_id] * torch.ones(1).to(self.device), uav_rewards[uav_id]*torch.ones(1).to(self.device), self.last_map_encoding_history.clone())
        

    def act_mobility(self, env):
        uav_directions = []
        uav_speeds = []
        for agent_id in range(self.env.n_UAV):
            uav_gridx = max(0, min(int((self.UAV_target_positions[agent_id, 0] - self.env.min_range_x) // self.grid_width), self.grid_num_x-1))
            uav_gridy = max(0, min(int((self.UAV_target_positions[agent_id, 1] - self.env.min_range_y) // self.grid_width), self.grid_num_y-1))
            direction_action = self.direction_action[agent_id] # direction_action是弧度制
            direction = direction_action
            speed = 0
            dist = self.env.distance(self.env.UAVs[agent_id].position, self.UAV_target_positions[agent_id,:])
            if self.uav_directions[agent_id] != 4: # 4代表停止
                speed = min(self.env.uav_max_speed, dist / self.env.time_step)
            uav_directions.append(direction)
            uav_speeds.append(speed)
        return uav_directions, uav_speeds

    def act_mining_and_pay(self, env):
        return super().act_mining_and_pay(env)
    
    def act_CPU_allocation(self, env):
        for device in env.UAVs + env.BSs:
            if len(device.task_queue)>0:
                self.total_comp_time += env.TTI 
        
        cpu_allocation_for_fog_nodes = []
        serving_vehicles = env.serving_vehicles.values()
        uavs = env.UAVs
        bss = env.BSs
        devices = list(serving_vehicles) + list(uavs) + list(bss)
        for device in devices:
            task_len = len(device.task_queue)
            if task_len > 0:
                info_dict = {}
                cheat_or_not = np.zeros(shape=(task_len), dtype='bool')
                info_dict['device'] = device
                cpu_alloc = np.zeros(shape=(task_len), dtype='float')
                cpu_alloc[0] = 1
                info_dict['CPU_allocation'] = cpu_alloc
                info_dict['is_to_cheat'] = cheat_or_not
                cpu_allocation_for_fog_nodes.append(info_dict)
        return cpu_allocation_for_fog_nodes

    
    def act_RB_allocation(self, env):
        activated_offloading_tasks_with_RB_Nos = np.zeros((len(env.offloading_tasks), env.n_RB), dtype='bool')
        # 简单点，直接平均分配，不过需要考虑V2V_band, V2U_band, V2I_band: 6, 6, 8
        V2U_RB = self.args.V2U_RB
        V2V_RB = self.args.V2V_RB
        V2I_RB = self.args.V2I_RB
        V2U_cnt = 0
        V2V_cnt = 0
        V2I_cnt = 0
        max_serving_cnt = 20
        if len(env.offloading_tasks) > 0:
            # 遍历每个任务的routing[-1]，根据所在位置的横纵坐标添加到列表，然后排序，按照序号来进行资源平均分配，这样尽量保障干扰发生在比较远的两个车辆
            position_list = []
            serve_num = (len(env.offloading_tasks))
            for i in range(len(env.offloading_tasks)):
                device = env.offloading_tasks[i]['task'].routing[-1]
                last_transmit_time = env.offloading_tasks[i]['task'].last_transmit_time
                mode = env.offloading_tasks[i]['mode']
                if mode == 'V2U' or mode == 'U2V':
                    V2U_cnt += 1
                    if V2U_cnt > max_serving_cnt:
                        continue
                elif mode == 'V2I' or mode == 'I2V':
                    V2I_cnt += 1
                    if V2I_cnt > max_serving_cnt:
                        continue
                position_list.append((device.position, i, last_transmit_time))
                if len(position_list) > serve_num:
                    break
            
            serve_num = min(serve_num, len(position_list))
            V2U_cnt = 0
            V2V_cnt = 0
            V2I_cnt = 0
            position_list.sort(key=lambda x: last_transmit_time*10000 + x[0][0]//50 * 100 + x[0][1]//50)
            for pidx in range(serve_num):
                i = position_list[pidx][1]
                mode = env.offloading_tasks[i]['mode']
                if mode == 'V2U' or mode == 'U2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2U_cnt
                    V2U_cnt += 1
                elif mode == 'V2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2V_cnt
                    V2V_cnt += 1
                elif mode == 'V2I' or mode == 'I2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2I_cnt
                    V2I_cnt += 1
            avg_V2V_RB_num = max(1, int(self.args.V2V_RB / max(1, V2V_cnt)))
            avg_V2U_RB_num = max(1, int(self.args.V2U_RB / max(1, V2U_cnt)))
            avg_V2I_RB_num = max(1, int(self.args.V2I_RB / max(1, V2I_cnt)))
            for pidx in range(serve_num):
                i = position_list[pidx][1]
                mode = env.offloading_tasks[i]['mode']
                RB_cnt = env.offloading_tasks[i]['RB_cnt']
                if mode == 'V2U' or mode == 'U2V':
                    activated_offloading_tasks_with_RB_Nos[i, RB_cnt*avg_V2U_RB_num%env.n_RB:(RB_cnt+1)*avg_V2U_RB_num%env.n_RB] = True
                elif mode == 'V2V':
                    activated_offloading_tasks_with_RB_Nos[i, V2V_RB + RB_cnt*avg_V2V_RB_num%env.n_RB : V2V_RB + (RB_cnt+1)*avg_V2V_RB_num%env.n_RB] = True
                elif mode == 'V2I' or mode == 'I2V':
                    activated_offloading_tasks_with_RB_Nos[i, V2V_RB + V2U_RB + RB_cnt*avg_V2I_RB_num%env.n_RB : V2V_RB + V2U_RB + (RB_cnt+1)*avg_V2I_RB_num%env.n_RB] = True
        return activated_offloading_tasks_with_RB_Nos

    def act_offloading(self, env):
        # return []
        vehicle_list = env.vehicle_by_index
        V2VRate = self.get_V2VRateWithoutBand(env) * self.args.V2V_RB / env.n_RB
        V2URate = self.get_V2URateWithoutBand(env) * self.args.V2U_RB / env.n_RB
        U2VRate = self.get_U2VRateWithoutBand(env) * self.args.V2U_RB / env.n_RB
        V2IRate = self.get_V2IRateWithoutBand(env) * self.args.V2I_RB / env.n_RB
        I2VRate = self.get_I2VRateWithoutBand(env) * self.args.V2I_RB / env.n_RB
        tot_serving = 0
        n_tt = math.ceil(env.time_step / env.TTI)
        tot_upt = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_reqt = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_taut = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_genv_idx = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_task_idx = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_genv_cnt = [set() for _ in range(self.n_agent)]
        task_path_dict_list = []
        offloading_tasks = [] # 用来记录这个time_step内所有的决策offloading的tasks，方便计算reward
        # 添加task的数据，遍历env.to_offload_tasks，并且看他的generate_vehicle在哪个基站的覆盖范围内
        self.to_offload_task_ids = []
        for task_idx, task in enumerate(env.to_offload_tasks):
            veh = task.g_veh
            arrival_time_slot = max(0,int((task.start_time - env.cur_time) / env.TTI))
            device_idx = veh.assigned_to
            if device_idx == -1:
                continue
            device = env.UAVs[device_idx] if device_idx < env.n_UAV else env.BSs[device_idx - env.n_UAV]
            comm_range = self.args.UAV_communication_range if device_idx < env.n_UAV else self.args.RSU_communication_range
            if env.distance(veh.position, device.position) > comm_range:
                veh.assigned_to = -1
                continue
            self.to_offload_task_ids.append(task.id)
            tot_upt[device_idx][arrival_time_slot].append(task.ini_data_size)
            tot_reqt[device_idx][arrival_time_slot].append(task.ini_cpu)
            tot_taut[device_idx][arrival_time_slot].append(task.ddl // env.TTI)
            vbidx = env.vehicle_by_index.index(veh)
            tot_genv_idx[device_idx][arrival_time_slot].append(vbidx) # 通过index可以定位所有的传输车辆，把每个任务的车辆看成是不同的k即可
            tot_task_idx[device_idx][arrival_time_slot].append(task_idx)
            tot_genv_cnt[device_idx].add(vbidx)

        for i in range(env.n_UAV):
            task_cnt = 0
            n_kt = 0
            n_jt = 1 # 直接假设无人机等价于两个车,并且放在最后
            veh_ids = []
            sv_ids = []
            tv_ids = []
            cpu_res = []
            # 添加serving vehicle的数据
            for vidx, veh in enumerate(vehicle_list):
                if veh.assigned_to == i:
                    veh_ids.append(vidx)
                    intRate = V2URate[vidx, i] * U2VRate[i, :] / (V2URate[vidx, i] + U2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                        cpu_res.append(veh.CPU_frequency)
            cpu_res.append(env.UAVs[i].CPU_frequency)
            tot_serving += n_jt
            n_kt = len(tot_genv_cnt[i]) # 这个存储的是set
            if n_kt > 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                task_cnt = 0
                assigned_subscript = []
                tv_ids = list(tot_genv_cnt[i])
                for tt in range(n_tt):
                    task_num = len(tot_upt[i][tt]) # 每个时隙的任务数量最多只有1个
                    for gv_task in range(task_num):
                        k = tot_genv_idx[i][tt][gv_task] # 车辆在v_by_index里面的index
                        region_tv_index = tv_ids.index(k)
                        upt[tt, region_tv_index] = tot_upt[i][tt][gv_task]
                        reqt[tt, region_tv_index] = tot_reqt[i][tt][gv_task]
                        taut[tt, region_tv_index] = tot_taut[i][tt][gv_task]
                        
                        assigned_subscript.append((tt, region_tv_index, gv_task))
                        task_cnt += 1
                if task_cnt > 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    gt[:len(tv_ids), -1, 0] = V2URate[tv_ids, i]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    task_assignment = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = n_tt, F_jt = cpu_res, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = env.TTI, Bt = env.bandwidth)
                    for (tt, k, sub_cnt) in assigned_subscript:
                        j = np.argmax(task_assignment[tt,k,:])
                        if task_assignment[tt,k,j] == 0:
                            continue
                        k = tv_ids[k]
                        X_device = None
                        to_relay = False
                        if j + 1 >= n_jt: # 无人机
                            j = i # 无人机的idx
                            X_type = 'UAV'
                            X_device = env.UAVs[j]
                        else:
                            j = sv_ids[j]
                            X_type = 'Veh'
                            X_device = vehicle_list[j]
                            intRate = V2URate[k, i] * U2VRate[i, :] / (V2URate[k, i] + U2VRate[i, :])
                            idxs = intRate[:] > V2VRate[k, :]
                            to_relay = idxs[j]
                            if X_device in vehicle_list[k].neighbor_vehicles:
                                self.served_within_range += 1
                                dist_v_indice = vehicle_list[k].neighbor_vehicles.index(X_device)
                                self.served_as_nth_veh[dist_v_indice+1] += 1
                            else:
                                self.served_within_range += 0
                        offload_path = [{
                                'X_device':X_device
                            }]    
                        if to_relay and X_type == 'Veh': # 如果卸载到车，并且这个车是通过UAV进行的relay
                            relay_device = env.UAVs[i]
                            offload_path.insert(0, {
                                'X_device':relay_device
                            })
                        task_path_dict_list.append({
                            'task':env.to_offload_tasks[tot_task_idx[i][tt][sub_cnt]],
                            'offload_path': offload_path,
                            'task_type':'offload'
                        })
                        offloading_tasks.append(task_path_dict_list[-1]['task'])

        for bsi in range(env.n_BS):
            i = bsi + env.n_UAV
            task_cnt = 0
            n_kt = 0
            n_jt = 1 # BS=10Veh
            veh_ids = []
            sv_ids = []
            tv_ids = []
            cpu_res = []
            for vidx, veh in enumerate(vehicle_list):
                if veh.assigned_to == i:
                    veh_ids.append(vidx)
                    intRate = V2IRate[vidx, bsi] * I2VRate[bsi, :] / (V2IRate[vidx, bsi] + I2VRate[bsi, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                        cpu_res.append(veh.CPU_frequency)
            cpu_res.append(env.BSs[bsi].CPU_frequency)
            tot_serving += n_jt
            n_kt = len(tot_genv_cnt[i])
            if n_kt > 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                task_cnt = 0
                assigned_subscript = []
                tv_ids = list(tot_genv_cnt[i])
                for tt in range(n_tt):
                    task_num = len(tot_upt[i][tt]) # 每个时隙的任务数量最多只有1个
                    for gv_task in range(task_num):
                        k = tot_genv_idx[i][tt][gv_task] # 车辆在v_by_index里面的index
                        region_tv_index = tv_ids.index(k)
                        upt[tt, region_tv_index] = tot_upt[i][tt][gv_task]
                        reqt[tt, region_tv_index] = tot_reqt[i][tt][gv_task]
                        taut[tt, region_tv_index] = tot_taut[i][tt][gv_task]
                        assigned_subscript.append((tt, region_tv_index, gv_task))
                        task_cnt += 1
                if task_cnt > 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    gt[:len(tv_ids), -1, 0] = V2IRate[tv_ids, bsi]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    task_assignment = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = n_tt, F_jt = cpu_res, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = env.TTI, Bt = env.bandwidth)
                    for (tt, k, sub_cnt) in assigned_subscript:
                        j = np.argmax(task_assignment[tt,k,:])
                        if task_assignment[tt,k,j] == 0:
                            continue
                        k = tv_ids[k]
                        X_type='Veh'
                        to_relay = False
                        if j + 1 >= n_jt: # BS
                            j = bsi # BS的id
                            X_type = 'RSU'
                            X_device = env.BSs[j]
                            self.served_within_range += 1
                            self.served_as_nth_veh[0] += 1
                        else:
                            j = sv_ids[j]
                            X_type = 'Veh'
                            X_device = vehicle_list[j]
                            intRate = V2IRate[k, bsi] * I2VRate[bsi, :] / (V2IRate[k, bsi] + I2VRate[bsi, :])
                            idxs = intRate[:] > V2VRate[k, :]
                            to_relay = idxs[j]
                            if X_device in vehicle_list[k].neighbor_vehicles:
                                self.served_within_range += 1
                                dist_v_indice = vehicle_list[k].neighbor_vehicles.index(X_device)
                                self.served_as_nth_veh[dist_v_indice+1] += 1
                            else:
                                self.served_within_range += 0 # debug
                            
                        offload_path = [{
                            'X_device':X_device
                            }]    
                        if to_relay and X_type == 'Veh': # 如果卸载到车，并且这个车是通过UAV进行的relay
                            relay_device = env.BSs[bsi]
                            offload_path.insert(0,{
                                'X_device':relay_device
                            })
                        task_path_dict_list.append({
                            'task':env.to_offload_tasks[tot_task_idx[i][tt][sub_cnt]],
                            'offload_path': offload_path,
                            'task_type':'offload'
                        })
                        offloading_tasks.append(task_path_dict_list[-1]['task'])
        self.cur_offloading_tasks = offloading_tasks
        return task_path_dict_list

    def act_verification(self, env):
        return super().act_verification(env)

    def get_V2VRateWithoutBand(self, env):
        V2V_Signal = 10 ** ((env.V2V_power_dB - env.V2VChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2V_Rate_without_Bandwidth = np.log2(1 + np.divide(V2V_Signal, env.sig2))
        return V2V_Rate_without_Bandwidth

    def get_V2URateWithoutBand(self, env):
        V2U_Signal = 10 ** ((env.V2U_power_dB - env.V2UChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2U_Rate_without_Bandwidth = np.log2(1 + np.divide(V2U_Signal, env.sig2))
        return V2U_Rate_without_Bandwidth

    def get_V2IRateWithoutBand(self, env):
        V2I_Signal = 10 ** ((env.V2I_power_dB - env.V2IChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2I_Rate_without_Bandwidth = np.log2(1 + np.divide(V2I_Signal, env.sig2))
        return V2I_Rate_without_Bandwidth

    def get_I2VRateWithoutBand(self, env):
        # env.I2VChannel_with_fastfading 是 env.V2IChannel_with_fastfading 的转置
        I2V_Signal = 10 ** ((env.I2V_power_dB - env.I2VChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        I2V_Rate_without_Bandwidth = np.log2(1 + np.divide(I2V_Signal, env.sig2))
        return I2V_Rate_without_Bandwidth

    def get_U2VRateWithoutBand(self, env):
        U2V_Signal = 10 ** ((env.U2V_power_dB - env.U2VChannel_with_fastfading) / 10)[:,:,0]
        U2V_Rate_without_Bandwidth = np.log2(1 + np.divide(U2V_Signal, env.sig2))
        return U2V_Rate_without_Bandwidth

    def get_V2VAsU2URateWithoutBand(self, env):
        # 把vehicle里面的V2V信道替换为V2U信道
        V2VaU_Signal = 10 ** ((env.V2U_power_dB - env.V2VChannel_with_fastfading + 35) / 10)[:,:,0]
        V2VaU_Rate_without_Bandwidth = np.log2(1 + np.divide(V2VaU_Signal, env.sig2))
        return V2VaU_Rate_without_Bandwidth
