#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque
import os
import random
import sys
sys.path.append("..")
from memoryBuffer.replayBuffer import ReplayBuffer, SuperviseLearningBuffer
from models.components import REGISTRY as registry_net_frame
import common.utlis as U
#from argument.dqnArgs import args
from argument.argManage import args
from common.utlis import set_seed

# todo: 判断需要更全面，添加报错机制
net_frmae = registry_net_frame[args.net_frame]
if 'cnn' in args.net_frame:
    print("\nWarning: must def convs!")


class DQN(object):
    def __init__(self, state_dim, n_action):
        self.replay_buffer = ReplayBuffer(args.replay_size)
        self.epsilon = args.initial_epsilon
        self.state_dim = state_dim
        self.n_action = n_action

    def load_parms(self):
        raise NotImplementedError

    def create_training_method(self):
        raise NotImplementedError

    def perceive(self):
        raise NotImplementedError

    def train_network(self):
        raise NotImplementedError

    def update_target_net(self):
        raise NotImplementedError

    def sample_action(self):
        raise NotImplementedError

    def greedy_action(self):
        raise NotImplementedError


class DQN2013(DQN):
    def __init__(self, state_dim, n_action, is_train=False, is_based=False, scope=None):
        super(DQN2013, self).__init__(state_dim, n_action)
        self.scope = scope

        self.is_train = is_train
        self.is_based = is_based

        self.bool_defaule_action = False

        self.gamma = args.gamma
        self.learning_rate = args.learning_rate

        self.save_path = args.save_path
        # todo： 保存路径里面添加 checkpoint_floder_name
        self.checkpoint_folder_name = args.checkpoint_folder_name
        self.file_name = args.file_name


        self.flag_target_net = args.flag_target_net     # 是否使用 target_network （2013 or 2015）

        self.model = registry_net_frame[args.net_frame](self.state_dim, self.n_action)
        if self.flag_target_net:
            self.target_model = registry_net_frame[args.net_frame](self.state_dim, self.n_action)

        if U.is_cuda():
            self.model = self.model.cuda()
            if self.flag_target_net:
                self.target_model = self.target_model.cuda() 
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load_parms()
        
        
    def _load_parms(self):
        # todo: if self.scope:  pkl_file_path = xxxx    else:  pkl_save_path =  xxx   [in __init__()]
        if self.is_train:
            if self.is_based:
                self.epsilon = INITIAL_EPSILON * 0.5
                self.model.load_state_dict(torch.load(self.save_path + "/" + self.checkpoint_folder_name+"/" +self.scope + self.file_name))
                #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
                # else:
                #     print("Could not find old network weights")
        else:
            file_path = self.save_path + "/" + self.checkpoint_folder_name+"/" + self.scope + self.file_name
            if os.path.exists(file_path):
                self.model.load_state_dict(torch.load(file_path))
                print("\n\n\n=======Successfully loaded:" + file_path + "========")
            else:
                print("\n\n\n ========== LoadNone: default action for use_agent ===============")
                print("Could not find old network weights, the agent will keep choosing default_action = 2 in [0,1,2]")
                print("=========== please make ture your save_path is correct ==============\n")
                self.bool_defaule_action = True
                

    def train(self):
        assert len(self.replay_buffer) >= args.batch_size
        state, action, reward, next_state, done = self.replay_buffer.sample(args.batch_size)

        state      = U.Variable(torch.FloatTensor(state.astype(np.float32)))
        next_state = U.Variable(torch.FloatTensor(next_state.astype(np.float32)))
        action     = U.Variable(torch.LongTensor(action))
        reward     = U.Variable(torch.FloatTensor(reward))
        done       = U.Variable(torch.FloatTensor(done))

        q_values      = self.model(state)
        if self.flag_target_net:
            next_q_values = self.target_model(next_state)
        else:
            next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value_max = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value_max * (1 - done)
        
        loss = (q_value - U.Variable(expected_q_value.detach())).pow(2).mean()            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        assert self.flag_target_net
        self.target_model.load_state_dict(self.model.state_dict())
        print("update target network")

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward,
                                   next_state, done)
        if len(self.replay_buffer) > args.replay_size:
            self.replay_buffer.pop()
        if len(self.replay_buffer) > args.batch_size:
            self.train()

    def store_data(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) > args.replay_size:
            self.replay_buffer.pop()
        self.replay_buffer.store(state, action, reward, next_state, done)

    def egreedy_action(self, state, epsilon_decay=1):
        if epsilon_decay:
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon - 0.000005
            else:
                self.epsilon = self.epsilon * args.decay_rate

        if random.random() > self.epsilon:
            state   = U.Variable(torch.FloatTensor(state.astype(np.float32)).unsqueeze(0))
            q_value = self.model(state)
            action_max_value, index = torch.max(q_value, 1)
            action = index.item()
            # action  = q_value.max(1)[1].data[0]
#             print(action)
        else:
            action = random.randrange(self.n_action)
        return action

    def max_action(self, state):
        if self.bool_defaule_action:
            return 2
        else:
            state   = U.Variable(torch.FloatTensor(state.astype(np.float32)).unsqueeze(0))
            q_value = self.model(state)
            action_max_value, index = torch.max(q_value, 1)
            action = index.item()
            return action

    def save_model(self, iter_num=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if iter_num is None:
            torch.save(self.model.state_dict(), self.save_path + "/" + self.checkpoint_folder_name+"/" + self.scope + self.file_name)
        else:
            torch.save(self.model.state_dict(), self.save_path + "/" + self.checkpoint_folder_name+"/" + str(iter_num) + self.scope + self.file_name)


class DQN4NFSP(DQN):
    def __init__(self, state_dim, action_dim, scope, is_train=1, is_based=0):
        super(DQN4NFSP, self).__init__(state_dim, action_dim)
        self.buffer_rl = self.replay_buffer                         # 强化学习使用的buffer，简单的对rl_buffer对象进行重命名
        self.buffer_sl = SuperviseLearningBuffer(100000) # 监督学习使用的buffer    args.sl_size
        self.scope = scope

        self.is_train = is_train
        self.is_based = is_based
        self.flag_target_net = args.flag_target_net     # 是否使用 target_network （2013 or 2015）

        self.gamma = args.gamma
        self.learning_rate_rl = 0.0005
        self.learning_rate_sl = 0.001
        # self.learning_rate_rl = args.learning_rate_rl
        # self.learning_rate_sl = args.learning_rate_sl

        self.eta = 0.1
        # self.eta = args.eta

        self.save_path = args.save_path
        # todo： 保存路径里面添加 checkpoint_floder_name
        self.checkpoint_folder_name = args.checkpoint_folder_name
        self.file_name = args.file_name

        self.model_rl = registry_net_frame[args.net_frame](self.state_dim, self.n_action)
        self.model_sl = registry_net_frame[args.net_frame](self.state_dim, self.n_action)

        if self.flag_target_net:
            self.target_model_rl = registry_net_frame[args.net_frame](self.state_dim, self.n_action)

        if U.is_cuda():
            self.model_rl = self.model_rl.cuda()
            self.model_sl = self.model_sl.cuda()
            if self.flag_target_net:
                self.target_model_rl = self.target_model_rl.cuda() 
        
        self.optimizer_rl = optim.Adam(self.model_rl.parameters(), lr=self.learning_rate_rl)
        self.optimizer_sl = optim.Adam(self.model_sl.parameters(), lr=self.learning_rate_sl)
        self._load_parms()
        
        
    def _load_parms(self, iter_num=None):
        # todo: if self.scope:  pkl_file_path = xxxx    else:  pkl_save_path =  xxx   [in __init__()]
        # levin：这里删除了 is_based 功能
        if iter_num is not None:
            file_path = self.save_path + str(iter_num) + self.scope + self.file_name
        else:
            file_path = self.save_path + self.scope + self.file_name
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model_rl.load_state_dict(checkpoint["model_rl"])
            self.model_sl.load_state_dict(checkpoint["model_sl"])
            # self.target_model_rl.load_state_dict(checkpoint["target_model_rl"])
            print("\n\n\n=======Successfully loaded:" + file_path + "========")
        else:
            print("\n\n\n ========== LoadNone: default action for use_agent ===============")
            print("Could not find old network weights, the agent will keep choosing default_action = 2 in [0,1,2]")
            print("=========== please make ture your save_path is correct ==============\n")
            self.bool_defaule_action = True

    def save_model(self, iter_num=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if iter_num is None:
            torch.save({'model_rl': self.model_rl.state_dict(), \
                        'model_sl': self.model_sl.state_dict(), \
                        'target_model_rl': self.target_model_rl.state_dict()}, \
                self.save_path + self.scope + self.file_name)
        else:
            torch.save({'model_rl': self.model_rl.state_dict(), \
                        'model_sl': self.model_sl.state_dict()}, \
                        # 'target_model_rl': self.target_model_rl.state_dict()}, \
                self.save_path + str(iter_num) + self.scope + self.file_name)
                
    def train_rl(self):
        assert len(self.buffer_rl) >= args.batch_size
        state, action, reward, next_state, done = self.buffer_rl.sample(args.batch_size)

        state      = U.Variable(torch.FloatTensor(state.astype(np.float32)))
        next_state = U.Variable(torch.FloatTensor(next_state.astype(np.float32)))
        action     = U.Variable(torch.LongTensor(action))
        reward     = U.Variable(torch.FloatTensor(reward))
        done       = U.Variable(torch.FloatTensor(done))

        q_values   = self.model_rl(state)
        if self.flag_target_net:
            next_q_values = self.target_model_rl(next_state)
        else:
            next_q_values = self.model_rl(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value_max = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value_max * (1 - done)
        
        loss_rl = (q_value - U.Variable(expected_q_value.detach())).pow(2).mean()            
        self.optimizer_rl.zero_grad()
        loss_rl.backward()
        self.optimizer_rl.step()

    def train_sl(self):
        assert len(self.buffer_sl) >= args.batch_size
        state, action = self.buffer_sl.sample(args.batch_size)

        state  = U.Variable(torch.FloatTensor(state.astype(np.float32)))
        action = U.Variable(torch.LongTensor(action))

        logits = self.model_sl(state)
        action_one_hot = F.one_hot(action, action.size()[0])
        logits_action = logits.gather(1, action.unsqueeze(1)).squeeze(1)

        loss_sl = -(torch.log(logits_action)).mean()
        self.optimizer_sl.zero_grad()
        loss_sl.backward()
        self.optimizer_sl.step()

    def update_target_net(self):
        assert self.flag_target_net
        self.target_model_rl.load_state_dict(self.model_rl.state_dict())
        print("update target network")

    def store_data_rl(self, state, action, reward, next_state, done):
        self.buffer_rl.store(state, action, reward, next_state, done)

    def store_data_sl(self, state, action):
        self.buffer_sl.store(state, action)    

    def NFSP_action(self, state, epsilon_decay=1, eta_decay=0):
        '''
        Param:
            self.eta:  probability for best_response or average_stargery
        '''
        if epsilon_decay:
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon - 0.00005
            else:
                self.epsilon = self.epsilon * args.decay_rate

        if random.random() > self.eta:
            is_best_response = False
            action = self.average_stargiey(state)
        else:
            is_best_response = True
            if random.random() > self.epsilon:
                action = self.best_response(state)
            else:
                action = random.randrange(self.n_action)
        return action, is_best_response

    def best_response(self, state):
        state   = U.Variable(torch.FloatTensor(state.astype(np.float32)).unsqueeze(0))
        q_value = self.model_rl(state)
        action_max_value, index = torch.max(q_value, 1)
        action = index.item()
        return action

    def average_stargiey(self, state):
        state   = U.Variable(torch.FloatTensor(state.astype(np.float32)).unsqueeze(0))
        logits = self.model_sl(state)
        action_max_value, index = torch.max(logits, 1)
        action = index.item()
        return action