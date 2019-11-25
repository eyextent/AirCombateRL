#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DQN中使用的经验缓存池
'''
import collections
import numpy as np
import random
import sys
sys.path.append("..")
from argument.dqnArgs import args

class Buffer(object):
    '''
    基类
    '''
    def __init__(self, capacity, flag_piexl=0):
        self.replay_buffer = []
        self.capacity = capacity
        self.flag_piexl = flag_piexl
        self.size = len(self.replay_buffer)

    def store(self, **args):
        '''
        根据任务自己定义：
            如果是常规单智能体，子类中可以使用如下参数：
                state, action, reward, next_state, done
            如果需要存储多智能体，则酌情改变参数，
                注意：须与__init__()函数中self.Transition的定义对应
        '''
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def pop(self):
        self.replay_buffer.pop(0)

    def _update_size(self):
        '''
        每次在对replay_buffer进行操作后调用此函数，以便实时更新size数值。
        '''
        self.size = len(self.replay_buffer)

    def _piexl_processing(state, next_state):
        '''
        当 state 为pixel类型时，即图像类型的状态（观察）；
        则 进行此处理，可以节约存储空间。
        同时，使用时需要对从buffer中采样(sample)的state进行逆处理。
        '''
        assert np.amin(state) >= 0.0
        assert np.amax(state) <= 1.0

        # Class LazyFrame --> np.array()
        state = np.array(state)
        next_state = np.array(next_state)

        state  = (state * 255).round().astype(np.uint8)
        next_state = (next_state * 255).round().astype(np.uint8)

    def _piexl_rev_processing(state, next_state):
        state = state.astype(np.float32) / 255.0
        next_state = next_state.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.replay_buffer)



class ReplayBuffer(Buffer):
    def __init__(self, capacity, flag_piexl=0):
        super(ReplayBuffer, self).__init__(capacity, flag_piexl)
        self.Transition = collections.namedtuple("Transition" , ["state", "action", "reward", "next_state", "done"])

    def store(self, state, action, reward, next_state, done):
        if self.size > self.capacity:
            self.replay_buffer.pop(0)
            self._update_size()

        if self.flag_piexl:
            self._piexl_processing(state, next_state)

        self.replay_buffer.append(self.Transition(state, action , reward , next_state , float(done)))
        self._update_size()
 
    def sample(self, batch_size):
        batch_transition = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = map(np.array , zip(*batch_transition))

        if self.flag_piexl:
            self._piexl_re_processing(state, next_state)

        return state, action, reward, next_state, done


# buffer for momery replay
class ReplayBuffer_backup(Buffer):
    def __init__(self, capacity, flag_piexl=0):
        super(ReplayBuffer, self).__init__(capacity, flag_piexl)
        self.Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state" , "done"])
    
    def append(self, state, action, reward, next_state, done):
        # state      = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)
        
        self.replay_buffer.append(self.Transition(state, action, reward, next_state, float(done)))
    
    def sample(self, batch_size):
        batch_transition = random.sample(self.replay_buffer , batch_size)
                #***
        # batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array , zip(*batch_transition))
        # return np.concatenate(state), action, reward, np.concatenate(next_state), done
        return batch_transition
    
    def pop(self):
        self.replay_buffer.pop(0)

    