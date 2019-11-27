#!usr/bin/env python3
# -*- coding: utf-8 -*-

def env_reset(env, train_agent_name):
    state_blue, state_red = env.reset_selfPlay()
    state_train_agent, state_use_agent = alloc_state(state_blue, state_red, train_agent_name)
    return state_train_agent, state_use_agent

def env_step(env, action_train_agent, action_use_agent, train_agent_name):
    action_blue, action_red = alloc_action(action_train_agent, action_use_agent, train_agent_name)
    next_state_blue, next_state_red, reward_b, reward_r, done = env.step_selfPlay(action_blue, action_red) # todo-levin: 设置两种state，还是设置统一使用的state
    next_state_train_agent, next_state_use_agent = alloc_state(next_state_blue, next_state_red, train_agent_name)
    reward_train_agent, reward_use_agent = alloc_reward(reward_b, reward_r,train_agent_name)
    return next_state_train_agent, next_state_use_agent, reward_train_agent, done

def alloc_state(state_blue, state_red, train_agent_name):
    if train_agent_name == 'blue':
        state_train_agent = state_blue
        state_use_agent   = state_red
    else:
        state_train_agent = state_red
        state_use_agent   = state_blue
    return state_train_agent, state_use_agent

def alloc_reward(reward_b, reward_r,train_agent_name):
    if train_agent_name == 'blue':
        reward_train_agent = reward_b
        reward_use_agent   = reward_r
    else:
        reward_train_agent = reward_r
        reward_use_agent   = reward_b
    return reward_train_agent, reward_use_agent

def alloc_action(action_train_agent, action_use_action, train_agent_name):
    if train_agent_name == 'blue':
        action_blue = action_train_agent
        action_red  = action_use_action
    else:
        action_blue = action_use_action
        action_red  = action_train_agent
    return action_blue, action_red



def check_scheme(IsTrain_1, IsTrain_2, train_agent_name):
    '''
    在 trainer 中使用
    '''
    if IsTrain_1 ^ IsTrain_2:
        print("\n === train " + train_agent_name + " agent ===\n")
    elif IsTrain_1 | IsTrain_2:
        # levin-todo: raise Error
        print("\n!!! Error: both agent is training !!!\n")
    else:
        print("\n === test " + train_agent_name + " agent ===\n")