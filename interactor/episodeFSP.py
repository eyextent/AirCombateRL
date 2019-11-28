#!usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
sys.path.append('..')
from argument.dqnArgs import args

def run_NFSP(env, agent_blue, agent_red):
    
    for episode in range(args.episode):
        e_reward = 0            #总reward
        step = 0                #总步长数
        state_train_agent, state_use_agent
        state_blue, state_red = env.reset_selfPlay()
        while True:
            action_blue, is_best_blue = agent_blue.action(state_blue)
            action_red,  is_best_red  = agent_red.action(state_red)

            # store data for SL for training average stragery
            if is_best_blue:
                agent_blue.buffer_sl.store(state_blue, action_blue)
            if is_best_red:
                agent_red.buffer_sl.store(state_red, action_red)

            next_state_blue, next_state_red, reward_blue, reward_red, done = env.step_selfPlay(action_blue, action_red)

            # todo: both reward_blue and reward_red should be given
            agent_blue.buffer_rl.store(state_blue, action_blue, reward_blue, next_state_blue, done)
            agent_red.buffer_rl.store(state_red, action_red, reward_red, next_state_red, done)