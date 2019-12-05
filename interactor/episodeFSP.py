#!usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
sys.path.append('..')
from argument.dqnArgs import args

def run_NFSP(env, agent_blue, agent_red):
    if args.flag_is_train:
        for episode in range(args.episode):
            e_reward = 0            #总reward
            step = 0                #总步长数
            
            
            state_blue, state_red = env.reset_selfPlay()
            while True:
                action_blue, is_best_response_blue = agent_blue.NFSP_action(state_blue)
                action_red,  is_best_response_red  = agent_red.NFSP_action(state_red)

                # store data for SL for training average stragery
                if is_best_response_blue:
                    agent_blue.buffer_sl.store(state_blue, action_blue)
                if is_best_response_red:
                    agent_red.buffer_sl.store(state_red, action_red)

                next_state_blue, next_state_red, reward_blue, reward_red, done = env.step_selfPlay(action_blue, action_red)

                # todo: both reward_blue and reward_red should be given
                agent_blue.buffer_rl.store(state_blue, action_blue, reward_blue, next_state_blue, done)
                agent_red.buffer_rl.store(state_red, action_red, reward_red, next_state_red, done)

                if len(agent_blue.buffer_rl) > args.batch_size * 4:
                    agent_blue.train_rl()
                if len(agent_red.buffer_rl) > args.batch_size * 4:
                    agent_red.train_rl()

                if len(agent_blue.buffer_sl) > args.batch_size * 4:
                    agent_blue.train_sl()
                if len(agent_red.buffer_sl) > args.batch_size * 4:
                    agent_red.train_sl()

                state_blue, state_red = next_state_blue, next_state_red
                if done:
                    break

            if episode % 100 == 0:
                blue_suc_count = 0
                red_suc_count = 0
                draw_count = 0
                for ep in range(20):
                    state_blue, state_red = env.reset_selfPlay()
                    while True:
                        action_blue = agent_blue.best_response(state_blue)
                        action_red  = agent_red.average_stargiey(state_red)
                        next_state_blue, next_state_red, reward_blue, reward_red, done = env.step_selfPlay(action_blue, action_red)
                        state_blue, state_red = next_state_blue, next_state_red
                        if done:
                            if env.success == 1:
                                blue_suc_count += 1
                            elif env.success == -1:
                                red_suc_count += 1
                            draw_count += 1

                            break
                print('Episode: ', episode, "Blue success count:", blue_suc_count, "Red success count:", red_suc_count, "Draw count:", draw_count)

                if blue_suc_count > 18:
                    agent_blue.save_model(episode)
                    agent_red.save_model(episode)
    else:
        blue_suc_count = 0
        red_suc_count = 0
        draw_count = 0
        env.creat_ALG()
        for ep in range(100):
            state_blue, state_red = env.reset_selfPlay()
            
            while True:
                env.render()
                action_blue = agent_blue.best_response(state_blue)
                action_red  = agent_red.average_stargiey(state_red)
                next_state_blue, next_state_red, reward_blue, reward_red, done = env.step_selfPlay(action_blue, action_red)
                state_blue, state_red = next_state_blue, next_state_red
                if done:
                    print(env.success)
                    if env.success == 1:
                        blue_suc_count += 1
                    elif env.success == -1:
                        red_suc_count += 1
                    draw_count += 1
                    break
        print('Episode: ', ep , "Blue success count:", blue_suc_count, "Red success count:", red_suc_count, "Draw count:", draw_count)
