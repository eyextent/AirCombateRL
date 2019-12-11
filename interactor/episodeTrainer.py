#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
主函数逻辑：单进程
'''
# from argument.dqnArgs import args
from argument.argManage import args


def run_GuidenceEnv(env, train_agent):
    # ====  loop start ====
    if train_agent.is_train:
        suc_num = 0
        for episode in range(args.store):
            state = env.reset()
            if episode % 100 == 0:
                print('data collection: {} ,buffer capacity: {} '.format(episode / 100,
                                                                         len(train_agent.replay_buffer)))
            while True:
                action = train_agent.egreedy_action(state)
                next_state, reward, done = env.step(action)
                train_agent.store_data(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            if len(train_agent.replay_buffer) >= 100000:
                break
        # 开始训练
        for episode in range(args.episode):
            e_reward = 0
            step = 0
            state = env.reset()
            while True:
                action = train_agent.egreedy_action(state)
                next_state, reward, done = env.step(action)
                e_reward += reward
                train_agent.perceive(state, action, reward, next_state, done)
                state = next_state
                step += 1
                if done:
                    print('Episode: ', episode, 'Step', step, "Reward:", e_reward, train_agent.epsilon, env.acts)
                    break
            if episode % args.train_episode == 0:
                train_agent.save_model()
                total_reward = 0
                for i in range(args.test_episode):
                    state = env.reset()
                    step = 0
                    e_reward = 0
                    while True:
                        action = train_agent.max_action(state)
                        state, reward, done = env.step(action)
                        total_reward += reward
                        e_reward += reward
                        step += 1
                        if done:
                            break
                ave_reward = total_reward / args.test_episode
                print('episode: ', episode, "Avarge Reward:", ave_reward)
                if ave_reward >= 19.7:
                    suc_num += 1
                else:
                    suc_num = 0
                if suc_num >= 5:
                    break
