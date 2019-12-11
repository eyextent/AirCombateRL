#!usr/bin/env python3
# -*- coding: utf-8 -*-

import xlwt
import sys
sys.path.append('..')
import common.alloc as alloc
#from argument.dqnArgs import args
from argument.argManage import args

workbook=xlwt.Workbook(encoding='utf-8')
showbook = xlwt.Workbook(encoding="utf-8")
trainsheet=workbook.add_sheet('train', cell_overwrite_ok=True)      #训练数据
showsheet = showbook.add_sheet('show',cell_overwrite_ok=True)       #使用保存网络的数据
row0 = [u'episode',u'step',u'reward',u'success']
row1 = ['PATH_NUM','b_x','b_y','b_heading', 'b_bank','r_x','r_y','r_heading','ATA','AA']
for i in range(len(row0)):
    trainsheet.write(0, i, row0[i])
    showsheet.write(0,i,row0[i])


levin_debug = 0   # levin：debug专用，使用时设置为 0 即可

def run_AirCombat_selfPlay(env, train_agent, use_agent, train_agent_name):  
    '''
    Params：
        env:                class object
        train_agent:        class object
        use_agent:          class object
        train_agent_name:   str

    主要逻辑：
        将红、蓝智能体分为训练智能体、使用智能体进行训练；
        使用 common.alloc 模块 进行 红&蓝 与 训练&使用 之间的转换,
        完成训练和测试功能，并可以进行可视化。
    '''

    # ====  loop start ====
    if train_agent.is_train:       #训练模式(else:直接加载模型)
        suc_num = 0

        #经验池存储数据
        env.init_scen = 0
        for episode in range(args.store):
            #reset
            state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)

            if episode % 100 == 0:
                print('data collection: {} ,buffer capacity: {} '.format(episode / 100,
                                                                         len(train_agent.replay_buffer)))
            while True:
                #action
                # todo-levin: 修改 epsilon 递减机制
                action_train_agent = train_agent.egreedy_action(state_train_agent, epsilon_decay=args.epsilon_decay_during_obser)
                action_use_agent = use_agent.max_action(state_use_agent)
                if levin_debug:
                    action_use_agent = 2

                #next_state
                next_state_train_agent, next_state_use_agent, reward_train_agent, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)
                train_agent.store_data(state_train_agent, action_train_agent, reward_train_agent, next_state_train_agent, done)
                state_train_agent = next_state_train_agent
                if done:
                    break
            if(len(train_agent.replay_buffer) >= args.observe_step):
                break

        #开始训练
        for episode in range(args.episode):
            env.init_scen = 0       #训练时，蓝方飞机姿态随机
            e_reward = 0            #总reward
            step = 0                #总步长数
            state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
            while True:
                action_train_agent = train_agent.egreedy_action(state_train_agent)
                action_use_agent = use_agent.max_action(state_use_agent)
                if levin_debug:
                    action_use_agent = 2
                next_state_train_agent, next_state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

                e_reward += reward
                train_agent.perceive(state_train_agent, action_train_agent, reward, next_state_train_agent, done)
                state_train_agent = next_state_train_agent
                step += 1

                if done:
                    #print('Episode: ', episode, 'Step', step, "Reward:", e_reward, 'Success',env.success,'Advantage',env.advs)
                    break

            #训练过程中的阶段测试模式
            if (episode % args.train_episode == 0):
                train_agent.save_model()
                total_reward = 0
                total_step = 0
                blue_suc_count = 0
                red_suc_count = 0
                draw_count = 0
                for i in range(args.test_episode):
                    state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
                    step = 0
                    e_reward = 0

                    while True:
                        action_train_agent = train_agent.max_action(state_train_agent)
                        action_use_agent = use_agent.max_action(state_use_agent)
                        if levin_debug:
                            action_use_agent = 2
                        state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

                        total_reward += reward
                        total_step += 1
                        e_reward += reward
                        step += 1

                        if done:
                            if env.success == 1:
                                blue_suc_count += 1
                            elif env.success == -1:
                                red_suc_count += 1
                            else:
                                draw_count += 1

                            break
                ave_reward = total_reward / args.test_episode
                ave_step = total_step / args.test_episode
                trainsheet.write(int(episode / args.train_episode + 1), 0, (episode / args.train_episode + 1))
                trainsheet.write(int(episode / args.train_episode + 1), 1, ave_step)
                trainsheet.write(int(episode / args.train_episode + 1), 2, ave_reward)
                trainsheet.write(int(episode / args.train_episode + 1), 3, blue_suc_count)
                trainsheet.write(int(episode / args.train_episode + 1), 4, red_suc_count)
                trainsheet.write(int(episode / args.train_episode + 1), 5, draw_count)
                workbook.save(args.save_path + train_agent_name + '_data_train'  + '.xls')
                print('Episode: ', episode, "Blue success count:", blue_suc_count, "Red success count:", red_suc_count, "Draw count:", draw_count, "Average Reward:", ave_reward, train_agent.epsilon)
                if train_agent_name == 'blue':
                    if blue_suc_count >= 0.55 * args.test_episode and red_suc_count <= 0.05 * args.test_episode:
                        suc_num += 1
                    else:
                        suc_num = 0
                else:
                    if red_suc_count >= 0.55 * args.test_episode and blue_suc_count <= 0.05 * args.test_episode:
                        suc_num += 1
                    else:
                        suc_num = 0
                if suc_num >= 1:
                    train_agent.save_model(episode)
                    # pass
                # break
        

    else:    # 直接加载train_agent保存的模型，进行可视化
        for episode in range(args.episode):
            e_reward = 0
            step = 0
            tracesheet = showbook.add_sheet('trace' + str(episode + 1), cell_overwrite_ok=True)
            for i in range(len(row1)):
                tracesheet.write(0, i, row1[i])

            state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
            env.creat_ALG()
            tracesheet.write(step + 1, 0, step + 1)
            # tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
            # tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
            # tracesheet.write(step + 1, 3, float(env.ac_heading_b))
            # tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
            # tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
            # tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
            # tracesheet.write(step + 1, 7, float(env.ac_heading_r))
            # tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
            #tracesheet.write(step + 1, 8, float(env.ATA))
            #tracesheet.write(step + 1, 9, float(env.AA))

            while True:
                action_train_agent = train_agent.max_action(state_train_agent)
                action_use_agent = use_agent.max_action(state_use_agent)
                if levin_debug:
                    action_use_agent = 2
                state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

                e_reward += reward
                step += 1
                env.render()
                tracesheet.write(step + 1, 0, step + 1)
                # tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
                # tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
                # tracesheet.write(step + 1, 3, float(env.ac_heading_b))
                # tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
                # tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
                # tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
                # tracesheet.write(step + 1, 7, float(env.ac_heading_r))
                # tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
                #tracesheet.write(step + 1, 8, float(env.ATA))
                #tracesheet.write(step + 1, 9, float(env.AA))

                if done:
                    showsheet.write(episode + 1, 0, episode + 1)
                    showsheet.write(episode + 1, 1, step + 1)
                    showsheet.write(episode + 1, 2, e_reward)
                    print('Episode: ', episode, 'Step', step, "Reward:", e_reward,"Success:", env.success)
                    break
            showbook.save(args.save_path + train_agent_name + '_data_show' + '.xls')


def _test_loop(test_episode, flag_test_during_train):
    total_reward = 0
    total_step = 0
    blue_suc_count = 0
    red_suc_count = 0
    draw_count = 0

    for episode in range(args.episode):
        e_reward = 0
        step = 0

        state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)

        if not flag_test_during_train:
            env.creat_ALG()

            tracesheet = showbook.add_sheet('trace' + str(episode + 1), cell_overwrite_ok=True)
            for i in range(len(row1)):
                tracesheet.write(0, i, row1[i])
            tracesheet.write(step + 1, 0, step + 1)
            tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
            tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
            tracesheet.write(step + 1, 3, float(env.ac_heading_b))
            tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
            tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
            tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
            tracesheet.write(step + 1, 7, float(env.ac_heading_r))
            tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
            tracesheet.write(step + 1, 8, float(env.ATA))
            tracesheet.write(step + 1, 9, float(env.AA))

        while True:
            action_train_agent = train_agent.max_action(state_train_agent)
            action_use_agent = use_agent.max_action(state_use_agent)
            if levin_debug:
                action_use_agent = 2

            state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

            total_reward += reward
            total_step += 1
            e_reward += reward
            step += 1

            env.render()
            tracesheet.write(step + 1, 0, step + 1)
            tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
            tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
            tracesheet.write(step + 1, 3, float(env.ac_heading_b))
            tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
            tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
            tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
            tracesheet.write(step + 1, 7, float(env.ac_heading_r))
            tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
            tracesheet.write(step + 1, 8, float(env.ATA))
            tracesheet.write(step + 1, 9, float(env.AA))

            if done:
                showsheet.write(episode + 1, 0, episode + 1)
                showsheet.write(episode + 1, 1, step + 1)
                showsheet.write(episode + 1, 2, e_reward)
                print('Episode: ', episode, 'Step', step, "Reward:", e_reward,"Success:", env.success)
                break
        showbook.save(args.save_path + train_agent_name + '_data_show' + '.xls')


def _compute_suc_num():
    if train_agent_name == 'blue':
        if blue_suc_count >= 0.55 * args.test_episode and red_suc_count <= 0.05 * args.test_episode:
            suc_num += 1
        else:
            suc_num = 0
    else:
        if red_suc_count >= 0.55 * args.test_episode and blue_suc_count <= 0.05 * args.test_episode:
            suc_num += 1
        else:
            suc_num = 0
    if suc_num >= 1:
        train_agent.save_model(episode)



def run_AirCombat_selfPlay_change(env, train_agent, use_agent, train_agent_name):  
    '''
    Params：
        env:                class object
        train_agent:        class object
        use_agent:          class object
        train_agent_name:   str

    主要逻辑：
        将红、蓝智能体分为训练智能体、使用智能体进行训练；
        使用 common.alloc 模块 进行 红&蓝 与 训练&使用 之间的转换,
        完成训练和测试功能，并可以进行可视化。
    '''

    # ====  loop start ====
    if train_agent.is_train:       #训练模式(else:直接加载模型)
        suc_num = 0
        iter_train = 0

        flag_storing = 1  # 经验池存储数据

        for episode in range(args.episode):
            ep_reward = 0            #总reward
            step = 0                #总步长数

            # 如果是储存数据阶段，而不是训练阶段，定期输出已储存样本数
            if flag_storing and episode % 10 == 0:
                print('data collection: {} ,buffer capacity: {} '.format(episode / 10,
                                                                         len(train_agent.replay_buffer)))

            # obtain inital state
            state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
            while True:
                # 判断在observe阶段是否进行epsilon的递减
                if args.epsilon_decay_during_obser:
                    flag_epsilon_decay = 1
                else:
                    flag_epsilon_decay = 0 if flag_storing else 1
                # give the action
                action_train_agent = train_agent.egreedy_action(state_train_agent, flag_epsilon_decay)
                action_use_agent = use_agent.max_action(state_use_agent)
                # obtain next_state after taking action
                next_state_train_agent, next_state_use_agent, reward_train_agent, done = \
                                        alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)
                
                ep_reward = ep_reward + reward_train_agent
                step = step + 1
                if not flag_storing:
                    iter_train = iter_train + 1

                train_agent.store_data(state_train_agent, action_train_agent, reward_train_agent, next_state_train_agent, done)
                
                # 当储存样本数到达指定数目时，则开始训练
                if(len(train_agent.replay_buffer) >= args.observe_step):
                    flag_storing = 0
                
                # 如果是训练阶段，而不是储存数据阶段，agent 进行训练
                # todo: 训练的间隔次数设定，这里是 1
                if not flag_storing:
                    train_agent.train()
                    # todo-levin: 把2000做成参数
                    if iter_train % 2000 == 0 and args.flag_target_net:
                        train_agent.update_target_net()

                # tranfer to next_state
                state_train_agent = next_state_train_agent

                if done:
                    #print('Episode: ', episode, 'Step', step, "Reward:", e_reward, 'Success',env.success,'Advantage',env.advs)
                    break

            #训练过程中的阶段测试模式
            if (episode % args.train_episode == 0) and not flag_storing:
                train_agent.save_model()
                total_reward = 0
                total_step = 0
                blue_suc_count = 0
                red_suc_count = 0
                draw_count = 0
                for i in range(args.test_episode):
                    state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
                    step = 0
                    ep_reward = 0

                    while True:
                        action_train_agent = train_agent.max_action(state_train_agent)
                        action_use_agent = use_agent.max_action(state_use_agent)
                        if levin_debug:
                            action_use_agent = 2
                        state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

                        total_reward += reward
                        total_step += 1
                        ep_reward += reward
                        step += 1

                        if done:
                            if env.success == 1:
                                blue_suc_count += 1
                            elif env.success == -1:
                                red_suc_count += 1
                            else:
                                draw_count += 1

                            break
                ave_reward = total_reward / args.test_episode
                ave_step = total_step / args.test_episode
                trainsheet.write(int(episode / args.train_episode + 1), 0, (episode / args.train_episode + 1))
                trainsheet.write(int(episode / args.train_episode + 1), 1, ave_step)
                trainsheet.write(int(episode / args.train_episode + 1), 2, ave_reward)
                trainsheet.write(int(episode / args.train_episode + 1), 3, blue_suc_count)
                trainsheet.write(int(episode / args.train_episode + 1), 4, red_suc_count)
                trainsheet.write(int(episode / args.train_episode + 1), 5, draw_count)
                workbook.save(args.save_path + train_agent_name + '_data_train'  + '.xls')
                print('Episode: ', episode, "Blue success count:", blue_suc_count, "Red success count:", red_suc_count, "Draw count:", draw_count, "Average Reward:", ave_reward,  train_agent.epsilon)
                if train_agent_name == 'blue':
                    if blue_suc_count >= 0.55 * args.test_episode and red_suc_count <= 0.05 * args.test_episode:
                        suc_num += 1
                    else:
                        suc_num = 0
                else:
                    if red_suc_count >= 0.55 * args.test_episode and blue_suc_count <= 0.05 * args.test_episode:
                        suc_num += 1
                    else:
                        suc_num = 0
                if suc_num >= 1:
                    train_agent.save_model(episode)
                    # pass
                # break
        
    else:    # 直接加载train_agent保存的模型，进行可视化
        for episode in range(args.episode):
            e_reward = 0
            step = 0
            tracesheet = showbook.add_sheet('trace' + str(episode + 1), cell_overwrite_ok=True)
            for i in range(len(row1)):
                tracesheet.write(0, i, row1[i])

            state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)
            env.creat_ALG()
            tracesheet.write(step + 1, 0, step + 1)

            while True:
                action_train_agent = train_agent.max_action(state_train_agent)
                action_use_agent = use_agent.max_action(state_use_agent)
                if levin_debug:
                    action_use_agent = 2
                state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

                e_reward += reward
                step += 1
                env.render()
                tracesheet.write(step + 1, 0, step + 1)

                if done:
                    showsheet.write(episode + 1, 0, episode + 1)
                    showsheet.write(episode + 1, 1, step + 1)
                    showsheet.write(episode + 1, 2, e_reward)
                    print('Episode: ', episode, 'Step', step, "Reward:", e_reward,"Success:", env.success)
                    break
            showbook.save(args.save_path + train_agent_name + '_data_show' + '.xls')



def run_AirCombat_selfPlay_change(env, train_agent, use_agent, train_agent_name):  
    '''
    Params：
        env:                class object
        train_agent:        class object
        use_agent:          class object
        train_agent_name:   str

    主要逻辑：
        将红、蓝智能体分为训练智能体、使用智能体进行训练；
        使用 utlis.selfPlayUtlis模块 进行 红&蓝 与 训练&使用 之间的转换,
        完成训练和测试功能，并可以进行可视化。
    '''

    # ====  loop start ====
    if train_agent.is_train:       #训练模式(else:直接加载模型)
        suc_num = 0

        #经验池存储数据
        env.init_scen = 0
        _train_by_episode()


def _train_by_step(n_iters ):
    pass


def _test_loop(test_episode, flag_test_during_train):
    total_reward = 0
    total_step = 0
    blue_suc_count = 0
    red_suc_count = 0
    draw_count = 0

    for episode in range(args.episode):
        e_reward = 0
        step = 0

        state_train_agent, state_use_agent = alloc.env_reset(env, train_agent_name)

        if not flag_test_during_train:
            env.creat_ALG()

            tracesheet = showbook.add_sheet('trace' + str(episode + 1), cell_overwrite_ok=True)
            for i in range(len(row1)):
                tracesheet.write(0, i, row1[i])
            tracesheet.write(step + 1, 0, step + 1)
            tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
            tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
            tracesheet.write(step + 1, 3, float(env.ac_heading_b))
            tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
            tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
            tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
            tracesheet.write(step + 1, 7, float(env.ac_heading_r))
            tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
            tracesheet.write(step + 1, 8, float(env.ATA))
            tracesheet.write(step + 1, 9, float(env.AA))

        while True:
            action_train_agent = train_agent.max_action(state_train_agent)
            action_use_agent = use_agent.max_action(state_use_agent)

            state_train_agent, state_use_agent, reward, done = alloc.env_step(env, action_train_agent, action_use_agent, train_agent_name)

            total_reward += reward
            total_step += 1
            e_reward += reward
            step += 1

            env.render()
            tracesheet.write(step + 1, 0, step + 1)
            tracesheet.write(step + 1, 1, float(env.ac_pos_b[0]))
            tracesheet.write(step + 1, 2, float(env.ac_pos_b[1]))
            tracesheet.write(step + 1, 3, float(env.ac_heading_b))
            tracesheet.write(step + 1, 4, float(env.ac_bank_angle_b))
            tracesheet.write(step + 1, 5, float(env.ac_pos_r[0]))
            tracesheet.write(step + 1, 6, float(env.ac_pos_r[1]))
            tracesheet.write(step + 1, 7, float(env.ac_heading_r))
            tracesheet.write(step + 1, 8, float(env.ac_bank_angle_r))
            tracesheet.write(step + 1, 8, float(env.ATA))
            tracesheet.write(step + 1, 9, float(env.AA))

            if done:
                showsheet.write(episode + 1, 0, episode + 1)
                showsheet.write(episode + 1, 1, step + 1)
                showsheet.write(episode + 1, 2, e_reward)
                print('Episode: ', episode, 'Step', step, "Reward:", e_reward,"Success:", env.success)
                break
        showbook.save(args.save_path + train_agent_name + '_data_show' + '.xls')


def _compute_suc_num():
    if train_agent_name == 'blue':
        if blue_suc_count >= 0.55 * args.test_episode and red_suc_count <= 0.05 * args.test_episode:
            suc_num += 1
        else:
            suc_num = 0
    else:
        if red_suc_count >= 0.55 * args.test_episode and blue_suc_count <= 0.05 * args.test_episode:
            suc_num += 1
        else:
            suc_num = 0
    if suc_num >= 1:
        train_agent.save_model(episode)