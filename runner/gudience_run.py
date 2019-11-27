import torch
import sys

from interactor.episodeSelfPlay import run_guidence

sys.path.append('..')
import envs
from models.dqn import DQN2013
from argument.dqnArgs import args
from common.utlis import set_seed
if __name__ == '__main__':
    args.Sum_Oil = 100
    args.map_area = 10000
    args.env_name = "guidence"
    set_seed(args.seed)
    env = envs.make(args.env_name)
    train_agent = DQN2013(env.state_dim, env.action_dim, is_train=True, is_based=False, scope="guidence")
    run_guidence(env, train_agent)