import sys
import envs
from models.dqn import DQN2013
from argument.argManage import args
from common.utlis import set_seed
from interactor.episodeTrainer import run_GuidenceEnv

sys.path.append('..')



def run():
    env = envs.make(args.env_name)
    train_agent = DQN2013(env.state_dim, env.action_dim, is_train=True, is_based=False, scope="guidence")
    run_GuidenceEnv(env, train_agent)


if __name__ == '__main__':
    args.Sum_Oil = 100
    args.map_area = 10000
    args.env_name = "guidence"
    args.experiment_name = "guidence"
    set_seed(args.seed)
    run()
