import sys
sys.path.append('..')
import envs
from models.dqn import DQN4NFSP as DQN
#from argument.dqnArgs import args
from common.utlis import set_seed
from interactor.episodeFSP import run_NFSP
from common.config import merge
from argument.argManage import args
def run():

    print(args)
    env = envs.make(args.env_name)

 
    red_agent  = DQN(env.state_dim, env.action_dim, is_train=1, scope='red')
    blue_agent = DQN(env.state_dim, env.action_dim, is_train=1, scope='blue')
    run_NFSP(env, blue_agent, red_agent)


if __name__ == "__main__":
    '''
    参数传递；
    参数打印；
    高阶逻辑；
    等...
    '''

    set_seed(args.seed)
    run()