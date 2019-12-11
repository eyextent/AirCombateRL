import sys
sys.path.append('..')
import envs
from models.dqn import DQN4NFSP as DQN
#from argument.dqnArgs import args
from common.utlis import set_seed
from interactor.episodeFSP import run_NFSP
from common.config import merge
from argument.argManage import args
from sacred import Experiment
from sacred.observers import FileStorageObserver
from common.config import args_wrapper_checkpoint_folder
from common.config import add_ex_config_obs



ex = Experiment('NFSP1A1U')

@ex.main
def my_main():
    args.seed = 555
    set_seed(args.seed)
    if args.flag_is_train:
        args_wrapper_checkpoint_folder(args, ex.current_run._id)
    print(args.save_path)
    run()

def run():
    env = envs.make(args.env_name)


    blue_agent = DQN(env.state_dim, env.action_dim, is_train=1, scope='blue')
    red_agent  = DQN(env.state_dim, env.action_dim, is_train=1, scope='red')
    run_NFSP(env, blue_agent, red_agent)


if __name__ == "__main__":
    '''
    参数传递；
    参数打印；
    高阶逻辑；
    等...
    '''
    # 添加观察者和配置文件
    add_ex_config_obs(ex, args, result_path=1)
    ex.run_commandline()