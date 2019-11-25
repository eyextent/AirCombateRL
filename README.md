

# 整体框架

```
参数管理模块
	argumnet/
环境模块
	envs/
算法模块
	memoryBuffer/
	models/
	interactor/
	runner/ 	
结果记录与保存模块
	logger
其他（工具函数等）
	utlis/
```



## 参数管理模块

管理所有的超参数

```
argument/
	__init__.py
	# 以下文件包含具体的超参数
	# 在写程序时定义超参数就在这里面直接定义
	dqnArgs.py 
	pgArgs.py
	xxArgs.py
	...
```

- 说明：

  方便统一管理超参数。

  - 参数的定义与使用

    当编写相关函数需要定义超参数时，直接在此模块使用 `parser.add_argument` 添加参数；

    定义之后，import 该 args 之后，使用 args.xxx 来使用超参数。

  - 注意名字类参数的提示信息

    设置为str的参数变量，如游戏名字、神经网络结构名字等，需要在help中列出所有的名字列表，

    以便定义参数时，通过help命令获得具体名字提示。

  - args_wrapper函数

    主要是对保存路径进行封装，防止覆盖。



## 环境模块

```
envs/
	__init__.py
	units.py
	airCombateEnv.py
```

说明：

- `__init__`.py

  主要包含`REGISTRY` 字典，对创建的环境进行注册，方便通过环境名字加载对应的环境。

  使用举例：

  创建一个新环境的类，使用 “airCobate” 名字进行注册，之后在算法模块使用名字创建环境：

  ```python
  import envs
  env = envs.make("airCobate")
  ```

- units.py

  负责定义飞机等作战单位，继承 `Aircraft`基类 来构建 新的飞机类型。

  文件最后有个`REGISTRY`字典，每次创建新的飞机类型之后在 `REGISTRY` 中进行注册，方便通过传递名字来构建飞机。

- airCombateEnv.py

  包含基类 `env`，给出算法与环境的接口。

  以及根据任务的不同通过`_get_reward`函数来给出奖励值；同时可以在该函数内进行reward shaping。

  另外，在构造函数`__init__(self,)`中，必须给出 self.state_dim 和 self.action_dim，方便算法模块构建相应的神经网络等。

  创建新的环境类后，需要在 `__init__.py`中进行环境的注册。

  

  - 构造多个智能体的逻辑（示例中红方和蓝方均为1架飞机）：

    - 首先在参数管理模块中，设置 --red_unit_type_list=[type1, type2 ...]，想要几架飞机就在列表中添加几个元素。blue_unit_type也做同样操作。

    - 然后在环境 envs/airCombateEnv.py 中的实现方法：

      ```python
      # 以红方飞机为例
      for name in args.red_unit_type_list:  # args.red_unit_type_list为飞机类型名字列表
          red_unit = registry_unints[name]()	
          self.red_unit_list.append(red_unit) # self.red_unit_list为 类的实例列表
      # 【注意：】
      # 如果想在构造飞机时，初始化不同的飞机属性参数，实现方法有2种：
      # ① 定义为不同的类，即不同的type
      # ② 在参数模块定义red_unit_type_list等时，使用字典替代列表，即
      #	--red_unit_type_list={type1：(参数1), type2：(参数2)}
      #	使用时：
      #		for name,arg in args.red_unit_type_list: 
      #    		red_unit = registry_unints[name](*arg)	
      #    		self.red_unit_list.append(red_unit) 
          
          
      # 当进行移动、取参量等操作时，使用遍历操作
      for unit in self.red_unit_list:
          unit.move(action)
      ```

  - 一个智能体控制多个单位的逻辑：

    当使用一个智能体控制己方多个单位时，其实就是状态空间变为多个单位的联结状态空间，动作空间根据算法而定：最简单的就是联结动作空间，根据每个单位的动作维度划分联结动作空间。

    另一种实现方法是采用 independent learners的方法。




## 算法模块

```
memoryBuffer/
	replayBuffer.py		# 主要是值函数使用的buffer （基础、优先级经验回放）
	trajBuffer.py		# 主要是actor-critic相关的算法使用的buffer，需要保存一个trajectory
	...			# todo: 多智能体使用的Buffer 
models/
	netFrame.py		# 定义各种神经网络结构	
	dqn.py			# 使用神经网络构建DQN类
interactor/
	episodeSelfPlay.py		# 进行（红、蓝）博弈训练
	episodeTrainer.py		# 单智能体训练
	parallelSelfPlay.py		# 多环境并行化博弈训练（未实现...）
	parallelTrainer.py		# 多环境并行化单智能体训练（未实现...）
runner/
	selfPlay.py		# 主要包含红、蓝交替的逻辑，其实就是main.py
	...			# 其他方法的main.py
```

说明：

- memoryBuffer/

  创建各类与经验回放技术相关的类，包含基类 `Buffer` 。

  其中，在使用基类中需要注意：

  - `self.Transition`

    子类在继承 `Buffer`后，需要定义自己的

    例子见子类 `ReplayBufferTransition(Buffer)`

  - `_piexl_processing` 和 `_piexl_rev_processing`

    仅当状态为图像时使用，用于节约内存

- models/

  用来创建各种算法模型

  - netFrame.py

    在定义完神经网络机构之后，同样在文件最后需要进行名字的注册。

    目前实现的网络结构主要包括：mlp 和 cnn-mlp

    待实现：rnn-mlp（lstm-mlp），cnn-rnn-mlp

  - dqn.py

    包含基类 `DQN`，通过继承实现 DQN2013 和 DQN2015 等

- interactor/

  将方法和环境的对象作为输入，完成智能体与环境交互的逻辑，包括模型训练、测试等实现。

- runner/

  可理解为main函数的实现，程序启动文件。

  在博弈训练中可能针对trainer多进行一次封装，如交换训练方等。




## 其他（工具函数等）

包含各类其他模块需要使用的工具类函数。

```
utlis/
	alloc.py	# 和博弈训练相关的工具函数
	utlis.py	# 其他工具函数，包括 设置随机种子等在其他模块常用的函数
```



## 结果记录与保存模块

主要是结果的保存和记录。





# 部分函数解释:

## 1.注册 --- `REGISTRY` 

`REGISTRY` 实际上就是一个字典，字典的value为某个类或者函数，key就是人为设定的名字。

以构建net_frame为例：

- 构建

  ```python
  # 创建相关的函数和类（value的内容）
  def net_frame_mlp():
      xxx
      
  def net_frame_cnn_to_mlp():
  	xxx
      
  # 创建完函数或类之后进行注册    
  REGISTRY = {}
  REGISTRY["mlp"] = net_frame_mlp				# “mlp”为人为设定的名字，net_frame_mlp为函数名
  REGISTRY["cnn2mlp"] = net_frame_cnn_to_mlp
  
  那么，
  REGISTRY["mlp"]()  就是  net_frame_mlp()
  ```

- 使用

  ```python
  # 加载REGISTRY
  from models.netFrame import REGISTRY as registry_net_frame
  
  registry_net_frame["mlp"](*args) 就等同于 net_frame_mlp(*args) # *args为函数参数
  所以只需要通过参数传递 "mlp"等名字 就可以使用指定的神经网络结构
  ```

## 2.经验回放的collections.namedtuple使用

- 定义

  ```python
  class ReplayBufferTransition(Buffer):
      def __init__(self, capacity, flag_piexl=0):
          super(ReplayBufferTransition, self).__init__(capacity, flag_piexl)
          self.Transition = collections.namedtuple("Transition" , 
                  ["state", "action", "reward", "next_state", "done", "episode_return"])
          
  # 继承基类Buffer，使用super()函数重写基类__init__()函数，添加 self.Transition
  # 即使用上述定义的 collections.namedtuple 来保存 transition<s, a, r, s', done>
  ```

- 使用：

  ```python
  # 加载经验缓存池类
  from memoryBuffer.replayBuffer import ReplayBufferTransition
  
  # 构建经验缓存池
  buffer = ReplayBufferTransition(50)
  # 存储
  buffer.store(1,2,3,4,5)
  buffer.store(6,7,8,9,10)
  print(buffer.replay_buffer)
  
  ==>
  [Transition(state=1, action=2, reward=3, next_state=4, done=5.0, episode_return=0),    Transition(state=6, action=7, reward=8, next_state=9, done=10.0, episode_return=0)]
  ```


## 3.argument中的args_wrapper函数

主要是对结果的保存路径进行封装，防止覆盖。

结果均放置在result目录内，定义自己的结果存放目录有两种方法：

- 修改--experiment_name参数内容：

  举例：

  ```python
  --experiment_name = 'your_name/experiment_name'
  ```

- 在args_wrapper函数中修改 `your name` 内容

  ```python
  def args_wrapper(args):
      # 主要是对重复训练的保存路径进行封装
      # None
      args_origin.save_path = args_origin.source_path + '/' + args_origin.experiment_name + '/'
      # args_origin.save_path = args_origin.source_path + '/' 'your name' + '/' + args_origin.experiment_name + '/'
      if not os.path.exists(args_origin.save_path):
      	os.makedirs(args_origin.save_path)
      return args
  
  args = args_wrapper(args_origin)
  ```

  





# 待解决

+ [ ] logger.py 的编写 以及 替代原代码

+ [ ] 结果保存文件夹自动命名

  现在只是指定experiment_name来更换结果保存文件夹，还应该添加模式指定自动编码方式进行储存等模式









