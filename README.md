# PushT Human - 交互式推方块环境

一个基于gym-pusht的交互式强化学习环境，支持人类键盘控制、策略切换、轨迹记录和数据上传功能。

## 🚀 功能特点

- **🎮 人机交互**: 支持键盘和鼠标控制智能体移动
- **🔄 策略切换**: 一键切换用户控制和AI策略控制
- **📊 轨迹记录**: 自动记录状态、动作、奖励等完整轨迹数据
- **🎬 轨迹回放**: 支持回放已记录的轨迹，包含初始状态信息
- **🤖 强化学习训练**: 基于stable_baselines3的PPO和SAC算法训练
- **📈 实验管理**: 集成WandB进行实验跟踪和可视化
- **☁️ 数据上传**: 支持将数据上传到Hugging Face Hub
- **⚙️ 配置管理**: 基于Hydra的灵活配置系统
- **🎯 多观测模式**: 支持状态向量、像素图像等多种观测类型

## 📦 安装依赖

```bash
# 安装基础依赖
pip install pygame gymnasium hydra-core omegaconf
pip install datasets huggingface_hub
pip install numpy opencv-python

# 安装强化学习训练依赖
pip install stable-baselines3[extra] wandb torch

# 安装gym-pusht环境 (可编辑模式)
cd gym-pusht
pip install -e .
```

## 🎮 使用方法

### 基础运行

```bash
python pusht_human.py
```

### 强化学习训练

#### PPO训练

```bash
# 使用默认配置训练PPO
python RL/train_ppo.py

# 自定义训练参数
python RL/train_ppo.py training.total_timesteps=500000
python RL/train_ppo.py ppo.learning_rate=1e-4
python RL/train_ppo.py env.n_envs=8

# 禁用WandB日志
python RL/train_ppo.py wandb.enabled=false

# 修改模型保存路径
python RL/train_ppo.py save.model_dir=models/ppo_experiment1
```

#### SAC训练

```bash
# 使用默认配置训练SAC
python RL/train_sac.py

# 自定义训练参数
python RL/train_sac.py training.total_timesteps=500000
python RL/train_sac.py sac.learning_rate=1e-4
python RL/train_sac.py sac.buffer_size=200000

# 调整探索参数
python RL/train_sac.py sac.ent_coef=0.1

# 修改实验名称
python RL/train_sac.py experiment.name=SAC_PushT_LargeBuffer
```

#### WandB配置

```bash
# 首次使用需要登录WandB
wandb login

# 设置项目名称
python RL/train_ppo.py experiment.project=my-pusht-experiments

# 离线模式运行
python RL/train_ppo.py wandb.mode=offline
```

### 自定义配置运行

```bash
# 修改游戏轮数
python pusht_human.py data.num_episodes=10

# 修改观测类型
python pusht_human.py env.obs_type=pixels

# 修改帧率
python pusht_human.py control.fps=15

# 启用自动上传
python pusht_human.py upload.auto_upload=true upload.repo_id=your-username/pusht-demo
```

### 仅上传已有数据

```bash
python pusht_human.py upload_only=true upload.repo_id=your-username/pusht-demo
```

### 轨迹回放

```bash
# 回放所有轨迹（自动播放）
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl

# 回放指定轨迹
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl --episode_id=0

# 手动逐步回放
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl --manual_play

# 调整回放速度
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl --delay=0.05

# 调整轨迹间间隔（默认2秒）
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl --inter_episode_delay=1

# 连续回放无间隔
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.pkl --inter_episode_delay=0

# 支持JSON格式
python replay_trajectories.py --data_path=data/pusht_trajectories/trajectories_5episodes.json
```

#### 回放控制说明

**自动播放模式**:
- 轨迹将自动按照原始动作序列执行
- 多个轨迹会自动连续播放，默认轨迹间间隔2秒
- 按 `Q` 键可随时退出回放

**手动播放模式**:
- 按 `空格` 键执行下一步
- 按 `Q` 键退出回放
- 可以仔细观察每一步的执行结果

**回放参数**:
- `--delay`: 控制自动播放时每步之间的延迟时间
- `--inter_episode_delay`: 控制轨迹间的间隔时间（设为0可连续播放）
- `--episode_id`: 指定回放特定轨迹的ID

## ⌨️ 控制说明

| 按键 | 功能 |
|------|------|
| `W` | 向上移动智能体 |
| `S` | 向下移动智能体 |
| `A` | 向左移动智能体 |
| `D` | 向右移动智能体 |
| `空格` | 切换用户控制/策略控制模式 |
| `R` | 重置当前环境 |
| `Q` | 退出游戏 |

## 📁 项目结构

```
.
├── pusht_human.py              # 主程序文件
├── replay_trajectories.py      # 轨迹回放脚本
├── utils.py                    # 工具函数模块
├── RL/                         # 强化学习训练模块
│   ├── train_ppo.py           # PPO训练脚本
│   └── train_sac.py           # SAC训练脚本
├── configs/
│   ├── pusht_human.yaml       # 主程序配置文件
│   ├── pusht_human_mouse.yaml # 鼠标控制配置文件
│   ├── replay.yaml            # 回放配置文件
│   └── rl/                    # 强化学习配置目录
│       ├── ppo.yaml          # PPO训练配置
│       └── sac.yaml          # SAC训练配置
├── data/
│   └── pusht_trajectories/    # 轨迹数据保存目录
├── models/                    # 训练模型保存目录
│   ├── ppo/                  # PPO模型
│   └── sac/                  # SAC模型
├── logs/                      # 训练日志目录
│   ├── ppo/                  # PPO日志
│   └── sac/                  # SAC日志
├── gym-pusht/                 # PushT环境源码
└── README.md                  # 说明文档
```

## ⚙️ 配置说明

### 环境配置 (`env`)
- `obs_type`: 观测类型，可选 `state`、`pixels`、`environment_state_agent_pos`、`pixels_agent_pos`
- `max_episode_steps`: 每轮最大步数 (默认300)
- `success_threshold`: 成功覆盖率阈值 (默认0.95)

### 控制配置 (`control`)
- `fps`: 渲染帧率 (默认10)
- `keyboard_move_speed`: 键盘移动速度 (默认10)
- `user_control`: 初始是否用户控制 (默认true)
- `keys`: 键盘映射配置

### 数据配置 (`data`)
- `num_episodes`: 游戏轮数 (默认5)
- `save_dir`: 数据保存目录 (默认"data/pusht_trajectories")
- `save_format`: 保存格式，可选 `pickle`、`json`、`npz` (默认pickle)
- `dataset_name`: 数据集名称 (默认"pusht_human_demo")

### 策略配置 (`policy`)
- `type`: 策略类型，当前支持 `random` (默认random)
- `random_seed`: 随机种子 (默认42)

### 上传配置 (`upload`)
- `hf_token`: Hugging Face token (从环境变量获取)
- `repo_id`: 仓库ID (默认"pusht-human-demo")
- `private`: 是否私有 (默认false)
- `auto_upload`: 游戏结束后自动上传 (默认false)

### 强化学习配置 (`RL`)

#### PPO配置 (`configs/rl/ppo.yaml`)
- **实验配置**:
  - `experiment.name`: 实验名称
  - `experiment.project`: WandB项目名称
  - `experiment.tags`: 实验标签
- **环境配置**:
  - `env.obs_type`: 观测类型 (推荐`pixels_agent_pos`)
  - `env.n_envs`: 并行环境数量 (默认4)
- **PPO算法参数**:
  - `ppo.learning_rate`: 学习率 (默认3e-4)
  - `ppo.n_steps`: 每次更新收集步数 (默认2048)
  - `ppo.batch_size`: 批次大小 (默认64)
  - `ppo.gamma`: 折扣因子 (默认0.99)
- **训练配置**:
  - `training.total_timesteps`: 总训练步数 (默认200000)
  - `training.eval_freq`: 评估频率 (默认10000)
- **WandB配置**:
  - `wandb.enabled`: 是否启用WandB (默认true)
  - `wandb.mode`: 运行模式 (online/offline/disabled)

#### SAC配置 (`configs/rl/sac.yaml`)
- **SAC算法参数**:
  - `sac.learning_rate`: 学习率 (默认3e-4)
  - `sac.buffer_size`: 经验回放缓冲区大小 (默认100000)
  - `sac.learning_starts`: 开始学习的步数 (默认10000)
  - `sac.ent_coef`: 熵系数 (默认auto)
  - `sac.tau`: 软更新参数 (默认0.005)

## 📊 数据格式

### 轨迹数据结构

```python
@dataclass
class TrajectoryStep:
    observation: Any          # 观测数据
    action: np.ndarray       # 动作向量 [x, y]
    reward: float           # 奖励值
    terminated: bool        # 是否终止
    truncated: bool         # 是否截断
    info: Dict[str, Any]    # 额外信息

@dataclass 
class Episode:
    steps: List[TrajectoryStep]  # 所有步骤
    episode_id: int             # 轮次ID
    total_reward: float         # 总奖励
    success: bool              # 是否成功
    length: int                # 步数
    initial_state: Dict[str, Any]  # 初始状态信息（用于回放）
    # 包含: agent_pos, block_pos, block_angle, goal_pose
```

### 保存格式

- **Pickle**: 完整的Python对象，包含所有数据类型
- **JSON**: 文本格式，便于查看和处理
- **NPZ**: NumPy压缩格式，适合大规模数据分析

## 🤗 Hugging Face集成

### 设置Token

```bash
# 方法1: 环境变量
export HF_TOKEN=your_huggingface_token

# 方法2: 配置文件
python pusht_human.py upload.hf_token=your_huggingface_token
```

### 数据集格式

上传到Hugging Face的数据集包含以下字段：

- `episode_id`: 轮次编号
- `step_id`: 步骤编号  
- `observation`: 观测数据
- `action`: 动作数据
- `reward`: 奖励值
- `terminated`: 终止标志
- `truncated`: 截断标志
- `episode_success`: 轮次成功标志
- `episode_length`: 轮次长度
- `episode_total_reward`: 轮次总奖励

## 🎯 使用场景

### 1. 强化学习研究
- 收集人类专家演示数据
- 进行模仿学习(Imitation Learning)研究
- 训练强化学习智能体(PPO/SAC)
- 对比人类和AI策略的行为差异

### 2. 算法基准测试
- 在PushT环境上评估不同RL算法性能
- 对比像素观测和状态观测的效果
- 研究多模态观测(像素+位置)的优势

### 3. 数据集构建
- 创建高质量的人类演示数据集
- 为行为克隆(Behavioral Cloning)提供训练数据
- 构建基准测试数据集

### 4. 教学演示
- 直观展示强化学习环境
- 让学生体验智能体决策过程
- 对比不同策略的效果
- 可视化训练过程和性能曲线

## 📈 性能优化

### 强化学习训练优化
1. **观测类型选择**: 
   - `pixels_agent_pos`: 最佳性能，结合视觉和位置信息
   - `pixels`: 纯视觉学习，更具挑战性
   - `state`: 最快训练速度
2. **并行环境**: PPO使用多个并行环境加速数据收集
3. **超参数调优**: 
   - 学习率: 3e-4是良好的起点
   - PPO n_steps: 根据任务长度调整
   - SAC buffer_size: 更大的缓冲区通常效果更好
4. **WandB监控**: 实时监控训练进度和超参数效果

### 提高数据质量
1. **调整移动速度**: 根据需要调整`keyboard_move_speed`
2. **适当帧率**: 平衡流畅度和控制精度
3. **多轮游戏**: 收集足够的数据样本

## 🐛 常见问题

### Q: 强化学习训练显存不足？
A: 降低并行环境数量 (`env.n_envs`)，或使用较小的网络架构。

### Q: WandB无法连接？
A: 检查网络连接，或使用 `wandb.mode=offline` 进行离线训练。

### Q: 训练收敛慢？
A: 尝试调整学习率，增加网络容量，或使用预训练模型。

### Q: SAC训练不稳定？
A: 检查熵系数设置，确保经验回放缓冲区足够大。

### Q: 键盘控制不响应？
A: 确保pygame窗口处于焦点状态，点击窗口后再操作。

### Q: 数据上传失败？
A: 检查Hugging Face token是否正确设置，网络连接是否正常。

### Q: 游戏卡顿？
A: 降低帧率设置或使用更轻量的观测类型。

### Q: 找不到保存的数据？
A: 检查`data.save_dir`配置，确保目录存在且有写权限。

## 🔧 扩展开发

### 添加新的RL算法

在`RL/`目录下创建新的训练脚本：

```python
# RL/train_new_algorithm.py
from stable_baselines3 import NewAlgorithm

def setup_model(cfg, env):
    model = NewAlgorithm(
        policy=cfg.algorithm.policy,
        env=env,
        # 其他算法特定参数
    )
    return model
```

### 自定义特征提取器

```python
# 在训练脚本中添加自定义特征提取器
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # 定义网络架构
        
    def forward(self, observations):
        # 实现前向传播
        return features
```

### 添加新策略

在`utils.py`中继承基础策略类：

```python
class YourPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        # 实现你的策略逻辑
        return action
```

## 📄 许可证

本项目基于gym-pusht环境构建，遵循相应的开源许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目！

---

**注意**: 使用前请确保已正确安装gym-pusht环境，详见项目根目录下的安装说明。 