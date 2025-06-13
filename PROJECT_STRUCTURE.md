# HIRL项目结构说明

## 重构概览

本次重构将原始混乱的代码库重新组织为模块化、可维护的结构。主要改进包括：

1. **模块化设计**: 将功能按职责分离到不同模块
2. **清晰的层次结构**: 核心逻辑、控制器、数据管理、可视化等分离
3. **统一的接口**: 标准化的类和方法设计
4. **完整的文档**: 详细的代码注释和使用说明

## 新的目录结构

```
HIRL/
├── main.py                       # 🚀 主程序入口
├── replay.py                     # 🎬 轨迹回放工具
├── README.md                     # 📚 项目文档
├── PROJECT_STRUCTURE.md          # 📋 结构说明
├── requirements.txt              # 📦 依赖列表
├── .gitignore                    # 🚫 忽略文件
│
├── src/                          # 💻 核心源代码
│   ├── __init__.py               # 包初始化
│   ├── core/                     # 🔧 核心模块
│   │   ├── __init__.py
│   │   ├── data_types.py         # 数据类型定义
│   │   ├── environment.py        # 环境管理
│   │   └── game.py              # 主游戏逻辑
│   ├── controllers/              # 🎮 输入控制器
│   │   ├── __init__.py
│   │   ├── keyboard_controller.py
│   │   └── mouse_controller.py
│   ├── data/                     # 💾 数据管理
│   │   ├── __init__.py
│   │   ├── data_manager.py       # 数据保存/加载
│   │   └── huggingface_uploader.py # HF上传
│   ├── visualization/            # 🎨 可视化模块
│   │   ├── __init__.py
│   │   ├── display.py           # 游戏显示
│   │   └── replay.py            # 轨迹回放
│   └── training/                 # 🤖 强化学习训练
│       └── RL/                   # (从原RL目录移动)
│           ├── train_ppo.py
│           ├── train_sac.py
│           └── test_rl_setup.py
│
├── configs/                      # ⚙️ 配置文件
│   ├── pusht_human.yaml
│   ├── pusht_human_mouse.yaml
│   ├── replay.yaml
│   └── rl/
│       ├── ppo.yaml
│       └── sac.yaml
│
├── analysis/                     # 📊 数据分析工具
│   ├── analyze_human_vs_ai_actions.py
│   └── analyze_pixels_agent_pos_trajectories.ipynb
│
├── scripts/                      # 🔨 实用脚本
│   ├── test_human_action_tracking.py
│   ├── test_mixed_control.py
│   ├── test_resolution_comparison.py
│   ├── test_trajectory_recording.py
│   ├── debug_initial_state.py
│   └── debug_observation.py
│
├── demos/                        # 🎯 演示文件
│   ├── human_vs_ai_actions.png
│   └── resolution_comparison.png
│
├── gym-pusht/                    # 🏋️ PushT环境
│   └── ...
│
├── test/                         # 🧪 测试目录
│   ├── pusht.py
│   └── test_gym_pusht.py
│
├── VLM_Spatial/                  # 🤖 视觉语言模型 (空)
│
├── data/                         # 💾 数据存储 (运行时创建)
│   └── pusht_trajectories/
│
├── models/                       # 🎯 训练模型 (运行时创建)
│   ├── ppo/
│   └── sac/
│
└── logs/                         # 📝 训练日志 (运行时创建)
    ├── ppo/
    └── sac/
```

## 核心模块说明

### 🔧 Core模块 (`src/core/`)

**data_types.py**: 定义核心数据结构
- `TrajectoryStep`: 单步轨迹数据
- `Episode`: 完整轨迹数据
- `ExperimentConfig`: 实验配置

**environment.py**: 环境管理
- `PushTEnvironment`: 封装gym-pusht环境
- `RandomPolicy`: 随机策略实现

**game.py**: 主游戏逻辑
- `PushTGame`: 交互式游戏主类
- 统一的游戏循环和状态管理

### 🎮 Controllers模块 (`src/controllers/`)

**keyboard_controller.py**: 键盘输入处理
- WASD移动控制
- 特殊按键处理
- 窗口焦点管理

**mouse_controller.py**: 鼠标输入处理
- 鼠标位置追踪
- 点击拖拽控制
- 移动平滑处理

### 💾 Data模块 (`src/data/`)

**data_manager.py**: 数据管理核心
- 多格式保存支持 (Pickle/JSON/NPZ)
- 轨迹加载和序列化
- 数据统计分析

**huggingface_uploader.py**: 云端上传
- Hugging Face Hub集成
- 数据集格式转换
- 自动上传功能

### 🎨 Visualization模块 (`src/visualization/`)

**display.py**: 游戏显示管理
- Pygame渲染控制
- 状态信息覆盖
- 倒计时和消息显示

**replay.py**: 轨迹回放功能
- 自动/手动回放模式
- 环境状态恢复
- 回放控制接口

### 🤖 Training模块 (`src/training/`)

包含原有的强化学习训练脚本，支持PPO和SAC算法训练。

## 迁移指南

### 从旧版本迁移

如果你使用的是重构前的代码，需要进行以下调整：

1. **导入路径更新**:
```python
# 旧版本
from utils import DataManager, KeyboardController

# 新版本  
from src.data import DataManager
from src.controllers import KeyboardController
```

2. **运行命令更新**:
```bash
# 旧版本
python pusht_human.py

# 新版本
python main.py
```

3. **配置文件兼容**: 配置文件保持向后兼容，无需修改。

### 保留的文件

为了平滑过渡，以下原始文件暂时保留：
- `pusht_human.py` - 原主程序
- `replay_trajectories.py` - 原回放脚本  
- `utils.py` - 原工具函数

建议逐步迁移到新的模块化API。

## 开发建议

### 添加新功能

1. **新控制器**: 在`src/controllers/`下添加
2. **新环境**: 扩展`src/core/environment.py`
3. **新可视化**: 在`src/visualization/`下添加
4. **新分析工具**: 在`analysis/`下添加

### 代码风格

- 遵循PEP 8规范
- 使用类型注解
- 添加详细的文档字符串
- 保持模块间的低耦合

### 测试

- 在`scripts/`目录下添加测试脚本
- 使用配置文件进行参数化测试
- 确保向后兼容性

## 性能优化

重构后的代码在以下方面有所改进：

1. **模块化加载**: 按需导入，减少启动时间
2. **内存管理**: 优化数据结构，减少内存占用
3. **代码复用**: 减少重复代码，提高维护性
4. **类型安全**: 使用类型注解，减少运行时错误

## 总结

本次重构将HIRL从一个功能性原型转变为一个专业的研究平台。新的模块化结构不仅提高了代码的可维护性，也为未来的功能扩展奠定了良好的基础。

建议开发者逐步熟悉新的API设计，并在新项目中使用重构后的模块。 