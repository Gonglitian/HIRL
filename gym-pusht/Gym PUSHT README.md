# gym-pusht 环境分析

## 环境概述

gym-pusht 是一个基于 Gymnasium 的强化学习环境，模拟了一个推方块任务。在这个环境中，智能体（圆形）需要将一个 T 形方块推到指定的目标区域。

## 环境特点

- **物理仿真**: 使用 pymunk 物理引擎进行精确的物理模拟
- **连续控制**: 支持连续的动作空间
- **多观测模式**: 支持状态向量、像素图像等多种观测类型
- **可视化**: 支持实时渲染和图像输出

## 动作空间 (Action Space)

```python
动作空间: Box(0.0, 512.0, (2,), float32)
```

- **类型**: 连续动作空间
- **维度**: 2维向量 `[x, y]`
- **范围**: `[0, 512]` 
- **含义**: 表示智能体的目标位置坐标
- **控制方式**: PD控制器，智能体会向目标位置移动

### 动作示例
```python
action = [256.0, 300.0]  # 将智能体移动到 (256, 300) 位置
```

## 观测空间 (Observation Space)

环境支持四种不同的观测类型：

### 1. 状态观测 (`obs_type="state"`)
```python
观测空间: Box(0.0, [512. 512. 512. 512. 6.283185], (5,), float64)
```

- **维度**: 5维向量
- **内容**: `[agent_x, agent_y, block_x, block_y, block_angle]`
- **范围**: 
  - `agent_x, agent_y`: [0, 512] 智能体位置
  - `block_x, block_y`: [0, 512] 方块位置  
  - `block_angle`: [0, 2π] 方块角度（弧度）

### 2. 环境状态+智能体位置 (`obs_type="environment_state_agent_pos"`)
```python
观测空间: Dict({
    'environment_state': Box(0.0, 512.0, (16,), float64),
    'agent_pos': Box(0.0, 512.0, (2,), float64)
})
```

- **environment_state**: 16维向量，表示T形方块的8个关键点坐标
- **agent_pos**: 2维向量，表示智能体位置

### 3. 像素观测 (`obs_type="pixels"`)
```python
观测空间: Box(0, 255, (96, 96, 3), uint8)
```

- **类型**: RGB图像
- **尺寸**: 96×96×3
- **范围**: [0, 255]

### 4. 像素+智能体位置 (`obs_type="pixels_agent_pos"`)
```python
观测空间: Dict({
    'pixels': Box(0, 255, (96, 96, 3), uint8),
    'agent_pos': Box(0.0, 512.0, (2,), float64)
})
```

- **pixels**: 96×96×3 RGB图像
- **agent_pos**: 2维智能体位置向量

## 奖励机制

### 奖励计算
```python
coverage = intersection_area / goal_area
reward = np.clip(coverage / success_threshold, 0.0, 1.0)
```

- **基础**: 方块在目标区域的覆盖率
- **计算方式**: `覆盖面积 / 目标区域面积`
- **标准化**: 除以成功阈值 (0.95) 并限制在 [0, 1] 范围内
- **最大奖励**: 1.0 (方块完全在目标区域内)

### 奖励特点
- **连续奖励**: 随着方块进入目标区域的程度逐渐增加
- **稀疏性**: 只有当方块与目标区域有重叠时才有奖励
- **归一化**: 奖励值在 [0, 1] 范围内

## 终止条件

### 成功终止
```python
success_threshold = 0.95  # 95% 覆盖率
terminated = coverage > success_threshold
```

- **条件**: 方块在目标区域的覆盖率 > 95%
- **状态**: `terminated = True`, `is_success = True`

### 其他终止条件
- **最大步数**: 300步 (在环境注册时设定)
- **截断**: `truncated = False` (当前实现中不使用)

## 环境参数

### 物理参数
- **控制频率**: 10 Hz
- **物理步长**: 0.01s
- **PD控制器**: kp=100, kv=20
- **重力**: (0, 0) 无重力
- **阻尼**: 可配置，默认为0

### 几何参数
- **环境尺寸**: 512×512 像素
- **智能体**: 半径15的圆形
- **方块**: T形，由两个矩形组成
- **目标姿态**: [256, 256, π/4] (x, y, θ)

## 使用示例

### 基础使用
```python
import gymnasium as gym
import gym_pusht

# 创建环境
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")

# 重置环境
obs, info = env.reset()
print(f"初始观测: {obs}")
print(f"观测空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 执行随机动作
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"步骤 {step}: 奖励={reward:.4f}, 覆盖率={info['coverage']:.4f}")
    
    if terminated:
        print("任务完成！")
        break

env.close()
```

### 不同观测类型测试
```python
# 测试不同观测类型
obs_types = ["state", "pixels", "environment_state_agent_pos", "pixels_agent_pos"]

for obs_type in obs_types:
    env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type)
    obs, info = env.reset()
    print(f"观测类型: {obs_type}")
    print(f"观测空间: {env.observation_space}")
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    else:
        print(f"  观测形状: {obs.shape}")
    env.close()
```

## 环境配置选项

### 创建环境时的参数
```python
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="state",                    # 观测类型
    render_mode="rgb_array",            # 渲染模式
    block_cog=None,                     # 方块重心
    damping=None,                       # 阻尼系数
    observation_width=96,               # 观测图像宽度
    observation_height=96,              # 观测图像高度
    visualization_width=680,            # 可视化图像宽度
    visualization_height=680            # 可视化图像高度
)
```

## 技术细节

### 物理引擎兼容性
- **Pymunk 版本**: 兼容 7.0+ 版本
- **碰撞检测**: 使用新的 `on_collision` API
- **修复内容**: 解决了 `add_collision_handler` 方法在新版本中被移除的问题

### 坐标系统
- **原点**: 左上角 (0, 0)
- **x轴**: 向右递增
- **y轴**: 向下递增
- **角度**: 弧度制，顺时针为正

### 渲染系统
- **图形库**: pygame + OpenCV
- **默认尺寸**: 512×512 (内部), 96×96 (观测)
- **颜色编码**: 
  - 智能体: 蓝色 (RoyalBlue)
  - 方块: 灰色 (LightSlateGray)  
  - 目标区域: 绿色 (LightGreen)

## 性能优化建议

1. **观测类型选择**: 
   - 训练阶段使用 `state` 类型（计算效率高）
   - 需要视觉信息时使用 `pixels` 类型

2. **批量处理**: 
   - 可以创建多个环境实例进行并行训练

3. **渲染控制**:
   - 训练时使用 `render_mode="rgb_array"`
   - 调试时使用 `render_mode="human"`

## 常见问题

### Q: 环境安装失败？
A: 确保使用可编辑安装：`pip install -e gym-pusht/`

### Q: Pymunk版本兼容性问题？
A: 当前版本已修复pymunk 7.0+的兼容性问题，确保使用最新代码。

### Q: 奖励总是0？
A: 检查方块是否与目标区域有重叠，只有重叠时才有奖励。

### Q: 任务太难完成？
A: 可以调整成功阈值或增加最大步数限制。

## 相关资源

- **Gymnasium**: https://gymnasium.farama.org/
- **Pymunk**: http://www.pymunk.org/
- **环境注册**: 在 `gym_pusht/__init__.py` 中定义 