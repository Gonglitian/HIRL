# HIRL 数据格式选择指南

## 问题背景

默认的pickle格式会保存对象的类型信息（如`src.core.data_types.TrajectoryStep`），导致加载数据时需要相同的模块结构。这会在数据共享、长期存储和跨环境使用时造成问题。

## 推荐的纯数据格式

### 1. HDF5格式 (推荐用于大数据)

**优点：**
- 高效的压缩和存储
- 支持大规模数据
- 跨平台兼容
- 无类依赖，纯数据格式
- 支持层次化数据结构

**使用场景：**
- 大量轨迹数据（>1000个回合）
- 需要高性能读写
- 长期数据存储

**示例代码：**
```python
from src.data.data_manager import DataManager

# 保存数据
manager = DataManager(save_dir="data", save_format="hdf5")
manager.add_episode(episode)
manager.save_data("my_trajectories")  # 生成 my_trajectories.h5

# 加载数据（返回纯字典）
data = manager.load_data("data/my_trajectories.h5")
```

### 2. JSON格式 (推荐用于小数据和调试)

**优点：**
- 人类可读
- 跨语言支持
- 无类依赖
- 易于调试和检查

**缺点：**
- 文件较大
- 加载速度较慢

**使用场景：**
- 小规模数据集（<100个回合）
- 需要人工检查数据
- 跨语言数据交换

### 3. CSV格式 (推荐用于数据分析)

**优点：**
- 最通用的格式
- 可直接在Excel/Pandas中打开
- 无类依赖
- 易于数据分析

**缺点：**
- 扁平化数据结构
- 文件较大

**使用场景：**
- 数据分析和可视化
- 与非Python工具集成
- 简单的统计分析

### 4. NPZ格式 (推荐用于纯数值数据)

**优点：**
- NumPy原生格式
- 高效压缩
- 快速加载

**缺点：**
- 主要适用于数值数据
- 丢失部分元数据

**使用场景：**
- 强化学习训练数据
- 只需要观测和动作数据

## 使用示例

### 设置不同的保存格式

```python
# HDF5格式（推荐）
manager = DataManager(save_dir="data", save_format="hdf5")

# JSON格式（调试用）
manager = DataManager(save_dir="data", save_format="json")

# CSV格式（分析用）
manager = DataManager(save_dir="data", save_format="csv")

# NPZ格式（训练用）
manager = DataManager(save_dir="data", save_format="npz")
```

### 加载和使用数据

```python
# 加载数据（所有格式都返回纯字典）
data = manager.load_data("path/to/file.h5")

# 数据结构示例
print(data[0])  # 第一个回合
{
    'episode_id': 0,
    'total_reward': 15.5,
    'success': True,
    'length': 100,
    'steps': [
        {
            'observation': [0.1, 0.2],
            'action': [1.0, 0.0],
            'reward': 0.1,
            'terminated': False,
            'truncated': False,
            'is_human_action': True
        },
        # ... 更多步骤
    ]
}
```

### 数据转换

如果您已有pickle格式的数据，可以这样转换：

```python
# 加载旧的pickle数据
old_manager = DataManager(save_dir="old_data", save_format="pickle")
episodes = old_manager.load_data("old_file.pkl")

# 转换为新格式
new_manager = DataManager(save_dir="new_data", save_format="hdf5")
for episode in episodes:
    new_manager.add_episode(episode)
new_manager.save_data("converted_data")
```

## 性能对比

| 格式 | 文件大小 | 保存速度 | 加载速度 | 可读性 | 兼容性 |
|------|----------|----------|----------|--------|--------|
| HDF5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| JSON | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CSV  | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| NPZ  | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Pickle | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |

## 建议

1. **新项目**：使用HDF5格式作为默认选择
2. **调试阶段**：使用JSON格式便于检查数据
3. **数据分析**：使用CSV格式便于分析工具处理
4. **已有项目**：逐步迁移到HDF5格式

## 安装依赖

```bash
pip install h5py pandas
```

以上格式都能确保您的数据是"纯数据"，不依赖特定的Python类结构，便于长期存储和跨环境使用。 