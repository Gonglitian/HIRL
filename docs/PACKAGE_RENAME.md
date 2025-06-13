# HIRL 包重命名更新

## 🔄 重大更新

### 包目录重命名：`src` → `HIRL`

为了让项目更加规范化和专业化，我们将源代码目录从 `src` 重命名为 `HIRL`，使其成为一个正式的Python包。

## 📦 新的包结构

```
HIRL/
├── main.py                 # 主程序入口
├── replay.py              # 轨迹回放工具
├── HIRL/                  # 🆕 核心Python包
│   ├── __init__.py        # 包初始化文件
│   ├── core/              # 核心模块
│   │   ├── __init__.py
│   │   ├── data_types.py  # 数据类型定义
│   │   ├── environment.py # 环境管理
│   │   └── game.py        # 主游戏逻辑
│   ├── controllers/       # 控制器模块
│   │   ├── __init__.py
│   │   ├── keyboard_controller.py
│   │   └── mouse_controller.py
│   ├── data/              # 数据管理模块
│   │   ├── __init__.py
│   │   ├── data_manager.py
│   │   └── huggingface_uploader.py
│   ├── visualization/     # 可视化模块
│   │   ├── __init__.py
│   │   ├── display.py
│   │   └── replay.py
│   └── training/          # 训练模块
│       └── __init__.py
├── configs/               # 配置文件
├── scripts/               # 辅助脚本
└── docs/                  # 文档
```

## 🚀 新的导入方式

### 推荐的导入方式

```python
# 从HIRL包导入核心类
from HIRL.core import PushTGame
from HIRL.controllers import KeyboardController, MouseController
from HIRL.data import DataManager, HuggingFaceUploader
from HIRL.visualization import GameDisplay
from HIRL.visualization.replay import TrajectoryReplayer

# 或者从包的顶层导入（推荐）
from HIRL import PushTGame, DataManager, KeyboardController
```

### 具体模块导入

```python
# 导入数据类型
from HIRL.core.data_types import TrajectoryStep, Episode

# 导入环境管理
from HIRL.core.environment import PushTEnvironment, RandomPolicy

# 导入控制器
from HIRL.controllers.keyboard_controller import KeyboardController
from HIRL.controllers.mouse_controller import MouseController

# 导入数据管理
from HIRL.data.data_manager import DataManager
from HIRL.data.huggingface_uploader import HuggingFaceUploader

# 导入可视化
from HIRL.visualization.display import GameDisplay
from HIRL.visualization.replay import TrajectoryReplayer
```

## 📋 更新的文件

### 主程序文件

- ✅ `main.py` - 更新为 `from HIRL.core import PushTGame`
- ✅ `replay.py` - 更新为 `from HIRL.visualization.replay import TrajectoryReplayer`

### 辅助脚本

- ✅ `scripts/convert_data_format.py` - 更新导入路径
- ✅ `scripts/test_config_params.py` - 更新导入路径

### 包文件

- ✅ `HIRL/__init__.py` - 配置包的公共接口
- ✅ 所有子模块的 `__init__.py` - 正确导出类和函数

## 🔧 开发者指南

### 包的使用

1. **直接导入主要类**：
   ```python
   from HIRL import PushTGame
   game = PushTGame(config)
   ```

2. **从子模块导入**：
   ```python
   from HIRL.core import PushTGame
   from HIRL.data import DataManager
   ```

3. **导入整个包**：
   ```python
   import HIRL
   game = HIRL.PushTGame(config)
   ```

### 包的扩展

要添加新功能到HIRL包：

1. 在相应的子模块中添加新文件
2. 在子模块的 `__init__.py` 中导出新类/函数
3. 在顶层 `HIRL/__init__.py` 中添加到 `__all__` 列表

## ⚠️ 向后兼容性

### 已更新的导入

所有项目内部的导入都已更新：
- ❌ `from src.core import ...` (旧)
- ✅ `from HIRL.core import ...` (新)

### 用户代码更新

如果您有自定义脚本使用了旧的导入方式，请更新：

```python
# 旧的导入方式 (不再工作)
from src.core import PushTGame
from src.data import DataManager

# 新的导入方式
from HIRL.core import PushTGame
from HIRL.data import DataManager
```

## 🎉 优势

1. **专业化**：HIRL现在是一个正式的Python包
2. **标准化**：遵循Python包的标准组织方式
3. **易用性**：可以直接 `from HIRL import ...`
4. **可扩展性**：更容易添加新功能和模块
5. **可发布性**：将来可以轻松发布到PyPI

## 🧪 测试

使用测试脚本验证包结构：

```bash
# 测试包导入是否正常
python3 test_import.py

# 测试主程序
python3 main.py

# 测试回放功能
python3 replay.py
```

包重命名完成！现在HIRL是一个标准的Python包，使用更加方便和专业。🚀 