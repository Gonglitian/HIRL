"""
HIRL - 人机交互强化学习平台
Human-in-the-Loop Reinforcement Learning Platform

一个支持人机交互的强化学习研究平台，基于PushT环境实现。
"""

__version__ = "2.0.0"
__author__ = "HIRL Team"
__email__ = "hirl@example.com"

# 核心模块导入
from .core import (
    TrajectoryStep, Episode, ExperimentConfig,
    PushTEnvironment, RandomPolicy, PushTGame
)

from .controllers import KeyboardController, MouseController
from .data import DataManager, HuggingFaceUploader
from .visualization import GameDisplay

__all__ = [
    # 数据类型
    'TrajectoryStep', 'Episode', 'ExperimentConfig',
    
    # 环境和策略
    'PushTEnvironment', 'RandomPolicy',
    
    # 主游戏类
    'PushTGame',
    
    # 控制器
    'KeyboardController', 'MouseController',
    
    # 数据管理
    'DataManager', 'HuggingFaceUploader',
    
    # 可视化
    'GameDisplay'
] 