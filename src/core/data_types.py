"""
HIRL核心数据结构模块
定义轨迹、回合等核心数据类型
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TrajectoryStep:
    """单步轨迹数据"""
    observation: Any
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    is_human_action: bool = False  # 标记是否为人类控制的动作


@dataclass 
class Episode:
    """单轮游戏数据"""
    steps: List[TrajectoryStep]
    episode_id: int
    total_reward: float
    success: bool
    length: int
    initial_state: Dict[str, Any]  # 包含agent_pos, block_pos, block_angle, goal_pose等


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    name: str
    description: str
    env_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any] 