"""
环境管理模块
封装PushT环境的创建和管理
"""

import gymnasium as gym
import gym_pusht
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging


class PushTEnvironment:
    """PushT环境管理器"""
    
    def __init__(self, 
                 obs_type: str = "pixels_agent_pos",
                 render_mode: str = "rgb_array",
                 observation_width: int = 512,
                 observation_height: int = 512,
                 max_episode_steps: int = 300):
        """
        初始化环境
        
        Args:
            obs_type: 观测类型
            render_mode: 渲染模式
            observation_width: 观测图像宽度
            observation_height: 观测图像高度
            max_episode_steps: 最大步数
        """
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.max_episode_steps = max_episode_steps
        
        # 创建环境
        self.env = gym.make(
            "gym_pusht/PushT-v0", 
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height
        )
        
        logging.info(f"PushT环境已创建，观测类型: {obs_type}")
        
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """重置环境"""
        return self.env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        return self.env.step(action)
    
    def get_agent_position(self) -> np.ndarray:
        """获取智能体当前位置"""
        if hasattr(self.env, 'unwrapped'):
            env = self.env.unwrapped
            if hasattr(env, 'agent'):
                return np.array(env.agent.position, dtype=np.float32)
        return np.array([256.0, 256.0], dtype=np.float32)  # 默认中心位置
    
    def get_initial_state_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """从info中提取初始状态信息"""
        return {
            'agent_pos': np.array(info['pos_agent']).tolist(),
            'block_pos': info['block_pose'][:2].tolist(),  # [x, y]
            'block_angle': float(info['block_pose'][2]),   # angle
            'goal_pose': info['goal_pose'].tolist()
        }
    
    def close(self):
        """关闭环境"""
        if self.env:
            self.env.close()
    
    @property
    def action_space(self):
        """动作空间"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """观测空间"""
        return self.env.observation_space


class RandomPolicy:
    """随机策略"""
    
    def __init__(self, action_space, seed: int = 42):
        """
        初始化随机策略
        
        Args:
            action_space: 动作空间
            seed: 随机种子
        """
        self.action_space = action_space
        np.random.seed(seed)
        logging.info(f"随机策略已初始化，种子: {seed}")
    
    def get_action(self, observation: Any) -> np.ndarray:
        """获取随机动作"""
        return self.action_space.sample() 