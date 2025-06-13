"""
轨迹回放模块
负责回放已记录的轨迹数据
"""

import logging
import pygame
import gymnasium as gym
import gym_pusht
import numpy as np
from typing import List, Any, Dict, Optional
from pathlib import Path

from ..core.data_types import Episode, TrajectoryStep
from ..data import DataManager


class TrajectoryReplayer:
    """轨迹回放器"""
    
    def __init__(self, env_id: str = "gym_pusht/PushT-v0", **env_kwargs):
        """
        初始化回放器
        
        Args:
            env_id: 环境ID
            **env_kwargs: 环境参数
        """
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.env = None
        self.data_manager = DataManager("temp")
        
        # 初始化pygame (用于处理事件)
        pygame.init()
        
        logging.info(f"轨迹回放器初始化完成，环境: {env_id}")
    
    def load_episodes(self, data_path: str) -> List[Episode]:
        """
        加载轨迹数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            Episode列表
        """
        return self.data_manager.load_data(data_path)
    
    def replay_episode(self, episode: Episode, auto_play: bool = True, delay: float = 0.1):
        """
        回放单个轨迹
        
        Args:
            episode: 要回放的轨迹
            auto_play: 是否自动播放
            delay: 自动播放时的延迟(秒)
        """
        logging.info(f"开始回放轨迹 {episode.episode_id}")
        
        # 创建环境
        if self.env is None:
            self.env = gym.make(self.env_id, **self.env_kwargs)
        
        # 重置到初始状态
        self._reset_to_initial_state(episode)
        
        # 渲染初始状态
        self.env.render()
        
        if auto_play:
            logging.info("自动回放模式 - 按Q键退出")
        else:
            logging.info("手动回放模式 - 按空格键下一步，Q键退出")
        
        # 逐步回放
        for step_idx, step in enumerate(episode.steps):
            # 检查退出事件
            if self._check_quit_events():
                logging.info("用户中断回放")
                break
            
            # 手动模式等待用户输入
            if not auto_play:
                if not self._wait_for_space():
                    break
            
            # 执行动作
            try:
                obs, reward, terminated, truncated, info = self.env.step(step.action)
                self.env.render()
                
                # 显示步骤信息
                control_type = "人类" if step.is_human_action else "AI"
                logging.debug(f"步骤 {step_idx + 1}/{len(episode.steps)}: "
                            f"动作={step.action}, 奖励={step.reward:.3f}, "
                            f"控制={control_type}")
                
                # 检查是否提前结束
                if terminated or truncated:
                    logging.info(f"轨迹在第{step_idx + 1}步结束")
                    break
                
            except Exception as e:
                logging.error(f"回放步骤{step_idx}时出错: {e}")
                break
            
            # 自动播放延迟
            if auto_play and delay > 0:
                pygame.time.wait(int(delay * 1000))
        
        logging.info(f"轨迹 {episode.episode_id} 回放完成")
    
    def replay_all_episodes(self, episodes: List[Episode], 
                          auto_play: bool = True, 
                          delay: float = 0.1, 
                          inter_episode_delay: float = 2.0):
        """
        回放所有轨迹
        
        Args:
            episodes: 轨迹列表
            auto_play: 是否自动播放
            delay: 步间延迟
            inter_episode_delay: 轨迹间延迟
        """
        logging.info(f"开始回放 {len(episodes)} 个轨迹")
        
        for i, episode in enumerate(episodes):
            logging.info(f"回放轨迹 {i + 1}/{len(episodes)}: ID={episode.episode_id}")
            
            # 回放单个轨迹
            self.replay_episode(episode, auto_play, delay)
            
            # 轨迹间间隔
            if i < len(episodes) - 1 and inter_episode_delay > 0:
                logging.info(f"等待 {inter_episode_delay} 秒后开始下一个轨迹...")
                if auto_play:
                    pygame.time.wait(int(inter_episode_delay * 1000))
                    # 检查是否在等待期间退出
                    if self._check_quit_events():
                        break
        
        logging.info("所有轨迹回放完成")
    
    def _reset_to_initial_state(self, episode: Episode):
        """重置环境到轨迹的初始状态"""
        # 标准重置
        self.env.reset()
        
        # 尝试设置到精确的初始状态
        if hasattr(episode, 'initial_state') and episode.initial_state:
            try:
                self._set_environment_state(episode.initial_state)
            except Exception as e:
                logging.warning(f"无法设置精确初始状态: {e}")
    
    def _set_environment_state(self, initial_state: Dict[str, Any]):
        """设置环境状态（如果支持）"""
        # 这里可以根据具体环境实现状态设置
        # 对于PushT环境，可能需要直接操作环境内部状态
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'set_state'):
            try:
                self.env.unwrapped.set_state(initial_state)
            except Exception as e:
                logging.debug(f"环境不支持状态设置: {e}")
    
    def _check_quit_events(self) -> bool:
        """检查退出事件"""
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return True
        return False
    
    def _wait_for_space(self) -> bool:
        """等待空格键按下，返回True继续，False退出"""
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return False
                    elif event.key == pygame.K_SPACE:
                        return True
            pygame.time.wait(50)  # 避免CPU占用过高
    
    def close(self):
        """关闭回放器"""
        if self.env:
            self.env.close()
        pygame.quit()
        logging.info("轨迹回放器已关闭") 