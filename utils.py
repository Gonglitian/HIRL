"""
PushT人机交互工具函数模块
包含数据保存、键盘处理、策略管理等功能
"""

import os
import pickle
import json
import numpy as np
import pygame
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import logging

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
    # 新增：初始状态元数据，用于轨迹回放
    initial_state: Dict[str, Any]  # 包含agent_pos, block_pos, block_angle, goal_pose等

class MouseController:
    """鼠标控制器"""
    
    def __init__(self, smoothing: float = 0.5, click_to_move: bool = False):
        self.smoothing = smoothing
        self.click_to_move = click_to_move
        self.last_mouse_pos = None
        self.current_target = None
        self.mouse_pressed = False
        
    def update_mouse_state(self, events: List[pygame.event.Event]) -> Dict[str, bool]:
        """更新鼠标状态并返回特殊按键动作"""
        actions = {}
        
        # 确保事件被处理
        pygame.event.pump()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self.mouse_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键
                    self.mouse_pressed = False
            elif event.type == pygame.WINDOWFOCUSGAINED:
                logging.info("✓ 窗口获得焦点")
            elif event.type == pygame.WINDOWFOCUSLOST:
                logging.info("⚠ 窗口失去焦点 - 请点击游戏窗口")
        
        return actions
    
    def get_mouse_action(self) -> Optional[np.ndarray]:
        """根据鼠标位置获取动作"""
        mouse_pos = pygame.mouse.get_pos()
        
        # 检查是否需要点击才移动
        if self.click_to_move and not self.mouse_pressed:
            return None
        
        # 将pygame坐标转换为环境坐标 (pygame: (0,0)在左上，环境: (0,0)在左上)
        # pygame窗口大小通常是512x512，环境坐标也是0-512
        target_pos = np.array(mouse_pos, dtype=np.float32)
        
        # 限制在环境边界内
        target_pos = np.clip(target_pos, [0, 0], [512, 512])
        
        # 应用平滑
        if self.smoothing > 0 and self.current_target is not None:
            # 使用指数移动平均进行平滑
            self.current_target = (
                self.smoothing * self.current_target + 
                (1 - self.smoothing) * target_pos
            )
        else:
            self.current_target = target_pos
        
        self.last_mouse_pos = mouse_pos
        return self.current_target.copy()

class KeyboardController:
    """键盘控制器"""
    
    def __init__(self, key_mapping: Dict[str, str], move_speed: float = 10.0, setup_focus: bool = True):
        self.key_mapping = {v: k for k, v in key_mapping.items()}  # 反向映射
        self.move_speed = move_speed
        self.pressed_keys = set()
        if setup_focus:
            self._setup_window_focus()
    
    def _setup_window_focus(self):
        """设置窗口焦点（Windows优化）"""
        import os
        if os.name == 'nt':  # Windows
            try:
                import win32gui
                import win32con
                hwnd = pygame.display.get_wm_info()["window"]
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                logging.info("✓ 已设置窗口获得键盘焦点")
            except ImportError:
                logging.info("⚠ 未安装pywin32，请手动点击窗口获得焦点")
            except Exception as e:
                logging.debug(f"⚠ 窗口焦点设置失败: {e}")
    
    def update_pressed_keys(self, events: List[pygame.event.Event]) -> Dict[str, bool]:
        """更新按键状态"""
        actions = {}
        
        # 确保事件被处理
        pygame.event.pump()
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                if key_name in self.key_mapping:
                    actions[self.key_mapping[key_name]] = True
                    logging.debug(f"检测到按键: {key_name} -> {self.key_mapping[key_name]}")
            elif event.type == pygame.WINDOWFOCUSGAINED:
                logging.info("✓ 窗口获得焦点")
            elif event.type == pygame.WINDOWFOCUSLOST:
                logging.info("⚠ 窗口失去焦点 - 请点击游戏窗口")
        
        return actions
    
    def get_movement_action(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """根据实时按键状态计算移动动作"""
        # 获取实时按键状态
        keys = pygame.key.get_pressed()
        
        dx, dy = 0, 0
        
        # 检查移动键
        if keys[pygame.K_w]:  # up
            dy -= self.move_speed
        if keys[pygame.K_s]:  # down  
            dy += self.move_speed
        if keys[pygame.K_a]:  # left
            dx -= self.move_speed
        if keys[pygame.K_d]:  # right
            dx += self.move_speed
            
        if dx == 0 and dy == 0:
            return None
            
        new_pos = current_pos + np.array([dx, dy])
        # 限制在环境边界内
        new_pos = np.clip(new_pos, [0, 0], [512, 512])
        
        return new_pos.astype(np.float32)

class RandomPolicy:
    """随机策略"""
    
    def __init__(self, action_space, seed: int = 42):
        self.action_space = action_space
        np.random.seed(seed)
    
    def get_action(self, observation: Any) -> np.ndarray:
        """获取随机动作"""
        return self.action_space.sample()

class DataManager:
    """数据管理器"""
    
    def __init__(self, save_dir: str, save_format: str = "pickle"):
        self.save_dir = Path(save_dir)
        self.save_format = save_format
        self.episodes: List[Episode] = []
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def add_episode(self, episode: Episode):
        """添加一轮游戏数据"""
        self.episodes.append(episode)
        logging.info(f"添加第{episode.episode_id}轮游戏数据, 步数: {episode.length}, 成功: {episode.success}")
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """保存所有数据"""
        if filename is None:
            filename = f"trajectories_{len(self.episodes)}episodes"
        
        filepath = self.save_dir / f"{filename}.{self.save_format}"
        
        if self.save_format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self.episodes, f)
        elif self.save_format == "json":
            # 转换为JSON可序列化格式
            json_data = self._episodes_to_json()
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        elif self.save_format == "npz":
            # 转换为numpy数组格式
            np_data = self._episodes_to_numpy()
            np.savez_compressed(filepath, **np_data)
        
        logging.info(f"数据已保存到: {filepath}")
        return str(filepath)
    
    def _episodes_to_json(self) -> Dict[str, Any]:
        """转换为JSON格式"""
        json_episodes = []
        for episode in self.episodes:
            json_steps = []
            for step in episode.steps:
                json_step = {
                    'observation': step.observation.tolist() if hasattr(step.observation, 'tolist') else step.observation,
                    'action': step.action.tolist() if hasattr(step.action, 'tolist') else step.action,
                    'reward': float(step.reward),
                    'terminated': bool(step.terminated),
                    'truncated': bool(step.truncated),
                    'info': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in step.info.items()}
                }
                json_steps.append(json_step)
            
            json_episode = {
                'steps': json_steps,
                'episode_id': episode.episode_id,
                'total_reward': float(episode.total_reward),
                'success': bool(episode.success),
                'length': int(episode.length),
                'initial_state': {
                    k: (v.tolist() if hasattr(v, 'tolist') else v) 
                    for k, v in episode.initial_state.items()
                }
            }
            json_episodes.append(json_episode)
        
        return {'episodes': json_episodes}
    
    def _episodes_to_numpy(self) -> Dict[str, np.ndarray]:
        """转换为numpy格式"""
        # 收集所有数据
        all_observations = []
        all_actions = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        episode_starts = []
        episode_successes = []
        
        for episode in self.episodes:
            episode_starts.append(len(all_observations))
            episode_successes.append(episode.success)
            
            for step in episode.steps:
                all_observations.append(step.observation)
                all_actions.append(step.action)
                all_rewards.append(step.reward)
                all_terminated.append(step.terminated)
                all_truncated.append(step.truncated)
        
        return {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'terminated': np.array(all_terminated),
            'truncated': np.array(all_truncated),
            'episode_starts': np.array(episode_starts),
            'episode_successes': np.array(episode_successes)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.episodes:
            return {}
        
        success_rate = sum(ep.success for ep in self.episodes) / len(self.episodes)
        avg_length = np.mean([ep.length for ep in self.episodes])
        avg_reward = np.mean([ep.total_reward for ep in self.episodes])
        
        return {
            'total_episodes': len(self.episodes),
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'avg_total_reward': avg_reward,
            'total_steps': sum(ep.length for ep in self.episodes)
        }

class HuggingFaceUploader:
    """Hugging Face数据集上传器"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('HF_TOKEN')
        self.api = HfApi(token=self.token) if self.token else None
    
    def upload_dataset(self, data_path: str, repo_id: str, private: bool = False) -> str:
        """上传数据集到Hugging Face Hub"""
        if not self.api:
            raise ValueError("需要Hugging Face token才能上传数据集")
        
        # 加载数据
        if data_path.endswith('.pickle'):
            with open(data_path, 'rb') as f:
                episodes = pickle.load(f)
        else:
            raise ValueError("目前只支持pickle格式的数据上传")
        
        # 转换为Hugging Face Dataset格式
        dataset = self._episodes_to_hf_dataset(episodes)
        
        # 上传到Hub
        dataset.push_to_hub(repo_id, private=private, token=self.token)
        
        logging.info(f"数据集已上传到: https://huggingface.co/datasets/{repo_id}")
        return f"https://huggingface.co/datasets/{repo_id}"
    
    def _episodes_to_hf_dataset(self, episodes: List[Episode]) -> DatasetDict:
        """转换为Hugging Face Dataset格式"""
        data = {
            'episode_id': [],
            'step_id': [],
            'observation': [],
            'action': [],
            'reward': [],
            'terminated': [],
            'truncated': [],
            'episode_success': [],
            'episode_length': [],
            'episode_total_reward': [],
            'initial_agent_pos': [],
            'initial_block_pos': [],
            'initial_block_angle': [],
            'goal_pose': []
        }
        
        for episode in episodes:
            for step_id, step in enumerate(episode.steps):
                data['episode_id'].append(episode.episode_id)
                data['step_id'].append(step_id)
                data['observation'].append(step.observation.tolist() if hasattr(step.observation, 'tolist') else step.observation)
                data['action'].append(step.action.tolist() if hasattr(step.action, 'tolist') else step.action)
                data['reward'].append(step.reward)
                data['terminated'].append(step.terminated)
                data['truncated'].append(step.truncated)
                data['episode_success'].append(episode.success)
                data['episode_length'].append(episode.length)
                data['episode_total_reward'].append(episode.total_reward)
                # 添加初始状态信息
                data['initial_agent_pos'].append(episode.initial_state.get('agent_pos', []))
                data['initial_block_pos'].append(episode.initial_state.get('block_pos', []))
                data['initial_block_angle'].append(episode.initial_state.get('block_angle', 0.0))
                data['goal_pose'].append(episode.initial_state.get('goal_pose', []))
        
        dataset = Dataset.from_dict(data)
        return DatasetDict({'train': dataset})

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


class PushTReplayEnv:
    """PushT环境回放类
    
    用于回放已记录的轨迹，不修改原始环境源码
    """
    
    def __init__(self, base_env, render_mode="human"):
        """
        初始化回放环境
        
        Args:
            base_env: 基础PushT环境实例
            render_mode: 渲染模式
        """
        self.base_env = base_env
        # 获取实际的PushT环境（去除TimeLimit等包装器）
        self.actual_env = self._get_actual_env(base_env)
        self.render_mode = render_mode
        self.current_episode = None
        self.current_step = 0
        self.replay_finished = False
        
    def _get_actual_env(self, env):
        """获取实际的PushT环境，去除包装器"""
        # 如果有unwrapped属性，直接返回
        if hasattr(env, 'unwrapped'):
            return env.unwrapped
        # 如果有env属性，递归查找
        elif hasattr(env, 'env'):
            return self._get_actual_env(env.env)
        # 否则返回原环境
        else:
            return env
        
    def reset_to_episode(self, episode: Episode):
        """重置环境到指定轨迹的初始状态"""
        self.current_episode = episode
        self.current_step = 0
        self.replay_finished = False
        
        # 从初始状态元数据恢复环境状态
        initial_state = episode.initial_state
        
        # 先正常reset环境以初始化所有组件
        obs, info = self.base_env.reset()
        
        # 然后直接设置物理状态，避免额外的物理步进
        self._set_state_without_physics_step(
            agent_pos=initial_state['agent_pos'],
            block_pos=initial_state['block_pos'], 
            block_angle=initial_state['block_angle']
        )
        
        # 获取设置后的观测和信息
        obs = self.actual_env.get_obs()
        # info = self.actual_env._get_info()
        
        logging.info(f"回放环境已重置到轨迹{episode.episode_id}的初始状态")
        logging.info(f"  Agent位置: {initial_state['agent_pos']}")
        logging.info(f"  Block位置: {initial_state['block_pos']}")
        logging.info(f"  Block角度: {initial_state['block_angle']:.3f}")
        logging.info(f"  轨迹总步数: {episode.length}")
        
        return obs, info
    
    def _set_state_without_physics_step(self, agent_pos, block_pos, block_angle):
        """直接设置状态，不运行物理步进"""
        # 使用实际的环境来设置状态
        actual_env = self.actual_env
        
        # 直接设置agent位置
        actual_env.agent.position = agent_pos
        
        # 对于block，需要先设置角度再设置位置
        # 因为设置角度会相对于重心旋转，可能改变几何位置
        actual_env.block.angle = block_angle
        actual_env.block.position = block_pos
        
        # 重置速度，避免之前的动量影响
        actual_env.agent.velocity = (0, 0)
        actual_env.block.velocity = (0, 0)
        actual_env.block.angular_velocity = 0
        
        # 重置碰撞计数
        actual_env.n_contact_points = 0
    
    def step_replay(self):
        """执行回放的下一步"""
        if self.current_episode is None:
            raise ValueError("请先调用reset_to_episode()设置轨迹")
            
        if self.current_step >= len(self.current_episode.steps):
            self.replay_finished = True
            logging.info("轨迹回放完成")
            return None, None, None, None, {"replay_finished": True}
        
        # 获取当前步骤的动作
        step_data = self.current_episode.steps[self.current_step]
        action = step_data.action
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # 添加回放信息
        info.update({
            "replay_step": self.current_step,
            "replay_total_steps": len(self.current_episode.steps),
            "replay_finished": self.replay_finished,
            "original_reward": step_data.reward,
            "original_terminated": step_data.terminated,
            "original_truncated": step_data.truncated
        })
        
        self.current_step += 1
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染当前状态"""
        return self.base_env.render()
    
    def close(self):
        """关闭环境"""
        self.base_env.close()


class TrajectoryReplayer:
    """轨迹回放器
    
    提供完整的轨迹回放功能，包括自动回放和交互式回放
    """
    
    def __init__(self, env_class, env_kwargs=None):
        """
        初始化回放器
        
        Args:
            env_class: 环境类
            env_kwargs: 环境初始化参数
        """
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        
    def load_episodes(self, data_path: str) -> List[Episode]:
        """从文件加载轨迹数据"""
        data_path = Path(data_path)
        
        if data_path.suffix in ['.pkl', '.pickle']:
            with open(data_path, 'rb') as f:
                episodes = pickle.load(f)
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            episodes = self._json_to_episodes(data)
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")
        
        logging.info(f"已加载{len(episodes)}条轨迹数据")
        return episodes
    
    def _json_to_episodes(self, data: Dict[str, Any]) -> List[Episode]:
        """将JSON数据转换为Episode对象"""
        episodes = []
        
        for ep_data in data['episodes']:
            steps = []
            for step_data in ep_data['steps']:
                step = TrajectoryStep(
                    observation=np.array(step_data['observation']),
                    action=np.array(step_data['action']),
                    reward=step_data['reward'],
                    terminated=step_data['terminated'],
                    truncated=step_data['truncated'],
                    info=step_data['info']
                )
                steps.append(step)
            
            episode = Episode(
                steps=steps,
                episode_id=ep_data['episode_id'],
                total_reward=ep_data['total_reward'],
                success=ep_data['success'],
                length=ep_data['length'],
                initial_state=ep_data['initial_state']
            )
            episodes.append(episode)
        
        return episodes
    
    def replay_episode(self, episode: Episode, auto_play: bool = True, delay: float = 0.1):
        """
        回放单个轨迹
        
        Args:
            episode: 要回放的轨迹
            auto_play: 是否自动播放（False为手动逐步播放）
            delay: 自动播放时的步间延迟（秒）
        """
        # 创建环境和回放器
        env = self.env_class(**self.env_kwargs)
        replay_env = PushTReplayEnv(env, render_mode="human")
        
        try:
            # 重置到轨迹初始状态
            obs, info = replay_env.reset_to_episode(episode)
            replay_env.render()
            
            if auto_play:
                logging.info(f"开始自动回放轨迹{episode.episode_id}...")
                import time
                
                while not replay_env.replay_finished:
                    obs, reward, terminated, truncated, info = replay_env.step_replay()
                    if obs is not None:
                        replay_env.render()
                        time.sleep(delay)
                    
                    # 检查退出事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            logging.info("用户退出回放")
                            return
                
                logging.info("自动回放完成")
                
            else:
                logging.info(f"开始手动回放轨迹{episode.episode_id}...")
                logging.info("按Space键执行下一步，按Q键退出")
                
                while not replay_env.replay_finished:
                    # 等待用户输入
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                                logging.info("用户退出回放")
                                return
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                waiting = False
                    
                    # 执行下一步
                    obs, reward, terminated, truncated, info = replay_env.step_replay()
                    if obs is not None:
                        replay_env.render()
                        logging.info(f"步骤 {info.get('replay_step', 0)}/{info.get('replay_total_steps', 0)}")
                
                logging.info("手动回放完成")
                
        finally:
            replay_env.close()
    
    def replay_all_episodes(self, episodes: List[Episode], auto_play: bool = True, delay: float = 0.1, inter_episode_delay: float = 2.0):
        """
        回放所有轨迹
        
        Args:
            episodes: 轨迹列表
            auto_play: 是否自动播放
            delay: 自动播放时的步间延迟（秒）
            inter_episode_delay: 轨迹间间隔时间（秒）
        """
        for i, episode in enumerate(episodes):
            logging.info(f"回放轨迹 {i+1}/{len(episodes)}")
            
            self.replay_episode(episode, auto_play=auto_play, delay=delay)
            
            if i < len(episodes) - 1:
                # 轨迹间短暂间隔，然后自动继续下一个
                if inter_episode_delay > 0:
                    logging.info(f"轨迹 {i+1} 回放完成，{inter_episode_delay}秒后自动开始下一个轨迹...")
                    import time
                    time.sleep(inter_episode_delay)
                else:
                    logging.info(f"轨迹 {i+1} 回放完成，立即开始下一个轨迹...") 