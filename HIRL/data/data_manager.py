"""
HIRL数据管理模块
处理轨迹数据的保存、加载和管理
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from ..core.data_types import Episode, TrajectoryStep


class DataManager:
    """数据管理器"""
    
    def __init__(self, save_dir: str, save_format: str = "hdf5"):
        """
        初始化数据管理器
        
        Args:
            save_dir: 保存目录
            save_format: 保存格式，支持 "hdf5", "json", "csv", "npz", "pickle"
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证保存格式
        valid_formats = ["hdf5", "json", "csv", "npz", "pickle"]
        if save_format not in valid_formats:
            raise ValueError(f"不支持的保存格式: {save_format}，支持的格式: {valid_formats}")
        
        if save_format == "hdf5" and not HDF5_AVAILABLE:
            logging.warning("HDF5不可用，自动切换到JSON格式")
            save_format = "json"
            
        self.save_format = save_format
        self.episodes: List[Episode] = []
        
        logging.info(f"数据管理器初始化完成，保存格式: {save_format}")
    
    def add_episode(self, episode: Episode):
        """添加回合数据"""
        self.episodes.append(episode)
        logging.debug(f"添加了第{len(self.episodes)}个回合，步数: {episode.length}")
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """
        保存轨迹数据
        
        Args:
            filename: 文件名（不含扩展名）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"trajectories_{len(self.episodes)}episodes"
        
        if self.save_format == "pickle":
            file_path = self.save_dir / f"{filename}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(self.episodes, f)
        elif self.save_format == "json":
            file_path = self.save_dir / f"{filename}.json"
            data = self._episodes_to_pure_json()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif self.save_format == "npz":
            file_path = self.save_dir / f"{filename}.npz"
            data = self._episodes_to_numpy()
            np.savez_compressed(file_path, **data)
        elif self.save_format == "hdf5":
            file_path = self.save_dir / f"{filename}.h5"
            self._save_to_hdf5(file_path)
        elif self.save_format == "csv":
            file_path = self.save_dir / f"{filename}.csv"
            self._save_to_csv(file_path)
        else:
            raise ValueError(f"不支持的保存格式: {self.save_format}")
        
        logging.info(f"数据已保存到: {file_path}")
        return str(file_path)
    
    def _episodes_to_pure_json(self) -> Dict[str, Any]:
        """将Episode数据转换为纯JSON格式（无类引用）"""
        episodes_data = []
        
        for episode in self.episodes:
            steps_data = []
            for step in episode.steps:
                # 处理观测数据
                if isinstance(step.observation, dict):
                    obs_data = {}
                    for key, value in step.observation.items():
                        if isinstance(value, np.ndarray):
                            obs_data[key] = value.tolist()
                        else:
                            obs_data[key] = value
                elif isinstance(step.observation, np.ndarray):
                    obs_data = step.observation.tolist()
                else:
                    obs_data = step.observation
                
                step_data = {
                    'observation': obs_data,
                    'action': step.action.tolist() if isinstance(step.action, np.ndarray) else step.action,
                    'reward': float(step.reward),
                    'terminated': bool(step.terminated),
                    'truncated': bool(step.truncated),
                    'info': step.info,
                    'is_human_action': bool(step.is_human_action)
                }
                steps_data.append(step_data)
            
            episode_data = {
                'episode_id': episode.episode_id,
                'total_reward': float(episode.total_reward),
                'success': bool(episode.success),
                'length': episode.length,
                'initial_state': episode.initial_state,
                'steps': steps_data
            }
            episodes_data.append(episode_data)
        
        return {
            'episodes': episodes_data, 
            'total_episodes': len(self.episodes),
            'format_version': '1.0',
            'description': 'HIRL trajectory data in pure JSON format'
        }
    
    def _save_to_hdf5(self, file_path: Path):
        """保存到HDF5格式（高效且纯数据）"""
        with h5py.File(file_path, 'w') as f:
            # 保存元数据
            f.attrs['total_episodes'] = len(self.episodes)
            f.attrs['format_version'] = '1.0'
            f.attrs['description'] = 'HIRL trajectory data in HDF5 format'
            
            # 为每个episode创建一个组
            for i, episode in enumerate(self.episodes):
                ep_group = f.create_group(f'episode_{i}')
                
                # 保存episode元数据
                ep_group.attrs['episode_id'] = episode.episode_id
                ep_group.attrs['total_reward'] = episode.total_reward
                ep_group.attrs['success'] = episode.success
                ep_group.attrs['length'] = episode.length
                
                # 保存初始状态
                init_group = ep_group.create_group('initial_state')
                for key, value in episode.initial_state.items():
                    if isinstance(value, (int, float, bool)):
                        init_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        init_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple)):
                        init_group.create_dataset(key, data=np.array(value))
                
                # 保存步骤数据
                steps_group = ep_group.create_group('steps')
                
                # 收集所有步骤的数据
                observations = []
                actions = []
                rewards = []
                terminated = []
                truncated = []
                is_human_action = []
                
                for step in episode.steps:
                    # 处理观测
                    if isinstance(step.observation, dict) and 'agent_pos' in step.observation:
                        observations.append(step.observation['agent_pos'])
                    elif isinstance(step.observation, np.ndarray):
                        observations.append(step.observation.flatten())
                    else:
                        observations.append([0, 0])  # 默认值
                    
                    actions.append(step.action)
                    rewards.append(step.reward)
                    terminated.append(step.terminated)
                    truncated.append(step.truncated)
                    is_human_action.append(step.is_human_action)
                
                # 保存为数组
                steps_group.create_dataset('observations', data=np.array(observations))
                steps_group.create_dataset('actions', data=np.array(actions))
                steps_group.create_dataset('rewards', data=np.array(rewards))
                steps_group.create_dataset('terminated', data=np.array(terminated))
                steps_group.create_dataset('truncated', data=np.array(truncated))
                steps_group.create_dataset('is_human_action', data=np.array(is_human_action))

    def _save_to_csv(self, file_path: Path):
        """保存到CSV格式（最通用的纯数据格式）"""
        rows = []
        
        for episode in self.episodes:
            for step_idx, step in enumerate(episode.steps):
                row = {
                    'episode_id': episode.episode_id,
                    'step_idx': step_idx,
                    'reward': step.reward,
                    'terminated': step.terminated,
                    'truncated': step.truncated,
                    'is_human_action': step.is_human_action,
                    'episode_total_reward': episode.total_reward,
                    'episode_success': episode.success,
                    'episode_length': episode.length
                }
                
                # 添加动作数据
                if isinstance(step.action, np.ndarray):
                    for i, action_val in enumerate(step.action):
                        row[f'action_{i}'] = action_val
                
                # 添加观测数据
                if isinstance(step.observation, dict):
                    for key, value in step.observation.items():
                        if isinstance(value, np.ndarray):
                            if value.ndim == 1:
                                for i, obs_val in enumerate(value):
                                    row[f'obs_{key}_{i}'] = obs_val
                            else:
                                row[f'obs_{key}'] = str(value.tolist())
                        else:
                            row[f'obs_{key}'] = value
                elif isinstance(step.observation, np.ndarray):
                    for i, obs_val in enumerate(step.observation.flatten()):
                        row[f'obs_{i}'] = obs_val
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载轨迹数据（返回纯字典格式，无类依赖）
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的纯数据列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.suffix == '.pkl':
            # 仍然支持pickle格式的加载
            with open(file_path, 'rb') as f:
                episodes = pickle.load(f)
            return self._episodes_to_dict_list(episodes)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['episodes']
        elif file_path.suffix == '.h5':
            return self._load_from_hdf5(file_path)
        elif file_path.suffix == '.csv':
            return self._load_from_csv(file_path)
        elif file_path.suffix == '.npz':
            return self._load_from_npz(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _load_from_hdf5(self, file_path: Path) -> List[Dict[str, Any]]:
        """从HDF5格式加载数据"""
        episodes_data = []
        
        with h5py.File(file_path, 'r') as f:
            for ep_name in f.keys():
                if ep_name.startswith('episode_'):
                    ep_group = f[ep_name]
                    
                    # 加载步骤数据
                    steps_data = []
                    if 'steps' in ep_group:
                        steps_group = ep_group['steps']
                        observations = np.array(steps_group['observations'])
                        actions = np.array(steps_group['actions'])
                        rewards = np.array(steps_group['rewards'])
                        terminated = np.array(steps_group['terminated'])
                        truncated = np.array(steps_group['truncated'])
                        is_human_action = np.array(steps_group['is_human_action'])
                        
                        for i in range(len(observations)):
                            step_data = {
                                'observation': observations[i].tolist(),
                                'action': actions[i].tolist(),
                                'reward': float(rewards[i]),
                                'terminated': bool(terminated[i]),
                                'truncated': bool(truncated[i]),
                                'is_human_action': bool(is_human_action[i])
                            }
                            steps_data.append(step_data)
                    
                    episode_data = {
                        'episode_id': int(ep_group.attrs['episode_id']),
                        'total_reward': float(ep_group.attrs['total_reward']),
                        'success': bool(ep_group.attrs['success']),
                        'length': int(ep_group.attrs['length']),
                        'steps': steps_data
                    }
                    episodes_data.append(episode_data)
        
        return episodes_data

    def _load_from_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """从CSV格式加载数据"""
        df = pd.read_csv(file_path)
        episodes_data = []
        
        for episode_id in df['episode_id'].unique():
            episode_df = df[df['episode_id'] == episode_id].sort_values('step_idx')
            
            steps_data = []
            for _, row in episode_df.iterrows():
                # 提取观测数据
                obs_data = {}
                action_data = []
                
                for col in row.index:
                    if col.startswith('obs_'):
                        obs_data[col] = row[col]
                    elif col.startswith('action_'):
                        action_data.append(row[col])
                
                step_data = {
                    'observation': obs_data,
                    'action': action_data,
                    'reward': row['reward'],
                    'terminated': row['terminated'],
                    'truncated': row['truncated'],
                    'is_human_action': row['is_human_action']
                }
                steps_data.append(step_data)
            
            episode_data = {
                'episode_id': int(episode_id),
                'total_reward': float(episode_df.iloc[0]['episode_total_reward']),
                'success': bool(episode_df.iloc[0]['episode_success']),
                'length': int(episode_df.iloc[0]['episode_length']),
                'steps': steps_data
            }
            episodes_data.append(episode_data)
        
        return episodes_data

    def _load_from_npz(self, file_path: Path) -> List[Dict[str, Any]]:
        """从NPZ格式加载数据"""
        data = np.load(file_path)
        
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        episode_lengths = data['episode_lengths']
        
        episodes_data = []
        start_idx = 0
        
        for ep_idx, length in enumerate(episode_lengths):
            end_idx = start_idx + length
            
            steps_data = []
            for i in range(start_idx, end_idx):
                step_data = {
                    'observation': observations[i].tolist(),
                    'action': actions[i].tolist(),
                    'reward': float(rewards[i]),
                    'terminated': False,
                    'truncated': False,
                    'is_human_action': False
                }
                steps_data.append(step_data)
            
            episode_data = {
                'episode_id': ep_idx,
                'total_reward': float(rewards[start_idx:end_idx].sum()),
                'success': False,  # NPZ格式不保存此信息
                'length': int(length),
                'steps': steps_data
            }
            episodes_data.append(episode_data)
            start_idx = end_idx
        
        return episodes_data

    def _episodes_to_dict_list(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """将Episode对象转换为纯字典列表"""
        episodes_data = []
        
        for episode in episodes:
            steps_data = []
            for step in episode.steps:
                step_data = {
                    'observation': step.observation,
                    'action': step.action.tolist() if isinstance(step.action, np.ndarray) else step.action,
                    'reward': float(step.reward),
                    'terminated': bool(step.terminated),
                    'truncated': bool(step.truncated),
                    'info': step.info,
                    'is_human_action': bool(step.is_human_action)
                }
                steps_data.append(step_data)
            
            episode_data = {
                'episode_id': episode.episode_id,
                'total_reward': float(episode.total_reward),
                'success': bool(episode.success),
                'length': episode.length,
                'initial_state': episode.initial_state,
                'steps': steps_data
            }
            episodes_data.append(episode_data)
        
        return episodes_data

    def _episodes_to_numpy(self) -> Dict[str, np.ndarray]:
        """将Episode数据转换为numpy格式（用于npz保存）"""
        # 简化版本，主要保存数值数据
        all_observations = []
        all_actions = []
        all_rewards = []
        episode_lengths = []
        
        for episode in self.episodes:
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            
            for step in episode.steps:
                if isinstance(step.observation, dict) and 'agent_pos' in step.observation:
                    episode_obs.append(step.observation['agent_pos'])
                else:
                    episode_obs.append(step.observation)
                episode_actions.append(step.action)
                episode_rewards.append(step.reward)
            
            all_observations.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            episode_lengths.append(len(episode.steps))
        
        return {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'episode_lengths': np.array(episode_lengths)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.episodes:
            return {}
        
        total_steps = sum(ep.length for ep in self.episodes)
        success_rate = sum(ep.success for ep in self.episodes) / len(self.episodes)
        avg_reward = sum(ep.total_reward for ep in self.episodes) / len(self.episodes)
        avg_length = total_steps / len(self.episodes)
        
        return {
            'total_episodes': len(self.episodes),
            'total_steps': total_steps,
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'average_length': avg_length,
            'format': self.save_format
        } 