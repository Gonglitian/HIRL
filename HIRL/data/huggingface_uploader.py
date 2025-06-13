"""
Hugging Face数据上传模块
负责将轨迹数据上传到Hugging Face Hub
"""

import os
from typing import List, Optional
from pathlib import Path
import logging

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from ..core.data_types import Episode


class HuggingFaceUploader:
    """Hugging Face数据上传器"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化上传器
        
        Args:
            token: Hugging Face API token，如果为None则从环境变量获取
        """
        self.token = token or os.getenv('HF_TOKEN')
        if not self.token:
            logging.warning("未提供Hugging Face token，无法上传数据")
        
        self.api = HfApi(token=self.token) if self.token else None
    
    def upload_dataset(self, data_path: str, repo_id: str, private: bool = False) -> str:
        """
        上传数据集到Hugging Face Hub
        
        Args:
            data_path: 数据文件路径
            repo_id: 仓库ID (用户名/数据集名)
            private: 是否私有仓库
            
        Returns:
            数据集URL
        """
        if not self.api:
            raise ValueError("未配置Hugging Face token")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据
        from .data_manager import DataManager
        data_manager = DataManager(save_dir="temp")
        episodes = data_manager.load_data(str(data_path))
        
        # 转换为Hugging Face Dataset格式
        dataset_dict = self._episodes_to_hf_dataset(episodes)
        
        # 上传数据集
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=self.token
        )
        
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        logging.info(f"数据集已上传到: {dataset_url}")
        
        return dataset_url
    
    def _episodes_to_hf_dataset(self, episodes: List[Episode]) -> DatasetDict:
        """将Episode数据转换为Hugging Face Dataset格式"""
        
        # 准备数据
        episode_data = []
        step_data = []
        
        for episode in episodes:
            # Episode级别数据
            episode_data.append({
                'episode_id': episode.episode_id,
                'total_reward': episode.total_reward,
                'success': episode.success,
                'length': episode.length,
                'initial_state': str(episode.initial_state)  # 转换为字符串
            })
            
            # Step级别数据
            for step_idx, step in enumerate(episode.steps):
                step_record = {
                    'episode_id': episode.episode_id,
                    'step_id': step_idx,
                    'action': step.action.tolist(),
                    'reward': step.reward,
                    'terminated': step.terminated,
                    'truncated': step.truncated,
                    'is_human_action': step.is_human_action
                }
                
                # 处理观测数据
                if isinstance(step.observation, dict):
                    if 'agent_pos' in step.observation:
                        step_record['agent_pos'] = step.observation['agent_pos'].tolist()
                    if 'pixels' in step.observation:
                        # 对于像素数据，只保存形状信息
                        step_record['pixels_shape'] = list(step.observation['pixels'].shape)
                else:
                    step_record['observation'] = step.observation.tolist() if hasattr(step.observation, 'tolist') else step.observation
                
                step_data.append(step_record)
        
        # 创建Dataset
        episodes_dataset = Dataset.from_list(episode_data)
        steps_dataset = Dataset.from_list(step_data)
        
        return DatasetDict({
            'episodes': episodes_dataset,
            'steps': steps_dataset
        }) 