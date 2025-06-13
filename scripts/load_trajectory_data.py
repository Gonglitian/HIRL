#!/usr/bin/env python3
"""
轨迹数据加载脚本
解决pickle文件中src模块依赖的问题
"""

import pickle
import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# 现在可以安全导入src模块
from src.core.data_types import TrajectoryStep, Episode


def load_trajectory_data(data_path: str):
    """
    安全加载轨迹数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        加载的Episode列表
    """
    try:
        with open(data_path, 'rb') as f:
            episodes = pickle.load(f)
        print(f"✅ 成功加载了 {len(episodes)} 个episode")
        return episodes
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def analyze_episodes(episodes):
    """分析轨迹数据"""
    if not episodes:
        return
    
    print("\n📊 轨迹数据分析:")
    print(f"总轨迹数: {len(episodes)}")
    
    for i, episode in enumerate(episodes):
        success_str = "成功" if episode.success else "失败"
        print(f"Episode {i}: ID={episode.episode_id}, 步数={episode.length}, "
              f"奖励={episode.total_reward:.3f}, {success_str}")
        
        # 分析人类vs AI动作
        human_actions = sum(1 for step in episode.steps if step.is_human_action)
        ai_actions = len(episode.steps) - human_actions
        print(f"  - 人类动作: {human_actions}, AI动作: {ai_actions}")


if __name__ == "__main__":
    # 示例用法
    data_path = "/home/glt/Projects/HIRL/data/pusht_human_mouse_trajectories/trajectories_1episodes.pkl"
    
    # 加载数据
    episodes = load_trajectory_data(data_path)
    
    # 分析数据
    if episodes:
        analyze_episodes(episodes) 