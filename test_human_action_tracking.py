#!/usr/bin/env python3
"""
测试human action标记功能
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_trajectory_data(data_path):
    """分析轨迹数据中的human action标记"""
    print(f"=== 分析轨迹数据: {data_path} ===")
    
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    print(f"总episode数: {len(data)}")
    
    total_steps = 0
    human_steps = 0
    ai_steps = 0
    
    for episode_idx, episode in enumerate(data):
        print(f"\n--- Episode {episode_idx + 1} ---")
        print(f"总步数: {len(episode.steps)}")
        print(f"成功: {episode.success}")
        print(f"总奖励: {episode.total_reward:.4f}")
        
        episode_human_steps = 0
        episode_ai_steps = 0
        
        # 分析每个步骤
        for step_idx, step in enumerate(episode.steps):
            total_steps += 1
            
            if step.is_human_action:
                human_steps += 1
                episode_human_steps += 1
            else:
                ai_steps += 1
                episode_ai_steps += 1
            
            # 显示前5步和后5步的详细信息
            if step_idx < 5 or step_idx >= len(episode.steps) - 5:
                control_type = "Human" if step.is_human_action else "AI"
                print(f"  步骤 {step_idx:3d}: {control_type:5s} | 奖励={step.reward:.4f} | 动作={step.action}")
        
        print(f"该episode - Human步数: {episode_human_steps}, AI步数: {episode_ai_steps}")
        print(f"Human比例: {episode_human_steps / len(episode.steps) * 100:.1f}%")
    
    print(f"\n=== 总体统计 ===")
    print(f"总步数: {total_steps}")
    print(f"Human控制步数: {human_steps}")
    print(f"AI控制步数: {ai_steps}")
    print(f"Human控制比例: {human_steps / total_steps * 100:.1f}%")
    print(f"AI控制比例: {ai_steps / total_steps * 100:.1f}%")
    
    return {
        'total_steps': total_steps,
        'human_steps': human_steps,
        'ai_steps': ai_steps,
        'human_ratio': human_steps / total_steps if total_steps > 0 else 0
    }

def test_new_trajectory_recording():
    """测试新的轨迹记录功能"""
    from utils import TrajectoryStep
    import gymnasium as gym
    import gym_pusht
    
    print("=== 测试TrajectoryStep新字段 ===")
    
    # 创建测试步骤
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, info = env.reset()
    action = env.action_space.sample()
    
    # 测试human action
    step_human = TrajectoryStep(
        observation=obs.copy(),
        action=action.copy(),
        reward=0.5,
        terminated=False,
        truncated=False,
        info=info.copy(),
        is_human_action=True
    )
    
    # 测试AI action
    step_ai = TrajectoryStep(
        observation=obs.copy(),
        action=action.copy(),
        reward=0.3,
        terminated=False,
        truncated=False,
        info=info.copy(),
        is_human_action=False
    )
    
    print(f"Human action step: is_human_action = {step_human.is_human_action}")
    print(f"AI action step: is_human_action = {step_ai.is_human_action}")
    
    # 测试默认值
    step_default = TrajectoryStep(
        observation=obs.copy(),
        action=action.copy(),
        reward=0.1,
        terminated=False,
        truncated=False,
        info=info.copy()
    )
    
    print(f"Default step: is_human_action = {step_default.is_human_action}")
    
    env.close()
    print("✓ TrajectoryStep测试通过")

if __name__ == "__main__":
    # 测试新的数据结构
    test_new_trajectory_recording()
    
    # 查找现有的轨迹数据文件
    data_dir = Path("data")
    if data_dir.exists():
        pickle_files = list(data_dir.glob("**/*.pickle"))
        
        if pickle_files:
            print(f"\n找到 {len(pickle_files)} 个轨迹数据文件:")
            for file_path in pickle_files:
                print(f"  {file_path}")
            
            # 分析最新的文件
            latest_file = max(pickle_files, key=lambda f: f.stat().st_mtime)
            print(f"\n分析最新文件: {latest_file}")
            
            try:
                stats = analyze_trajectory_data(latest_file)
                print(f"\n✓ 数据分析完成")
            except Exception as e:
                print(f"分析轨迹数据时出错: {e}")
                print("可能是因为数据文件格式与新的TrajectoryStep不兼容")
                print("建议重新生成轨迹数据")
        else:
            print("\n未找到轨迹数据文件")
            print("建议运行: python pusht_human.py 生成新的轨迹数据")
    else:
        print("\ndata目录不存在")
        print("建议运行: python pusht_human.py 生成新的轨迹数据") 