#!/usr/bin/env python3
"""
测试轨迹记录的pixels数据
"""

import pickle
import numpy as np
import gymnasium as gym
import gym_pusht
from utils import TrajectoryStep, Episode, DataManager

def test_single_trajectory():
    """测试单个轨迹的记录"""
    print("=== 测试单个轨迹记录 ===")
    
    # 创建环境
    env = gym.make(
        "gym_pusht/PushT-v0", 
        obs_type="pixels_agent_pos",
        render_mode="rgb_array"
    )
    
    # 重置环境
    obs, info = env.reset(seed=42)
    trajectory_steps = []
    
    print(f"初始观测类型: {type(obs)}")
    print(f"初始观测键: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    
    # 执行几步
    for step in range(5):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        # 记录轨迹步骤 - 使用step后的观测
        step_data = TrajectoryStep(
            observation=next_obs.copy() if isinstance(next_obs, dict) else next_obs.copy(),
            action=action.copy(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=next_info.copy()
        )
        trajectory_steps.append(step_data)
        
        # 检查观测
        if isinstance(next_obs, dict):
            pixels = next_obs.get('pixels')
            agent_pos = next_obs.get('agent_pos')
            print(f"步骤 {step}: pixels shape={pixels.shape if pixels is not None else None}, "
                  f"agent_pos={agent_pos}, reward={reward:.4f}")
            
            # 检查记录的数据
            recorded_obs = step_data.observation
            recorded_pixels = recorded_obs.get('pixels') if isinstance(recorded_obs, dict) else None
            print(f"  记录的pixels shape={recorded_pixels.shape if recorded_pixels is not None else None}")
        
        obs = next_obs
        info = next_info
        
        if terminated or truncated:
            break
    
    env.close()
    
    # 创建episode并保存
    episode = Episode(
        steps=trajectory_steps,
        episode_id=0,
        total_reward=sum(step.reward for step in trajectory_steps),
        success=False,
        length=len(trajectory_steps),
        initial_state={'agent_pos': [256, 400], 'block_pos': [256, 300], 'block_angle': 0, 'goal_pose': [256, 256, np.pi/4]}
    )
    
    # 保存数据
    data_manager = DataManager("test_data", "pickle")
    data_manager.add_episode(episode)
    data_path = data_manager.save_data("test_pixels_trajectory")
    
    print(f"\n轨迹已保存到: {data_path}")
    
    # 加载并验证
    print("\n=== 验证保存的数据 ===")
    with open(data_path, 'rb') as f:
        loaded_episodes = pickle.load(f)
    
    if loaded_episodes:
        episode = loaded_episodes[0]
        print(f"加载了 {len(episode.steps)} 个步骤")
        
        for i, step in enumerate(episode.steps):
            obs = step.observation
            if isinstance(obs, dict):
                pixels = obs.get('pixels')
                agent_pos = obs.get('agent_pos')
                print(f"步骤 {i}: pixels是否为None={pixels is None}, "
                      f"pixels shape={pixels.shape if pixels is not None else None}, "
                      f"agent_pos={agent_pos}")
            else:
                print(f"步骤 {i}: 观测不是字典类型: {type(obs)}")

if __name__ == "__main__":
    test_single_trajectory()
    print("\n=== 测试完成 ===") 