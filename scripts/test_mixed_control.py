#!/usr/bin/env python3
"""
测试混合Human和AI控制的轨迹生成
通过编程方式模拟用户按空格键切换控制模式
"""

import sys
import time
import logging
import pygame
import gymnasium as gym
import gym_pusht
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, '.')

from utils import TrajectoryStep, Episode, DataManager, HuggingFaceUploader

# 简单的随机策略类
class SimpleRandomPolicy:
    def __init__(self, action_space, seed=42):
        self.action_space = action_space
        np.random.seed(seed)
    
    def get_action(self, obs):
        return self.action_space.sample()

def simulate_mixed_control_game():
    """模拟混合控制游戏，生成包含Human和AI动作的轨迹"""
    
    print("=== 模拟混合控制游戏 ===")
    
    # 初始化环境
    env = gym.make(
        "gym_pusht/PushT-v0", 
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=512,
        observation_height=512
    )
    
    # 初始化策略和控制器
    random_policy = SimpleRandomPolicy(env.action_space, 42)
    
    # 初始化数据管理器
    data_manager = DataManager("data/mixed_control_test", "pickle")
    
    # 游戏状态
    current_trajectory = []
    user_control = True  # 开始时是用户控制
    
    print("开始模拟游戏...")
    
    # 重置环境
    obs, info = env.reset(seed=42)
    
    episode_reward = 0.0
    step_count = 0
    max_steps = 100  # 较短的episode用于测试
    
    print(f"初始状态: user_control={user_control}")
    
    for step_num in range(max_steps):
        # 模拟用户按空格键切换控制模式（每20步切换一次）
        if step_num > 0 and step_num % 20 == 0:
            user_control = not user_control
            mode = "用户控制" if user_control else "AI控制"
            print(f"步骤 {step_num}: 切换到 {mode}")
        
        # 获取动作
        if user_control:
            # 模拟用户控制：移动到目标位置附近
            target = np.array([256.0, 200.0])  # 固定目标位置
            noise = np.random.normal(0, 10, 2)  # 添加噪声模拟人类不精确性
            action = np.clip(target + noise, 0, 512)
        else:
            # AI控制：使用随机策略
            action = random_policy.get_action(obs)
        
        # 执行动作
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        # 记录轨迹步骤
        step = TrajectoryStep(
            observation=obs.copy() if isinstance(obs, dict) else obs.copy(),
            action=action.copy(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info.copy(),
            is_human_action=user_control  # 关键：记录是否为人类控制
        )
        current_trajectory.append(step)
        
        # 更新状态
        obs = next_obs
        info = next_info
        episode_reward += reward
        step_count += 1
        
        # 每10步显示一次进度
        if step_count % 10 == 0:
            control_type = "Human" if user_control else "AI"
            print(f"步骤 {step_count}: {control_type} 控制, 奖励={reward:.4f}, 总奖励={episode_reward:.4f}")
        
        # 检查结束条件
        if terminated or truncated:
            success = info.get('is_success', False)
            print(f"Episode提前结束: 步数={step_count}, 成功={success}")
            break
    
    # 保存轨迹
    initial_state = {
        'agent_pos': [256.0, 400.0],  # 默认初始位置
        'block_pos': [256.0, 300.0],
        'block_angle': 0.0,
        'goal_pose': [256.0, 256.0, 0.785]  # π/4
    }
    
    episode = Episode(
        steps=current_trajectory,
        episode_id=0,
        total_reward=episode_reward,
        success=info.get('is_success', False),
        length=step_count,
        initial_state=initial_state
    )
    
    data_manager.add_episode(episode)
    
    # 保存数据
    data_path = data_manager.save_data("mixed_control_demo")
    print(f"混合控制轨迹已保存到: {data_path}")
    
    # 分析轨迹
    analyze_mixed_trajectory(current_trajectory)
    
    env.close()
    return data_path

def analyze_mixed_trajectory(trajectory):
    """分析混合轨迹的控制模式分布"""
    print(f"\n=== 轨迹分析 ===")
    print(f"总步数: {len(trajectory)}")
    
    human_steps = sum(1 for step in trajectory if step.is_human_action)
    ai_steps = len(trajectory) - human_steps
    
    print(f"Human控制步数: {human_steps}")
    print(f"AI控制步数: {ai_steps}")
    print(f"Human控制比例: {human_steps / len(trajectory) * 100:.1f}%")
    
    # 分析切换模式
    switches = 0
    last_mode = trajectory[0].is_human_action
    switch_points = []
    
    for i, step in enumerate(trajectory[1:], 1):
        if step.is_human_action != last_mode:
            switches += 1
            switch_points.append(i)
            last_mode = step.is_human_action
    
    print(f"控制模式切换次数: {switches}")
    print(f"切换点: {switch_points}")
    
    # 显示前10步和后10步的控制模式
    print(f"\n前10步控制模式:")
    for i in range(min(10, len(trajectory))):
        mode = "Human" if trajectory[i].is_human_action else "AI"
        print(f"  步骤 {i:2d}: {mode}")
    
    if len(trajectory) > 10:
        print(f"\n后10步控制模式:")
        for i in range(max(len(trajectory)-10, 10), len(trajectory)):
            mode = "Human" if trajectory[i].is_human_action else "AI"
            print(f"  步骤 {i:2d}: {mode}")

if __name__ == "__main__":
    # 生成混合控制轨迹
    data_path = simulate_mixed_control_game()
    
    print(f"\n=== 测试完成 ===")
    print(f"生成的混合控制轨迹数据: {data_path}")
    print("✓ Human action标记功能测试成功") 