#!/usr/bin/env python3
"""
测试pixels_agent_pos观测类型的调试脚本 - 高分辨率版本
"""

import gymnasium as gym
import gym_pusht
import numpy as np
import copy

# 创建环境 - 使用512x512高分辨率
env = gym.make(
    "gym_pusht/PushT-v0", 
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
    observation_width=512,   # 高分辨率观测
    observation_height=512
)

print("=== 环境信息 ===")
print(f"观测空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 重置环境
obs, info = env.reset(seed=42)

print("\n=== 初始观测分析 ===")
print(f"观测类型: {type(obs)}")
print(f"观测键: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")

if isinstance(obs, dict):
    if 'pixels' in obs:
        pixels = obs['pixels']
        print(f"pixels类型: {type(pixels)}")
        print(f"pixels shape: {pixels.shape}")
        print(f"pixels数据类型: {pixels.dtype}")
        print(f"pixels是否为None: {pixels is None}")
        print(f"pixels值范围: [{pixels.min()}, {pixels.max()}]" if pixels is not None else "pixels为None")
    
    if 'agent_pos' in obs:
        agent_pos = obs['agent_pos']
        print(f"agent_pos类型: {type(agent_pos)}")
        print(f"agent_pos shape: {agent_pos.shape}")
        print(f"agent_pos值: {agent_pos}")

# 测试copy方法
print("\n=== 测试copy()方法 ===")
obs_copy = obs.copy()
print(f"copy后类型: {type(obs_copy)}")
print(f"copy后键: {list(obs_copy.keys()) if isinstance(obs_copy, dict) else 'Not a dict'}")

if isinstance(obs_copy, dict):
    if 'pixels' in obs_copy:
        pixels_copy = obs_copy['pixels']
        print(f"copy后pixels类型: {type(pixels_copy)}")
        print(f"copy后pixels是否为None: {pixels_copy is None}")
        if pixels_copy is not None:
            print(f"copy后pixels shape: {pixels_copy.shape}")

# 测试深拷贝
print("\n=== 测试深拷贝 ===")
try:
    obs_deepcopy = copy.deepcopy(obs)
    print("深拷贝成功")
    if isinstance(obs_deepcopy, dict) and 'pixels' in obs_deepcopy:
        pixels_deepcopy = obs_deepcopy['pixels']
        print(f"深拷贝后pixels是否为None: {pixels_deepcopy is None}")
        if pixels_deepcopy is not None:
            print(f"深拷贝后pixels shape: {pixels_deepcopy.shape}")
except Exception as e:
    print(f"深拷贝失败: {e}")

# 执行一步
print("\n=== 执行一步动作 ===")
action = env.action_space.sample()
print(f"动作: {action}")

next_obs, reward, terminated, truncated, next_info = env.step(action)
print(f"下一个观测类型: {type(next_obs)}")

if isinstance(next_obs, dict):
    if 'pixels' in next_obs:
        next_pixels = next_obs['pixels']
        print(f"step后pixels是否为None: {next_pixels is None}")
        if next_pixels is not None:
            print(f"step后pixels shape: {next_pixels.shape}")

env.close()
print("\n=== 测试完成 ===") 