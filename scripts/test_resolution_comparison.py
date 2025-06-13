#!/usr/bin/env python3
"""
比较不同分辨率的图像质量
"""

import gymnasium as gym
import gym_pusht
import numpy as np
import matplotlib.pyplot as plt

# 创建低分辨率环境 (96x96)
env_low = gym.make(
    "gym_pusht/PushT-v0", 
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
    observation_width=96,
    observation_height=96
)

# 创建高分辨率环境 (512x512)
env_high = gym.make(
    "gym_pusht/PushT-v0", 
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
    observation_width=512,
    observation_height=512
)

print("=== 分辨率比较测试 ===")

# 重置环境到相同状态
seed = 42
obs_low, _ = env_low.reset(seed=seed)
obs_high, _ = env_high.reset(seed=seed)

# 获取像素数据
pixels_low = obs_low['pixels']
pixels_high = obs_high['pixels']

print(f"低分辨率图像: {pixels_low.shape}")
print(f"高分辨率图像: {pixels_high.shape}")

# 计算图像统计信息
print(f"\n低分辨率图像统计:")
print(f"  大小: {pixels_low.nbytes / 1024:.1f} KB")
print(f"  值范围: [{pixels_low.min()}, {pixels_low.max()}]")
print(f"  平均值: {pixels_low.mean():.1f}")

print(f"\n高分辨率图像统计:")
print(f"  大小: {pixels_high.nbytes / 1024:.1f} KB")
print(f"  值范围: [{pixels_high.min()}, {pixels_high.max()}]")
print(f"  平均值: {pixels_high.mean():.1f}")

# 保存图像进行比较
print(f"\n保存图像到文件以便比较...")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(pixels_low)
plt.title('96x96 分辨率 (模糊)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pixels_high)
plt.title('512x512 分辨率 (清晰)')
plt.axis('off')

plt.tight_layout()
plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
print("图像对比已保存到: resolution_comparison.png")

# 关闭环境
env_low.close()
env_high.close()

print("\n=== 测试完成 ===")
print("结论:")
print("- 动作空间 [0,512] 是物理坐标系，表示智能体可以移动的范围")
print("- 观测图像分辨率可以独立设置 (96x96 或 512x512)")
print("- 512x512 分辨率图像更清晰，包含更多细节")
print("- 高分辨率图像占用更多内存，但提供更好的视觉质量") 