import gym_pusht
import gymnasium as gym
import numpy as np

def test_gym_pusht():
    """测试gym-pusht环境的各种功能"""
    
    print("=== 测试gym-pusht环境 ===")
    
    # 创建环境
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    print("✓ 环境创建成功")
    
    # 获取环境信息
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"✓ 环境重置成功，观察形状: {obs.shape}")
    print(f"初始观察: {obs}")
    
    # 测试步进
    action = env.action_space.sample()  # 随机动作
    print(f"执行动作: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ 步进成功，新观察形状: {obs.shape}")
    print(f"奖励: {reward}")
    print(f"结束: {terminated}")
    print(f"截断: {truncated}")
    
    # 测试多步
    print("\n=== 执行10步随机动作 ===")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 奖励={reward:.4f}, 结束={terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print("环境已重置")
    
    # 测试渲染
    img = env.render()
    print(f"✓ 渲染成功，图像形状: {img.shape}")
    
    # 关闭环境
    env.close()
    print("✓ 环境关闭成功")
    
    print("\n=== 所有测试通过！gym-pusht环境工作正常 ===")

if __name__ == "__main__":
    test_gym_pusht() 