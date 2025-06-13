#!/usr/bin/env python3
"""
分析Human和AI动作的差异
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_action_patterns(data_path):
    """分析Human和AI动作模式的差异"""
    print(f"=== 分析动作模式: {data_path} ===")
    
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    human_actions = []
    ai_actions = []
    
    # 收集Human和AI的动作
    for episode in data:
        for step in episode.steps:
            if step.is_human_action:
                human_actions.append(step.action)
            else:
                ai_actions.append(step.action)
    
    human_actions = np.array(human_actions)
    ai_actions = np.array(ai_actions)
    
    print(f"Human动作数量: {len(human_actions)}")
    print(f"AI动作数量: {len(ai_actions)}")
    
    if len(human_actions) > 0 and len(ai_actions) > 0:
        # 分析统计特性
        print(f"\nHuman动作统计:")
        print(f"  X坐标: 均值={human_actions[:, 0].mean():.2f}, 标准差={human_actions[:, 0].std():.2f}")
        print(f"  Y坐标: 均值={human_actions[:, 1].mean():.2f}, 标准差={human_actions[:, 1].std():.2f}")
        
        print(f"\nAI动作统计:")
        print(f"  X坐标: 均值={ai_actions[:, 0].mean():.2f}, 标准差={ai_actions[:, 0].std():.2f}")
        print(f"  Y坐标: 均值={ai_actions[:, 1].mean():.2f}, 标准差={ai_actions[:, 1].std():.2f}")
        
        # 可视化动作分布
        plt.figure(figsize=(15, 5))
        
        # X坐标分布
        plt.subplot(1, 3, 1)
        plt.hist(human_actions[:, 0], bins=20, alpha=0.7, label='Human', color='blue')
        plt.hist(ai_actions[:, 0], bins=20, alpha=0.7, label='AI', color='red')
        plt.xlabel('X坐标')
        plt.ylabel('频次')
        plt.title('X坐标分布对比')
        plt.legend()
        
        # Y坐标分布
        plt.subplot(1, 3, 2)
        plt.hist(human_actions[:, 1], bins=20, alpha=0.7, label='Human', color='blue')
        plt.hist(ai_actions[:, 1], bins=20, alpha=0.7, label='AI', color='red')
        plt.xlabel('Y坐标')
        plt.ylabel('频次')
        plt.title('Y坐标分布对比')
        plt.legend()
        
        # 2D散点图
        plt.subplot(1, 3, 3)
        plt.scatter(human_actions[:, 0], human_actions[:, 1], alpha=0.6, label='Human', color='blue', s=10)
        plt.scatter(ai_actions[:, 0], ai_actions[:, 1], alpha=0.6, label='AI', color='red', s=10)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('动作分布 (2D)')
        plt.legend()
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        
        plt.tight_layout()
        plt.savefig('human_vs_ai_actions.png', dpi=150, bbox_inches='tight')
        print(f"\n动作分布图已保存到: human_vs_ai_actions.png")
        
        # 分析动作序列的连续性
        analyze_action_consistency(human_actions, ai_actions)
    
    else:
        print("数据不足：需要同时包含Human和AI动作才能进行对比分析")

def analyze_action_consistency(human_actions, ai_actions):
    """分析动作的连续性和一致性"""
    print(f"\n=== 动作一致性分析 ===")
    
    # 计算相邻动作之间的距离
    def action_distances(actions):
        if len(actions) < 2:
            return np.array([])
        return np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    
    human_distances = action_distances(human_actions)
    ai_distances = action_distances(ai_actions)
    
    if len(human_distances) > 0:
        print(f"Human动作连续性:")
        print(f"  平均移动距离: {human_distances.mean():.2f}")
        print(f"  移动距离标准差: {human_distances.std():.2f}")
        print(f"  最大移动距离: {human_distances.max():.2f}")
    
    if len(ai_distances) > 0:
        print(f"AI动作连续性:")
        print(f"  平均移动距离: {ai_distances.mean():.2f}")
        print(f"  移动距离标准差: {ai_distances.std():.2f}")
        print(f"  最大移动距离: {ai_distances.max():.2f}")
    
    # 分析动作的"人类特征"
    if len(human_actions) > 0:
        print(f"\nHuman动作特征:")
        # 检查是否有重复动作（人类可能会短暂停留）
        repeated_actions = sum(1 for i in range(len(human_actions)-1) 
                             if np.allclose(human_actions[i], human_actions[i+1], atol=1.0))
        print(f"  重复/相似动作数量: {repeated_actions}")
        print(f"  重复动作比例: {repeated_actions/(len(human_actions)-1)*100:.1f}%")

def compare_reward_patterns(data_path):
    """比较Human和AI获得的奖励模式"""
    print(f"\n=== 奖励模式对比 ===")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    human_rewards = []
    ai_rewards = []
    
    for episode in data:
        for step in episode.steps:
            if step.is_human_action:
                human_rewards.append(step.reward)
            else:
                ai_rewards.append(step.reward)
    
    if len(human_rewards) > 0 and len(ai_rewards) > 0:
        print(f"Human平均奖励: {np.mean(human_rewards):.4f}")
        print(f"AI平均奖励: {np.mean(ai_rewards):.4f}")
        print(f"Human累计奖励: {np.sum(human_rewards):.4f}")
        print(f"AI累计奖励: {np.sum(ai_rewards):.4f}")
        
        # 统计正奖励的比例
        human_positive = sum(1 for r in human_rewards if r > 0)
        ai_positive = sum(1 for r in ai_rewards if r > 0)
        
        print(f"Human获得正奖励步数: {human_positive}/{len(human_rewards)} ({human_positive/len(human_rewards)*100:.1f}%)")
        print(f"AI获得正奖励步数: {ai_positive}/{len(ai_rewards)} ({ai_positive/len(ai_rewards)*100:.1f}%)")

if __name__ == "__main__":
    # 查找包含混合控制的数据文件
    data_dir = Path("data")
    mixed_files = list(data_dir.glob("**/mixed_control_demo.pickle"))
    
    if mixed_files:
        latest_file = max(mixed_files, key=lambda f: f.stat().st_mtime)
        print(f"分析文件: {latest_file}")
        
        analyze_action_patterns(latest_file)
        compare_reward_patterns(latest_file)
        
        print(f"\n=== 分析完成 ===")
        print("✓ Human vs AI动作模式分析完成")
        print("查看生成的图表: human_vs_ai_actions.png")
    else:
        print("未找到混合控制数据文件")
        print("请先运行: python test_mixed_control.py") 