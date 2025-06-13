#!/usr/bin/env python3
"""
调试初始状态脚本
检查轨迹记录的初始状态是否正确
"""

import pickle
import numpy as np
from pathlib import Path

def main():
    """检查轨迹文件中的初始状态信息"""
    data_path = Path("data/pusht_trajectories/pusht_human_demo.pickle")
    
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        return
    
    # 加载轨迹数据
    with open(data_path, 'rb') as f:
        episodes = pickle.load(f)
    
    print(f"加载了 {len(episodes)} 条轨迹")
    
    for i, episode in enumerate(episodes):
        print(f"\n=== 轨迹 {i} (ID: {episode.episode_id}) ===")
        print(f"成功: {episode.success}")
        print(f"步数: {episode.length}")
        print(f"总奖励: {episode.total_reward:.4f}")
        
        # 检查初始状态
        initial_state = episode.initial_state
        print(f"\n初始状态:")
        print(f"  Agent位置: {initial_state['agent_pos']}")
        print(f"  Block位置: {initial_state['block_pos']}")
        print(f"  Block角度: {initial_state['block_angle']:.4f}")
        print(f"  Goal位置: {initial_state['goal_pose']}")
        
        # 检查第一步的观测
        if episode.steps:
            first_step = episode.steps[0]
            print(f"\n第一步轨迹:")
            print(f"  观测: {first_step.observation}")
            print(f"  动作: {first_step.action}")
            
            # 对于state观测类型，检查一致性
            if len(first_step.observation) == 5:  # state观测
                obs_agent_pos = first_step.observation[:2]
                obs_block_pos = first_step.observation[2:4]
                obs_block_angle = first_step.observation[4]
                
                print(f"\n一致性检查:")
                agent_diff = np.linalg.norm(np.array(obs_agent_pos) - np.array(initial_state['agent_pos']))
                block_diff = np.linalg.norm(np.array(obs_block_pos) - np.array(initial_state['block_pos']))
                
                # 角度差异需要考虑模运算
                angle1 = obs_block_angle
                angle2 = initial_state['block_angle'] % (2 * np.pi)
                angle_diff = min(abs(angle1 - angle2), abs(angle1 - angle2 + 2*np.pi), abs(angle1 - angle2 - 2*np.pi))
                
                print(f"  Agent位置差异: {agent_diff:.6f}")
                print(f"  Block位置差异: {block_diff:.6f}")
                print(f"  Block角度差异: {angle_diff:.6f} (原始: {obs_block_angle:.4f} vs {initial_state['block_angle']:.4f})")
                
                if agent_diff > 0.1 or block_diff > 0.1 or angle_diff > 0.1:
                    print("  ⚠️ 检测到明显差异！")
                else:
                    print("  ✓ 初始状态与第一步观测基本一致")

if __name__ == "__main__":
    main() 