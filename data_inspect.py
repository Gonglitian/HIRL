#!/usr/bin/env python3
"""
HIRL数据检查工具
使用DataManager读取H5轨迹数据并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# 导入HIRL包
from HIRL.data import DataManager
from HIRL.core.data_types import TrajectoryStep, Episode

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def inspect_h5_data(file_path: str, episode_idx: int = 0, step_idx: int = 0):
    """
    检查H5轨迹数据并显示指定步骤的图像
    
    Args:
        file_path: H5文件路径
        episode_idx: 要查看的回合索引
        step_idx: 要查看的步骤索引
    """
    print(f"🔍 正在检查数据文件: {file_path}")
    
    # 创建临时DataManager来加载数据
    data_manager = DataManager(save_dir="/tmp", save_format="hdf5")
    
    try:
        # 加载数据
        print("📂 正在加载数据...")
        episodes_data = data_manager.load_data(file_path)
        
        print(f"✅ 成功加载数据！")
        print(f"📊 数据统计:")
        print(f"   总回合数: {len(episodes_data)}")
        
        if not episodes_data:
            print("❌ 没有找到数据")
            return
        
        # 显示每个回合的基本信息
        for i, episode in enumerate(episodes_data):
            print(f"   回合 {i}: {episode['length']} 步, "
                  f"奖励: {episode['total_reward']:.3f}, "
                  f"成功: {episode['success']}")
        
        # 检查指定回合
        if episode_idx >= len(episodes_data):
            print(f"❌ 回合索引 {episode_idx} 超出范围 (0-{len(episodes_data)-1})")
            return
        
        episode = episodes_data[episode_idx]
        print(f"\n🎯 查看回合 {episode_idx}:")
        print(f"   回合ID: {episode['episode_id']}")
        print(f"   总步数: {episode['length']}")
        print(f"   总奖励: {episode['total_reward']:.3f}")
        print(f"   是否成功: {episode['success']}")
        
        steps = episode['steps']
        if step_idx >= len(steps):
            print(f"❌ 步骤索引 {step_idx} 超出范围 (0-{len(steps)-1})")
            return
        
        step = steps[step_idx]
        print(f"\n🔍 查看步骤 {step_idx}:")
        print(f"   奖励: {step['reward']}")
        print(f"   是否终止: {step['terminated']}")
        print(f"   是否截断: {step['truncated']}")
        print(f"   人类动作: {step['is_human_action']}")
        print(f"   动作: {step['action']}")
        
        # 分析观测数据
        observation = step['observation']
        print(f"   观测数据类型: {type(observation)}")
        
        if isinstance(observation, dict):
            print(f"   观测数据键: {list(observation.keys())}")
            if 'pixels' in observation:
                pixels = observation['pixels']
                print(f"   像素数据形状: {pixels.shape if hasattr(pixels, 'shape') else type(pixels)}")
                print(f"   像素数据类型: {type(pixels)}")
            if 'agent_pos' in observation:
                agent_pos = observation['agent_pos']
                print(f"   智能体位置: {agent_pos}")
        elif isinstance(observation, list):
            print(f"   观测数据长度: {len(observation)}")
            if len(observation) > 0:
                print(f"   观测数据示例: {observation[:5]}...")  # 显示前5个元素
        
        # 尝试获取像素数据进行可视化
        print(f"\n🖼️  尝试获取图像数据...")
        
        pixels_data = None
        if isinstance(observation, dict) and 'pixels' in observation:
            pixels_data = observation['pixels']
            print(f"   从观测字典中找到pixels数据，形状: {pixels_data.shape if hasattr(pixels_data, 'shape') else type(pixels_data)}")
        
        # 检查是否是像素观测（兼容旧格式）
        elif isinstance(observation, list) and len(observation) > 100:
            print(f"   检测到可能的像素数据，长度: {len(observation)}")
            
            # 尝试重塑为图像
            obs_array = np.array(observation)
            print(f"   观测数组形状: {obs_array.shape}")
            
            # 尝试不同的图像尺寸
            possible_shapes = [
                (512, 512, 3),  # RGB图像
                (256, 256, 3),  # 较小的RGB图像
                (512, 512),     # 灰度图像
                (256, 256),     # 较小的灰度图像
            ]
            
            for shape in possible_shapes:
                if obs_array.size == np.prod(shape):
                    print(f"   🎯 匹配图像形状: {shape}")
                    pixels_data = obs_array.reshape(shape)
                    break
        
        # 如果找到了像素数据，显示图像
        if pixels_data is not None and hasattr(pixels_data, 'shape') and len(pixels_data.shape) >= 2:
            print(f"   🎯 找到像素数据，形状: {pixels_data.shape}")
            
            # 显示图像
            plt.figure(figsize=(10, 8))
            
            if len(pixels_data.shape) == 3:  # RGB图像
                # 确保像素值在正确范围内
                if pixels_data.max() <= 1.0:
                    img_display = pixels_data
                else:
                    img_display = pixels_data / 255.0
                plt.imshow(img_display)
            else:  # 灰度图像
                plt.imshow(pixels_data, cmap='gray')
            
            plt.title(f'Episode {episode_idx}, Step {step_idx}\n'
                     f'Reward: {step["reward"]:.3f}, '
                     f'Human Action: {step["is_human_action"]}')
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图像
            save_path = f"step_{episode_idx}_{step_idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   💾 图像已保存为: {save_path}")
            
            plt.show()
            return
        
        # 如果没有找到图像数据，显示观测信息
        print("   ⚠️  未检测到标准图像数据格式")
        print("   📋 观测数据详情:")
        if isinstance(observation, dict):
            print(f"       数据类型: 字典")
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    print(f"       {key}: {type(value)} - 形状{value.shape}")
                else:
                    print(f"       {key}: {type(value)} - {value}")
        elif isinstance(observation, list):
            print(f"       数据类型: 列表")
            print(f"       长度: {len(observation)}")
            if len(observation) <= 20:
                print(f"       内容: {observation}")
        else:
            print(f"       数据类型: {type(observation)}")
            print(f"       内容: {observation}")
            
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        import traceback
        traceback.print_exc()

def show_multiple_steps(file_path: str, episode_idx: int = 0, max_steps: int = 6):
    """
    显示一个回合中的多个步骤图像
    
    Args:
        file_path: H5文件路径
        episode_idx: 回合索引
        max_steps: 最大显示步数
    """
    print(f"🎬 显示回合 {episode_idx} 的多个步骤...")
    
    data_manager = DataManager(save_dir="/tmp", save_format="hdf5")
    
    try:
        episodes_data = data_manager.load_data(file_path)
        
        if episode_idx >= len(episodes_data):
            print(f"❌ 回合索引超出范围")
            return
            
        episode = episodes_data[episode_idx]
        steps = episode['steps']
        
        # 计算要显示的步骤
        step_indices = np.linspace(0, len(steps)-1, min(max_steps, len(steps)), dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, step_idx in enumerate(step_indices):
            if i >= len(axes):
                break
                
            step = steps[step_idx]
            observation = step['observation']
            
            # 尝试获取像素数据
            pixels_data = None
            if isinstance(observation, dict) and 'pixels' in observation:
                pixels_data = observation['pixels']
            elif isinstance(observation, list) and len(observation) > 100:
                obs_array = np.array(observation)
                
                # 尝试512x512x3形状
                if obs_array.size == 512 * 512 * 3:
                    pixels_data = obs_array.reshape(512, 512, 3)
            
            if pixels_data is not None and hasattr(pixels_data, 'shape') and len(pixels_data.shape) >= 2:
                if len(pixels_data.shape) == 3 and pixels_data.max() > 1.0:
                    pixels_data = pixels_data / 255.0
                
                axes[i].imshow(pixels_data)
                axes[i].set_title(f'Step {step_idx}\nReward: {step["reward"]:.3f}')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Step {step_idx}\nNo Image Data', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(step_indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Episode {episode_idx} - Multi-Step Visualization', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ 显示多步骤失败: {e}")

def main():
    """主函数"""
    # 数据文件路径
    data_file = "/home/glt/Projects/HIRL/data/pusht_human_mouse_trajectories/50_trajs.h5"
    
    print("=" * 60)
    print("HIRL 数据检查工具")
    print("=" * 60)
    
    # 检查文件是否存在
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 基本数据检查
    inspect_h5_data(data_file, episode_idx=0, step_idx=0)
    
    print("\n" + "="*40)
    
    # 显示多个步骤
    show_multiple_steps(data_file, episode_idx=0, max_steps=6)

if __name__ == "__main__":
    main()
