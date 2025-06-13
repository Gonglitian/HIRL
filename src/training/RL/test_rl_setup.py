#!/usr/bin/env python3
"""
RL环境设置测试脚本

用于验证stable_baselines3、gym-pusht等依赖是否正确安装

用法:
python RL/test_rl_setup.py
"""

import sys
import traceback

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    try:
        import gymnasium as gym
        print("✅ gymnasium 导入成功")
    except ImportError as e:
        print(f"❌ gymnasium 导入失败: {e}")
        return False
    
    try:
        import gym_pusht
        print("✅ gym_pusht 导入成功")
    except ImportError as e:
        print(f"❌ gym_pusht 导入失败: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"✅ stable_baselines3 导入成功 (版本: {stable_baselines3.__version__})")
    except ImportError as e:
        print(f"❌ stable_baselines3 导入失败: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO, SAC
        print("✅ PPO, SAC 导入成功")
    except ImportError as e:
        print(f"❌ PPO, SAC 导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch 导入成功 (版本: {torch.__version__})")
    except ImportError as e:
        print(f"❌ torch 导入失败: {e}")
        return False
    
    try:
        import hydra
        print(f"✅ hydra 导入成功")
    except ImportError as e:
        print(f"❌ hydra 导入失败: {e}")
        return False
    
    try:
        import wandb
        print(f"✅ wandb 导入成功")
    except ImportError as e:
        print(f"❌ wandb 导入失败: {e}")
        return False
    
    return True

def test_env_creation():
    """测试PushT环境创建"""
    print("\n🔍 测试环境创建...")
    
    try:
        import gymnasium as gym
        import gym_pusht
        
        # 测试不同观测类型
        obs_types = ["state", "pixels", "pixels_agent_pos"]
        
        for obs_type in obs_types:
            try:
                env = gym.make(
                    "gym_pusht/PushT-v0",
                    obs_type=obs_type,
                    render_mode="rgb_array"
                )
                obs, info = env.reset()
                print(f"✅ {obs_type} 环境创建成功")
                print(f"   观测空间: {env.observation_space}")
                print(f"   动作空间: {env.action_space}")
                env.close()
            except Exception as e:
                print(f"❌ {obs_type} 环境创建失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        import gymnasium as gym
        import gym_pusht
        from stable_baselines3 import PPO, SAC
        
        # 创建环境
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array"
        )
        
        # 测试PPO模型创建
        try:
            ppo_model = PPO(
                "MultiInputPolicy",
                env,
                verbose=0,
                device="auto"
            )
            print(f"✅ PPO模型创建成功 (设备: {ppo_model.device})")
        except Exception as e:
            print(f"❌ PPO模型创建失败: {e}")
            return False
        
        # 测试SAC模型创建
        try:
            sac_model = SAC(
                "MultiInputPolicy",
                env,
                verbose=0,
                device="auto"
            )
            print(f"✅ SAC模型创建成功 (设备: {sac_model.device})")
        except Exception as e:
            print(f"❌ SAC模型创建失败: {e}")
            return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🔍 测试GPU可用性...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU可用: {gpu_count}个设备")
            print(f"   GPU 0: {gpu_name}")
        else:
            print("⚠️  GPU不可用，将使用CPU训练")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("🧪 PushT强化学习环境设置测试")
    print("=" * 50)
    
    all_passed = True
    
    # 运行所有测试
    all_passed &= test_imports()
    all_passed &= test_env_creation()
    all_passed &= test_model_creation()
    all_passed &= test_gpu_availability()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！环境配置正确。")
        print("\n📚 接下来可以运行:")
        print("   python RL/train_ppo.py")
        print("   python RL/train_sac.py")
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        print("\n🔧 可能需要安装:")
        print("   pip install stable-baselines3[extra] wandb torch")
        print("   cd gym-pusht && pip install -e .")
    print("=" * 50)

if __name__ == "__main__":
    main() 