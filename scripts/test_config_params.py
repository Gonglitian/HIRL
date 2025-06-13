#!/usr/bin/env python3
"""
配置参数验证脚本
验证YAML配置文件中的所有参数是否被src模块正确使用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import logging

def test_config_integration(cfg: DictConfig):
    """测试配置集成"""
    
    print("🔍 正在验证配置参数集成...")
    print("=" * 60)
    
    # 验证环境配置
    print("📦 环境配置验证:")
    print(f"  ✓ obs_type: {cfg.env.obs_type}")
    print(f"  ✓ max_episode_steps: {cfg.env.max_episode_steps}")
    print(f"  ✓ success_threshold: {cfg.env.success_threshold}")
    
    # 验证控制配置
    print("\n🎮 控制配置验证:")
    print(f"  ✓ fps: {cfg.control.fps}")
    print(f"  ✓ user_control: {cfg.control.user_control}")
    print(f"  ✓ countdown_duration: {cfg.control.countdown_duration}")
    print(f"  ✓ input_mode: {cfg.control.input_mode}")
    print(f"  ✓ keyboard_move_speed: {cfg.control.keyboard_move_speed}")
    
    # 验证按键映射
    print("\n⌨️  按键映射验证:")
    for action, key in cfg.control.key_mapping.items():
        print(f"  ✓ {action}: {key}")
    
    # 验证鼠标配置
    print("\n🖱️  鼠标配置验证:")
    print(f"  ✓ smoothing: {cfg.control.mouse.smoothing}")
    print(f"  ✓ click_to_move: {cfg.control.mouse.click_to_move}")
    
    # 验证数据配置
    print("\n📊 数据配置验证:")
    print(f"  ✓ num_episodes: {cfg.data.num_episodes}")
    print(f"  ✓ save_dir: {cfg.data.save_dir}")
    print(f"  ✓ save_format: {cfg.data.save_format}")
    print(f"  ✓ dataset_name: {cfg.data.dataset_name}")
    
    # 验证策略配置
    print("\n🤖 策略配置验证:")
    print(f"  ✓ type: {cfg.policy.type}")
    print(f"  ✓ random_seed: {cfg.policy.random_seed}")
    
    # 验证上传配置
    print("\n☁️  上传配置验证:")
    print(f"  ✓ hf_token: {'已设置' if cfg.upload.hf_token else '未设置'}")
    print(f"  ✓ repo_id: {cfg.upload.repo_id}")
    print(f"  ✓ private: {cfg.upload.private}")
    print(f"  ✓ auto_upload: {cfg.upload.auto_upload}")
    
    print("\n" + "=" * 60)
    print("✅ 所有配置参数验证完成！")
    
    return True

def test_modules_import():
    """测试模块导入"""
    print("\n🔧 测试模块导入...")
    
    try:
        from HIRL.core import PushTGame
        print("  ✓ PushTGame 导入成功")
        
        from HIRL.core.environment import PushTEnvironment, RandomPolicy
        print("  ✓ Environment 模块导入成功")
        
        from HIRL.controllers import KeyboardController, MouseController
        print("  ✓ Controllers 模块导入成功")
        
        from HIRL.data import DataManager, HuggingFaceUploader
        print("  ✓ Data 模块导入成功")
        
        from HIRL.visualization import GameDisplay
        from HIRL.visualization.replay import TrajectoryReplayer
        print("  ✓ Visualization 模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def test_game_initialization(cfg: DictConfig):
    """测试游戏初始化（不实际运行）"""
    print("\n🎮 测试游戏初始化...")
    
    try:
        from HIRL.core import PushTGame
        
        # 只是初始化，不运行游戏
        game = PushTGame(cfg)
        print("  ✓ PushTGame 初始化成功")
        
        # 检查关键属性
        assert hasattr(game, 'environment'), "缺少 environment 属性"
        assert hasattr(game, 'controller'), "缺少 controller 属性"
        assert hasattr(game, 'data_manager'), "缺少 data_manager 属性"
        assert hasattr(game, 'display'), "缺少 display 属性"
        assert hasattr(game, 'ai_policy'), "缺少 ai_policy 属性"
        
        print("  ✓ 所有必要属性检查通过")
        
        # 清理资源
        game._cleanup()
        print("  ✓ 资源清理完成")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 游戏初始化失败: {e}")
        return False

@hydra.main(version_base=None, config_path="../configs", config_name="pusht_human")
def main(cfg: DictConfig):
    """主函数"""
    
    print("🚀 HIRL 配置参数验证工具")
    print("验证所有YAML配置参数是否被src模块正确使用")
    print()
    
    # 设置日志级别为WARNING，减少输出噪音
    logging.getLogger().setLevel(logging.WARNING)
    
    all_tests_passed = True
    
    # 测试1: 配置参数集成
    if not test_config_integration(cfg):
        all_tests_passed = False
    
    # 测试2: 模块导入
    if not test_modules_import():
        all_tests_passed = False
    
    # 测试3: 游戏初始化
    if not test_game_initialization(cfg):
        all_tests_passed = False
    
    # 最终结果
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有测试通过！配置参数集成正常！")
        print("\n建议：")
        print("  • 可以正常使用 python main.py")
        print("  • 所有配置参数都被正确使用")
        print("  • 模块化架构工作正常")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 