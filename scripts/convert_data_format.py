#!/usr/bin/env python3
"""
数据格式转换脚本
将旧的pickle格式轨迹数据转换为新的纯数据格式
"""

import argparse
import logging
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager


def convert_data(input_file: str, output_format: str, output_dir: str = None):
    """
    转换数据格式
    
    Args:
        input_file: 输入文件路径
        output_format: 输出格式 (hdf5, json, csv, npz)
        output_dir: 输出目录（可选）
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        logging.error(f"输入文件不存在: {input_file}")
        return False
    
    # 设置输出目录
    if output_dir is None:
        output_dir = input_path.parent / "converted"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建数据管理器进行转换
        logging.info(f"开始转换 {input_file} 到 {output_format} 格式...")
        
        # 加载原始数据（支持pickle格式）
        old_manager = DataManager(save_dir=str(input_path.parent), save_format="pickle")
        
        if input_path.suffix == '.pkl':
            # 加载pickle数据
            import pickle
            with open(input_path, 'rb') as f:
                episodes = pickle.load(f)
            
            # 创建新的数据管理器
            new_manager = DataManager(save_dir=str(output_dir), save_format=output_format)
            
            # 添加所有回合数据
            for episode in episodes:
                new_manager.add_episode(episode)
            
            # 保存为新格式
            output_filename = f"{input_path.stem}_converted"
            output_path = new_manager.save_data(output_filename)
            
            logging.info(f"转换完成! 输出文件: {output_path}")
            logging.info(f"转换了 {len(episodes)} 个回合的数据")
            
            # 显示统计信息
            stats = new_manager.get_statistics()
            logging.info(f"统计信息: {stats}")
            
        else:
            # 已经是新格式，尝试直接加载和转换
            data = old_manager.load_data(str(input_path))
            logging.info(f"加载了 {len(data)} 个回合的纯数据")
            
            # 创建新格式管理器并保存
            new_manager = DataManager(save_dir=str(output_dir), save_format=output_format)
            # 注意：这里需要将纯数据转换回Episode对象
            logging.warning("输入文件已经是纯数据格式，请手动处理转换")
            
        return True
        
    except Exception as e:
        logging.error(f"转换失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="转换轨迹数据格式")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("--format", "-f", 
                       choices=["hdf5", "json", "csv", "npz"], 
                       default="hdf5",
                       help="输出格式 (默认: hdf5)")
    parser.add_argument("--output-dir", "-o", 
                       help="输出目录 (默认: 输入文件同目录下的converted文件夹)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 执行转换
    success = convert_data(args.input_file, args.format, args.output_dir)
    
    if success:
        print(f"✅ 数据转换成功！格式: {args.format}")
    else:
        print("❌ 数据转换失败！")
        sys.exit(1)


if __name__ == "__main__":
    main() 