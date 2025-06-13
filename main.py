#!/usr/bin/env python3
"""
HIRL - 人机交互强化学习平台
Human-in-the-Loop Reinforcement Learning Platform

主程序入口文件
"""

import logging
import hydra
from omegaconf import DictConfig

# 导入HIRL包
from HIRL.core import PushTGame


def setup_logging(level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def manual_upload_dataset(cfg: DictConfig):
    """手动上传数据集"""
    from HIRL.data import HuggingFaceUploader
    
    uploader = HuggingFaceUploader(cfg.upload.hf_token)
    
    if not uploader.token:
        logging.error("未配置Hugging Face token，无法上传")
        return
    
    # 查找数据文件
    data_dir = Path(cfg.data.save_dir)
    if not data_dir.exists():
        logging.error(f"数据目录不存在: {data_dir}")
        return
    
    # 寻找最新的数据文件
    data_files = list(data_dir.glob("*.pkl")) + list(data_dir.glob("*.json"))
    if not data_files:
        logging.error("未找到数据文件")
        return
    
    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    logging.info(f"找到数据文件: {latest_file}")
    
    try:
        url = uploader.upload_dataset(
            str(latest_file),
            cfg.upload.repo_id,
            cfg.upload.private
        )
        logging.info(f"数据已成功上传到: {url}")
    except Exception as e:
        logging.error(f"上传失败: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="pusht_human")
def main(cfg: DictConfig):
    """主函数"""
    setup_logging(cfg.get("log_level", "INFO"))
    
    logging.info("启动HIRL - 人机交互强化学习平台")
    logging.info(f"配置: {cfg}")
    
    # 检查是否只是上传模式
    if cfg.get("upload_only", False):
        manual_upload_dataset(cfg)
        return
    
    # 创建并运行游戏
    try:
        game = PushTGame(cfg)
        game.run()
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        raise


if __name__ == "__main__":
    main() 