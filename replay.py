#!/usr/bin/env python3
"""
HIRL轨迹回放工具
支持回放已记录的轨迹数据

用法:
python replay.py
python replay.py data_path=data/pusht_trajectories/trajectories_5episodes.pkl
python replay.py auto_play=false manual_play=true
"""

import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path

# 设置模块搜索路径
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.visualization.replay import TrajectoryReplayer


def setup_logging(level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@hydra.main(version_base=None, config_path="configs", config_name="replay")
def main(cfg: DictConfig):
    """主函数"""
    setup_logging("INFO")
    
    logging.info("启动HIRL轨迹回放系统")
    
    # 处理播放模式
    auto_play = not cfg.get('manual_play', False) and cfg.get('auto_play', True)
    
    # 检查数据文件
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        logging.error(f"数据文件不存在: {data_path}")
        return
    
    logging.info(f"准备回放轨迹: {data_path}")
    logging.info(f"播放模式: {'自动' if auto_play else '手动'}")
    
    # 创建回放器
    env_kwargs = {
        'obs_type': cfg.env.obs_type,
        'render_mode': cfg.env.render_mode
    }
    
    replayer = TrajectoryReplayer(env_id='gym_pusht/PushT-v0', **env_kwargs)
    
    try:
        # 加载轨迹数据
        episodes = replayer.load_episodes(str(data_path))
        
        if not episodes:
            logging.error("没有找到可回放的轨迹数据")
            return
        
        # 显示轨迹信息
        if cfg.get('show_info', True):
            logging.info("=== 轨迹数据信息 ===")
            for i, episode in enumerate(episodes):
                success_str = "成功" if episode.success else "失败"
                logging.info(f"轨迹 {i}: ID={episode.episode_id}, 步数={episode.length}, "
                           f"奖励={episode.total_reward:.3f}, {success_str}")
        
        # 回放轨迹
        if cfg.get('episode_id') is not None:
            # 回放指定轨迹
            target_episodes = [ep for ep in episodes if ep.episode_id == cfg.episode_id]
            if not target_episodes:
                logging.error(f"未找到ID为{cfg.episode_id}的轨迹")
                return
            
            logging.info(f"回放指定轨迹: {cfg.episode_id}")
            replayer.replay_episode(target_episodes[0], auto_play=auto_play, delay=cfg.delay)
        else:
            # 回放所有轨迹
            logging.info("回放所有轨迹")
            replayer.replay_all_episodes(
                episodes, 
                auto_play=auto_play, 
                delay=cfg.delay, 
                inter_episode_delay=cfg.inter_episode_delay
            )
            
    except KeyboardInterrupt:
        logging.info("用户中断回放")
    except Exception as e:
        logging.error(f"回放过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        replayer.close()
        logging.info("回放结束")


if __name__ == "__main__":
    main() 