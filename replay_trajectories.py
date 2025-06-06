#!/usr/bin/env python3
"""
PushT轨迹回放脚本

用法:
python replay_trajectories.py
python replay_trajectories.py data_path=data/pusht_trajectories/trajectories_5episodes.pkl
python replay_trajectories.py auto_play=false manual_play=true
python replay_trajectories.py delay=0.05 inter_episode_delay=0
"""

import logging
import gymnasium as gym
import gym_pusht  # 注册PushT环境
import pygame
from pathlib import Path
import hydra
from omegaconf import DictConfig

from utils import TrajectoryReplayer, setup_logging

@hydra.main(version_base=None, config_path="configs", config_name="replay")
def main(cfg: DictConfig):
    """主函数"""
    setup_logging("INFO")
    
    logging.info("启动 PushT 轨迹回放系统")
    logging.info(f"配置: {cfg}")
    
    # 处理播放模式
    if cfg.manual_play:
        auto_play = False
    else:
        auto_play = cfg.auto_play
    
    # 检查数据文件是否存在
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        logging.error(f"数据文件不存在: {data_path}")
        return
    
    logging.info(f"准备回放轨迹: {data_path}")
    logging.info(f"自动播放: {auto_play}")
    if auto_play:
        logging.info(f"步间延迟: {cfg.delay}秒")
        logging.info(f"轨迹间间隔: {cfg.inter_episode_delay}秒")
    
    # 创建环境配置
    env_kwargs = {
        'obs_type': cfg.env.obs_type,
        'render_mode': cfg.env.render_mode
    }
    
    # 创建回放器
    replayer = TrajectoryReplayer(
        env_class=lambda **kwargs: gym.make('gym_pusht/PushT-v0', **kwargs),
        env_kwargs=env_kwargs
    )
    
    try:
        # 加载轨迹数据
        episodes = replayer.load_episodes(str(data_path))
        
        if not episodes:
            logging.error("没有找到可回放的轨迹数据")
            return
        
        # 显示轨迹信息
        if cfg.show_info:
            logging.info("=== 轨迹数据信息 ===")
            for i, episode in enumerate(episodes):
                success_str = "成功" if episode.success else "失败"
                logging.info(f"轨迹 {i}: ID={episode.episode_id}, 步数={episode.length}, "
                            f"奖励={episode.total_reward:.3f}, {success_str}")
                
                if cfg.show_initial_state:
                    logging.info(f"  初始状态 - Agent: {episode.initial_state['agent_pos']}, "
                                f"Block: {episode.initial_state['block_pos']}, "
                                f"Angle: {episode.initial_state['block_angle']:.3f}")
        
        # 初始化pygame (用于处理事件)
        pygame.init()
        pygame.display.set_mode((512, 512))  # 创建一个临时窗口
        
        # 回放轨迹
        if cfg.episode_id is not None:
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
            replayer.replay_all_episodes(episodes, auto_play=auto_play, delay=cfg.delay, inter_episode_delay=cfg.inter_episode_delay)
            
    except KeyboardInterrupt:
        logging.info("用户中断回放")
    except Exception as e:
        logging.error(f"回放过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        logging.info("回放结束")

if __name__ == "__main__":
    main() 