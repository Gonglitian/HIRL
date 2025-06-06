#!/usr/bin/env python3
"""
PushT环境PPO训练脚本

使用stable_baselines3训练PPO智能体
支持像素+智能体位置观测，集成WandB日志记录

用法:
python RL/train_ppo.py
python RL/train_ppo.py training.total_timesteps=500000
python RL/train_ppo.py ppo.learning_rate=1e-4 wandb.enabled=false
"""

import os
import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

class WandBCallback(BaseCallback):
    """WandB日志回调"""
    
    def __init__(self, cfg: DictConfig, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.log_freq = cfg.wandb.log_freq
        self.wandb_enabled = cfg.wandb.enabled
        
    def _on_step(self) -> bool:
        # 只在WandB启用且达到记录频率时记录
        if not self.wandb_enabled or self.n_calls % self.log_freq != 0:
            return True
            
        # 记录基本训练信息
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            wandb.log({
                'episode_reward': ep_info['r'],
                'episode_length': ep_info['l'],
                'timesteps': self.num_timesteps,
                'n_calls': self.n_calls
            })
        
        # 记录训练统计信息
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if isinstance(info, dict):
                    # 记录任务特定信息
                    if 'is_success' in info:
                        wandb.log({
                            'success_rate': float(info['is_success']),
                            'coverage': info.get('coverage', 0.0),
                            'timesteps': self.num_timesteps
                        })
        
        return True

def setup_environment(cfg: DictConfig):
    """设置训练和评估环境"""
    
    def make_env():
        env = gym.make(
            cfg.env.name,
            obs_type=cfg.env.obs_type,
            render_mode=cfg.env.render_mode
        )
        env = Monitor(env)
        return env
    
    # 创建训练环境
    if cfg.env.n_envs > 1:
        train_env = make_vec_env(make_env, n_envs=cfg.env.n_envs)
    else:
        train_env = make_env()
    
    # 创建评估环境
    eval_env = make_env()
    
    logging.info(f"环境设置完成: {cfg.env.name}")
    logging.info(f"观测类型: {cfg.env.obs_type}")
    logging.info(f"训练环境数量: {cfg.env.n_envs}")
    
    return train_env, eval_env

def setup_callbacks(cfg: DictConfig, eval_env):
    """设置训练回调"""
    callbacks = []
    
    # 评估回调
    if cfg.callbacks.eval:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=cfg.save.model_dir,
            log_path=cfg.save.log_dir,
            eval_freq=cfg.training.eval_freq,
            n_eval_episodes=cfg.training.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        logging.info(f"评估回调已设置: 每{cfg.training.eval_freq}步评估{cfg.training.n_eval_episodes}轮")
    
    # 检查点回调
    if cfg.callbacks.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=cfg.training.save_freq,
            save_path=cfg.save.model_dir,
            name_prefix="ppo_pusht"
        )
        callbacks.append(checkpoint_callback)
        logging.info(f"检查点回调已设置: 每{cfg.training.save_freq}步保存")
    
    # WandB回调
    if cfg.wandb.enabled and cfg.callbacks.wandb:
        wandb_callback = WandBCallback(cfg)
        callbacks.append(wandb_callback)
        logging.info("WandB回调已添加")
    
    return CallbackList(callbacks) if callbacks else None

def setup_model(cfg: DictConfig, env):
    """设置PPO模型"""
    
    # 构建策略参数
    policy_kwargs = OmegaConf.to_container(cfg.ppo.policy_kwargs, resolve=True)
    
    # 处理activation_fn字符串转换为torch函数
    if 'activation_fn' in policy_kwargs:
        activation_fn_str = policy_kwargs['activation_fn']
        if activation_fn_str == 'tanh':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.Tanh
        elif activation_fn_str == 'relu':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.ReLU
        elif activation_fn_str == 'leaky_relu':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.LeakyReLU
        else:
            # 默认使用Tanh
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.Tanh
    
    # 创建PPO模型
    model = PPO(
        policy=cfg.ppo.policy,
        env=env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        clip_range_vf=cfg.ppo.clip_range_vf,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )
    
    # 设置日志记录器
    if cfg.callbacks.tensorboard:
        logger = configure(cfg.save.log_dir, ["stdout", "tensorboard"])
        model.set_logger(logger)
    
    logging.info("PPO模型已创建")
    logging.info(f"策略类型: {cfg.ppo.policy}")
    logging.info(f"学习率: {cfg.ppo.learning_rate}")
    logging.info(f"设备: {model.device}")
    
    return model

def setup_wandb(cfg: DictConfig):
    """设置WandB日志记录"""
    if not cfg.wandb.enabled:
        return
    
    # 初始化WandB
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        notes=cfg.experiment.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        save_code=cfg.wandb.save_code
    )
    
    logging.info(f"WandB已初始化: 项目={cfg.experiment.project}, 实验={cfg.experiment.name}")

@hydra.main(version_base=None, config_path="../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    """主训练函数"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("开始PPO训练")
    logging.info(f"配置: {OmegaConf.to_yaml(cfg)}")
    
    # 创建保存目录
    os.makedirs(cfg.save.model_dir, exist_ok=True)
    os.makedirs(cfg.save.log_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置WandB
    setup_wandb(cfg)
    
    try:
        # 设置环境
        train_env, eval_env = setup_environment(cfg)
        
        # 设置模型
        model = setup_model(cfg, train_env)
        
        # 设置回调
        callbacks = setup_callbacks(cfg, eval_env)
        
        # 开始训练
        logging.info(f"开始训练，总步数: {cfg.training.total_timesteps}")
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(cfg.save.model_dir, "final_model")
        model.save(final_model_path)
        logging.info(f"最终模型已保存: {final_model_path}")
        
        # 上传模型到WandB
        if cfg.wandb.enabled:
            try:
                wandb.save(f"{final_model_path}.zip")
                logging.info("模型已上传到WandB")
            except (OSError, PermissionError) as e:
                logging.warning(f"WandB模型上传失败（权限问题）: {e}")
                logging.info("训练日志已正常记录到WandB")
        
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        raise
    
    finally:
        # 清理资源
        if hasattr(train_env, 'close'):
            train_env.close()
        if hasattr(eval_env, 'close'):
            eval_env.close()
        
        if cfg.wandb.enabled:
            wandb.finish()
        
        logging.info("PPO训练完成")

if __name__ == "__main__":
    main() 