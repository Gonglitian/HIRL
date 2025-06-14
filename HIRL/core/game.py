"""
主游戏逻辑模块
管理交互式PushT游戏的核心逻辑
"""

import pygame
import numpy as np
from typing import Optional, List
from omegaconf import DictConfig
import logging
import time

from .data_types import TrajectoryStep, Episode
from .environment import PushTEnvironment, RandomPolicy
from ..controllers import KeyboardController, MouseController
from ..data import DataManager, HuggingFaceUploader
from ..visualization import GameDisplay


class PushTGame:
    """交互式PushT游戏主类"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化游戏
        
        Args:
            cfg: 游戏配置
        """
        self.cfg = cfg
        
        # 初始化环境
        self.environment = PushTEnvironment(
            obs_type=cfg.env.obs_type,
            observation_width=512,
            observation_height=512,
            max_episode_steps=cfg.env.max_episode_steps
        )
        
        # 初始化控制器
        self._setup_controllers(cfg)
        
        # 初始化显示
        self.display = GameDisplay(window_size=512)
        
        # 初始化数据管理
        self.data_manager = DataManager(cfg.data.save_dir, cfg.data.save_format)
        self.uploader = HuggingFaceUploader(cfg.upload.hf_token)
        
        # 初始化策略
        self._setup_policy(cfg.policy)
        
        # 游戏状态
        self.user_control = cfg.control.user_control
        self.running = True
        self.current_episode = 0
        self.fps = cfg.control.fps
        self.clock = pygame.time.Clock()
        
        # 当前轨迹数据
        self.current_trajectory: List[TrajectoryStep] = []
        self.current_obs = None
        self.current_info = None
        
        logging.info("HIRL游戏初始化完成")
        self._log_controls()
    
    def _setup_controllers(self, cfg: DictConfig):
        """设置输入控制器"""
        self.input_mode = cfg.control.input_mode
        
        if self.input_mode == "keyboard":
            keys_dict = {
                'up': cfg.control.key_mapping.up,
                'down': cfg.control.key_mapping.down,
                'left': cfg.control.key_mapping.left,
                'right': cfg.control.key_mapping.right,
                'toggle_control': cfg.control.key_mapping.toggle_control,
                'quit': cfg.control.key_mapping.quit,
                'reset': cfg.control.key_mapping.reset
            }
            self.controller = KeyboardController(keys_dict, cfg.control.keyboard_move_speed)
            
        elif self.input_mode == "mouse":
            self.controller = MouseController(
                smoothing=cfg.control.mouse.smoothing,
                click_to_move=cfg.control.mouse.click_to_move
            )
            
        else:
            raise ValueError(f"不支持的输入模式: {self.input_mode}")
        
        # 特殊按键控制器
        special_keys = {
            'toggle_control': cfg.control.key_mapping.toggle_control,
            'quit': cfg.control.key_mapping.quit,
            'reset': cfg.control.key_mapping.reset
        }
        self.special_controller = KeyboardController(special_keys, 0)
    
    def _setup_policy(self, policy_cfg):
        """设置AI策略"""
        if policy_cfg.type == "random":
            self.ai_policy = RandomPolicy(self.environment.action_space, policy_cfg.random_seed)
            logging.info(f"使用随机策略，种子: {policy_cfg.random_seed}")
        elif policy_cfg.type == "trained":
            # 未来可以在这里加载训练好的模型
            logging.warning("训练策略暂未实现，使用随机策略")
            self.ai_policy = RandomPolicy(self.environment.action_space, policy_cfg.random_seed)
        else:
            raise ValueError(f"不支持的策略类型: {policy_cfg.type}")
    
    def _log_controls(self):
        """输出控制说明"""
        logging.info("=" * 50)
        logging.info("控制说明:")
        if self.input_mode == "keyboard":
            logging.info("  WASD: 移动智能体")
        elif self.input_mode == "mouse":
            logging.info("  鼠标: 控制智能体移动")
        logging.info("  空格键: 切换用户/AI控制模式")
        logging.info("  R键: 重置环境")
        logging.info("  Q键: 退出游戏")
        logging.info("=" * 50)
    
    def run(self):
        """主游戏循环"""
        logging.info(f"开始游戏，计划进行 {self.cfg.data.num_episodes} 轮")
        
        try:
            while self.running and self.current_episode < self.cfg.data.num_episodes:
                self._play_episode()
                self.current_episode += 1
            
            self._finish_game()
            
        except KeyboardInterrupt:
            logging.info("用户中断游戏")
            self._save_current_data()
        except Exception as e:
            logging.error(f"游戏运行出错: {e}")
            self._save_current_data()
            raise
        finally:
            self._cleanup()
    
    def _play_episode(self):
        """进行一轮游戏"""
        logging.info(f"开始第 {self.current_episode + 1} 轮游戏")
        
        # 重置环境
        self.current_obs, self.current_info = self.environment.reset()
        self.current_trajectory = []
        
        # 获取初始状态信息
        initial_state = self.environment.get_initial_state_info(self.current_info)
        
        # 初始化状态变量
        episode_reward = 0.0
        step_count = 0
        
        # 渲染初始状态（包含状态信息）
        control_mode = "Human Control" if self.user_control else "AI Control"
        pixels = self._get_current_pixels()
        self.display.render_game_state(pixels, step_count, episode_reward, self.current_info, control_mode)
        
        # 用户准备阶段
        if self.user_control and not self._countdown_start(self.cfg.control.countdown_duration):
            self.running = False
            return
        
        # 游戏主循环
        
        while True:
            # 处理输入并获取动作
            action = self._get_action()
            
            if action is None:  # 退出信号
                logging.info("用户请求退出游戏")
                # 保存当前轨迹（如果有的话）
                if self.current_trajectory:
                    episode = Episode(
                        steps=self.current_trajectory,
                        episode_id=self.current_episode,
                        total_reward=episode_reward,
                        success=False,  # 提前退出视为未成功
                        length=step_count,
                        initial_state=initial_state
                    )
                    self.data_manager.add_episode(episode)
                self.running = False
                break
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.environment.step(action)
            
            # 更新当前状态
            self.current_obs = obs
            self.current_info = info
            
            # 记录轨迹
            step = TrajectoryStep(
                observation=obs.copy() if isinstance(obs, dict) else obs.copy(),
                action=action.copy(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info.copy(),
                is_human_action=self.user_control
            )
            self.current_trajectory.append(step)
            
            episode_reward += reward
            step_count += 1
            
            # 一次性渲染完整的游戏状态（避免闪烁）
            control_mode = "Human Control" if self.user_control else "AI Control"
            pixels = self._get_current_pixels()
            self.display.render_game_state(pixels, step_count, episode_reward, info, control_mode)
            
            # 检查游戏结束条件
            max_steps_reached = step_count >= self.cfg.env.max_episode_steps
            if terminated or truncated or max_steps_reached:
                # 检查成功条件
                coverage = info.get('coverage', 0.0)
                success = info.get('is_success', False) or coverage >= self.cfg.env.success_threshold
                
                if max_steps_reached:
                    logging.info(f"第{self.current_episode + 1}轮达到最大步数限制({self.cfg.env.max_episode_steps})")
                
                logging.info(f"第{self.current_episode + 1}轮结束，步数: {step_count}，"
                           f"奖励: {episode_reward:.3f}，覆盖率: {coverage:.3f}，成功: {success}")
                
                # 保存轨迹
                episode = Episode(
                    steps=self.current_trajectory,
                    episode_id=self.current_episode,
                    total_reward=episode_reward,
                    success=success,
                    length=step_count,
                    initial_state=initial_state
                )
                self.data_manager.add_episode(episode)
                break
            
            # 控制帧率
            self.clock.tick(self.fps)
    
    def _get_action(self) -> Optional[np.ndarray]:
        """获取当前动作"""
        events = pygame.event.get()
        
        # 处理特殊按键
        special_actions = self.special_controller.process_events(events)
        
        # 检查退出
        if special_actions.get('quit'):
            return None
        
        # 检查重置
        if special_actions.get('reset'):
            self.current_obs, self.current_info = self.environment.reset()
            # 重置后只渲染像素，不渲染状态信息（会在主循环中更新）
            self._render_current_state()
            logging.info("环境已重置")
        
        # 检查控制模式切换
        if special_actions.get('toggle_control'):
            self.user_control = not self.user_control
            mode = "Human Control" if self.user_control else "AI Control"
            logging.info(f"Switched to: {mode}")
            self.display.show_message(f"Switched to: {mode}", duration_ms=1000)
        
        # 获取移动动作
        if self.user_control:
            # 用户控制
            if self.input_mode == "keyboard":
                current_pos = self.environment.get_agent_position()
                return self.controller.get_movement_action(current_pos)
            elif self.input_mode == "mouse":
                self.controller.process_events(events)
                return self.controller.get_mouse_action()
        else:
            # AI控制
            return self.ai_policy.get_action(self.current_obs)
    
    def _get_current_pixels(self) -> np.ndarray:
        """获取当前状态的像素数据"""
        if isinstance(self.current_obs, dict) and 'pixels' in self.current_obs:
            return self.current_obs['pixels']
        else:
            # 如果没有像素数据，返回黑屏
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _render_current_state(self):
        """渲染当前状态（仅像素数据）"""
        pixels = self._get_current_pixels()
        self.display.render_pixels(pixels)
    
    def _countdown_start(self, duration: int = 3) -> bool:
        """开始倒计时"""
        for i in range(duration, 0, -1):
            # 检查退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return False
            
            self.display.show_countdown(i)
            time.sleep(1)
        
        self.display.show_countdown(0)
        time.sleep(0.5)
        return True
    
    def _save_current_data(self):
        """保存当前已收集的数据"""
        if not self.data_manager.episodes:
            logging.info("没有数据需要保存")
            return None
            
        try:
            saved_path = self.data_manager.save_data(self.cfg.data.dataset_name)
            
            # 显示统计信息
            stats = self.data_manager.get_statistics()
            logging.info(f"数据已保存: {saved_path}")
            logging.info(f"游戏统计:")
            logging.info(f"  总轮数: {stats['total_episodes']}")
            logging.info(f"  成功率: {stats['success_rate']:.3f}")
            logging.info(f"  平均奖励: {stats['average_reward']:.3f}")
            logging.info(f"  平均步数: {stats['average_length']:.1f}")
            
            # 自动上传
            if self.cfg.upload.auto_upload and self.uploader.token:
                try:
                    url = self.uploader.upload_dataset(
                        saved_path, 
                        self.cfg.upload.repo_id, 
                        self.cfg.upload.private
                    )
                    logging.info(f"数据已上传到: {url}")
                except Exception as e:
                    logging.error(f"上传失败: {e}")
            
            return saved_path
            
        except Exception as e:
            logging.error(f"保存数据失败: {e}")
            return None

    def _finish_game(self):
        """游戏结束处理"""
        logging.info("游戏结束")
        self._save_current_data()
        self.display.show_message("Game Over, Thanks for Playing!", duration_ms=3000)
    
    def _cleanup(self):
        """清理资源"""
        self.environment.close()
        self.display.close()
        logging.info("资源清理完成") 