"""
交互式PushT环境
支持用户键盘控制和策略切换，记录轨迹数据并上传到Hugging Face
"""

import pygame
import gymnasium as gym
import gym_pusht
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, List
import logging

from utils import (
    KeyboardController, MouseController, RandomPolicy, DataManager, HuggingFaceUploader,
    TrajectoryStep, Episode, setup_logging
)

class PushTHuman:
    """交互式PushT环境主类"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # 初始化环境 - 使用更高分辨率的观测
        self.env = gym.make(
            "gym_pusht/PushT-v0", 
            obs_type=cfg.env.obs_type,
            render_mode="rgb_array",
            observation_width=512,   # 提高观测分辨率到512x512
            observation_height=512   # 与物理坐标系一致
        )
        
        # 初始化控制器
        self.input_mode = cfg.control.input_mode
        
        if self.input_mode == "keyboard":
            # 键盘控制模式
            keys_dict = {
                'up': cfg.control.key_mapping.up,
                'down': cfg.control.key_mapping.down,
                'left': cfg.control.key_mapping.left,
                'right': cfg.control.key_mapping.right,
                'toggle_control': cfg.control.key_mapping.toggle_control,
                'quit': cfg.control.key_mapping.quit,
                'reset': cfg.control.key_mapping.reset
            }
            self.keyboard_controller = KeyboardController(
                keys_dict, 
                cfg.control.keyboard_move_speed
            )
            self.mouse_controller = None
            
        elif self.input_mode == "mouse":
            # 鼠标控制模式
            self.mouse_controller = MouseController(
                smoothing=cfg.control.mouse.smoothing,
                click_to_move=cfg.control.mouse.click_to_move
            )
            self.keyboard_controller = None
            
        else:
            raise ValueError(f"不支持的输入模式: {self.input_mode}")
        
        # 特殊按键控制器（用于toggle_control, quit, reset）
        special_keys_dict = {
            'toggle_control': cfg.control.key_mapping.toggle_control,
            'quit': cfg.control.key_mapping.quit,
            'reset': cfg.control.key_mapping.reset
        }
        self.special_keys_controller = KeyboardController(special_keys_dict, 0)
        
        # 初始化策略
        self.random_policy = RandomPolicy(self.env.action_space, cfg.policy.random_seed)
        
        # 初始化数据管理器
        self.data_manager = DataManager(cfg.data.save_dir, cfg.data.save_format)
        
        # 初始化上传器
        self.uploader = HuggingFaceUploader(cfg.upload.hf_token)
        
        # 游戏状态
        self.user_control = cfg.control.user_control
        self.running = True
        self.current_episode = 0
        self.clock = pygame.time.Clock()
        self.fps = cfg.control.fps
        
        # 初始化pygame显示
        pygame.init()
        pygame.display.init()
        # 设置显示窗口大小 - 使用较大的尺寸以便观看
        self.window_size = 512  # 显示窗口大小
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("PushT Human - 初始化中...")
        
        # 当前轨迹
        self.current_trajectory: List[TrajectoryStep] = []
        self.current_obs = None
        self.current_info = None
        
        logging.info("PushT Human 环境初始化完成")
        logging.info("=" * 50)
        logging.info("控制说明:")
        
        if self.input_mode == "keyboard":
            logging.info("  输入模式: 键盘控制")
            logging.info("  WASD: 移动agent")
        elif self.input_mode == "mouse":
            logging.info("  输入模式: 鼠标控制")
            if cfg.control.mouse.click_to_move:
                logging.info("  鼠标: 按住左键并移动鼠标控制agent")
            else:
                logging.info("  鼠标: 移动鼠标悬停位置控制agent")
            logging.info(f"  平滑系数: {cfg.control.mouse.smoothing}")
        
        logging.info("  空格键: 切换用户/AI控制模式")
        logging.info("  R键: 重置环境")
        logging.info("  Q键: 退出游戏")
        logging.info("=" * 50)
    
    def run(self):
        """主游戏循环"""
        logging.info(f"开始游戏，计划进行 {self.cfg.data.num_episodes} 轮")
        
        while self.running and self.current_episode < self.cfg.data.num_episodes:
            self.play_episode()
            self.current_episode += 1
        
        # 游戏结束处理
        self.finish_game()
    
    def play_episode(self):
        """进行一轮游戏"""
        logging.info(f"开始第 {self.current_episode + 1} 轮游戏")
        
        # 重置环境和轨迹
        self.current_obs, self.current_info = self.env.reset()
        self.current_trajectory = []
        
        # 渲染初始状态
        if isinstance(self.current_obs, dict) and 'pixels' in self.current_obs:
            self.render_pixels(self.current_obs['pixels'])
        else:
            self.screen.fill((0, 0, 0))
            pygame.display.flip()
        
        # 保存初始状态信息用于轨迹回放
        initial_state = {
            'agent_pos': np.array(self.current_info['pos_agent']).tolist(),
            'block_pos': self.current_info['block_pose'][:2].tolist(),  # [x, y]
            'block_angle': float(self.current_info['block_pose'][2]),   # angle
            'goal_pose': self.current_info['goal_pose'].tolist()
        }
        
        # 添加用户提示
        if self.user_control:
            logging.info("=== 用户控制模式 ===")
            if self.input_mode == "keyboard":
                logging.info("使用WASD控制agent")
            elif self.input_mode == "mouse":
                logging.info("使用鼠标控制agent")
            logging.info("空格键：切换控制模式，R：重置，Q：退出")
            
            # 倒计时给用户准备时间
            if not self.countdown_start(self.cfg.control.countdown_duration):
                self.running = False
                return
        
        episode_reward = 0.0
        step_count = 0
        
        while True:
            # 处理事件和获取动作
            action = self.handle_events_and_get_action()
            
            if action is None:  # 退出游戏
                self.running = False
                break
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 渲染像素图像
            if isinstance(obs, dict) and 'pixels' in obs:
                self.render_pixels(obs['pixels'])
            else:
                # 如果不是pixels_agent_pos类型，显示黑屏或错误信息
                self.screen.fill((0, 0, 0))
                pygame.display.flip()
            # 记录轨迹 - 记录执行动作后的观测
            step = TrajectoryStep(
                observation=obs.copy() if isinstance(obs, dict) else obs.copy(),
                action=action.copy(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info.copy(),
                is_human_action=self.user_control  # 记录是否为人类控制
            )
            self.current_trajectory.append(step)
            
            # 更新状态
            self.current_obs = obs
            self.current_info = info
            episode_reward += reward
            step_count += 1
            
            # 显示状态信息
            self.display_status(step_count, episode_reward, info)
            
            # 控制帧率
            self.clock.tick(self.fps)
            
            # 检查结束条件
            if terminated or truncated or step_count >= self.cfg.env.max_episode_steps:
                success = info.get('is_success', False)
                logging.info(f"第 {self.current_episode + 1} 轮结束: 步数={step_count}, 奖励={episode_reward:.4f}, 成功={success}")
                
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
    
    def handle_events_and_get_action(self) -> Optional[np.ndarray]:
        """处理事件并获取动作"""
        events = pygame.event.get()
        
        # 处理退出事件
        for event in events:
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                logging.debug(f"检测到按键: {key_name}")
        
        # 处理特殊按键（toggle_control, quit, reset）
        special_actions = self.special_keys_controller.update_pressed_keys(events)
        
        if 'quit' in special_actions:
            logging.info("退出游戏")
            return None
        
        if 'reset' in special_actions:
            logging.info("重置环境")
            self.current_obs, self.current_info = self.env.reset()
            self.current_trajectory = []
            
            # 渲染重置后的状态
            if isinstance(self.current_obs, dict) and 'pixels' in self.current_obs:
                self.render_pixels(self.current_obs['pixels'])
            else:
                self.screen.fill((0, 0, 0))
                pygame.display.flip()
            
            # 重置后给用户准备时间
            if self.user_control:
                if not self.countdown_start(self.cfg.control.countdown_duration):
                    return None  # 用户在倒计时期间退出
        
        if 'toggle_control' in special_actions:
            self.user_control = not self.user_control
            mode = "用户控制" if self.user_control else "策略控制"
            logging.info(f"切换到: {mode}")
        
        # 获取动作
        if self.user_control:
            # 用户控制模式
            if self.input_mode == "keyboard":
                # 键盘控制
                key_actions = self.keyboard_controller.update_pressed_keys(events)
                agent_pos = self.get_agent_position()
                action = self.keyboard_controller.get_movement_action(agent_pos)
                
                if action is not None and not np.array_equal(action, agent_pos):
                    logging.debug(f"键盘移动: 从 {agent_pos} 到 {action}")
                
                if action is None:
                    action = agent_pos  # 无按键输入时，保持当前位置
                    
            elif self.input_mode == "mouse":
                # 鼠标控制
                self.mouse_controller.update_mouse_state(events)
                action = self.mouse_controller.get_mouse_action()
                
                if action is not None:
                    agent_pos = self.get_agent_position()
                    distance = np.linalg.norm(action - agent_pos)
                    if distance > 1.0:  # 只在移动距离较大时打印日志
                        logging.debug(f"鼠标移动: 目标 {action}, 当前 {agent_pos}, 距离 {distance:.1f}")
                
                if action is None:
                    action = self.get_agent_position()  # 保持当前位置
        else:
            # 策略控制模式
            action = self.random_policy.get_action(self.current_obs)
        
        return action
    
    def get_agent_position(self) -> np.ndarray:
        """获取智能体当前位置"""
        if self.cfg.env.obs_type == "state":
            return self.current_obs[:2]  # [agent_x, agent_y]
        elif self.cfg.env.obs_type == "environment_state_agent_pos":
            return self.current_obs["agent_pos"]
        elif self.cfg.env.obs_type == "pixels_agent_pos":
            return self.current_obs["agent_pos"]
        else:
            # 对于纯像素观测，从info中获取
            return np.array(self.current_info["pos_agent"])
    
    def render_pixels(self, pixels: np.ndarray):
        """渲染pixels图像到pygame窗口"""
        if pixels is not None and len(pixels.shape) == 3:
            # 检查图像尺寸
            height, width = pixels.shape[:2]
            
            # 转换为pygame surface
            surface = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))  # pygame需要转置
            
            # 如果图像尺寸与窗口尺寸一致，直接显示；否则缩放
            if width == self.window_size and height == self.window_size:
                # 直接显示，无需缩放（清晰）
                self.screen.blit(surface, (0, 0))
            else:
                # 需要缩放（适用于96x96等低分辨率）
                scaled_surface = pygame.transform.scale(surface, (self.window_size, self.window_size))
                self.screen.blit(scaled_surface, (0, 0))
            
            pygame.display.flip()
        else:
            # 如果没有pixels数据，显示黑屏
            self.screen.fill((0, 0, 0))
            pygame.display.flip()

    def display_status(self, step_count: int, episode_reward: float, info: dict):
        """显示状态信息在窗口标题"""
        mode = "用户控制" if self.user_control else "策略控制"
        coverage = info.get('coverage', 0.0)
        success = info.get('is_success', False)
        
        title = (f"PushT Human - 第{self.current_episode + 1}轮 - 步数:{step_count} - "
                f"奖励:{episode_reward:.3f} - 覆盖:{coverage:.3f} - 模式:{mode} - "
                f"成功:{success} | WASD:移动 空格:切换模式 R:重置 Q:退出")
        
        pygame.display.set_caption(title)
    
    def finish_game(self):
        """游戏结束处理"""
        logging.info("游戏结束，处理数据...")
        
        # 显示统计信息
        stats = self.data_manager.get_statistics()
        logging.info(f"游戏统计: {stats}")
        
        # 保存数据
        data_path = self.data_manager.save_data(self.cfg.data.dataset_name)
        
        # 自动上传
        if self.cfg.upload.auto_upload:
            try:
                url = self.uploader.upload_dataset(
                    data_path, 
                    self.cfg.upload.repo_id, 
                    self.cfg.upload.private
                )
                logging.info(f"数据已自动上传: {url}")
            except Exception as e:
                logging.error(f"自动上传失败: {e}")
        
        # 关闭环境
        self.env.close()
        pygame.quit()
        
        logging.info("程序结束")

    def countdown_start(self, duration: int = 3):
        """倒计时开始，给用户反应时间"""
        if self.input_mode == "mouse":
            agent_pos = self.get_agent_position()
            logging.info(f"=== 倒计时 {duration} 秒 ===")
            logging.info(f"Agent当前位置: ({agent_pos[0]:.0f}, {agent_pos[1]:.0f})")
            logging.info("请将鼠标移动到agent位置准备开始...")
        else:
            logging.info(f"=== 倒计时 {duration} 秒 ===")
            logging.info("准备开始游戏...")
        
        for i in range(duration, 0, -1):
            # 渲染当前状态，让用户看到agent位置
            if isinstance(self.current_obs, dict) and 'pixels' in self.current_obs:
                self.render_pixels(self.current_obs['pixels'])
            else:
                # 如果没有pixels数据，显示黑屏
                self.screen.fill((0, 0, 0))
                pygame.display.flip()
            
            # 更新窗口标题显示倒计时
            title = f"PushT Human - 倒计时 {i} 秒 - 准备开始"
            pygame.display.set_caption(title)
            
            logging.info(f"开始倒计时: {i}...")
            
            # 等待1秒，但保持响应
            start_time = pygame.time.get_ticks()
            while pygame.time.get_ticks() - start_time < 1000:
                # 处理退出事件，允许在倒计时期间退出
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        key_name = pygame.key.name(event.key)
                        if key_name == 'q':
                            logging.info("用户在倒计时期间退出")
                            return False
                
                # 继续渲染
                self.env.render()
                self.clock.tick(30)  # 保持流畅的渲染
        
        logging.info("倒计时结束，开始游戏！")
        return True

def manual_upload_dataset(cfg: DictConfig):
    """手动上传数据集"""
    uploader = HuggingFaceUploader(cfg.upload.hf_token)
    
    # 查找最新的数据文件
    from pathlib import Path
    data_dir = Path(cfg.data.save_dir)
    
    if not data_dir.exists():
        logging.error(f"数据目录不存在: {data_dir}")
        return
    
    # 查找pickle文件
    pickle_files = list(data_dir.glob("*.pickle"))
    if not pickle_files:
        logging.error("未找到pickle格式的数据文件")
        return
    
    # 使用最新的文件
    latest_file = max(pickle_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"准备上传文件: {latest_file}")
    
    try:
        url = uploader.upload_dataset(
            str(latest_file),
            cfg.upload.repo_id,
            cfg.upload.private
        )
        logging.info(f"数据上传成功: {url}")
    except Exception as e:
        logging.error(f"上传失败: {e}")

@hydra.main(version_base=None, config_path="configs", config_name="pusht_human")
def main(cfg: DictConfig):
    """主函数"""
    setup_logging()
    
    logging.info("启动 PushT Human 交互环境")
    logging.info(f"配置: {cfg}")
    
    # 检查是否只是上传数据
    if cfg.get('upload_only', False):
        manual_upload_dataset(cfg)
        return
    
    # 创建并运行游戏
    game = PushTHuman(cfg)
    
    try:
        game.run()
    except KeyboardInterrupt:
        logging.info("用户中断游戏")
        game.env.close()
        pygame.quit()
    except Exception as e:
        logging.error(f"游戏运行错误: {e}")
        game.env.close()
        pygame.quit()
        raise

if __name__ == "__main__":
    main() 