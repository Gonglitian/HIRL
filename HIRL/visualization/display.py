"""
可视化显示模块
处理pygame渲染和状态显示
"""

import pygame
import numpy as np
from typing import Dict, Any
import logging


class GameDisplay:
    """游戏显示管理器"""
    
    def __init__(self, window_size: int = 512):
        """
        初始化显示管理器
        
        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        
        # 初始化pygame显示
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("HIRL - Human-in-the-Loop RL Platform")
        
        # 字体设置
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        logging.info(f"游戏显示初始化完成，窗口大小: {window_size}x{window_size}")
    
    def render_pixels(self, pixels: np.ndarray, update_display: bool = True):
        """
        渲染像素图像
        
        Args:
            pixels: 像素数组 (H, W, 3)
            update_display: 是否立即更新显示
        """
        if len(pixels.shape) == 3 and pixels.shape[2] == 3:
            # 确保像素值在正确范围内
            if pixels.max() <= 1.0:
                pixels = (pixels * 255).astype(np.uint8)
            else:
                pixels = pixels.astype(np.uint8)
            
            # 转换为pygame surface
            surface = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
            
            # 缩放到窗口大小
            if surface.get_size() != (self.window_size, self.window_size):
                surface = pygame.transform.scale(surface, (self.window_size, self.window_size))
            
            self.screen.blit(surface, (0, 0))
        else:
            # 如果不是有效的像素数据，显示黑屏
            self.screen.fill((0, 0, 0))
            self._draw_text("Invalid pixel data", self.window_size // 2, self.window_size // 2, 
                          color=(255, 255, 255), center=True)
        
        if update_display:
            pygame.display.flip()
    
    def render_status(self, step_count: int, episode_reward: float, 
                     info: Dict[str, Any], control_mode: str, update_display: bool = True):
        """
        渲染状态信息覆盖层
        
        Args:
            step_count: 步数
            episode_reward: 累计奖励
            info: 环境信息
            control_mode: 控制模式
            update_display: 是否立即更新显示
        """
        # 创建半透明覆盖层
        overlay = pygame.Surface((self.window_size, 100))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        
        # 绘制状态信息
        y_offset = 10
        status_texts = [
            f"Steps: {step_count}  Reward: {episode_reward:.3f}",
            f"Control: {control_mode}",
            f"Coverage: {info.get('coverage', 0):.3f}  Success: {info.get('is_success', False)}"
        ]
        
        for text in status_texts:
            text_surface = self.font.render(text, True, (255, 255, 255))
            overlay.blit(text_surface, (10, y_offset))
            y_offset += 25
        
        self.screen.blit(overlay, (0, self.window_size - 100))
        
        if update_display:
            pygame.display.flip()
    
    def render_game_state(self, pixels: np.ndarray, step_count: int, episode_reward: float, 
                         info: Dict[str, Any], control_mode: str):
        """
        一次性渲染完整的游戏状态（像素+状态信息）
        避免多次display.flip()调用造成的闪烁
        
        Args:
            pixels: 像素数组 (H, W, 3)
            step_count: 步数
            episode_reward: 累计奖励
            info: 环境信息
            control_mode: 控制模式
        """
        # 先渲染像素数据，但不更新显示
        self.render_pixels(pixels, update_display=False)
        
        # 再渲染状态信息，并一次性更新显示
        self.render_status(step_count, episode_reward, info, control_mode, update_display=True)
    
    def show_countdown(self, seconds: int):
        """
        显示倒计时
        
        Args:
            seconds: 剩余秒数
        """
        self.screen.fill((0, 0, 0))
        
        # 绘制倒计时
        countdown_text = str(seconds) if seconds > 0 else "Start!"
        text_surface = self.large_font.render(countdown_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.window_size // 2, self.window_size // 2))
        self.screen.blit(text_surface, text_rect)
        
        # 绘制提示信息
        if seconds > 0:
            hint_text = "Get ready to play..."
            hint_surface = self.font.render(hint_text, True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(self.window_size // 2, self.window_size // 2 + 50))
            self.screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
    
    def show_message(self, message: str, color=(255, 255, 255), duration_ms: int = 2000):
        """
        显示临时消息
        
        Args:
            message: 消息文本
            color: 文本颜色
            duration_ms: 显示时长(毫秒)
        """
        # 创建覆盖层
        overlay = pygame.Surface((self.window_size, 60))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        
        # 绘制消息
        text_surface = self.font.render(message, True, color)
        text_rect = text_surface.get_rect(center=(self.window_size // 2, 30))
        overlay.blit(text_surface, text_rect)
        
        self.screen.blit(overlay, (0, self.window_size // 2 - 30))
        pygame.display.flip()
        
        # 等待指定时间
        pygame.time.wait(duration_ms)
    
    def _draw_text(self, text: str, x: int, y: int, color=(255, 255, 255), center=False):
        """
        在指定位置绘制文本
        
        Args:
            text: 文本内容
            x, y: 位置坐标
            color: 文本颜色
            center: 是否居中显示
        """
        text_surface = self.font.render(text, True, color)
        if center:
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, (x, y))
    
    def close(self):
        """关闭显示"""
        pygame.quit() 