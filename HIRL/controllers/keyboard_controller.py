"""
键盘控制器模块
处理用户键盘输入转换为环境动作
"""

import pygame
import numpy as np
from typing import Dict, List, Optional
import logging


class KeyboardController:
    """键盘控制器"""
    
    def __init__(self, key_mapping: Dict[str, str], move_speed: float = 10.0):
        """
        初始化键盘控制器
        
        Args:
            key_mapping: 按键映射字典 {'action': 'key'}
            move_speed: 移动速度
        """
        self.key_mapping = {v: k for k, v in key_mapping.items()}  # 反向映射
        self.move_speed = move_speed
        self._setup_window_focus()
    
    def _setup_window_focus(self):
        """设置窗口焦点优化"""
        import os
        if os.name == 'nt':  # Windows
            try:
                import win32gui
                import win32con
                hwnd = pygame.display.get_wm_info()["window"]
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                logging.info("✓ 窗口已获得键盘焦点")
            except ImportError:
                logging.info("⚠ 未安装pywin32，请手动点击窗口获得焦点")
            except Exception as e:
                logging.debug(f"⚠ 窗口焦点设置失败: {e}")
    
    def process_events(self, events: List[pygame.event.Event]) -> Dict[str, bool]:
        """
        处理pygame事件，返回触发的动作
        
        Args:
            events: pygame事件列表
            
        Returns:
            动作字典 {'action': True/False}
        """
        actions = {}
        
        pygame.event.pump()
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                if key_name in self.key_mapping:
                    actions[self.key_mapping[key_name]] = True
                    logging.debug(f"按键触发: {key_name} -> {self.key_mapping[key_name]}")
            elif event.type == pygame.WINDOWFOCUSGAINED:
                logging.info("✓ 窗口获得焦点")
            elif event.type == pygame.WINDOWFOCUSLOST:
                logging.info("⚠ 窗口失去焦点 - 请点击游戏窗口")
        
        return actions
    
    def get_movement_action(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        根据当前按键状态计算移动动作
        
        Args:
            current_pos: 当前位置
            
        Returns:
            新位置坐标，如果没有移动返回None
        """
        keys = pygame.key.get_pressed()
        
        dx, dy = 0, 0
        
        # 检查移动键
        # 简化实现：检查配置文件中的标准按键
        if keys[pygame.K_w]:  # 上 (默认配置)
            dy -= self.move_speed
        if keys[pygame.K_s]:  # 下  
            dy += self.move_speed
        if keys[pygame.K_a]:  # 左
            dx -= self.move_speed
        if keys[pygame.K_d]:  # 右
            dx += self.move_speed
            
        if dx == 0 and dy == 0:
            return None
            
        new_pos = current_pos + np.array([dx, dy])
        # 限制在环境边界内
        new_pos = np.clip(new_pos, [0, 0], [512, 512])
        
        return new_pos.astype(np.float32) 