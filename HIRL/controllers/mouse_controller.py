"""
鼠标控制器模块
处理用户鼠标输入转换为环境动作
"""

import pygame
import numpy as np
from typing import List, Optional, Dict
import logging


class MouseController:
    """鼠标控制器"""
    
    def __init__(self, smoothing: float = 0.5, click_to_move: bool = False):
        """
        初始化鼠标控制器
        
        Args:
            smoothing: 鼠标移动平滑系数 (0-1)
            click_to_move: 是否需要按住鼠标才移动
        """
        self.smoothing = smoothing
        self.click_to_move = click_to_move
        self.last_mouse_pos = None
        self.current_target = None
        self.mouse_pressed = False
        
    def process_events(self, events: List[pygame.event.Event]) -> Dict[str, bool]:
        """
        处理pygame鼠标事件
        
        Args:
            events: pygame事件列表
            
        Returns:
            特殊按键动作字典
        """
        actions = {}
        
        pygame.event.pump()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self.mouse_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键
                    self.mouse_pressed = False
            elif event.type == pygame.WINDOWFOCUSGAINED:
                logging.info("✓ 窗口获得焦点")
            elif event.type == pygame.WINDOWFOCUSLOST:
                logging.info("⚠ 窗口失去焦点 - 请点击游戏窗口")
        
        return actions
    
    def get_mouse_action(self) -> Optional[np.ndarray]:
        """
        根据鼠标位置获取目标动作
        
        Returns:
            目标位置坐标，如果无效返回None
        """
        mouse_pos = pygame.mouse.get_pos()
        
        # 检查是否需要点击才移动
        if self.click_to_move and not self.mouse_pressed:
            return None
        
        # 将pygame坐标转换为环境坐标
        target_pos = np.array(mouse_pos, dtype=np.float32)
        
        # 限制在环境边界内
        target_pos = np.clip(target_pos, [0, 0], [512, 512])
        
        # 应用平滑
        if self.smoothing > 0 and self.current_target is not None:
            # 使用指数移动平均进行平滑
            self.current_target = (
                self.smoothing * self.current_target + 
                (1 - self.smoothing) * target_pos
            )
        else:
            self.current_target = target_pos
        
        self.last_mouse_pos = mouse_pos
        return self.current_target.copy() 