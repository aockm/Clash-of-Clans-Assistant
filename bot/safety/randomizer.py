"""
Randomizer - 随机化模块

实现点击坐标随机化、延迟随机化，模拟人工操作。
"""

import random
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class Randomizer:
    """
    随机化器
    
    提供坐标随机偏移、延迟随机化等功能。
    """
    
    def __init__(
        self,
        click_offset: int = 15,
        delay_range: Tuple[float, float] = (0.3, 0.8),
        long_delay_range: Tuple[float, float] = (1.0, 3.0)
    ):
        """
        初始化随机化器
        
        Args:
            click_offset: 点击坐标随机偏移范围（像素）
            delay_range: 普通延迟范围（秒）
            long_delay_range: 长延迟范围（秒）
        """
        self.click_offset = click_offset
        self.delay_range = delay_range
        self.long_delay_range = long_delay_range
        
        logger.info(
            f"Randomizer 初始化: offset=±{click_offset}px, "
            f"delay={delay_range}"
        )
    
    def randomize_click(self, x: int, y: int) -> Tuple[int, int]:
        """
        随机化点击坐标
        
        Args:
            x: 原始 X 坐标
            y: 原始 Y 坐标
            
        Returns:
            随机化后的坐标 (x, y)
        """
        offset_x = random.randint(-self.click_offset, self.click_offset)
        offset_y = random.randint(-self.click_offset, self.click_offset)
        
        new_x = x + offset_x
        new_y = y + offset_y
        
        logger.debug(f"坐标随机化: ({x}, {y}) -> ({new_x}, {new_y})")
        return (new_x, new_y)
    
    def random_delay(self, short: bool = False) -> None:
        """
        随机延迟
        
        Args:
            short: 是否使用短延迟
        """
        if short:
            delay = random.uniform(0.1, 0.3)
        else:
            delay = random.uniform(*self.delay_range)
        
        logger.debug(f"随机延迟: {delay:.2f}s")
        time.sleep(delay)
    
    def random_long_delay(self) -> None:
        """随机长延迟"""
        delay = random.uniform(*self.long_delay_range)
        logger.debug(f"随机长延迟: {delay:.2f}s")
        time.sleep(delay)
    
    def random_wait(self, min_seconds: float, max_seconds: float) -> None:
        """
        随机等待
        
        Args:
            min_seconds: 最小等待时间
            max_seconds: 最大等待时间
        """
        wait_time = random.uniform(min_seconds, max_seconds)
        time.sleep(wait_time)
    
    def should_skip(self, probability: float = 0.1) -> bool:
        """
        随机决定是否跳过某个操作
        
        Args:
            probability: 跳过概率 (0.0 - 1.0)
            
        Returns:
            是否跳过
        """
        return random.random() < probability
    
    def random_choice(self, choices: list):
        """
        随机选择
        
        Args:
            choices: 选项列表
            
        Returns:
            随机选择的选项
        """
        return random.choice(choices)
    
    def set_click_offset(self, offset: int) -> None:
        """设置点击偏移范围"""
        self.click_offset = offset
        logger.info(f"点击偏移更新为: ±{offset}px")
    
    def set_delay_range(self, delay_range: Tuple[float, float]) -> None:
        """设置延迟范围"""
        self.delay_range = delay_range
        logger.info(f"延迟范围更新为: {delay_range}")
    
    def __repr__(self) -> str:
        return (
            f"Randomizer(offset=±{self.click_offset}, "
            f"delay={self.delay_range})"
        )
