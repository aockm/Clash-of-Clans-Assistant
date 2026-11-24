"""
Session Controller - 会话控制模块

控制每次运行的时长、攻击次数，避免过于规律的行为模式。
"""

import time
import random
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """会话配置"""
    max_duration: float = 3600.0  # 最大运行时长（秒）
    min_duration: float = 1800.0  # 最小运行时长（秒）
    max_attacks: int = 12         # 最大攻击次数
    min_attacks: int = 8          # 最小攻击次数
    break_probability: float = 0.1  # 每次攻击后休息概率
    break_duration: tuple = (30.0, 120.0)  # 休息时长范围（秒）


class SessionController:
    """
    会话控制器
    
    管理运行时长、攻击次数、休息时间。
    """
    
    def __init__(self, config: Optional[SessionConfig] = None):
        """
        初始化会话控制器
        
        Args:
            config: 会话配置，None 使用默认配置
        """
        self.config = config or SessionConfig()
        
        # 随机化本次会话的目标
        self.target_duration = random.uniform(
            self.config.min_duration,
            self.config.max_duration
        )
        self.target_attacks = random.randint(
            self.config.min_attacks,
            self.config.max_attacks
        )
        
        # 运行状态
        self._start_time = time.time()
        self._attack_count = 0
        self._is_running = True
        self._last_break_time = 0.0
        
        logger.info(
            f"会话控制器初始化: 目标时长={self.target_duration:.0f}s, "
            f"目标攻击={self.target_attacks}次"
        )
    
    @property
    def elapsed_time(self) -> float:
        """已运行时长（秒）"""
        return time.time() - self._start_time
    
    @property
    def remaining_time(self) -> float:
        """剩余时长（秒）"""
        return max(0, self.target_duration - self.elapsed_time)
    
    @property
    def attack_count(self) -> int:
        """已攻击次数"""
        return self._attack_count
    
    @property
    def remaining_attacks(self) -> int:
        """剩余攻击次数"""
        return max(0, self.target_attacks - self._attack_count)
    
    def should_continue(self) -> bool:
        """
        检查是否应该继续运行
        
        Returns:
            应该继续返回 True
        """
        if not self._is_running:
            return False
        
        # 检查时长
        if self.elapsed_time >= self.target_duration:
            logger.info(f"达到目标时长: {self.target_duration:.0f}s")
            return False
        
        # 检查攻击次数
        if self._attack_count >= self.target_attacks:
            logger.info(f"达到目标攻击次数: {self.target_attacks}")
            return False
        
        return True
    
    def record_attack(self) -> None:
        """记录一次攻击"""
        self._attack_count += 1
        logger.debug(f"攻击计数: {self._attack_count}/{self.target_attacks}")
    
    def should_take_break(self) -> bool:
        """
        检查是否应该休息
        
        Returns:
            应该休息返回 True
        """
        # 检查休息概率
        if random.random() < self.config.break_probability:
            # 检查距离上次休息是否足够
            if time.time() - self._last_break_time > 300:  # 至少 5 分钟间隔
                return True
        return False
    
    def take_break(self) -> None:
        """执行休息"""
        break_duration = random.uniform(*self.config.break_duration)
        logger.info(f"开始休息: {break_duration:.0f}s")
        
        self._last_break_time = time.time()
        time.sleep(break_duration)
        
        logger.info("休息结束")
    
    def stop(self) -> None:
        """停止会话"""
        self._is_running = False
        logger.info("会话已停止")
    
    def get_stats(self) -> dict:
        """
        获取会话统计
        
        Returns:
            统计信息字典
        """
        return {
            "elapsed_time": self.elapsed_time,
            "target_duration": self.target_duration,
            "attack_count": self._attack_count,
            "target_attacks": self.target_attacks,
            "is_running": self._is_running,
        }
    
    def reset(self) -> None:
        """重置会话"""
        self._start_time = time.time()
        self._attack_count = 0
        self._is_running = True
        self._last_break_time = 0.0
        
        # 重新随机化目标
        self.target_duration = random.uniform(
            self.config.min_duration,
            self.config.max_duration
        )
        self.target_attacks = random.randint(
            self.config.min_attacks,
            self.config.max_attacks
        )
        
        logger.info("会话已重置")
    
    def __repr__(self) -> str:
        return (
            f"SessionController(elapsed={self.elapsed_time:.0f}s, "
            f"attacks={self._attack_count}/{self.target_attacks})"
        )
