"""
Resource Collector - 资源收集功能

在主村庄状态下自动收集资源（点击金矿、圣水收集器、暗黑重油钻井）。
可合并为宏动作，依次点击所有资源建筑。
"""

import time
import random
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class ResourceBuilding:
    """资源建筑信息"""
    name: str
    template_name: str
    collect_template: Optional[str] = None  # 收集气泡模板
    position: Optional[Tuple[int, int]] = None  # 已知位置


class ResourceCollector:
    """
    资源收集器
    
    自动检测并点击资源建筑上的收集气泡。
    支持金矿、圣水收集器、暗黑重油钻井。
    """
    
    # 默认资源建筑配置
    DEFAULT_BUILDINGS = [
        ResourceBuilding("金矿", "gold_mine", "gold_collect_bubble"),
        ResourceBuilding("圣水收集器", "elixir_collector", "elixir_collect_bubble"),
        ResourceBuilding("暗黑重油钻井", "dark_elixir_drill", "dark_collect_bubble"),
    ]
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        click_callback=None
    ):
        """
        初始化资源收集器
        
        Args:
            state_machine: 状态机实例
            template_matcher: 模板匹配器
            click_callback: 点击回调函数 (x, y) -> bool
        """
        self.sm = state_machine
        self.matcher = template_matcher
        self.click = click_callback or self._default_click
        
        # 资源建筑列表
        self.buildings: List[ResourceBuilding] = []
        
        # 已发现建筑的缓存位置
        self._building_positions: Dict[str, Tuple[int, int]] = {}
        
        # 统计信息
        self._collect_count = 0
        self._last_collect_time = 0.0
        
        # 配置
        self.collect_interval = 30.0  # 最小收集间隔（秒）
        self.click_delay = (0.3, 0.8)  # 点击间隔范围（秒）
        
        logger.info("资源收集器初始化完成")
    
    def add_building(self, building: ResourceBuilding) -> None:
        """添加资源建筑"""
        self.buildings.append(building)
        logger.debug(f"添加资源建筑: {building.name}")
    
    def setup_default_buildings(self) -> None:
        """设置默认资源建筑"""
        for building in self.DEFAULT_BUILDINGS:
            self.buildings.append(building)
        logger.info(f"已设置 {len(self.DEFAULT_BUILDINGS)} 个默认资源建筑")
    
    def collect(self, frame: np.ndarray) -> int:
        """
        执行资源收集
        
        在帧中检测所有资源建筑的收集气泡并点击。
        
        Args:
            frame: 当前帧
            
        Returns:
            成功收集的资源建筑数量
        """
        # 检查状态
        if self.sm.current_state != GameState.MAIN_VILLAGE:
            logger.warning(f"当前状态不是主村庄: {self.sm.current_state.name}")
            return 0
        
        # 检查间隔
        if time.time() - self._last_collect_time < self.collect_interval:
            logger.debug("收集间隔未到")
            return 0
        
        collected = 0
        
        # 检测并点击收集气泡
        for building in self.buildings:
            if building.collect_template:
                result = self.matcher.match(
                    frame, 
                    building.collect_template,
                    threshold=0.7
                )
                
                if result.found:
                    # 点击收集气泡
                    center = result.center
                    if self.click(center[0], center[1]):
                        collected += 1
                        self._collect_count += 1
                        logger.info(f"收集资源: {building.name} at {center}")
                        
                        # 随机延迟
                        delay = random.uniform(*self.click_delay)
                        time.sleep(delay)
        
        if collected > 0:
            self._last_collect_time = time.time()
            logger.info(f"本次收集了 {collected} 个资源建筑")
        
        return collected
    
    def collect_at_known_positions(self) -> int:
        """
        在已知位置收集资源
        
        使用缓存的建筑位置直接点击。
        
        Returns:
            成功收集的数量
        """
        if self.sm.current_state != GameState.MAIN_VILLAGE:
            return 0
        
        collected = 0
        
        for building in self.buildings:
            if building.name in self._building_positions:
                pos = self._building_positions[building.name]
                if self.click(pos[0], pos[1]):
                    collected += 1
                    self._collect_count += 1
                    logger.debug(f"点击已知位置: {building.name} at {pos}")
                    
                    delay = random.uniform(*self.click_delay)
                    time.sleep(delay)
        
        return collected
    
    def scan_buildings(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        扫描并记录资源建筑位置
        
        Args:
            frame: 当前帧
            
        Returns:
            [(建筑名, 位置), ...] 列表
        """
        found = []
        
        for building in self.buildings:
            result = self.matcher.match(
                frame,
                building.template_name,
                threshold=0.7
            )
            
            if result.found:
                pos = result.center
                self._building_positions[building.name] = pos
                found.append((building.name, pos))
                logger.debug(f"发现资源建筑: {building.name} at {pos}")
        
        return found
    
    def _default_click(self, x: int, y: int) -> bool:
        """默认点击实现（仅日志）"""
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def collect_count(self) -> int:
        """总收集次数"""
        return self._collect_count
    
    @property
    def known_buildings(self) -> Dict[str, Tuple[int, int]]:
        """已知建筑位置"""
        return self._building_positions.copy()
    
    def clear_cache(self) -> None:
        """清除建筑位置缓存"""
        self._building_positions.clear()
    
    def __repr__(self) -> str:
        return (
            f"ResourceCollector(buildings={len(self.buildings)}, "
            f"collected={self._collect_count})"
        )
