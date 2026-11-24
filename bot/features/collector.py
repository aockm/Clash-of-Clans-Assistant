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

import cv2
import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher, MatchResult

logger = logging.getLogger(__name__)

# 收集气泡球体 HSV 颜色范围（视频尺度实测，与 YOLO 标注一致）
BUBBLE_HSV_RANGES = {
    "金矿": ((18, 120, 150), (35, 255, 255)),      # 亮黄球
    "圣水收集器": ((140, 120, 120), (170, 255, 255)),  # 品红球
    "暗黑重油钻井": ((125, 80, 40), (145, 255, 130)),   # 深紫球
}
# 村庄扫描区域（避开顶部/右侧 HUD 资源栏），视频尺度 800x360
BUBBLE_SCAN_ROI = (150, 55, 600, 300)  # x0, y0, x1, y1


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
    
    def detect_bubbles(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        基于 HSV 颜色 + 形状检测收集气泡
        
        气泡球体为极高饱和的纯色圆斑（金币黄/圣水晶红/黑油深紫），
        圆形度过滤可排除草地、金矿堆等误检。
        
        Args:
            frame: 当前帧（视频尺度）
            
        Returns:
            [(建筑名, (x, y)), ...] 气泡中心位置列表
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        x0, y0, x1, y1 = BUBBLE_SCAN_ROI
        x1, y1 = min(x1, w), min(y1, h)
        
        found: List[Tuple[str, Tuple[int, int]]] = []
        for name, (lo, hi) in BUBBLE_HSV_RANGES.items():
            mask = cv2.inRange(hsv, lo, hi)
            area_mask = np.zeros((h, w), np.uint8)
            area_mask[y0:y1, x0:x1] = 255
            mask = cv2.bitwise_and(mask, mask, mask=area_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 40:
                    continue
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if bh == 0 or not (0.5 <= bw / bh <= 2.0):
                    continue
                blob = (labels == i).astype(np.uint8)
                cnt = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                perim = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perim ** 2 + 1e-6)
                if circularity < 0.7:
                    continue
                # 球体核心应有大量极饱和像素（草地/土路没有）
                if (hsv[:, :, 1][labels == i] >= 240).mean() < 0.05:
                    continue
                cx, cy = int(cents[i][0]), int(cents[i][1])
                found.append((name, (cx, cy)))
                logger.debug(f"发现收集气泡: {name} at ({cx},{cy}) area={area}")
        
        return found
    
    def collect(self, frame: np.ndarray) -> int:
        """
        执行资源收集
        
        在帧中检测所有收集气泡并点击。
        
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
        
        bubbles = self.detect_bubbles(frame)
        collected = 0
        
        for name, (cx, cy) in bubbles:
            if self.click(cx, cy):
                collected += 1
                self._collect_count += 1
                self._building_positions[name] = (cx, cy)
                logger.info(f"收集资源: {name} at ({cx},{cy})")
                
                # 随机延迟
                delay = random.uniform(*self.click_delay)
                time.sleep(delay)
        
        if collected > 0:
            self._last_collect_time = time.time()
            logger.info(f"本次收集了 {collected} 个资源气泡")
        
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
