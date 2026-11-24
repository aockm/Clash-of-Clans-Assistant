"""
Building Upgrader - 建筑升级功能

在主村庄状态下自动升级建筑。
流程：MainVillage → BuilderMenu → 选择建筑 → 确认升级。
需要 OCR 读取工人状态和升级费用。
"""

import time
import random
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher
from ..vision.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


class UpgradePriority(Enum):
    """升级优先级"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    URGENT = auto()


@dataclass
class UpgradeTarget:
    """升级目标"""
    name: str
    building_template: str
    upgrade_button_template: str = "upgrade_button"
    priority: UpgradePriority = UpgradePriority.NORMAL
    max_cost: int = 0  # 最大可接受费用，0 表示不限
    position: Optional[Tuple[int, int]] = None


@dataclass
class BuilderStatus:
    """工人状态"""
    total: int = 0
    busy: int = 0
    
    @property
    def available(self) -> int:
        return self.total - self.busy
    
    def __repr__(self) -> str:
        return f"BuilderStatus(total={self.total}, busy={self.busy}, free={self.available})"


class BuildingUpgrader:
    """
    建筑升级器
    
    自动检测空闲工人并升级建筑。
    支持优先级和费用预算策略。
    """
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        ocr_engine: OCREngine,
        click_callback=None
    ):
        """
        初始化建筑升级器
        
        Args:
            state_machine: 状态机
            template_matcher: 模板匹配器
            ocr_engine: OCR 引擎
            click_callback: 点击回调 (x, y) -> bool
        """
        self.sm = state_machine
        self.matcher = template_matcher
        self.ocr = ocr_engine
        self.click = click_callback or self._default_click
        
        # 升级目标列表
        self.targets: List[UpgradeTarget] = []
        
        # 统计
        self._upgrade_count = 0
        self._total_cost = 0
        
        # 配置
        self.action_delay = (0.5, 1.2)
        self.confirm_delay = (0.3, 0.8)
        
        logger.info("建筑升级器初始化完成")
    
    def add_target(self, target: UpgradeTarget) -> None:
        """添加升级目标"""
        self.targets.append(target)
        # 按优先级排序
        self.targets.sort(key=lambda t: t.priority.value, reverse=True)
        logger.debug(f"添加升级目标: {target.name} (优先级: {target.priority.name})")
    
    def get_builder_status(self, frame: np.ndarray) -> Optional[BuilderStatus]:
        """
        获取工人状态
        
        Args:
            frame: 当前帧
            
        Returns:
            BuilderStatus 或 None
        """
        # 查找工人图标区域
        result = self.matcher.match(frame, "builder_icon", threshold=0.7)
        if not result.found:
            return None
        
        # OCR 读取工人数量（需要配置 ROI）
        x, y = result.center
        roi = (x + 20, y - 10, 60, 30)
        
        ocr_result = self.ocr.recognize_digits(frame, roi=roi)
        if ocr_result.success and ocr_result.value is not None:
            return BuilderStatus(total=ocr_result.value, busy=0)
        
        return None
    
    def get_upgrade_cost(self, frame: np.ndarray) -> Optional[int]:
        """
        获取升级费用
        
        Args:
            frame: 当前帧
            
        Returns:
            升级费用或 None
        """
        result = self.matcher.match(frame, "upgrade_cost_area", threshold=0.6)
        if not result.found:
            return None
        
        x, y, w, h = result.bbox
        roi = (x, y, w, h)
        
        ocr_result = self.ocr.recognize_digits(frame, roi=roi, allow_suffix=True)
        if ocr_result.success and ocr_result.value is not None:
            return ocr_result.value
        
        return None
    
    def try_upgrade(self, frame: np.ndarray) -> bool:
        """
        尝试执行一次升级
        
        流程：检测升级按钮 → 检查费用 → 点击升级 → 确认
        
        Args:
            frame: 当前帧
            
        Returns:
            升级成功返回 True
        """
        if self.sm.current_state != GameState.MAIN_VILLAGE:
            logger.warning(f"当前状态不是主村庄: {self.sm.current_state.name}")
            return False
        
        for target in self.targets:
            # 查找建筑
            building = self.matcher.match(frame, target.building_template, threshold=0.7)
            if not building.found:
                continue
            
            # 查找升级按钮
            upgrade_btn = self.matcher.match(
                frame, target.upgrade_button_template,
                threshold=0.7,
                roi=building.bbox  # 在建筑区域内搜索
            )
            
            if not upgrade_btn.found:
                continue
            
            # 检查费用
            if target.max_cost > 0:
                cost = self.get_upgrade_cost(frame)
                if cost is not None and cost > target.max_cost:
                    logger.info(
                        f"跳过 {target.name}: 费用 {cost} 超出预算 {target.max_cost}"
                    )
                    continue
            
            # 点击升级
            center = upgrade_btn.center
            if self.click(center[0], center[1]):
                logger.info(f"点击升级: {target.name} at {center}")
                time.sleep(random.uniform(*self.action_delay))
                
                self._upgrade_count += 1
                return True
        
        logger.debug("没有可升级的建筑")
        return False
    
    def run_upgrade_cycle(self, frame: np.ndarray) -> int:
        """
        执行一轮升级循环
        
        尝试升级所有可用的目标建筑。
        
        Args:
            frame: 当前帧
            
        Returns:
            成功升级的建筑数量
        """
        upgraded = 0
        
        for target in self.targets:
            if self.try_upgrade(frame):
                upgraded += 1
                time.sleep(random.uniform(*self.action_delay))
        
        if upgraded > 0:
            logger.info(f"本轮升级了 {upgraded} 个建筑")
        
        return upgraded
    
    def _default_click(self, x: int, y: int) -> bool:
        """默认点击"""
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def upgrade_count(self) -> int:
        return self._upgrade_count
    
    def __repr__(self) -> str:
        return (
            f"BuildingUpgrader(targets={len(self.targets)}, "
            f"upgraded={self._upgrade_count})"
        )
