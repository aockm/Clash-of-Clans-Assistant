"""
Attack Searcher - 攻击搜索功能

搜索对手并选择目标。
流程：MainVillage → AttackMenu → FindMatch → 选择对手。
可结合 OCR 跳过资源过低的对手。
"""

import time
import random
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher
from ..vision.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class EnemyBaseInfo:
    """敌方基地信息"""
    gold: Optional[int] = None
    elixir: Optional[int] = None
    dark_elixir: Optional[int] = None
    town_hall_level: Optional[int] = None
    percent_loot: Optional[int] = None  # 可掠夺资源百分比
    
    @property
    def total_resources(self) -> int:
        return (self.gold or 0) + (self.elixir or 0) + (self.dark_elixir or 0) * 10
    
    def __repr__(self) -> str:
        return (
            f"EnemyBase(gold={self.gold}, elixir={self.elixir}, "
            f"dark={self.dark_elixir}, total={self.total_resources})"
        )


class AttackSearcher:
    """
    攻击搜索器
    
    自动搜索对手，根据资源条件决定是否攻击。
    """
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        ocr_engine: OCREngine,
        click_callback=None
    ):
        """
        初始化攻击搜索器
        
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
        
        # 搜索策略
        self.min_gold = 0           # 最低金币要求
        self.min_elixir = 0         # 最低圣水要求
        self.min_dark = 0           # 最低暗黑重油要求
        self.max_skip_count = 5     # 最大跳过次数后强制攻击
        
        # 统计
        self._search_count = 0
        self._skip_count = 0
        self._attack_count = 0
        
        # 配置
        self.action_delay = (0.5, 1.2)
        self.search_timeout = 30.0  # 搜索超时（秒）
        
        logger.info("攻击搜索器初始化完成")
    
    def set_strategy(
        self,
        min_gold: int = 0,
        min_elixir: int = 0,
        min_dark: int = 0,
        max_skip: int = 5
    ) -> None:
        """
        设置搜索策略
        
        Args:
            min_gold: 最低金币要求
            min_elixir: 最低圣水要求
            min_dark: 最低暗黑重油要求
            max_skip: 最大跳过次数
        """
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.min_dark = min_dark
        self.max_skip_count = max_skip
        logger.info(
            f"搜索策略: gold>={min_gold}, elixir>={min_elixir}, "
            f"dark>={min_dark}, max_skip={max_skip}"
        )
    
    def start_search(self, frame: np.ndarray) -> bool:
        """
        开始搜索对手
        
        点击攻击菜单中的"寻找对手"按钮。
        
        Args:
            frame: 当前帧
            
        Returns:
            成功开始搜索返回 True
        """
        if self.sm.current_state != GameState.MAIN_VILLAGE:
            logger.warning(f"当前状态不是主村庄: {self.sm.current_state.name}")
            return False
        
        # 查找攻击按钮
        attack_btn = self.matcher.match(frame, "attack_button", threshold=0.8)
        if not attack_btn.found:
            logger.warning("未找到攻击按钮")
            return False
        
        # 点击攻击按钮
        if self.click(*attack_btn.center):
            logger.info("点击攻击按钮")
            time.sleep(random.uniform(*self.action_delay))
            
            # 查找"寻找对手"按钮
            find_btn = self.matcher.match(frame, "find_match_button", threshold=0.8)
            if find_btn.found:
                if self.click(*find_btn.center):
                    logger.info("开始搜索对手")
                    self._search_count += 1
                    return True
        
        return False
    
    def analyze_enemy(self, frame: np.ndarray) -> EnemyBaseInfo:
        """
        分析敌方基地资源
        
        Args:
            frame: 当前帧（敌方基地画面）
            
        Returns:
            EnemyBaseInfo 敌方基地信息
        """
        info = EnemyBaseInfo()
        
        # OCR 读取资源数值
        # 金币区域（需要根据实际分辨率调整 ROI）
        gold_result = self.ocr.recognize_digits(
            frame, roi=(50, 10, 200, 40), allow_suffix=True
        )
        if gold_result.success:
            info.gold = gold_result.value
        
        # 圣水区域
        elixir_result = self.ocr.recognize_digits(
            frame, roi=(50, 50, 200, 40), allow_suffix=True
        )
        if elixir_result.success:
            info.elixir = elixir_result
        
        # 暗黑重油区域
        dark_result = self.ocr.recognize_digits(
            frame, roi=(50, 90, 200, 40), allow_suffix=True
        )
        if dark_result.success:
            info.dark_elixir = dark_result
        
        logger.debug(f"敌方基地分析: {info}")
        return info
    
    def should_attack(self, info: EnemyBaseInfo) -> bool:
        """
        判断是否应该攻击该对手
        
        Args:
            info: 敌方基地信息
            
        Returns:
            应该攻击返回 True
        """
        # 检查资源是否满足最低要求
        if self.min_gold > 0 and (info.gold or 0) < self.min_gold:
            return False
        
        if self.min_elixir > 0 and (info.elixir or 0) < self.min_elixir:
            return False
        
        if self.min_dark > 0 and (info.dark_elixir or 0) < self.min_dark:
            return False
        
        return True
    
    def skip_enemy(self, frame: np.ndarray) -> bool:
        """
        跳过当前对手，继续搜索
        
        Args:
            frame: 当前帧
            
        Returns:
            成功跳过返回 True
        """
        # 查找跳过按钮
        skip_btn = self.matcher.match(frame, "skip_button", threshold=0.7)
        if skip_btn.found:
            if self.click(*skip_btn.center):
                self._skip_count += 1
                logger.info(f"跳过对手 (累计跳过: {self._skip_count})")
                time.sleep(random.uniform(*self.action_delay))
                return True
        
        return False
    
    def run_search_cycle(self, frame: np.ndarray) -> Tuple[bool, Optional[EnemyBaseInfo]]:
        """
        执行一次搜索循环
        
        搜索对手 → 分析资源 → 决定是否攻击
        
        Args:
            frame: 当前帧
            
        Returns:
            (是否找到目标, 敌方基地信息)
        """
        # 分析敌方基地
        info = self.analyze_enemy(frame)
        
        # 判断是否攻击
        if self.should_attack(info):
            logger.info(f"找到满意目标: {info}")
            self._attack_count += 1
            self._skip_count = 0  # 重置跳过计数
            return (True, info)
        
        # 检查是否超过最大跳过次数
        if self._skip_count >= self.max_skip_count:
            logger.info(f"已达最大跳过次数 ({self.max_skip_count})，强制攻击")
            self._attack_count += 1
            self._skip_count = 0
            return (True, info)
        
        # 跳过当前对手
        self.skip_enemy(frame)
        return (False, info)
    
    def _default_click(self, x: int, y: int) -> bool:
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def search_count(self) -> int:
        return self._search_count
    
    @property
    def attack_count(self) -> int:
        return self._attack_count
    
    @property
    def skip_count(self) -> int:
        return self._skip_count
    
    def __repr__(self) -> str:
        return (
            f"AttackSearcher(searches={self._search_count}, "
            f"attacks={self._attack_count}, skips={self._skip_count})"
        )
