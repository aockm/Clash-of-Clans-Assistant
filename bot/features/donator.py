"""
Troop Donator - 捐兵功能

检测到部落聊天有捐兵请求时进入捐赠流程。
流程：MainVillage → ClanChat → 捐赠 → 返回。
"""

import time
import random
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..states.state_machine import GameState, StateMachine
from ..vision.template_matcher import TemplateMatcher
from ..vision.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class DonationRequest:
    """捐兵请求"""
    troop_name: str
    troop_template: str
    count: int = 1
    position: Optional[Tuple[int, int]] = None


class TroopDonator:
    """
    捐兵器
    
    自动检测部落聊天中的捐兵请求并捐赠对应兵种。
    处理"请求兵种不匹配"的跳过逻辑。
    """
    
    # 常见兵种配置
    DEFAULT_TROOPS = {
        "barbarian": "野蛮人",
        "archer": "弓箭手",
        "giant": "巨人",
        "goblin": "哥布林",
        "dragon": "飞龙",
        "hog_rider": "野猪骑士",
        "valkyrie": "武神",
        "witch": "女巫",
        "golem": "石头人",
        "lava_hound": "熔岩猎犬",
    }
    
    def __init__(
        self,
        state_machine: StateMachine,
        template_matcher: TemplateMatcher,
        ocr_engine: OCREngine,
        click_callback=None
    ):
        """
        初始化捐兵器
        
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
        
        # 待捐赠请求
        self.pending_requests: List[DonationRequest] = []
        
        # 统计
        self._donate_count = 0
        self._skip_count = 0
        
        # 配置
        self.action_delay = (0.5, 1.0)
        self.scroll_delay = (0.3, 0.6)
        
        logger.info("捐兵器初始化完成")
    
    def detect_donation_requests(self, frame: np.ndarray) -> List[DonationRequest]:
        """
        检测捐兵请求
        
        在部落聊天界面检测请求的兵种。
        
        Args:
            frame: 当前帧
            
        Returns:
            检测到的捐兵请求列表
        """
        requests = []
        
        # 检测捐兵请求图标
        request_icons = self.matcher.match_all(
            frame, "donation_request_icon",
            threshold=0.7,
            max_count=5
        )
        
        for icon in request_icons:
            # 在请求图标附近识别兵种
            x, y = icon.center
            roi = (x - 50, y - 20, 100, 40)
            
            # 尝试匹配各兵种模板
            for troop_key, troop_name in self.DEFAULT_TROOPS.items():
                troop_result = self.matcher.match(
                    frame, f"troop_{troop_key}",
                    threshold=0.6,
                    roi=roi
                )
                
                if troop_result.found:
                    request = DonationRequest(
                        troop_name=troop_name,
                        troop_template=f"troop_{troop_key}",
                        position=icon.center
                    )
                    requests.append(request)
                    logger.debug(f"检测到捐兵请求: {troop_name} at {icon.center}")
                    break
        
        return requests
    
    def donate(self, frame: np.ndarray, request: DonationRequest) -> bool:
        """
        执行捐兵
        
        Args:
            frame: 当前帧
            request: 捐兵请求
            
        Returns:
            捐赠成功返回 True
        """
        if request.position is None:
            logger.warning("捐兵请求位置未知")
            return False
        
        # 点击捐兵请求
        if not self.click(request.position[0], request.position[1]):
            return False
        
        time.sleep(random.uniform(*self.action_delay))
        
        # 查找对应兵种卡牌
        troop_card = self.matcher.match(
            frame, request.troop_template,
            threshold=0.7
        )
        
        if not troop_card.found:
            logger.warning(f"未找到兵种卡牌: {request.troop_name}")
            self._skip_count += 1
            return False
        
        # 点击兵种卡牌进行捐赠
        card_center = troop_card.center
        for _ in range(request.count):
            if self.click(card_center[0], card_center[1]):
                time.sleep(random.uniform(0.1, 0.3))
        
        self._donate_count += 1
        logger.info(f"捐赠成功: {request.troop_name} x{request.count}")
        return True
    
    def run_donation_cycle(self, frame: np.ndarray) -> int:
        """
        执行一轮捐赠循环
        
        Args:
            frame: 当前帧
            
        Returns:
            成功捐赠的数量
        """
        if self.sm.current_state not in (GameState.CLAN_CHAT, GameState.DONATE_TROOPS):
            logger.warning("当前状态不支持捐兵")
            return 0
        
        # 检测请求
        requests = self.detect_donation_requests(frame)
        
        if not requests:
            logger.debug("没有检测到捐兵请求")
            return 0
        
        donated = 0
        for request in requests:
            if self.donate(frame, request):
                donated += 1
                time.sleep(random.uniform(*self.action_delay))
        
        logger.info(f"本轮捐赠: {donated}/{len(requests)}")
        return donated
    
    def _default_click(self, x: int, y: int) -> bool:
        logger.info(f"[模拟点击] ({x}, {y})")
        return True
    
    @property
    def donate_count(self) -> int:
        return self._donate_count
    
    @property
    def skip_count(self) -> int:
        return self._skip_count
    
    def __repr__(self) -> str:
        return (
            f"TroopDonator(donated={self._donate_count}, "
            f"skipped={self._skip_count})"
        )
