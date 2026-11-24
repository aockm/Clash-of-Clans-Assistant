"""
State Classifier - 状态分类器

每帧接收截图，输出 GameState 和置信度。
检测优先级按"打断性"排序：弹窗 > 战斗 > 主村庄。
"""

import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

from .state_machine import GameState
from ..vision.template_matcher import TemplateMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """状态分类结果"""
    state: GameState
    confidence: float
    all_scores: Dict[GameState, float]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self) -> str:
        return f"ClassificationResult(state={self.state.name}, conf={self.confidence:.2f})"


class StateClassifier:
    """
    状态分类器
    
    使用模板匹配识别当前游戏状态。
    检测优先级：弹窗 > 战斗 > 主村庄（打断性强的优先）。
    """
    
    # 状态检测优先级（从高到低）
    PRIORITY_ORDER = [
        GameState.POPUP,           # 弹窗最高优先级
        GameState.MAINTENANCE,     # 维护公告
        GameState.BATTLE_RESULTS,  # 战斗结算
        GameState.ENEMY_BASE,      # 战斗中
        GameState.ATTACK_SEARCH,   # 搜索对手
        GameState.MY_ARMY,         # 我的军队
        GameState.ATTACK_MENU,     # 攻击菜单
        GameState.DONATE_TROOPS,   # 捐兵
        GameState.CLAN_CHAT,       # 部落聊天
        GameState.BUILDER_MENU,    # 工人菜单
        GameState.UPGRADE_CONFIRM, # 升级确认
        GameState.BUILDING_INFO,   # 建筑信息
        GameState.SHOP,            # 商店
        GameState.SETTINGS,        # 设置
        GameState.LOADING,         # 加载中
        GameState.MAIN_VILLAGE,    # 主村庄（默认）
        GameState.UNKNOWN,         # 未知
    ]
    
    def __init__(self, template_matcher: TemplateMatcher):
        """
        初始化状态分类器
        
        Args:
            template_matcher: 模板匹配器实例
        """
        self.matcher = template_matcher
        
        # 状态对应的模板名称映射
        self._state_templates: Dict[GameState, List[str]] = {}
        
        # 状态检测阈值
        self._thresholds: Dict[GameState, float] = {}
        
        # 初始化默认阈值
        for state in GameState:
            self._thresholds[state] = 0.8
        
        # 设置 ROI 区域（可选，用于加速检测）
        self._state_rois: Dict[GameState, Tuple[int, int, int, int]] = {}
        
        logger.info("状态分类器初始化完成")
    
    def register_state(
        self,
        state: GameState,
        template_names: List[str],
        threshold: float = 0.8,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        """
        注册状态检测模板
        
        Args:
            state: 游戏状态
            template_names: 用于检测该状态的模板名称列表
            threshold: 匹配阈值
            roi: 感兴趣区域 (x, y, w, h)
        """
        self._state_templates[state] = template_names
        self._thresholds[state] = threshold
        if roi:
            self._state_rois[state] = roi
        
        logger.info(
            f"注册状态 {state.name}: 模板={template_names}, "
            f"阈值={threshold}, ROI={roi}"
        )
    
    def classify(self, frame: np.ndarray) -> ClassificationResult:
        """
        分类当前帧的游戏状态
        
        按优先级顺序检测，找到第一个匹配的状态即返回。
        
        Args:
            frame: 输入帧（BGR 格式）
            
        Returns:
            ClassificationResult 分类结果
        """
        all_scores: Dict[GameState, float] = {}
        
        # 按优先级顺序检测
        for state in self.PRIORITY_ORDER:
            if state not in self._state_templates:
                continue
            
            templates = self._state_templates[state]
            threshold = self._thresholds.get(state, 0.8)
            roi = self._state_rois.get(state)
            
            # 对每个模板进行匹配
            best_match: Optional[MatchResult] = None
            best_confidence = 0.0
            
            for template_name in templates:
                result = self.matcher.match(
                    frame, template_name, 
                    threshold=threshold,
                    roi=roi
                )
                
                if result.found and result.confidence > best_confidence:
                    best_match = result
                    best_confidence = result.confidence
            
            # 记录该状态的得分
            all_scores[state] = best_confidence
            
            # 如果匹配成功，立即返回（优先级高的先返回）
            if best_match is not None:
                logger.debug(
                    f"状态分类: {state.name} "
                    f"(置信度: {best_confidence:.2f})"
                )
                return ClassificationResult(
                    state=state,
                    confidence=best_confidence,
                    all_scores=all_scores,
                    metadata={"match": best_match}
                )
        
        # 没有匹配到任何状态
        if all_scores:
            # 返回得分最高的状态（即使低于阈值）
            best_state = max(all_scores, key=all_scores.get)
            if all_scores[best_state] > 0.5:
                logger.debug(
                    f"状态分类（低置信度）: {best_state.name} "
                    f"(置信度: {all_scores[best_state]:.2f})"
                )
                return ClassificationResult(
                    state=best_state,
                    confidence=all_scores[best_state],
                    all_scores=all_scores
                )
        
        # 完全无法识别
        logger.debug("状态分类: UNKNOWN")
        return ClassificationResult(
            state=GameState.UNKNOWN,
            confidence=0.0,
            all_scores=all_scores
        )
    
    def classify_all(self, frame: np.ndarray) -> List[ClassificationResult]:
        """
        分类所有可能的状态（不按优先级截断）
        
        用于调试和分析。
        
        Args:
            frame: 输入帧
            
        Returns:
            按置信度降序排列的分类结果列表
        """
        results = []
        
        for state in GameState:
            if state not in self._state_templates:
                continue
            
            templates = self._state_templates[state]
            roi = self._state_rois.get(state)
            
            best_confidence = 0.0
            best_match = None
            
            for template_name in templates:
                result = self.matcher.match(frame, template_name, roi=roi)
                if result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_match = result
            
            results.append(ClassificationResult(
                state=state,
                confidence=best_confidence,
                all_scores={},
                metadata={"match": best_match} if best_match else {}
            ))
        
        # 按置信度降序排列
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
    
    def set_threshold(self, state: GameState, threshold: float) -> None:
        """设置状态检测阈值"""
        self._thresholds[state] = threshold
    
    def set_roi(self, state: GameState, roi: Tuple[int, int, int, int]) -> None:
        """设置状态检测的 ROI 区域"""
        self._state_rois[state] = roi
    
    def get_registered_states(self) -> List[GameState]:
        """获取已注册的状态列表"""
        return list(self._state_templates.keys())
    
    def __repr__(self) -> str:
        return f"StateClassifier(states={len(self._state_templates)})"
