"""
Fallback Handler - 异常恢复处理器

处理意外状态（掉线弹窗、维护公告、未知界面），执行恢复策略。
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field

from .state_machine import GameState

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """恢复动作类型"""
    WAIT = auto()           # 等待
    TAP = auto()            # 点击
    TAP_BACK = auto()       # 按返回键
    RESTART = auto()        # 重启应用
    RECONNECT = auto()      # 重新连接
    ABORT = auto()          # 中止任务


@dataclass
class RecoveryStep:
    """恢复步骤"""
    action: RecoveryAction
    description: str
    params: Dict = field(default_factory=dict)
    timeout: float = 5.0  # 步骤超时（秒）
    
    def __repr__(self) -> str:
        return f"RecoveryStep({self.action.name}: {self.description})"


@dataclass
class RecoveryPlan:
    """恢复计划"""
    trigger_states: List[GameState]  # 触发恢复的状态
    steps: List[RecoveryStep]        # 恢复步骤
    target_state: GameState          # 期望恢复到的状态
    max_retries: int = 3             # 最大重试次数
    
    def __repr__(self) -> str:
        return (
            f"RecoveryPlan(triggers={[s.name for s in self.trigger_states]}, "
            f"target={self.target_state.name}, steps={len(self.steps)})"
        )


class FallbackHandler:
    """
    异常恢复处理器
    
    管理恢复计划，在检测到异常状态时执行恢复策略。
    """
    
    def __init__(self):
        """初始化恢复处理器"""
        # 恢复计划: {trigger_state: RecoveryPlan}
        self._plans: Dict[GameState, RecoveryPlan] = {}
        
        # 恢复回调
        self._on_recovery_start: List[Callable] = []
        self._on_recovery_success: List[Callable] = []
        self._on_recovery_failed: List[Callable] = []
        
        # 统计信息
        self._recovery_count = 0
        self._success_count = 0
        self._failed_count = 0
        
        # 重试跟踪
        self._retry_counts: Dict[GameState, int] = {}
        
        logger.info("异常恢复处理器初始化完成")
    
    def register_plan(self, plan: RecoveryPlan) -> None:
        """
        注册恢复计划
        
        Args:
            plan: 恢复计划
        """
        for state in plan.trigger_states:
            self._plans[state] = plan
            logger.debug(f"注册恢复计划: {state.name} -> {plan.target_state.name}")
    
    def add_plan(
        self,
        trigger_states: List[GameState],
        target_state: GameState,
        steps: List[RecoveryStep],
        max_retries: int = 3
    ) -> None:
        """
        便捷方法：添加恢复计划
        
        Args:
            trigger_states: 触发状态列表
            target_state: 目标状态
            steps: 恢复步骤
            max_retries: 最大重试次数
        """
        plan = RecoveryPlan(
            trigger_states=trigger_states,
            steps=steps,
            target_state=target_state,
            max_retries=max_retries
        )
        self.register_plan(plan)
    
    def should_handle(self, state: GameState) -> bool:
        """
        检查是否需要处理该状态
        
        Args:
            state: 当前状态
            
        Returns:
            需要处理返回 True
        """
        return state in self._plans
    
    def get_plan(self, state: GameState) -> Optional[RecoveryPlan]:
        """
        获取恢复计划
        
        Args:
            state: 当前状态
            
        Returns:
            恢复计划，不存在时返回 None
        """
        return self._plans.get(state)
    
    def can_retry(self, state: GameState) -> bool:
        """
        检查是否可以重试
        
        Args:
            state: 当前状态
            
        Returns:
            可以重试返回 True
        """
        plan = self._plans.get(state)
        if not plan:
            return False
        
        current_count = self._retry_counts.get(state, 0)
        return current_count < plan.max_retries
    
    def increment_retry(self, state: GameState) -> int:
        """
        增加重试计数
        
        Args:
            state: 当前状态
            
        Returns:
            当前重试次数
        """
        count = self._retry_counts.get(state, 0) + 1
        self._retry_counts[state] = count
        return count
    
    def reset_retry(self, state: GameState) -> None:
        """
        重置重试计数
        
        Args:
            state: 当前状态
        """
        self._retry_counts.pop(state, None)
    
    def execute_recovery(self, state: GameState) -> bool:
        """
        执行恢复流程
        
        Args:
            state: 当前状态
            
        Returns:
            恢复成功返回 True，否则返回 False
        """
        plan = self._plans.get(state)
        if not plan:
            logger.warning(f"未找到恢复计划: {state.name}")
            return False
        
        # 检查重试次数
        if not self.can_retry(state):
            logger.error(f"已达最大重试次数: {state.name}")
            self._failed_count += 1
            self._notify_failed(state)
            return False
        
        # 通知恢复开始
        self._recovery_count += 1
        self._notify_start(state, plan)
        
        logger.info(
            f"开始恢复: {state.name} -> {plan.target_state.name} "
            f"(重试: {self._retry_counts.get(state, 0)}/{plan.max_retries})"
        )
        
        # 执行恢复步骤
        for i, step in enumerate(plan.steps):
            logger.info(f"执行恢复步骤 {i+1}/{len(plan.steps)}: {step}")
            
            success = self._execute_step(step)
            if not success:
                logger.warning(f"恢复步骤失败: {step}")
                self.increment_retry(state)
                return False
        
        # 恢复成功
        self._success_count += 1
        self.reset_retry(state)
        self._notify_success(state, plan)
        
        logger.info(f"恢复成功: {state.name} -> {plan.target_state.name}")
        return True
    
    def _execute_step(self, step: RecoveryStep) -> bool:
        """
        执行单个恢复步骤
        
        Args:
            step: 恢复步骤
            
        Returns:
            执行成功返回 True
        """
        # 这里只是记录步骤，实际执行由外部处理
        # 外部需要注册步骤执行器
        
        action_handlers = {
            RecoveryAction.WAIT: self._handle_wait,
            RecoveryAction.TAP: self._handle_tap,
            RecoveryAction.TAP_BACK: self._handle_tap_back,
            RecoveryAction.RESTART: self._handle_restart,
            RecoveryAction.RECONNECT: self._handle_reconnect,
            RecoveryAction.ABORT: self._handle_abort,
        }
        
        handler = action_handlers.get(step.action)
        if handler:
            return handler(step)
        
        logger.warning(f"未知的恢复动作: {step.action}")
        return False
    
    def _handle_wait(self, step: RecoveryStep) -> bool:
        """处理等待动作"""
        duration = step.params.get("duration", 2.0)
        logger.debug(f"等待 {duration} 秒")
        time.sleep(duration)
        return True
    
    def _handle_tap(self, step: RecoveryStep) -> bool:
        """处理点击动作"""
        x = step.params.get("x", 0)
        y = step.params.get("y", 0)
        logger.debug(f"点击 ({x}, {y})")
        # 实际点击由外部执行器处理
        return True
    
    def _handle_tap_back(self, step: RecoveryStep) -> bool:
        """处理返回键动作"""
        logger.debug("按返回键")
        return True
    
    def _handle_restart(self, step: RecoveryStep) -> bool:
        """处理重启动作"""
        logger.debug("重启应用")
        return True
    
    def _handle_reconnect(self, step: RecoveryStep) -> bool:
        """处理重连动作"""
        logger.debug("重新连接")
        return True
    
    def _handle_abort(self, step: RecoveryStep) -> bool:
        """处理中止动作"""
        logger.debug("中止任务")
        return True
    
    def on_recovery_start(self, func: Callable) -> Callable:
        """注册恢复开始回调（装饰器）"""
        self._on_recovery_start.append(func)
        return func
    
    def on_recovery_success(self, func: Callable) -> Callable:
        """注册恢复成功回调（装饰器）"""
        self._on_recovery_success.append(func)
        return func
    
    def on_recovery_failed(self, func: Callable) -> Callable:
        """注册恢复失败回调（装饰器）"""
        self._on_recovery_failed.append(func)
        return func
    
    def _notify_start(self, state: GameState, plan: RecoveryPlan) -> None:
        """通知恢复开始"""
        for callback in self._on_recovery_start:
            try:
                callback(state, plan)
            except Exception as e:
                logger.error(f"恢复开始回调失败: {e}")
    
    def _notify_success(self, state: GameState, plan: RecoveryPlan) -> None:
        """通知恢复成功"""
        for callback in self._on_recovery_success:
            try:
                callback(state, plan)
            except Exception as e:
                logger.error(f"恢复成功回调失败: {e}")
    
    def _notify_failed(self, state: GameState) -> None:
        """通知恢复失败"""
        for callback in self._on_recovery_failed:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"恢复失败回调失败: {e}")
    
    @property
    def stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "total": self._recovery_count,
            "success": self._success_count,
            "failed": self._failed_count,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._recovery_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._retry_counts.clear()
    
    def __repr__(self) -> str:
        return (
            f"FallbackHandler(plans={len(self._plans)}, "
            f"stats={self.stats})"
        )


def create_default_fallback_handler() -> FallbackHandler:
    """
    创建默认的恢复处理器
    
    配置常见的异常状态恢复策略。
    
    Returns:
        配置好的恢复处理器
    """
    handler = FallbackHandler()
    
    # 弹窗恢复
    handler.add_plan(
        trigger_states=[GameState.POPUP],
        target_state=GameState.MAIN_VILLAGE,
        steps=[
            RecoveryStep(
                action=RecoveryAction.WAIT,
                description="等待弹窗完全加载",
                params={"duration": 1.0}
            ),
            RecoveryStep(
                action=RecoveryAction.TAP,
                description="点击关闭按钮",
                params={"x": 540, "y": 1200}  # 假设的关闭按钮位置
            ),
        ],
        max_retries=3
    )
    
    # 维护公告恢复
    handler.add_plan(
        trigger_states=[GameState.MAINTENANCE],
        target_state=GameState.MAIN_VILLAGE,
        steps=[
            RecoveryStep(
                action=RecoveryAction.WAIT,
                description="等待维护结束",
                params={"duration": 30.0}
            ),
        ],
        max_retries=10
    )
    
    # 未知状态恢复
    handler.add_plan(
        trigger_states=[GameState.UNKNOWN],
        target_state=GameState.MAIN_VILLAGE,
        steps=[
            RecoveryStep(
                action=RecoveryAction.TAP_BACK,
                description="按返回键"
            ),
            RecoveryStep(
                action=RecoveryAction.WAIT,
                description="等待界面响应",
                params={"duration": 2.0}
            ),
        ],
        max_retries=5
    )
    
    # 加载状态恢复
    handler.add_plan(
        trigger_states=[GameState.LOADING],
        target_state=GameState.MAIN_VILLAGE,
        steps=[
            RecoveryStep(
                action=RecoveryAction.WAIT,
                description="等待加载完成",
                params={"duration": 10.0}
            ),
        ],
        max_retries=3
    )
    
    logger.info(f"默认恢复处理器创建完成: {handler}")
    return handler
