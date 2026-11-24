"""
State Machine - 状态机核心

定义游戏状态枚举和状态机核心逻辑。
每个游戏屏幕是一个 GameState，状态之间的迁移由视觉信号驱动。
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class GameState(Enum):
    """
    游戏状态枚举
    
    每个状态对应一个游戏屏幕或界面。
    """
    # 主界面相关
    MAIN_VILLAGE = auto()           # 主村庄
    BUILDING_INFO = auto()          # 建筑信息面板
    BUILDER_MENU = auto()           # 工人选择菜单
    UPGRADE_CONFIRM = auto()        # 升级确认对话框
    
    # 部落相关
    CLAN_CHAT = auto()              # 部落聊天
    CLAN_CASTLE = auto()            # 部落城堡界面
    DONATE_TROOPS = auto()          # 捐兵界面
    
    # 战斗相关
    ATTACK_MENU = auto()            # 攻击菜单（搜索对手）
    ATTACK_SEARCH = auto()          # 正在搜索对手
    MY_ARMY = auto()                # 我的军队（配兵/进攻页面）
    ENEMY_BASE = auto()             # 敌方基地（战斗画面）
    BATTLE_RESULTS = auto()         # 战斗结算界面
    
    # 商店/其他
    SHOP = auto()                   # 商店
    SETTINGS = auto()               # 设置界面
    
    # 异常状态
    LOADING = auto()                # 加载中
    POPUP = auto()                  # 弹窗（通用）
    MAINTENANCE = auto()            # 维护公告
    UNKNOWN = auto()                # 未知状态


@dataclass
class StateContext:
    """
    状态上下文
    
    包含当前状态的附加信息，如置信度、检测时间等。
    """
    state: GameState
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"StateContext(state={self.state.name}, conf={self.confidence:.2f})"


class StateMachine:
    """
    状态机
    
    管理游戏状态，处理状态转换，执行状态相关的回调函数。
    """
    
    def __init__(self, initial_state: GameState = GameState.UNKNOWN):
        """
        初始化状态机
        
        Args:
            initial_state: 初始状态
        """
        self._current_state = initial_state
        self._previous_state: Optional[GameState] = None
        self._state_enter_time = time.time()
        self._state_history: List[StateContext] = []
        
        # 状态转换回调
        self._on_enter_callbacks: Dict[GameState, List[Callable]] = {}
        self._on_exit_callbacks: Dict[GameState, List[Callable]] = {}
        self._on_transition_callbacks: List[Callable] = []
        
        # 状态确认配置
        self._confirm_frames: Dict[GameState, int] = {}
        self._pending_state: Optional[GameState] = None
        self._pending_count = 0
        
        # 统计信息
        self._transition_count = 0
        self._start_time = time.time()
        
        logger.info(f"状态机初始化，初始状态: {initial_state.name}")
    
    @property
    def current_state(self) -> GameState:
        """当前状态"""
        return self._current_state
    
    @property
    def previous_state(self) -> Optional[GameState]:
        """上一个状态"""
        return self._previous_state
    
    @property
    def state_duration(self) -> float:
        """当前状态持续时间（秒）"""
        return time.time() - self._state_enter_time
    
    @property
    def transition_count(self) -> int:
        """状态转换次数"""
        return self._transition_count
    
    def set_confirm_frames(self, state: GameState, frames: int) -> None:
        """
        设置状态确认帧数
        
        对于关键状态转换，需要连续多帧确认才执行转换。
        
        Args:
            state: 目标状态
            frames: 需要确认的帧数
        """
        self._confirm_frames[state] = frames
        logger.debug(f"设置 {state.name} 确认帧数: {frames}")
    
    def transition(self, new_state: GameState, confidence: float = 1.0, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        尝试转换到新状态
        
        Args:
            new_state: 目标状态
            confidence: 状态识别置信度
            metadata: 附加元数据
            
        Returns:
            转换成功返回 True，否则返回 False
        """
        # 检查是否需要多帧确认
        required_frames = self._confirm_frames.get(new_state, 1)
        
        if new_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = new_state
            self._pending_count = 1
        
        if self._pending_count < required_frames:
            logger.debug(
                f"等待状态确认: {new_state.name} "
                f"({self._pending_count}/{required_frames})"
            )
            return False
        
        # 执行状态转换
        return self._do_transition(new_state, confidence, metadata)
    
    def force_transition(self, new_state: GameState, 
                         metadata: Optional[Dict] = None) -> bool:
        """
        强制转换状态（跳过确认）
        
        Args:
            new_state: 目标状态
            metadata: 附加元数据
            
        Returns:
            转换成功返回 True
        """
        return self._do_transition(new_state, 1.0, metadata)
    
    def _do_transition(self, new_state: GameState, confidence: float,
                       metadata: Optional[Dict]) -> bool:
        """执行状态转换"""
        if new_state == self._current_state:
            return False
        
        old_state = self._current_state
        
        # 执行退出回调
        for callback in self._on_exit_callbacks.get(old_state, []):
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"退出回调执行失败: {e}")
        
        # 更新状态
        self._previous_state = old_state
        self._current_state = new_state
        self._state_enter_time = time.time()
        self._transition_count += 1
        
        # 重置确认计数
        self._pending_state = None
        self._pending_count = 0
        
        # 记录历史
        context = StateContext(
            state=new_state,
            confidence=confidence,
            metadata=metadata or {}
        )
        self._state_history.append(context)
        
        # 限制历史记录长度
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-50:]
        
        # 执行进入回调
        for callback in self._on_enter_callbacks.get(new_state, []):
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"进入回调执行失败: {e}")
        
        # 执行转换回调
        for callback in self._on_transition_callbacks:
            try:
                callback(old_state, new_state, confidence)
            except Exception as e:
                logger.error(f"转换回调执行失败: {e}")
        
        logger.info(
            f"状态转换: {old_state.name} -> {new_state.name} "
            f"(置信度: {confidence:.2f})"
        )
        
        return True
    
    def on_enter(self, state: GameState) -> Callable:
        """
        注册状态进入回调（装饰器）
        
        Args:
            state: 目标状态
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            if state not in self._on_enter_callbacks:
                self._on_enter_callbacks[state] = []
            self._on_enter_callbacks[state].append(func)
            return func
        return decorator
    
    def on_exit(self, state: GameState) -> Callable:
        """
        注册状态退出回调（装饰器）
        
        Args:
            state: 离开状态
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            if state not in self._on_exit_callbacks:
                self._on_exit_callbacks[state] = []
            self._on_exit_callbacks[state].append(func)
            return func
        return decorator
    
    def on_transition(self, func: Callable) -> Callable:
        """
        注册状态转换回调（装饰器）
        
        Args:
            func: 回调函数 (old_state, new_state, confidence)
            
        Returns:
            原函数
        """
        self._on_transition_callbacks.append(func)
        return func
    
    def add_enter_callback(self, state: GameState, callback: Callable) -> None:
        """添加状态进入回调"""
        if state not in self._on_enter_callbacks:
            self._on_enter_callbacks[state] = []
        self._on_enter_callbacks[state].append(callback)
    
    def add_exit_callback(self, state: GameState, callback: Callable) -> None:
        """添加状态退出回调"""
        if state not in self._on_exit_callbacks:
            self._on_exit_callbacks[state] = []
        self._on_exit_callbacks[state].append(callback)
    
    def add_transition_callback(self, callback: Callable) -> None:
        """添加状态转换回调"""
        self._on_transition_callbacks.append(callback)
    
    def get_history(self, count: int = 10) -> List[StateContext]:
        """
        获取状态历史
        
        Args:
            count: 返回的历史记录数量
            
        Returns:
            状态上下文列表
        """
        return self._state_history[-count:]
    
    def reset(self) -> None:
        """重置状态机"""
        self._current_state = GameState.UNKNOWN
        self._previous_state = None
        self._state_enter_time = time.time()
        self._pending_state = None
        self._pending_count = 0
        logger.info("状态机已重置")
    
    def __repr__(self) -> str:
        duration = time.time() - self._start_time
        return (
            f"StateMachine(state={self._current_state.name}, "
            f"transitions={self._transition_count}, "
            f"uptime={duration:.1f}s)"
        )
