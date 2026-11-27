from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bot.orchestrator import GameContext


class GameState(enum.Enum):
    UNKNOWN = "unknown"
    MAIN_VILLAGE = "main_village"
    ATTACK_MENU = "attack_menu"
    ARMY_SETUP = "army_setup"        # 进攻前的配兵确认页
    FIND_MATCH = "find_match"        # 搜索到对手、待决策
    BATTLE = "battle"
    RESULTS = "results"
    CLAN_CHAT = "clan_chat"
    BUILDER_MENU = "builder_menu"
    BUILDING_PANEL = "building_panel"  # 建筑信息面板 / 升级确认框
    SETTINGS = "settings"              # 设置弹窗
    SWITCH_ACCOUNT = "switch_account"  # Supercell ID 切换ID页
    POPUP = "popup"                  # 各类弹窗（最高打断优先级）


@dataclass
class StateResult:
    state: GameState
    confidence: float = 0.0


class StateHandler:
    """每个游戏状态对应一个处理器：进入时执行该状态下的默认动作。"""

    state: GameState = GameState.UNKNOWN

    def __init__(self, ctx: GameContext) -> None:
        self.ctx = ctx

    def on_enter(self, frame: np.ndarray) -> GameState | None:
        return None
