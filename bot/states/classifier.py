import numpy as np

from bot.states.base import GameState, StateResult
from bot.vision.template_matcher import TemplateMatcher


class StateClassifier:
    """按"打断性"优先级分类当前画面：弹窗 > 战斗 > 其他主流程。"""

    PRIORITY = [
        # 对手预览页会同时出现"结束战斗"和"下一个"，必须先判"下一个"才能把预览页和实战区分开
        (GameState.FIND_MATCH, "find_match_next_btn"),
        (GameState.BATTLE, "battle_end_battle_btn"),
        # 升级确认框 / 建筑升级列表：绿色按钮与结算按钮相似，必须排在 RESULTS 前面
        (GameState.BUILDING_PANEL, "upgrade_confirm_btn"),
        (GameState.BUILDING_PANEL, "builder_busy_dialog"),
        (GameState.BUILDER_MENU, "builder_menu_title"),
        (GameState.RESULTS, "results_return_home"),
        (GameState.ARMY_SETUP, "army_setup_attack_btn"),
        (GameState.ATTACK_MENU, "attack_menu_find_match_btn"),
        (GameState.SETTINGS, "settings_supercell_bar"),
        (GameState.SWITCH_ACCOUNT, "switch_account_title"),
        (GameState.MAIN_VILLAGE, "main_village_attack_btn"),
        # popup_close 是各面板通用的关闭 X，不具排他性，必须放最后
        # 真正的未知弹窗会落到 UNKNOWN，由 FallbackHandler 尝试关闭
    ]

    # 个别模板是大面积纯色按钮，默认阈值会误命中背景，单独收紧
    # results_return_home 是纯绿按钮，在配兵页纯色背景上会拿到 0.84，真实结算页为 1.00
    # 升级确认框/列表标题同理，统一用 0.90 防止误判
    THRESHOLDS = {
        "results_return_home": 0.90,
        "upgrade_confirm_btn": 0.90,
        "builder_menu_title": 0.90,
        "builder_busy_dialog": 0.90,
        "settings_supercell_bar": 0.90,
        "switch_account_title": 0.90,
    }

    def __init__(self, matcher: TemplateMatcher, confirm_frames: int = 2) -> None:
        self._matcher = matcher
        self._confirm = confirm_frames
        self._history: list[GameState] = []

    def classify(self, frame: np.ndarray) -> StateResult:
        for state, tpl in self.PRIORITY:
            if self._matcher.match(frame, tpl, threshold=self.THRESHOLDS.get(tpl, 0.8)):
                self._history.append(state)
                if self._confirmed(state):
                    return StateResult(state=state, confidence=1.0)
                return StateResult(state=GameState.UNKNOWN, confidence=0.5)
        self._history.append(GameState.UNKNOWN)
        return StateResult(state=GameState.UNKNOWN, confidence=0.0)

    def _confirmed(self, state: GameState) -> bool:
        recent = self._history[-self._confirm:]
        return len(recent) == self._confirm and all(s == state for s in recent)
