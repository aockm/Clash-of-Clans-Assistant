import logging

import numpy as np

from bot.states.base import GameState, StateHandler

logger = logging.getLogger(__name__)


class FallbackHandler(StateHandler):
    """处理 UNKNOWN / 弹窗等意外状态：尝试关闭弹窗，连续失败则重进游戏。"""

    state = GameState.POPUP

    MAX_ATTEMPTS = 5

    # 依次尝试：关闭 X、奖励/提示弹窗的“继续”
    DISMISS_TEMPLATES = ("popup_close", "popup_continue_btn")

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._attempts = 0

    def reset(self) -> None:
        self._attempts = 0

    def on_enter(self, frame: np.ndarray) -> GameState | None:
        self._attempts += 1
        if self._attempts > self.MAX_ATTEMPTS:
            logger.error("Fallback exceeded max attempts, restarting game")
            self._attempts = 0
            return GameState.UNKNOWN
        for name in self.DISMISS_TEMPLATES:
            m = self.ctx.matcher.match(frame, name, threshold=0.85)
            if m:
                self.ctx.actions.tap(*m.center)
                return None
        return None
