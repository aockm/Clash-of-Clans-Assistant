import logging

import numpy as np

from bot.states.base import GameState, StateHandler

logger = logging.getLogger(__name__)


class FallbackHandler(StateHandler):
    """处理 UNKNOWN / 弹窗等意外状态：尝试点掉弹窗；认不出的加载动画就安静等待。"""

    state = GameState.POPUP

    MAX_ATTEMPTS = 5

    # 依次尝试：奖励/提示弹窗的“继续”
    DISMISS_TEMPLATES = ("popup_continue_btn",)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._attempts = 0

    def reset(self) -> None:
        self._attempts = 0

    def on_enter(self, frame: np.ndarray) -> GameState | None:
        self._attempts += 1
        for name in self.DISMISS_TEMPLATES:
            m = self.ctx.matcher.match(frame, name, threshold=0.85)
            if m:
                self.ctx.actions.tap(*m.center)
                return None
        # 加载/动画中会连续出现 UNKNOWN，只在达到阈值时提醒一次，避免刷屏
        if self._attempts == self.MAX_ATTEMPTS:
            logger.info("连续 %d 帧未识别出状态（多为加载/动画），继续等画面恢复",
                        self.MAX_ATTEMPTS)
        return None
