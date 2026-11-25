from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.orchestrator import GameContext


class Feature:
    """业务功能基类：每个功能实现 run()，返回是否完成一轮。"""

    name: str = "feature"

    def __init__(self, ctx: GameContext) -> None:
        self.ctx = ctx

    def run(self) -> bool:
        raise NotImplementedError
