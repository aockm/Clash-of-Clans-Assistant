import logging

import adbutils

logger = logging.getLogger(__name__)


class AdbManager:
    """执行层 ADB 连接管理：负责 tap/swipe 等动作注入。"""

    def __init__(self, serial: str | None = None) -> None:
        self.client = adbutils.AdbClient(host="127.0.0.1", port=5037)
        self.device = self.client.device(serial) if serial else self.client.device()
        logger.info("ADB connected: %s", self.device.serial)

    def tap(self, x: int, y: int) -> None:
        self.device.click(x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> None:
        self.device.swipe(x1, y1, x2, y2, duration=duration)

    def screen_size(self) -> tuple[int, int]:
        w, h = self.device.window_size()
        return w, h
