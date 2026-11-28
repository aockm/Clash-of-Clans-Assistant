import random
import time

from bot.device.adb_manager import AdbManager
from bot.device.coordinate import CoordinateMapper


class SafeActions:
    """带随机化的动作注入：坐标抖动 + 延迟抖动，坐标从视频空间映射到设备空间。"""

    def __init__(self, adb: AdbManager, mapper: CoordinateMapper,
                 jitter_px: int, delay_range: tuple[float, float]) -> None:
        self._adb = adb
        self._mapper = mapper
        self._jitter = jitter_px
        self._delay = delay_range

    def tap(self, x: int, y: int, pause: bool = True) -> None:
        """pause=False 用于连续下兵这类需要快速连点的场景。"""
        dx, dy = self._mapper.to_device(x, y)
        dx += random.randint(-self._jitter, self._jitter)
        dy += random.randint(-self._jitter, self._jitter)
        self._adb.tap(dx, dy)
        if pause:
            time.sleep(random.uniform(*self._delay))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float) -> None:
        sx, sy = self._mapper.to_device(x1, y1)
        ex, ey = self._mapper.to_device(x2, y2)
        j = self._jitter
        self._adb.swipe(sx + random.randint(-j, j), sy + random.randint(-j, j),
                        ex + random.randint(-j, j), ey + random.randint(-j, j),
                        duration=duration)
        time.sleep(random.uniform(*self._delay))

    def back(self) -> None:
        self._adb.key_back()
        time.sleep(random.uniform(*self._delay))
