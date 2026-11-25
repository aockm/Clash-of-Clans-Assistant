import logging

import numpy as np

from bot.device.adb_manager import AdbManager

logger = logging.getLogger(__name__)


class FallbackCapture:
    """scrcpy 失败时的后备：adb exec-out screencap 截图。"""

    def __init__(self, adb: AdbManager) -> None:
        self._adb = adb

    def capture(self) -> np.ndarray:
        logger.warning("Using fallback ADB screencap")
        img = self._adb.device.screenshot()
        return np.array(img)
