import logging
import time

from wetest.scrcpy import AndroidDevice
from wetest.scrcpy.core._control import Action, CtrlConnection

logger = logging.getLogger(__name__)


class ScrcpyTouch:
    """通过 scrcpy 控制通道注入多点触控。

    adb input 只能模拟单指，双指捏合缩小地图必须走 scrcpy 控制通道；
    控制通道的坐标系与视频帧一致（如 1280x576）。
    """

    def __init__(self, serial: str) -> None:
        self._serial = serial
        self._ctrl: CtrlConnection | None = None

    def _connection(self) -> CtrlConnection | None:
        if self._ctrl is not None:
            return self._ctrl
        try:
            # launcher_config 留空：只连已有 server，连不上不重启（避免打断视频流）
            device = AndroidDevice(serial=self._serial)
            self._ctrl = CtrlConnection(device=device, timeout=5)
            info = self._ctrl.screen_info
            logger.info("触控通道已连接：%dx%d", info.Width, info.Height)
            return self._ctrl
        except Exception:
            logger.exception("连接 scrcpy 触控通道失败")
            self._ctrl = None
            return None

    def pinch_in(self, cx: int, cy: int, start_dist: int = 900,
                 end_dist: int = 120, steps: int = 10,
                 step_delay: float = 0.03) -> bool:
        """双指捏合（两指从远处靠拢），游戏内表现为地图缩小。"""
        ctrl = self._connection()
        if ctrl is None:
            return False
        half0, half1 = start_dist // 2, end_dist // 2
        try:
            ctrl.touch(cx - half0, cy, Action.DOWN, touch_id=0)
            ctrl.touch(cx + half0, cy, Action.DOWN, touch_id=1)
            for i in range(1, steps + 1):
                left = cx - half0 + (half0 - half1) * i // steps
                right = cx + half0 - (half0 - half1) * i // steps
                ctrl.touch(left, cy, Action.MOVE, touch_id=0)
                ctrl.touch(right, cy, Action.MOVE, touch_id=1)
                time.sleep(step_delay)
            ctrl.touch(cx - half1, cy, Action.UP, touch_id=0)
            ctrl.touch(cx + half1, cy, Action.UP, touch_id=1)
            return True
        except Exception:
            logger.exception("捏合手势注入失败，下次重新连接触控通道")
            self.close()
            return False

    def close(self) -> None:
        if self._ctrl is not None:
            try:
                self._ctrl.close()
            except Exception:
                pass
            self._ctrl = None
