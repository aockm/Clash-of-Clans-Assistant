import logging
import shutil
from pathlib import Path

import adbutils
import numpy as np
from wetest.scrcpy import AndroidDevice, LauncherConfig, VideoStreamScreenshot

from bot.device.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

# wetest-scrcpy 0.4.0 的 wheel 漏打包了 scrcpy-server，启动前需要补齐到包资源目录
_SERVER_RESOURCE = Path("resource") / "android" / "scrcpy-server"
DEFAULT_SERVER_PATH = Path("assets/scrcpy/scrcpy-server")


def ensure_server_binary(source: Path = DEFAULT_SERVER_PATH) -> None:
    """把项目内的 scrcpy-server 复制到 wetest-scrcpy 包资源目录（已存在则跳过）。"""
    import wetest.scrcpy as pkg

    target = Path(pkg.__file__).parent / _SERVER_RESOURCE
    if target.exists():
        return
    if not source.exists():
        raise FileNotFoundError(f"缺少 scrcpy-server：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    logger.info("scrcpy-server 已补齐到 %s", target)


class _BufferVideoStream(VideoStreamScreenshot):
    """在库自带解码线程的基础上，把每帧解码结果同步写入环形缓冲。"""

    def __init__(self, device: AndroidDevice, buffer: FrameBuffer,
                 timeout: float = 5.0) -> None:
        super().__init__(device=device, timeout=timeout)
        self._buffer = buffer

    def _process_video_frame(self, frame) -> None:
        super()._process_video_frame(frame)
        with self._lock:
            img = self._latest_frame
        if img is not None:
            # to_ndarray 每帧返回新数组，可直接交给缓冲持有
            self._buffer.push(img)


class ScrcpySession:
    """scrcpy 会话管理：自动拉起 server，解码 H.264 并写入环形缓冲。"""

    def __init__(self, serial: str | None, max_size: int, bit_rate: int,
                 frame_buffer: FrameBuffer) -> None:
        self.serial = serial
        self.max_size = max_size
        self.bit_rate = bit_rate
        self.buffer = frame_buffer
        self._device: AndroidDevice | None = None
        self._video: _BufferVideoStream | None = None

    def start(self) -> None:
        ensure_server_binary()
        serial = self.serial or adbutils.adb.device().serial
        self._device = AndroidDevice(
            serial=serial,
            launcher_config=LauncherConfig(max_size=self.max_size,
                                           bit_rate=self.bit_rate),
        )
        self._video = _BufferVideoStream(self._device, self.buffer)
        self._video.start()
        logger.info("scrcpy 视频流已启动：%s (max_size=%d)", serial, self.max_size)

    def stop(self) -> None:
        if self._video is not None:
            self._video.stop()
            self._video = None

    def wait_first_frame(self, timeout: float = 15.0) -> np.ndarray:
        """阻塞等待首帧，用于验证链路是否打通。"""
        if self.buffer.new_frame.wait(timeout):
            frame = self.buffer.get_latest()
            if frame is not None:
                return frame
        raise TimeoutError(f"等待首帧超时（{timeout}s）")

    def video_size(self) -> tuple[int, int]:
        frame = self.buffer.get_latest()
        if frame is None:
            raise RuntimeError("尚未拿到视频帧，无法确定视频尺寸")
        h, w = frame.shape[:2]
        return w, h
