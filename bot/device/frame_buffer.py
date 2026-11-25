import threading
from collections import deque

import numpy as np


class FrameBuffer:
    """环形缓冲：后台解码线程持续写入，视觉层 get_latest() 零等待取最新帧。"""

    def __init__(self, capacity: int = 10) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.new_frame = threading.Event()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(frame)
        self.new_frame.set()

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def clear_event(self) -> None:
        self.new_frame.clear()
