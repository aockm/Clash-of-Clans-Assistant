"""
Frame Buffer Module - 环形帧缓冲区

实现后台线程持续消费视频流，解码为 BGR 帧并写入环形缓冲。
视觉层通过 get_latest() 即时返回最新帧，无采集阻塞。
"""

import threading
import time
import logging
from typing import Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    环形帧缓冲区
    
    维护最近 N 帧，支持零等待获取最新帧。
    使用 threading.Event 通知新帧就绪，避免忙等。
    """
    
    def __init__(self, max_size: int = 10):
        """
        初始化帧缓冲区
        
        Args:
            max_size: 缓冲区最大帧数，默认 10 帧
        """
        self.max_size = max_size
        self._buffer: deque[np.ndarray] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._frame_count = 0
        self._last_frame_time = 0.0
        
    def put(self, frame: np.ndarray) -> None:
        """
        写入新帧到缓冲区
        
        Args:
            frame: BGR 格式的 numpy 数组
        """
        with self._lock:
            self._buffer.append(frame)
            self._frame_count += 1
            self._last_frame_time = time.time()
            self._frame_event.set()
    
    def get_latest(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        获取最新帧（零等待）
        
        Args:
            timeout: 等待新帧的超时时间（秒），None 表示不等待
            
        Returns:
            最新的 BGR 帧，超时或无帧时返回 None
        """
        if timeout is not None:
            # 等待新帧到达
            self._frame_event.wait(timeout=timeout)
        
        with self._lock:
            if not self._buffer:
                return None
            # 返回最新帧（队尾）
            frame = self._buffer[-1].copy()
            self._frame_event.clear()
            return frame
    
    def get_all(self) -> list:
        """获取缓冲区中所有帧（按时间顺序）"""
        with self._lock:
            return [f.copy() for f in self._buffer]
    
    @property
    def frame_count(self) -> int:
        """已处理的总帧数"""
        return self._frame_count
    
    @property
    def last_frame_time(self) -> float:
        """最后一帧的时间戳"""
        return self._last_frame_time
    
    @property
    def buffer_size(self) -> int:
        """当前缓冲区中的帧数"""
        with self._lock:
            return len(self._buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
            self._frame_event.clear()
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        with self._lock:
            return len(self._buffer) == 0


class FrameBufferManager:
    """
    帧缓冲管理器
    
    管理后台流消费者线程，持续从视频源解码帧并写入环形缓冲。
    支持帧率统计和帧尺寸查询。
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        target_fps: int = 60
    ):
        """
        初始化帧缓冲管理器
        
        Args:
            buffer_size: 环形缓冲区大小
            target_fps: 目标帧率（用于统计）
        """
        self.buffer = FrameBuffer(max_size=buffer_size)
        self.target_fps = target_fps
        
        # 流控制
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._source = None  # 视频源（scrcpy 或 fallback）
        
        # 统计信息
        self._decoded_frames = 0
        self._decode_start_time = 0.0
        self._frame_dimensions: Optional[Tuple[int, int]] = None
        
        # 回调函数
        self._on_frame_callback = None
        
    def set_source(self, source) -> None:
        """
        设置视频源
        
        Args:
            source: 视频源对象，需提供 recv_frame() 方法
        """
        self._source = source
        logger.info(f"视频源已设置: {type(source).__name__}")
    
    def set_on_frame_callback(self, callback) -> None:
        """
        设置帧回调函数
        
        Args:
            callback: 回调函数，签名 callback(frame: np.ndarray)
        """
        self._on_frame_callback = callback
    
    def start(self) -> bool:
        """
        启动流消费者线程
        
        Returns:
            启动成功返回 True，否则返回 False
        """
        if self._running:
            logger.warning("帧缓冲管理器已在运行中")
            return False
        
        if self._source is None:
            logger.error("未设置视频源，无法启动")
            return False
        
        self._running = True
        self._decode_start_time = time.time()
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            name="FrameConsumer",
            daemon=True
        )
        self._consumer_thread.start()
        logger.info("帧缓冲管理器已启动")
        return True
    
    def stop(self) -> None:
        """停止流消费者线程"""
        if not self._running:
            return
        
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)
            if self._consumer_thread.is_alive():
                logger.warning("流消费者线程未能及时退出")
        logger.info("帧缓冲管理器已停止")
    
    def _consume_loop(self) -> None:
        """流消费者线程主循环"""
        logger.info("流消费者线程已启动")
        
        while self._running:
            try:
                # 从视频源接收帧
                frame = self._source.recv_frame()
                
                if frame is None:
                    # 短暂等待后重试
                    time.sleep(0.001)
                    continue
                
                # 记录帧尺寸
                if self._frame_dimensions is None:
                    h, w = frame.shape[:2]
                    self._frame_dimensions = (w, h)
                    logger.info(f"帧尺寸: {w}x{h}")
                
                # 写入缓冲区
                self.buffer.put(frame)
                self._decoded_frames += 1
                
                # 触发回调
                if self._on_frame_callback:
                    try:
                        self._on_frame_callback(frame)
                    except Exception as e:
                        logger.error(f"帧回调执行失败: {e}")
                
            except Exception as e:
                logger.error(f"帧消费循环异常: {e}")
                time.sleep(0.01)  # 避免异常时 CPU 空转
        
        logger.info("流消费者线程退出")
    
    def get_latest_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        获取最新帧
        
        Args:
            timeout: 等待超时（秒）
            
        Returns:
            最新的 BGR 帧或 None
        """
        return self.buffer.get_latest(timeout=timeout)
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running
    
    @property
    def fps(self) -> float:
        """当前帧率"""
        elapsed = time.time() - self._decode_start_time
        if elapsed <= 0:
            return 0.0
        return self._decoded_frames / elapsed
    
    @property
    def frame_dimensions(self) -> Optional[Tuple[int, int]]:
        """帧尺寸 (width, height)"""
        return self._frame_dimensions
    
    @property
    def decoded_frames(self) -> int:
        """已解码帧数"""
        return self._decoded_frames
    
    def wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        """
        等待第一帧到达
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            成功收到帧返回 True，超时返回 False
        """
        start = time.time()
        while time.time() - start < timeout:
            if not self.buffer.is_empty():
                return True
            time.sleep(0.05)
        return False
