"""
Fallback Capture - ADB 截图后备采集

当 scrcpy 连接失败或首帧超时时，自动回退到 adb exec-out screencap。
保证程序不中断，记录警告日志。
"""

import time
import logging
import subprocess
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 cv2，如果不可用则延迟处理
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV 不可用，fallback 采集将返回原始 PNG 数据")


class FallbackCapture:
    """
    ADB 截图后备采集器
    
    使用 adb exec-out screencap -p 命令获取屏幕截图。
    性能较 scrcpy 低（约 2-5 秒/帧），但作为后备方案保证程序可用性。
    """
    
    def __init__(self, adb_path: str = "adb", device_serial: Optional[str] = None):
        """
        初始化后备采集器
        
        Args:
            adb_path: ADB 可执行文件路径
            device_serial: 设备序列号，None 表示自动选择
        """
        self.adb_path = adb_path
        self.device_serial = device_serial
        self._last_frame = None
        self._last_capture_time = 0.0
        self._capture_count = 0
        self._frame_interval = 0.5  # 最小采集间隔（秒），避免过于频繁
        
    def _build_adb_command(self) -> list:
        """构建 ADB 命令"""
        cmd = [self.adb_path]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(["exec-out", "screencap", "-p"])
        return cmd
    
    def recv_frame(self) -> Optional[np.ndarray]:
        """
        接收一帧图像
        
        使用 ADB screencap 命令获取屏幕截图并解码为 BGR numpy 数组。
        有最小采集间隔限制，避免过于频繁调用。
        
        Returns:
            BGR 格式的 numpy 数组，失败返回 None
        """
        # 检查采集间隔
        current_time = time.time()
        elapsed = current_time - self._last_capture_time
        if elapsed < self._frame_interval:
            # 返回上一帧
            return self._last_frame
        
        try:
            # 执行 ADB 命令
            cmd = self._build_adb_command()
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"ADB screencap 失败: {result.stderr.decode('utf-8', errors='replace')}")
                return self._last_frame
            
            # 获取 PNG 数据
            png_data = result.stdout
            
            if not png_data:
                logger.warning("ADB screencap 返回空数据")
                return self._last_frame
            
            # 解码 PNG 为 BGR 数组
            if CV2_AVAILABLE:
                frame = self._decode_png(png_data)
            else:
                logger.error("OpenCV 不可用，无法解码图像")
                return None
            
            if frame is not None:
                self._last_frame = frame
                self._last_capture_time = current_time
                self._capture_count += 1
                logger.debug(f"Fallback 采集成功，帧 #{self._capture_count}")
            
            return frame
            
        except subprocess.TimeoutExpired:
            logger.error("ADB screencap 超时")
            return self._last_frame
        except Exception as e:
            logger.error(f"Fallback 采集异常: {e}")
            return self._last_frame
    
    def _decode_png(self, png_data: bytes) -> Optional[np.ndarray]:
        """
        解码 PNG 数据为 BGR 数组
        
        Args:
            png_data: PNG 格式的图像数据
            
        Returns:
            BGR 格式的 numpy 数组，解码失败返回 None
        """
        if not CV2_AVAILABLE:
            return None
        
        try:
            # 将字节数据转换为 numpy 数组
            png_array = np.frombuffer(png_data, dtype=np.uint8)
            # 解码为 BGR 图像
            frame = cv2.imdecode(png_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"PNG 解码失败: {e}")
            return None
    
    def set_device(self, device_serial: str) -> None:
        """
        设置设备序列号
        
        Args:
            device_serial: 设备序列号
        """
        self.device_serial = device_serial
        logger.info(f"Fallback 采集器设置设备: {device_serial}")
    
    def set_frame_interval(self, interval: float) -> None:
        """
        设置最小采集间隔
        
        Args:
            interval: 采集间隔（秒）
        """
        self._frame_interval = max(0.1, interval)  # 最小 0.1 秒
        logger.info(f"Fallback 采集间隔设置为: {self._frame_interval}s")
    
    def get_last_frame(self) -> Optional[np.ndarray]:
        """获取最后一帧（不执行新的采集）"""
        return self._last_frame
    
    @property
    def capture_count(self) -> int:
        """已采集帧数"""
        return self._capture_count
    
    @property
    def last_capture_time(self) -> float:
        """最后采集时间戳"""
        return self._last_capture_time
    
    def __repr__(self) -> str:
        device = self.device_serial or "auto"
        return f"FallbackCapture(device={device}, captures={self._capture_count})"
