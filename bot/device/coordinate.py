"""
Coordinate Mapper - 坐标空间管理器

管理视频编码尺寸与设备像素尺寸的映射关系。
scrcpy 控制通道的坐标必须落在视频尺寸空间，而非设备像素空间。
视觉层检测到的坐标基于视频帧，执行层 ADB tap 需要设备像素坐标。
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """
    坐标映射器
    
    在视频帧坐标和设备像素坐标之间进行转换。
    支持缩放因子计算和坐标变换。
    """
    
    def __init__(
        self,
        video_width: int,
        video_height: int,
        device_width: int,
        device_height: int
    ):
        """
        初始化坐标映射器
        
        Args:
            video_width: 视频流宽度（像素）
            video_height: 视频流高度（像素）
            device_width: 设备屏幕宽度（像素）
            device_height: 设备屏幕高度（像素）
        """
        self.video_width = video_width
        self.video_height = video_height
        self.device_width = device_width
        self.device_height = device_height
        
        # 计算缩放因子
        self.scale_x = device_width / video_width
        self.scale_y = device_height / video_height
        
        logger.info(
            f"坐标映射器初始化: 视频={video_width}x{video_height}, "
            f"设备={device_width}x{device_height}, "
            f"缩放因子=({self.scale_x:.3f}, {self.scale_y:.3f})"
        )
    
    def video_to_device(self, x: int, y: int) -> Tuple[int, int]:
        """
        将视频坐标转换为设备坐标
        
        Args:
            x: 视频帧中的 X 坐标
            y: 视频帧中的 Y 坐标
            
        Returns:
            设备像素坐标 (x, y)
        """
        device_x = int(x * self.scale_x)
        device_y = int(y * self.scale_y)
        
        # 边界裁剪
        device_x = max(0, min(device_x, self.device_width - 1))
        device_y = max(0, min(device_y, self.device_height - 1))
        
        return (device_x, device_y)
    
    def device_to_video(self, x: int, y: int) -> Tuple[int, int]:
        """
        将设备坐标转换为视频坐标
        
        Args:
            x: 设备像素 X 坐标
            y: 设备像素 Y 坐标
            
        Returns:
            视频帧坐标 (x, y)
        """
        video_x = int(x / self.scale_x)
        video_y = int(y / self.scale_y)
        
        # 边界裁剪
        video_x = max(0, min(video_x, self.video_width - 1))
        video_y = max(0, min(video_y, self.video_height - 1))
        
        return (video_x, video_y)
    
    def update_video_size(self, width: int, height: int) -> None:
        """
        更新视频尺寸（当 scrcpy 调整分辨率时）
        
        Args:
            width: 新的视频宽度
            height: 新的视频高度
        """
        self.video_width = width
        self.video_height = height
        self.scale_x = self.device_width / width
        self.scale_y = self.device_height / height
        
        logger.info(
            f"视频尺寸更新: {width}x{height}, "
            f"新缩放因子=({self.scale_x:.3f}, {self.scale_y:.3f})"
        )
    
    def update_device_size(self, width: int, height: int) -> None:
        """
        更新设备尺寸（当设备旋转或分辨率变化时）
        
        Args:
            width: 新的设备宽度
            height: 新的设备高度
        """
        self.device_width = width
        self.device_height = height
        self.scale_x = width / self.video_width
        self.scale_y = height / self.video_height
        
        logger.info(
            f"设备尺寸更新: {width}x{height}, "
            f"新缩放因子=({self.scale_x:.3f}, {self.scale_y:.3f})"
        )
    
    def get_scale_factors(self) -> Tuple[float, float]:
        """
        获取缩放因子
        
        Returns:
            (scale_x, scale_y) 元组
        """
        return (self.scale_x, self.scale_y)
    
    def __repr__(self) -> str:
        return (
            f"CoordinateMapper(video={self.video_width}x{self.video_height}, "
            f"device={self.device_width}x{self.device_height})"
        )
