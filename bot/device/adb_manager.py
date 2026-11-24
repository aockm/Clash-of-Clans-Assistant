"""
ADB Manager - ADB 连接管理器

管理 ADB 连接，提供设备查询、连接、命令执行等功能。
用于执行层（ADB 点击注入）和设备状态查询。
"""

import subprocess
import logging
import re
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class ADBManager:
    """
    ADB 连接管理器
    
    封装 ADB 命令，提供设备连接管理和命令执行接口。
    """
    
    def __init__(self, adb_path: str = "adb"):
        """
        初始化 ADB 管理器
        
        Args:
            adb_path: ADB 可执行文件路径
        """
        self.adb_path = adb_path
        self.device_serial: Optional[str] = None
        self.connected = False
        
    def _run_adb_command(self, args: List[str], timeout: int = 10) -> Tuple[bool, str]:
        """
        执行 ADB 命令
        
        Args:
            args: ADB 命令参数列表
            timeout: 命令超时时间（秒）
            
        Returns:
            (success, output) 元组
        """
        cmd = [self.adb_path]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )
            
            if result.returncode == 0:
                return (True, result.stdout.strip())
            else:
                logger.error(f"ADB 命令失败: {' '.join(cmd)}")
                logger.error(f"错误输出: {result.stderr}")
                return (False, result.stderr.strip())
                
        except subprocess.TimeoutExpired:
            logger.error(f"ADB 命令超时: {' '.join(cmd)}")
            return (False, "命令超时")
        except Exception as e:
            logger.error(f"ADB 命令执行异常: {e}")
            return (False, str(e))
    
    def list_devices(self) -> List[str]:
        """
        列出已连接的设备
        
        Returns:
            设备序列号列表
        """
        success, output = self._run_adb_command(["devices"])
        if not success:
            return []
        
        devices = []
        lines = output.strip().split("\n")
        for line in lines[1:]:  # 跳过标题行
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        
        logger.info(f"发现 {len(devices)} 个设备: {devices}")
        return devices
    
    def connect(self, device_serial: Optional[str] = None) -> bool:
        """
        连接到指定设备
        
        Args:
            device_serial: 设备序列号，None 表示自动选择第一个设备
            
        Returns:
            连接成功返回 True，否则返回 False
        """
        devices = self.list_devices()
        
        if not devices:
            logger.error("未发现已连接的设备")
            return False
        
        if device_serial:
            if device_serial not in devices:
                logger.error(f"设备 {device_serial} 未连接")
                return False
            self.device_serial = device_serial
        else:
            # 自动选择第一个设备
            self.device_serial = devices[0]
            logger.info(f"自动选择设备: {self.device_serial}")
        
        self.connected = True
        logger.info(f"已连接到设备: {self.device_serial}")
        return True
    
    def disconnect(self) -> None:
        """断开设备连接"""
        self.device_serial = None
        self.connected = False
        logger.info("已断开设备连接")
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """
        获取设备屏幕尺寸
        
        Returns:
            (width, height) 元组，失败返回 None
        """
        success, output = self._run_adb_command(["shell", "wm", "size"])
        if not success:
            return None
        
        # 解析输出，格式: "Physical size: 1080x1920"
        match = re.search(r"(\d+)x(\d+)", output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            logger.info(f"设备屏幕尺寸: {width}x{height}")
            return (width, height)
        
        logger.error("无法解析屏幕尺寸")
        return None
    
    def get_screen_density(self) -> Optional[int]:
        """
        获取设备屏幕密度
        
        Returns:
            DPI 值，失败返回 None
        """
        success, output = self._run_adb_command(["shell", "wm", "density"])
        if not success:
            return None
        
        match = re.search(r"(\d+)", output)
        if match:
            dpi = int(match.group(1))
            logger.info(f"设备屏幕密度: {dpi} DPI")
            return dpi
        
        return None
    
    def tap(self, x: int, y: int) -> bool:
        """
        执行点击操作
        
        Args:
            x: X 坐标（设备像素）
            y: Y 坐标（设备像素）
            
        Returns:
            成功返回 True，否则返回 False
        """
        success, _ = self._run_adb_command(["shell", "input", "tap", str(x), str(y)])
        if success:
            logger.debug(f"点击: ({x}, {y})")
        return success
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        """
        执行滑动操作
        
        Args:
            x1: 起始 X 坐标
            y1: 起始 Y 坐标
            x2: 结束 X 坐标
            y2: 结束 Y 坐标
            duration_ms: 滑动持续时间（毫秒）
            
        Returns:
            成功返回 True，否则返回 False
        """
        success, _ = self._run_adb_command([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms)
        ])
        if success:
            logger.debug(f"滑动: ({x1},{y1}) -> ({x2},{y2}), {duration_ms}ms")
        return success
    
    def key_event(self, keycode: int) -> bool:
        """
        发送按键事件
        
        Args:
            keycode: 按键代码（如 KEYCODE_HOME = 3）
            
        Returns:
            成功返回 True，否则返回 False
        """
        success, _ = self._run_adb_command(["shell", "input", "keyevent", str(keycode)])
        return success
    
    def screencap(self) -> Optional[bytes]:
        """
        执行屏幕截图
        
        Returns:
            PNG 图像数据，失败返回 None
        """
        success, output = self._run_adb_command(
            ["exec-out", "screencap", "-p"],
            timeout=5
        )
        
        if success:
            # 返回原始字节数据
            return output.encode("latin-1") if isinstance(output, str) else output
        
        return None
    
    def is_screen_on(self) -> bool:
        """
        检查屏幕是否亮起
        
        Returns:
            屏幕亮起返回 True，否则返回 False
        """
        success, output = self._run_adb_command([
            "shell", "dumpsys", "power"
        ])
        
        if success:
            return "mScreenOn=true" in output or "Display Power: state=ON" in output
        
        return False
    
    def wake_screen(self) -> bool:
        """
        唤醒屏幕
        
        Returns:
            成功返回 True，否则返回 False
        """
        if self.is_screen_on():
            return True
        
        # 按电源键唤醒
        return self.key_event(26)  # KEYCODE_POWER
    
    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        device = self.device_serial or "none"
        return f"ADBManager({status}, device={device})"
