"""
Scrcpy Session - scrcpy 会话管理器

启动 scrcpy-server，建立 video socket 连接，管理会话生命周期。
使用 adb reverse 而非 adb forward（forward 路径实测 2~4 秒后静默停流）。
必须传 cleanup=false 防止 jar 被删除。
"""

import socket
import subprocess
import time
import logging
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 PyAV 用于 H.264 解码
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("PyAV 不可用，scrcpy 会话将无法解码 H.264 流")

# 尝试导入 cv2 用于图像格式转换
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ScrcpySession:
    """
    scrcpy 会话管理器
    
    负责：
    1. 推送 scrcpy-server 到设备
    2. 启动 scrcpy-server（通过 adb shell）
    3. 建立 adb reverse 端口转发
    4. 连接 video socket 和 control socket
    5. 接收 H.264 流并解码为 BGR 帧
    
    注意：
    - 必须使用 adb reverse（不是 forward）
    - 必须传 cleanup=false 防止 jar 被删除
    - video 和 control 两个 socket 必须同时连接
    """
    
    # scrcpy 默认参数
    DEFAULT_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
    DEFAULT_VIDEO_PORT = 27183
    DEFAULT_CONTROL_PORT = 27183
    DEFAULT_MAX_SIZE = 800  # 视频最大尺寸（降低以提高性能）
    DEFAULT_BIT_RATE = 4000000  # 4 Mbps
    DEFAULT_FRAME_RATE = 15  # 目标帧率
    
    def __init__(
        self,
        adb_path: str = "adb",
        device_serial: Optional[str] = None,
        scrcpy_server_path: Optional[str] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        bit_rate: int = DEFAULT_BIT_RATE,
        frame_rate: int = DEFAULT_FRAME_RATE,
        cleanup: bool = False  # 必须为 False，防止 jar 被删除
    ):
        """
        初始化 scrcpy 会话
        
        Args:
            adb_path: ADB 可执行文件路径
            device_serial: 设备序列号
            scrcpy_server_path: 本地 scrcpy-server.jar 路径，None 使用默认
            max_size: 视频最大尺寸（像素）
            bit_rate: 视频比特率
            frame_rate: 目标帧率
            cleanup: 是否清理 server 文件（必须为 False）
        """
        self.adb_path = adb_path
        self.device_serial = device_serial
        self.scrcpy_server_path = scrcpy_server_path
        self.max_size = max_size
        self.bit_rate = bit_rate
        self.frame_rate = frame_rate
        self.cleanup = cleanup
        
        # 会话状态
        self._running = False
        self._server_process: Optional[subprocess.Popen] = None
        self._video_socket: Optional[socket.socket] = None
        self._control_socket: Optional[socket.socket] = None
        
        # 视频解码
        self._codec = None
        self._video_width = 0
        self._video_height = 0
        
        # 帧缓冲
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        
        # 接收线程
        self._recv_thread: Optional[threading.Thread] = None
        
        logger.info(
            f"ScrcpySession 初始化: max_size={max_size}, "
            f"bit_rate={bit_rate}, fps={frame_rate}"
        )
    
    def _run_adb_command(self, args: list, timeout: int = 10) -> Tuple[bool, str]:
        """执行 ADB 命令"""
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
            return (result.returncode == 0, result.stdout.strip())
        except Exception as e:
            logger.error(f"ADB 命令失败: {e}")
            return (False, str(e))
    
    def push_server(self) -> bool:
        """
        推送 scrcpy-server.jar 到设备
        
        Returns:
            成功返回 True，否则返回 False
        """
        if self.scrcpy_server_path is None:
            logger.error("未指定 scrcpy-server.jar 路径")
            return False
        
        logger.info(f"推送 scrcpy-server 到设备: {self.DEFAULT_SERVER_PATH}")
        success, _ = self._run_adb_command([
            "push",
            self.scrcpy_server_path,
            self.DEFAULT_SERVER_PATH
        ])
        
        return success
    
    def setup_reverse_tunnel(self) -> bool:
        """
        建立 adb reverse 端口转发
        
        Returns:
            成功返回 True，否则返回 False
        """
        logger.info("建立 adb reverse 端口转发")
        
        # 移除旧的转发规则
        self._run_adb_command(["reverse", "--remove-all"])
        
        # 建立新的转发
        success, _ = self._run_adb_command([
            "reverse",
            f"localabstract:scrcpy",
            f"tcp:{self.DEFAULT_VIDEO_PORT}"
        ])
        
        return success
    
    def start_server(self) -> bool:
        """
        启动 scrcpy-server
        
        Returns:
            成功返回 True，否则返回 False
        """
        logger.info("启动 scrcpy-server")
        
        # 构建 server 启动命令
        cmd = [
            self.adb_path
        ]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        
        cmd.extend([
            "shell",
            f"CLASSPATH={self.DEFAULT_SERVER_PATH}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "2.1",  # scrcpy 版本
            f"log_level=info",
            f"max_size={self.max_size}",
            f"max_fps={self.frame_rate}",
            f"video_bit_rate={self.bit_rate}",
            f"cleanup=false",  # 必须为 false
            f"raw_stream=true"  # 原始流模式
        ])
        
        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            # 等待 server 启动
            time.sleep(1.0)
            
            # 检查进程是否存活
            if self._server_process.poll() is not None:
                logger.error("scrcpy-server 启动失败")
                return False
            
            logger.info("scrcpy-server 已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动 scrcpy-server 异常: {e}")
            return False
    
    def connect_sockets(self) -> bool:
        """
        连接 video 和 control sockets
        
        Returns:
            成功返回 True，否则返回 False
        """
        logger.info("连接 scrcpy sockets")
        
        try:
            # 等待端口就绪
            time.sleep(0.5)
            
            # 连接 video socket（通过 localhost 转发端口）
            self._video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._video_socket.settimeout(5.0)
            self._video_socket.connect(("127.0.0.1", self.DEFAULT_VIDEO_PORT))
            logger.info("Video socket 已连接")
            
            # 连接 control socket（可以保持空闲，但必须连接）
            self._control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._control_socket.settimeout(5.0)
            self._control_socket.connect(("127.0.0.1", self.DEFAULT_CONTROL_PORT + 1))
            logger.info("Control socket 已连接")
            
            return True
            
        except Exception as e:
            logger.error(f"连接 sockets 失败: {e}")
            return False
    
    def init_decoder(self) -> bool:
        """
        初始化 H.264 解码器
        
        Returns:
            成功返回 True，否则返回 False
        """
        if not AV_AVAILABLE:
            logger.error("PyAV 不可用，无法初始化解码器")
            return False
        
        try:
            # 创建 H.264 解码器
            self._codec = av.codec.CodecContext.create("h264", "r")
            logger.info("H.264 解码器已初始化")
            return True
        except Exception as e:
            logger.error(f"初始化解码器失败: {e}")
            return False
    
    def start(self) -> bool:
        """
        启动完整的 scrcpy 会话
        
        Returns:
            成功返回 True，否则返回 False
        """
        if self._running:
            logger.warning("会话已在运行中")
            return False
        
        logger.info("启动 scrcpy 会话...")
        
        # 1. 推送 server
        if self.scrcpy_server_path:
            if not self.push_server():
                logger.error("推送 server 失败")
                return False
        
        # 2. 建立反向隧道
        if not self.setup_reverse_tunnel():
            logger.error("建立反向隧道失败")
            return False
        
        # 3. 启动 server
        if not self.start_server():
            logger.error("启动 server 失败")
            return False
        
        # 4. 连接 sockets
        if not self.connect_sockets():
            logger.error("连接 sockets 失败")
            self.stop()
            return False
        
        # 5. 初始化解码器
        if not self.init_decoder():
            logger.error("初始化解码器失败")
            self.stop()
            return False
        
        # 6. 启动接收线程
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name="ScrcpyRecv",
            daemon=True
        )
        self._recv_thread.start()
        
        logger.info("scrcpy 会话已启动")
        return True
    
    def stop(self) -> None:
        """停止 scrcpy 会话"""
        logger.info("停止 scrcpy 会话")
        
        self._running = False
        
        # 关闭 sockets
        if self._video_socket:
            try:
                self._video_socket.close()
            except:
                pass
            self._video_socket = None
        
        if self._control_socket:
            try:
                self._control_socket.close()
            except:
                pass
            self._control_socket = None
        
        # 终止 server 进程
        if self._server_process:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=2.0)
            except:
                try:
                    self._server_process.kill()
                except:
                    pass
            self._server_process = None
        
        # 等待接收线程退出
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        
        logger.info("scrcpy 会话已停止")
    
    def _recv_loop(self) -> None:
        """接收线程主循环"""
        logger.info("接收线程已启动")
        
        while self._running:
            try:
                # 从 video socket 读取数据
                data = self._video_socket.recv(65536)
                
                if not data:
                    logger.warning("Video socket 连接关闭")
                    break
                
                # 解码 H.264 帧
                frame = self._decode_frame(data)
                
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_count += 1
                        
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"接收循环异常: {e}")
                time.sleep(0.01)
        
        logger.info("接收线程退出")
    
    def _decode_frame(self, data: bytes) -> Optional[np.ndarray]:
        """
        解码 H.264 数据为 BGR 帧
        
        Args:
            data: H.264 编码数据
            
        Returns:
            BGR 格式的 numpy 数组，解码失败返回 None
        """
        if not AV_AVAILABLE or self._codec is None:
            return None
        
        try:
            # 解码 H.264 包
            packets = self._codec.parse(data)
            frames = []
            
            for packet in packets:
                decoded = self._codec.decode(packet)
                frames.extend(decoded)
            
            if not frames:
                return None
            
            # 取最后一帧
            frame = frames[-1]
            
            # 转换为 BGR numpy 数组
            if frame.format != "bgr24":
                frame = frame.reformat(format="bgr24")
            
            bgr_array = frame.to_ndarray()
            
            # 更新视频尺寸
            if self._video_width == 0:
                self._video_height, self._video_width = bgr_array.shape[:2]
                logger.info(f"视频尺寸: {self._video_width}x{self._video_height}")
            
            return bgr_array
            
        except Exception as e:
            logger.debug(f"解码帧失败: {e}")
            return None
    
    def recv_frame(self) -> Optional[np.ndarray]:
        """
        接收最新帧（供 FrameBufferManager 调用）
        
        Returns:
            最新的 BGR 帧，无帧时返回 None
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None
    
    @property
    def is_running(self) -> bool:
        """会话是否运行中"""
        return self._running
    
    @property
    def video_size(self) -> Tuple[int, int]:
        """视频尺寸 (width, height)"""
        return (self._video_width, self._video_height)
    
    @property
    def frame_count(self) -> int:
        """已接收帧数"""
        return self._frame_count
    
    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"ScrcpySession({status}, video={self._video_width}x{self._video_height})"
