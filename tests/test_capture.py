"""
Test Capture - 图像采集模块测试

测试内容：
1. FrameBuffer 环形缓冲区功能
2. FrameBufferManager 流管理
3. CoordinateMapper 坐标映射
4. ADBManager 设备连接
5. FallbackCapture 后备采集
6. ScrcpySession 会话管理（需要实际设备）
"""

import sys
import time
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

import numpy as np

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(__file__).replace("\\tests\\test_capture.py", ""))

from bot.device.frame_buffer import FrameBuffer, FrameBufferManager
from bot.device.coordinate import CoordinateMapper
from bot.device.adb_manager import ADBManager
from bot.device.fallback_capture import FallbackCapture
from bot.device.scrcpy_session import ScrcpySession


class TestFrameBuffer(unittest.TestCase):
    """测试 FrameBuffer 环形缓冲区"""
    
    def setUp(self):
        """测试前准备"""
        self.buffer = FrameBuffer(max_size=5)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.buffer.max_size, 5)
        self.assertTrue(self.buffer.is_empty())
        self.assertEqual(self.buffer.frame_count, 0)
    
    def test_put_and_get(self):
        """测试写入和读取帧"""
        # 创建测试帧
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 写入帧
        self.buffer.put(frame1)
        self.assertFalse(self.buffer.is_empty())
        self.assertEqual(self.buffer.buffer_size, 1)
        
        self.buffer.put(frame2)
        self.assertEqual(self.buffer.buffer_size, 2)
        
        # 获取最新帧
        latest = self.buffer.get_latest()
        self.assertIsNotNone(latest)
        np.testing.assert_array_equal(latest, frame2)
    
    def test_buffer_overflow(self):
        """测试缓冲区溢出处理"""
        # 写入超过容量的帧
        for i in range(10):
            frame = np.full((100, 100, 3), i, dtype=np.uint8)
            self.buffer.put(frame)
        
        # 缓冲区应该保持在最大容量
        self.assertEqual(self.buffer.buffer_size, 5)
        
        # 最新帧应该是最后写入的
        latest = self.buffer.get_latest()
        self.assertEqual(latest[0, 0, 0], 9)
    
    def test_get_latest_timeout(self):
        """测试带超时的获取"""
        # 空缓冲区，立即返回 None
        result = self.buffer.get_latest(timeout=0.1)
        self.assertIsNone(result)
    
    def test_clear(self):
        """测试清空缓冲区"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.buffer.put(frame)
        self.assertFalse(self.buffer.is_empty())
        
        self.buffer.clear()
        self.assertTrue(self.buffer.is_empty())
        self.assertEqual(self.buffer.buffer_size, 0)
    
    def test_get_all(self):
        """测试获取所有帧"""
        frames = [np.full((100, 100, 3), i, dtype=np.uint8) for i in range(3)]
        for f in frames:
            self.buffer.put(f)
        
        all_frames = self.buffer.get_all()
        self.assertEqual(len(all_frames), 3)


class TestFrameBufferManager(unittest.TestCase):
    """测试 FrameBufferManager 流管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.manager = FrameBufferManager(buffer_size=10, target_fps=30)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.manager.target_fps, 30)
        self.assertFalse(self.manager.is_running)
        self.assertEqual(self.manager.decoded_frames, 0)
    
    def test_set_source(self):
        """测试设置视频源"""
        mock_source = Mock()
        self.manager.set_source(mock_source)
        self.assertIsNotNone(self.manager._source)
    
    def test_start_without_source(self):
        """测试无视频源时启动"""
        result = self.manager.start()
        self.assertFalse(result)
    
    def test_start_with_mock_source(self):
        """测试使用模拟视频源启动"""
        # 创建模拟视频源
        mock_source = Mock()
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_source.recv_frame.return_value = mock_frame
        
        self.manager.set_source(mock_source)
        
        # 启动管理器
        result = self.manager.start()
        self.assertTrue(result)
        self.assertTrue(self.manager.is_running)
        
        # 等待几帧
        time.sleep(0.5)
        
        # 停止
        self.manager.stop()
        self.assertFalse(self.manager.is_running)
        
        # 验证帧已接收
        self.assertGreater(self.manager.decoded_frames, 0)
    
    def test_wait_for_first_frame(self):
        """测试等待第一帧"""
        mock_source = Mock()
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_source.recv_frame.return_value = mock_frame
        
        self.manager.set_source(mock_source)
        self.manager.start()
        
        # 等待第一帧
        result = self.manager.wait_for_first_frame(timeout=2.0)
        self.assertTrue(result)
        
        self.manager.stop()
    
    def test_frame_dimensions(self):
        """测试帧尺寸检测"""
        mock_source = Mock()
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_source.recv_frame.return_value = mock_frame
        
        self.manager.set_source(mock_source)
        self.manager.start()
        
        # 等待帧处理
        time.sleep(0.3)
        
        # 检查帧尺寸
        dimensions = self.manager.frame_dimensions
        self.assertIsNotNone(dimensions)
        self.assertEqual(dimensions, (640, 480))
        
        self.manager.stop()


class TestCoordinateMapper(unittest.TestCase):
    """测试 CoordinateMapper 坐标映射器"""
    
    def setUp(self):
        """测试前准备"""
        self.mapper = CoordinateMapper(
            video_width=800,
            video_height=600,
            device_width=1080,
            device_height=1920
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.mapper.video_width, 800)
        self.assertEqual(self.mapper.video_height, 600)
        self.assertEqual(self.mapper.device_width, 1080)
        self.assertEqual(self.mapper.device_height, 1920)
    
    def test_video_to_device(self):
        """测试视频坐标转设备坐标"""
        # 中心点
        x, y = self.mapper.video_to_device(400, 300)
        self.assertEqual(x, 540)
        self.assertEqual(y, 960)
        
        # 原点
        x, y = self.mapper.video_to_device(0, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        
        # 右下角
        x, y = self.mapper.video_to_device(800, 600)
        self.assertEqual(x, 1079)  # 边界裁剪
        self.assertEqual(y, 1919)
    
    def test_device_to_video(self):
        """测试设备坐标转视频坐标"""
        # 中心点
        x, y = self.mapper.device_to_video(540, 960)
        self.assertEqual(x, 400)
        self.assertEqual(y, 300)
    
    def test_scale_factors(self):
        """测试缩放因子"""
        scale_x, scale_y = self.mapper.get_scale_factors()
        self.assertAlmostEqual(scale_x, 1.35)
        self.assertAlmostEqual(scale_y, 3.2)
    
    def test_update_video_size(self):
        """测试更新视频尺寸"""
        self.mapper.update_video_size(1024, 768)
        self.assertEqual(self.mapper.video_width, 1024)
        self.assertEqual(self.mapper.video_height, 768)
        
        # 缩放因子应该更新
        scale_x, scale_y = self.mapper.get_scale_factors()
        self.assertAlmostEqual(scale_x, 1080 / 1024)
        self.assertAlmostEqual(scale_y, 1920 / 768)


class TestADBManager(unittest.TestCase):
    """测试 ADBManager ADB 管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.manager = ADBManager()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.manager.adb_path, "adb")
        self.assertIsNone(self.manager.device_serial)
        self.assertFalse(self.manager.connected)
    
    @patch("subprocess.run")
    def test_list_devices(self, mock_run):
        """测试列出设备"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="List of devices attached\nABC123\tdevice\nDEF456\tdevice\n"
        )
        
        devices = self.manager.list_devices()
        self.assertEqual(len(devices), 2)
        self.assertIn("ABC123", devices)
        self.assertIn("DEF456", devices)
    
    @patch("subprocess.run")
    def test_connect(self, mock_run):
        """测试连接设备"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="List of devices attached\nABC123\tdevice\n"
        )
        
        result = self.manager.connect()
        self.assertTrue(result)
        self.assertTrue(self.manager.connected)
        self.assertEqual(self.manager.device_serial, "ABC123")
    
    @patch("subprocess.run")
    def test_get_screen_size(self, mock_run):
        """测试获取屏幕尺寸"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Physical size: 1080x1920"
        )
        
        size = self.manager.get_screen_size()
        self.assertIsNotNone(size)
        self.assertEqual(size, (1080, 1920))
    
    @patch("subprocess.run")
    def test_tap(self, mock_run):
        """测试点击操作"""
        mock_run.return_value = Mock(returncode=0, stdout="")
        
        result = self.manager.tap(100, 200)
        self.assertTrue(result)


class TestFallbackCapture(unittest.TestCase):
    """测试 FallbackCapture 后备采集器"""
    
    def setUp(self):
        """测试前准备"""
        self.capture = FallbackCapture()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.capture.adb_path, "adb")
        self.assertIsNone(self.capture.device_serial)
        self.assertEqual(self.capture._frame_interval, 0.5)
    
    def test_set_device(self):
        """测试设置设备"""
        self.capture.set_device("ABC123")
        self.assertEqual(self.capture.device_serial, "ABC123")
    
    def test_set_frame_interval(self):
        """测试设置采集间隔"""
        self.capture.set_frame_interval(1.0)
        self.assertEqual(self.capture._frame_interval, 1.0)
        
        # 测试最小值限制
        self.capture.set_frame_interval(0.01)
        self.assertEqual(self.capture._frame_interval, 0.1)


class TestScrcpySession(unittest.TestCase):
    """测试 ScrcpySession scrcpy 会话管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.session = ScrcpySession()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.session.adb_path, "adb")
        self.assertIsNone(self.session.device_serial)
        self.assertEqual(self.session.max_size, 800)
        self.assertFalse(self.session._running)
    
    def test_default_parameters(self):
        """测试默认参数"""
        self.assertEqual(self.session.DEFAULT_MAX_SIZE, 800)
        self.assertEqual(self.session.DEFAULT_BIT_RATE, 4000000)
        self.assertEqual(self.session.DEFAULT_FRAME_RATE, 15)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_frame_buffer_with_manager(self):
        """测试帧缓冲区与管理器集成"""
        # 创建管理器
        manager = FrameBufferManager(buffer_size=5)
        
        # 创建模拟视频源
        mock_source = Mock()
        frame_counter = [0]
        
        def generate_frame():
            frame = np.full((480, 640, 3), frame_counter[0] % 255, dtype=np.uint8)
            frame_counter[0] += 1
            time.sleep(0.05)  # 模拟 20fps
            return frame
        
        mock_source.recv_frame.side_effect = generate_frame
        
        # 设置视频源并启动
        manager.set_source(mock_source)
        manager.start()
        
        # 等待几帧
        time.sleep(0.5)
        
        # 验证帧已接收
        self.assertGreater(manager.decoded_frames, 0)
        self.assertIsNotNone(manager.frame_dimensions)
        
        # 获取最新帧
        frame = manager.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
        # 停止
        manager.stop()
    
    def test_coordinate_mapping_workflow(self):
        """测试坐标映射工作流"""
        # 模拟场景：视频 800x600，设备 1080x1920
        mapper = CoordinateMapper(800, 600, 1080, 1920)
        
        # 视觉层检测到视频坐标 (400, 300)
        video_x, video_y = 400, 300
        
        # 转换为设备坐标用于 ADB 点击
        device_x, device_y = mapper.video_to_device(video_x, video_y)
        
        # 验证转换正确
        self.assertEqual(device_x, 540)
        self.assertEqual(device_y, 960)
        
        # 反向转换验证
        back_x, back_y = mapper.device_to_video(device_x, device_y)
        self.assertEqual(back_x, video_x)
        self.assertEqual(back_y, video_y)


def run_basic_tests():
    """运行基本测试（不需要实际设备）"""
    print("\n" + "=" * 60)
    print("运行图像采集模块基本测试")
    print("=" * 60)
    
    # 测试 FrameBuffer
    print("\n[1/6] 测试 FrameBuffer 环形缓冲区...")
    buffer = FrameBuffer(max_size=5)
    
    # 写入测试帧
    for i in range(3):
        frame = np.full((100, 100, 3), i * 50, dtype=np.uint8)
        buffer.put(frame)
    
    assert buffer.buffer_size == 3, f"缓冲区大小错误: {buffer.buffer_size}"
    assert buffer.frame_count == 3, f"帧计数错误: {buffer.frame_count}"
    
    # 获取最新帧
    latest = buffer.get_latest()
    assert latest is not None, "获取帧失败"
    assert latest[0, 0, 0] == 100, f"帧值错误: {latest[0, 0, 0]}"
    
    print("  ✓ FrameBuffer 测试通过")
    
    # 测试 CoordinateMapper
    print("\n[2/6] 测试 CoordinateMapper 坐标映射器...")
    mapper = CoordinateMapper(800, 600, 1080, 1920)
    
    # 视频坐标转设备坐标
    device_x, device_y = mapper.video_to_device(400, 300)
    assert device_x == 540, f"X 坐标转换错误: {device_x}"
    assert device_y == 960, f"Y 坐标转换错误: {device_y}"
    
    # 设备坐标转视频坐标
    video_x, video_y = mapper.device_to_video(540, 960)
    assert video_x == 400, f"反向 X 转换错误: {video_x}"
    assert video_y == 300, f"反向 Y 转换错误: {video_y}"
    
    print("  ✓ CoordinateMapper 测试通过")
    
    # 测试 FrameBufferManager
    print("\n[3/6] 测试 FrameBufferManager 流管理器...")
    manager = FrameBufferManager(buffer_size=10)
    
    # 创建模拟视频源
    mock_source = Mock()
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_source.recv_frame.return_value = mock_frame
    
    manager.set_source(mock_source)
    assert manager.start(), "启动管理器失败"
    
    # 等待帧处理
    time.sleep(0.3)
    
    assert manager.decoded_frames > 0, "未接收到帧"
    assert manager.frame_dimensions == (640, 480), f"帧尺寸错误: {manager.frame_dimensions}"
    
    manager.stop()
    assert not manager.is_running, "管理器未停止"
    
    print("  ✓ FrameBufferManager 测试通过")
    
    # 测试 ADBManager（模拟）
    print("\n[4/6] 测试 ADBManager ADB 管理器...")
    adb = ADBManager()
    assert adb.adb_path == "adb", "ADB 路径错误"
    assert not adb.connected, "初始状态应为未连接"
    
    print("  ✓ ADBManager 测试通过")
    
    # 测试 FallbackCapture
    print("\n[5/6] 测试 FallbackCapture 后备采集器...")
    fallback = FallbackCapture()
    fallback.set_device("TEST123")
    assert fallback.device_serial == "TEST123", "设备设置失败"
    
    fallback.set_frame_interval(0.5)
    assert fallback._frame_interval == 0.5, "采集间隔设置失败"
    
    print("  ✓ FallbackCapture 测试通过")
    
    # 测试 ScrcpySession
    print("\n[6/6] 测试 ScrcpySession scrcpy 会话...")
    session = ScrcpySession()
    assert session.max_size == 800, "默认参数错误"
    assert not session._running, "初始状态应为未运行"
    
    print("  ✓ ScrcpySession 测试通过")
    
    print("\n" + "=" * 60)
    print("所有基本测试通过！")
    print("=" * 60)


def run_integration_tests():
    """运行集成测试"""
    print("\n" + "=" * 60)
    print("运行图像采集模块集成测试")
    print("=" * 60)
    
    # 测试完整工作流
    print("\n[集成测试] 完整采集工作流...")
    
    # 1. 创建帧缓冲管理器
    manager = FrameBufferManager(buffer_size=10)
    
    # 2. 创建模拟视频源
    class MockVideoSource:
        def __init__(self):
            self.frame_count = 0
        
        def recv_frame(self):
            # 生成测试帧
            frame = np.full((480, 640, 3), self.frame_count % 255, dtype=np.uint8)
            self.frame_count += 1
            time.sleep(0.033)  # 模拟 30fps
            return frame
    
    source = MockVideoSource()
    manager.set_source(source)
    
    # 3. 启动采集
    assert manager.start(), "启动采集失败"
    
    # 4. 等待第一帧
    assert manager.wait_for_first_frame(timeout=2.0), "等待第一帧超时"
    
    # 5. 持续采集一段时间
    time.sleep(1.0)
    
    # 6. 验证统计信息
    print(f"  已解码帧数: {manager.decoded_frames}")
    print(f"  当前帧率: {manager.fps:.2f} FPS")
    print(f"  帧尺寸: {manager.frame_dimensions}")
    
    assert manager.decoded_frames > 10, f"解码帧数过少: {manager.decoded_frames}"
    assert manager.fps > 10, f"帧率过低: {manager.fps}"
    
    # 7. 获取最新帧
    frame = manager.get_latest_frame()
    assert frame is not None, "获取最新帧失败"
    assert frame.shape == (480, 640, 3), f"帧形状错误: {frame.shape}"
    
    # 8. 停止采集
    manager.stop()
    
    print("  ✓ 集成测试通过")
    
    print("\n" + "=" * 60)
    print("集成测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行基本测试
    run_basic_tests()
    
    # 运行集成测试
    run_integration_tests()
    
    # 可选：运行 unittest 测试套件
    # unittest.main()
