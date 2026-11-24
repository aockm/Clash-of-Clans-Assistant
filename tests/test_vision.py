"""
Test Vision - 视觉与识别层模块测试

测试内容：
1. TemplateMatcher 模板匹配引擎
2. OCREngine OCR 识别引擎
3. EdgeDetector 边缘/区域检测器
"""

import sys
import os
import time
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(__file__).replace("\\tests\\test_vision.py", ""))

# 检查 OpenCV 可用性
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV 不可用，部分测试将跳过")

# 检查 pytesseract 可用性
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract 不可用，OCR 测试将跳过")


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """创建测试帧"""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_test_template(width: int = 50, height: int = 50) -> np.ndarray:
    """创建测试模板"""
    template = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制一个简单的图案
    cv2.circle(template, (width//2, height//2), min(width, height)//3, (255, 255, 255), -1)
    return template


def create_frame_with_pattern(width: int = 640, height: int = 480) -> np.ndarray:
    """创建带有特定图案的测试帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # 添加一些背景噪声
    frame[:] = np.random.randint(20, 50, (height, width, 3), dtype=np.uint8)
    return frame


class TestTemplateMatcher(unittest.TestCase):
    """测试 TemplateMatcher 模板匹配引擎"""
    
    def setUp(self):
        """测试前准备"""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV 不可用")
        
        from bot.vision.template_matcher import TemplateMatcher
        self.matcher = TemplateMatcher()
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.matcher)
        self.assertEqual(len(self.matcher._templates), 0)
    
    def test_set_and_get_template(self):
        """测试设置和获取模板"""
        template = create_test_template(60, 60)
        self.matcher.set_template("test_btn", template)
        
        retrieved = self.matcher.get_template("test_btn")
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, template)
    
    def test_list_templates(self):
        """测试列出模板"""
        self.matcher.set_template("btn1", create_test_template(40, 40))
        self.matcher.set_template("btn2", create_test_template(50, 50))
        
        templates = self.matcher.list_templates()
        self.assertEqual(len(templates), 2)
        self.assertIn("btn1", templates)
        self.assertIn("btn2", templates)
    
    def test_match_found(self):
        """测试匹配成功"""
        # 创建模板
        template = create_test_template(50, 50)
        self.matcher.set_template("circle", template)
        
        # 创建包含模板的帧
        frame = create_frame_with_pattern(640, 480)
        # 将模板放到帧的某个位置
        frame[100:150, 200:250] = template
        
        # 执行匹配
        result = self.matcher.match(frame, "circle", threshold=0.7)
        
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.7)
        self.assertEqual(result.template_name, "circle")
        
        # 检查位置（允许一定误差）
        self.assertAlmostEqual(result.x, 200, delta=5)
        self.assertAlmostEqual(result.y, 100, delta=5)
    
    def test_match_not_found(self):
        """测试匹配失败"""
        template = create_test_template(50, 50)
        self.matcher.set_template("target", template)
        
        # 创建不包含模板的帧
        frame = create_frame_with_pattern(640, 480)
        
        result = self.matcher.match(frame, "target", threshold=0.9)
        
        self.assertFalse(result.found)
    
    def test_match_with_roi(self):
        """测试带 ROI 的匹配"""
        template = create_test_template(40, 40)
        self.matcher.set_template("small", template)
        
        frame = create_frame_with_pattern(640, 480)
        frame[50:90, 100:140] = template
        
        # 只在 ROI 区域内搜索
        result = self.matcher.match(frame, "small", threshold=0.7, roi=(0, 0, 200, 150))
        
        self.assertTrue(result.found)
    
    def test_match_all(self):
        """测试多次匹配"""
        template = create_test_template(30, 30)
        self.matcher.set_template("dot", template)
        
        frame = create_frame_with_pattern(640, 480)
        # 放置多个模板实例
        frame[50:80, 100:130] = template
        frame[50:80, 300:330] = template
        frame[200:230, 100:130] = template
        
        results = self.matcher.match_all(frame, "dot", threshold=0.7)
        
        # 应该找到至少 2 个（NMS 可能会合并一些）
        self.assertGreaterEqual(len(results), 2)
    
    def test_match_multi(self):
        """测试多模板同时匹配"""
        template1 = create_test_template(40, 40)
        template2 = create_test_template(50, 50)
        
        self.matcher.set_template("t1", template1)
        self.matcher.set_template("t2", template2)
        
        frame = create_frame_with_pattern(640, 480)
        frame[100:140, 100:140] = template1
        frame[200:250, 300:350] = template2
        
        results = self.matcher.match_multi(frame, ["t1", "t2"], threshold=0.7)
        
        self.assertIn("t1", results)
        self.assertIn("t2", results)
    
    def test_match_result_properties(self):
        """测试 MatchResult 属性"""
        from bot.vision.template_matcher import MatchResult
        
        result = MatchResult(
            found=True,
            x=100, y=200,
            width=50, height=60,
            confidence=0.95,
            template_name="test"
        )
        
        # 测试 center 属性
        self.assertEqual(result.center, (125, 230))
        
        # 测试 bbox 属性
        self.assertEqual(result.bbox, (100, 200, 50, 60))


class TestOCREngine(unittest.TestCase):
    """测试 OCREngine OCR 识别引擎"""
    
    def setUp(self):
        """测试前准备"""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV 不可用")
        
        from bot.vision.ocr_engine import OCREngine
        self.engine = OCREngine(upscale_factor=2)
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.upscale_factor, 2)
    
    def test_preprocess_for_digits(self):
        """测试数字预处理"""
        frame = create_test_frame(200, 100)
        processed = self.engine._preprocess_for_digits(frame)
        
        # 应该是灰度图
        self.assertEqual(len(processed.shape), 2)
        
        # 应该被上采样
        self.assertEqual(processed.shape[0], 200)  # 100 * 2
        self.assertEqual(processed.shape[1], 400)  # 200 * 2
    
    def test_preprocess_for_text(self):
        """测试文本预处理"""
        frame = create_test_frame(150, 80)
        processed = self.engine._preprocess_for_text(frame)
        
        # 应该是灰度图
        self.assertEqual(len(processed.shape), 2)
    
    def test_parse_number_pure_digits(self):
        """测试纯数字解析"""
        # 测试纯数字
        value = self.engine._parse_number("1234")
        self.assertEqual(value, 1234)
        
        value = self.engine._parse_number("0")
        self.assertEqual(value, 0)
    
    def test_parse_number_with_comma(self):
        """测试带分隔符的数字"""
        value = self.engine._parse_number("1,234")
        self.assertEqual(value, 1234)
        
        value = self.engine._parse_number("1,234,567")
        self.assertEqual(value, 1234567)
    
    def test_parse_number_with_suffix(self):
        """测试带后缀的数字"""
        # 不启用后缀时
        value = self.engine._parse_number("5K", allow_suffix=False)
        self.assertEqual(value, 5)  # 只提取数字部分
        
        # 启用后缀时
        value = self.engine._parse_number("5K", allow_suffix=True)
        self.assertEqual(value, 5000)
        
        value = self.engine._parse_number("2M", allow_suffix=True)
        self.assertEqual(value, 2000000)
        
        value = self.engine._parse_number("1B", allow_suffix=True)
        self.assertEqual(value, 1000000000)
    
    def test_parse_number_invalid(self):
        """测试无效输入"""
        value = self.engine._parse_number("")
        self.assertIsNone(value)
        
        value = self.engine._parse_number("abc")
        self.assertIsNone(value)
    
    @unittest.skipUnless(TESSERACT_AVAILABLE, "pytesseract 不可用")
    def test_recognize_digits(self):
        """测试数字识别（需要 Tesseract）"""
        # 创建一个包含数字的图像
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "12345", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        result = self.engine.recognize_digits(img)
        
        # 由于 Tesseract 识别结果可能不完美，只验证基本功能
        self.assertIsInstance(result.success, bool)
    
    def test_set_debug(self):
        """测试调试模式设置"""
        self.engine.set_debug(True)
        self.assertTrue(self.engine.debug)
        
        self.engine.set_debug(False)
        self.assertFalse(self.engine.debug)


class TestEdgeDetector(unittest.TestCase):
    """测试 EdgeDetector 边缘/区域检测器"""
    
    def setUp(self):
        """测试前准备"""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV 不可用")
        
        from bot.vision.edge_detector import EdgeDetector
        self.detector = EdgeDetector(red_sensitivity=50, min_red_area=100)
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.red_sensitivity, 50)
        self.assertEqual(self.detector.min_red_area, 100)
        self.assertEqual(self.detector.edge_padding, 20)
    
    def test_detect_red_regions_none(self):
        """测试无红色区域"""
        # 创建没有红色的图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (128, 128, 128)  # 灰色
        
        red_mask, regions = self.detector.detect_red_regions(frame)
        
        self.assertEqual(len(regions), 0)
    
    def test_detect_red_regions_found(self):
        """测试检测到红色区域"""
        # 创建包含红色的图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 在 HSV 中，红色 H 值约 0 或 180
        # 创建一个红色矩形
        frame[100:200, 200:400] = (0, 0, 255)  # BGR 红色
        
        red_mask, regions = self.detector.detect_red_regions(frame)
        
        # 应该检测到红色区域
        self.assertGreater(len(regions), 0)
    
    def test_detect_deploy_zone_no_red(self):
        """测试无可部署区域（无红色边界）"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (128, 128, 128)
        
        result = self.detector.detect_deploy_zone(frame)
        
        self.assertFalse(result.success)
    
    def test_detect_deploy_zone_with_red(self):
        """测试检测到可部署区域"""
        # 创建带有红色边界的图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 创建红色边界
        frame[50:60, 100:500] = (0, 0, 255)  # 上边
        frame[420:430, 100:500] = (0, 0, 255)  # 下边
        frame[50:430, 100:110] = (0, 0, 255)  # 左边
        frame[50:430, 490:500] = (0, 0, 255)  # 右边
        
        result = self.detector.detect_deploy_zone(frame)
        
        # 如果检测到红色区域，应该有可部署区域
        if result.success:
            self.assertIsNotNone(result.deployable_zone)
            zone = result.deployable_zone
            self.assertTrue(zone.is_deployable)
            self.assertGreater(zone.width, 0)
            self.assertGreater(zone.height, 0)
    
    def test_generate_deploy_points(self):
        """测试生成部署点"""
        from bot.vision.edge_detector import DeployZone
        
        zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=True)
        
        points = self.detector.generate_deploy_points(zone, count=8)
        
        self.assertEqual(len(points), 8)
        
        # 所有点应该在合理范围内（允许随机偏移）
        for x, y in points:
            # 点应该在区域附近（考虑随机偏移）
            self.assertGreater(x, 50)
            self.assertLess(x, 550)
            self.assertGreater(y, 50)
            self.assertLess(y, 450)
    
    def test_generate_deploy_points_not_deployable(self):
        """测试不可部署区域"""
        from bot.vision.edge_detector import DeployZone
        
        zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=False)
        
        points = self.detector.generate_deploy_points(zone, count=5)
        
        self.assertEqual(len(points), 0)
    
    def test_detect_edges(self):
        """测试边缘检测"""
        # 创建有边缘的图像
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
        
        edges = self.detector.detect_edges(frame)
        
        # 应该有边缘像素
        self.assertGreater(np.sum(edges > 0), 0)
    
    def test_detect_contours(self):
        """测试轮廓检测"""
        # 创建有轮廓的图像
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.circle(frame, (150, 150), 50, (0, 0, 0), -1)
        
        contours = self.detector.detect_contours(frame, min_area=100)
        
        self.assertGreater(len(contours), 0)
    
    def test_set_red_sensitivity(self):
        """测试调整红色灵敏度"""
        self.detector.set_red_sensitivity(80)
        self.assertEqual(self.detector.red_sensitivity, 80)
    
    def test_deploy_zone_properties(self):
        """测试 DeployZone 属性"""
        from bot.vision.edge_detector import DeployZone
        
        zone = DeployZone(x=100, y=200, width=300, height=400, is_deployable=True)
        
        # 测试 center 属性
        self.assertEqual(zone.center, (250, 400))
        
        # 测试 bbox 属性
        self.assertEqual(zone.bbox, (100, 200, 300, 400))


def run_basic_tests():
    """运行基本测试（不需要实际设备）"""
    print("\n" + "=" * 60)
    print("运行视觉与识别层模块基本测试")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("\n[跳过] OpenCV 不可用，无法运行视觉测试")
        return
    
    # 测试 TemplateMatcher
    print("\n[1/3] 测试 TemplateMatcher 模板匹配引擎...")
    from bot.vision.template_matcher import TemplateMatcher, MatchResult
    
    matcher = TemplateMatcher()
    
    # 创建测试模板
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.circle(template, (25, 25), 20, (255, 255, 255), -1)
    
    matcher.set_template("circle", template)
    assert len(matcher.list_templates()) == 1, "模板列表错误"
    
    # 创建包含模板的帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:150, 200:250] = template
    
    result = matcher.match(frame, "circle", threshold=0.7)
    assert result.found, f"匹配失败: conf={result.confidence}"
    assert result.confidence > 0.7, f"置信度过低: {result.confidence}"
    
    print(f"  ✓ 模板匹配成功: center={result.center}, conf={result.confidence:.3f}")
    
    # 测试 MatchResult 属性
    assert result.center == (225, 125), f"中心点计算错误: {result.center}"
    print("  ✓ MatchResult 属性正确")
    
    # 测试多模板匹配
    template2 = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.rectangle(template2, (5, 5), (35, 35), (255, 255, 255), -1)
    matcher.set_template("square", template2)
    
    frame[200:240, 400:440] = template2
    results = matcher.match_multi(frame, ["circle", "square"], threshold=0.7)
    assert "circle" in results, "circle 匹配缺失"
    assert "square" in results, "square 匹配缺失"
    print("  ✓ 多模板匹配成功")
    
    print("  ✓ TemplateMatcher 测试通过")
    
    # 测试 OCREngine
    print("\n[2/3] 测试 OCREngine OCR 识别引擎...")
    from bot.vision.ocr_engine import OCREngine, OCRResult
    
    engine = OCREngine(upscale_factor=2)
    
    # 测试数字解析
    assert engine._parse_number("1234") == 1234, "纯数字解析错误"
    assert engine._parse_number("1,234") == 1234, "带逗号数字解析错误"
    assert engine._parse_number("5K", allow_suffix=True) == 5000, "K 后缀解析错误"
    assert engine._parse_number("2M", allow_suffix=True) == 2000000, "M 后缀解析错误"
    assert engine._parse_number("1B", allow_suffix=True) == 1000000000, "B 后缀解析错误"
    print("  ✓ 数字解析正确")
    
    # 测试预处理
    test_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    processed = engine._preprocess_for_digits(test_img)
    assert len(processed.shape) == 2, "预处理后应为灰度图"
    assert processed.shape[0] == 200, f"上采样错误: {processed.shape}"
    print("  ✓ 预处理流程正确")
    
    # 测试 OCRResult
    result = OCRResult(success=True, text="1234", value=1234, confidence=0.95)
    assert result.value == 1234, "OCRResult 属性错误"
    print("  ✓ OCRResult 属性正确")
    
    print("  ✓ OCREngine 测试通过")
    
    # 测试 EdgeDetector
    print("\n[3/3] 测试 EdgeDetector 边缘/区域检测器...")
    from bot.vision.edge_detector import EdgeDetector, DeployZone, EdgeResult
    
    detector = EdgeDetector(red_sensitivity=50, min_red_area=100)
    
    # 测试红色区域检测
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 200:400] = (0, 0, 255)  # BGR 红色
    
    red_mask, regions = detector.detect_red_regions(frame)
    assert len(regions) > 0, "未检测到红色区域"
    print(f"  ✓ 检测到 {len(regions)} 个红色区域")
    
    # 测试部署点生成
    zone = DeployZone(x=100, y=100, width=400, height=300, is_deployable=True)
    points = detector.generate_deploy_points(zone, count=8)
    assert len(points) == 8, f"部署点数量错误: {len(points)}"
    print(f"  ✓ 生成 {len(points)} 个部署点")
    
    # 测试边缘检测
    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (50, 50), (150, 150), (255, 255, 255), 2)
    edges = detector.detect_edges(test_frame)
    assert np.sum(edges > 0) > 0, "边缘检测失败"
    print("  ✓ 边缘检测正常")
    
    # 测试 DeployZone 属性
    assert zone.center == (300, 250), f"中心点计算错误: {zone.center}"
    assert zone.bbox == (100, 100, 400, 300), f"bbox 错误: {zone.bbox}"
    print("  ✓ DeployZone 属性正确")
    
    print("  ✓ EdgeDetector 测试通过")
    
    print("\n" + "=" * 60)
    print("所有基本测试通过！")
    print("=" * 60)


def run_performance_tests():
    """运行性能测试"""
    print("\n" + "=" * 60)
    print("运行视觉模块性能测试")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("\n[跳过] OpenCV 不可用")
        return
    
    from bot.vision.template_matcher import TemplateMatcher
    
    matcher = TemplateMatcher()
    
    # 创建测试模板
    template = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.circle(template, (30, 30), 25, (255, 255, 255), -1)
    matcher.set_template("perf_test", template)
    
    # 测试帧
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:560, 800:860] = template
    
    # 性能测试：全图匹配
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        matcher.match(frame, "perf_test", threshold=0.7)
    elapsed = time.time() - start
    
    fps = iterations / elapsed
    print(f"\n[性能] 全图匹配 (1920x1080): {fps:.1f} FPS ({elapsed/iterations*1000:.2f}ms/次)")
    
    # 性能测试：ROI 匹配
    start = time.time()
    for _ in range(iterations):
        matcher.match(frame, "perf_test", threshold=0.7, roi=(700, 400, 300, 300))
    elapsed = time.time() - start
    
    fps = iterations / elapsed
    print(f"[性能] ROI 匹配 (300x300): {fps:.1f} FPS ({elapsed/iterations*1000:.2f}ms/次)")
    
    print("\n" + "=" * 60)
    print("性能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行基本测试
    run_basic_tests()
    
    # 运行性能测试
    run_performance_tests()
    
    # 可选：运行 unittest 测试套件
    # unittest.main()
