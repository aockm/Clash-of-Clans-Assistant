"""
OCR Engine - OCR 识别引擎

使用 Tesseract 读取资源数值、工人状态、升级费用等。
配合 --psm 7（单行模式）和 tessedit_char_whitelist=0123456789 提升数字识别率。
预处理顺序：灰度 → 掩码提取 → 形态学清理 → 上采样 → 二值化。
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR 识别结果"""
    success: bool
    text: str = ""
    value: Optional[int] = None
    confidence: float = 0.0
    raw_output: str = ""
    
    def __repr__(self) -> str:
        if self.success:
            return f"OCRResult(text='{self.text}', value={self.value}, conf={self.confidence:.2f})"
        return "OCRResult(success=False)"


class OCREngine:
    """
    OCR 识别引擎
    
    针对 COC 游戏中的数字识别优化：
    - 资源数值（金币、圣水、暗黑重油）
    - 升级费用
    - 工人状态
    - 时间显示
    
    预处理流程：灰度 → 掩码提取 → 形态学清理 → 上采样 → 二值化
    """
    
    # 预设的 Tesseract 配置
    CONFIG_DIGITS = "--psm 7 -c tessedit_char_whitelist=0123456789"
    CONFIG_DIGITS_K = "--psm 7 -c tessedit_char_whitelist=0123456789KkMmBb"  # 支持 K/M/B 后缀
    CONFIG_TIME = "--psm 7 -c tessedit_char_whitelist=0123456789mdh"  # 时间格式
    CONFIG_TEXT = "--psm 7"
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        upscale_factor: int = 3,
        debug: bool = False
    ):
        """
        初始化 OCR 引擎
        
        Args:
            tesseract_cmd: Tesseract 可执行文件路径
            upscale_factor: 上采样倍数（小文字识别时放大）
            debug: 是否保存预处理中间图像用于调试
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV 不可用")
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract 不可用，OCR 功能将受限")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.upscale_factor = upscale_factor
        self.debug = debug
        self._debug_counter = 0
        
        logger.info(f"OCREngine 初始化，上采样倍数: {upscale_factor}")
    
    def recognize_digits(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        allow_suffix: bool = False
    ) -> OCRResult:
        """
        识别数字（资源数值、费用等）
        
        Args:
            frame: 输入帧（BGR 格式）
            roi: 感兴趣区域 (x, y, width, height)
            allow_suffix: 是否允许 K/M/B 后缀
            
        Returns:
            OCRResult 识别结果
        """
        # 提取 ROI
        if roi:
            x, y, w, h = roi
            img = frame[y:y+h, x:x+w].copy()
        else:
            img = frame.copy()
        
        # 预处理
        processed = self._preprocess_for_digits(img)
        
        # 选择配置
        config = self.CONFIG_DIGITS_K if allow_suffix else self.CONFIG_DIGITS
        
        # 执行 OCR
        return self._execute_ocr(processed, config, parse_digits=True, allow_suffix=allow_suffix)
    
    def recognize_badge_count(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> Optional[int]:
        """
        识别兵种卡数量角标（深色底白字小数字）

        Args:
            frame: 输入帧（BGR）
            roi: 角标区域 (x, y, width, height)

        Returns:
            识别到的数量，无数字返回 None
        """
        if not TESSERACT_AVAILABLE:
            return None
        x, y, w, h = roi
        img = frame[y:y+h, x:x+w]
        if img.size == 0:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(big, 160, 255, cv2.THRESH_BINARY)
        try:
            text = pytesseract.image_to_string(
                binary,
                config="--psm 7 -c tessedit_char_whitelist=0123456789"
            ).strip()
        except Exception as e:
            logger.debug(f"角标识别失败: {e}")
            return None
        return int(text) if text.isdigit() else None

    def recognize_text(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> OCRResult:
        """
        识别通用文本
        
        Args:
            frame: 输入帧
            roi: 感兴趣区域
            
        Returns:
            OCRResult 识别结果
        """
        if roi:
            x, y, w, h = roi
            img = frame[y:y+h, x:x+w].copy()
        else:
            img = frame.copy()
        
        # 预处理
        processed = self._preprocess_for_text(img)
        
        # 执行 OCR
        return self._execute_ocr(processed, self.CONFIG_TEXT, parse_digits=False)
    
    def recognize_time(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> OCRResult:
        """
        识别时间显示（如 "1d 2h", "30m"）
        
        Args:
            frame: 输入帧
            roi: 感兴趣区域
            
        Returns:
            OCRResult 识别结果
        """
        if roi:
            x, y, w, h = roi
            img = frame[y:y+h, x:x+w].copy()
        else:
            img = frame.copy()
        
        processed = self._preprocess_for_text(img)
        return self._execute_ocr(processed, self.CONFIG_TIME, parse_digits=False)
    
    def _preprocess_for_digits(self, img: np.ndarray) -> np.ndarray:
        """
        数字识别预处理流程
        
        灰度 → 掩码提取 → 形态学清理 → 上采样 → 二值化
        """
        # 1. 转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 2. 自适应二值化（处理不同背景）
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. 形态学清理（去除噪点）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 4. 上采样（提高小文字识别率）
        if self.upscale_factor > 1:
            h, w = cleaned.shape[:2]
            cleaned = cv2.resize(
                cleaned, 
                (w * self.upscale_factor, h * self.upscale_factor),
                interpolation=cv2.INTER_CUBIC
            )
        
        # 5. 保存调试图像
        if self.debug:
            self._save_debug(cleaned, "digits_preprocess")
        
        return cleaned
    
    def _preprocess_for_text(self, img: np.ndarray) -> np.ndarray:
        """文本识别预处理流程"""
        # 转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 上采样
        if self.upscale_factor > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(
                gray,
                (w * self.upscale_factor, h * self.upscale_factor),
                interpolation=cv2.INTER_CUBIC
            )
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if self.debug:
            self._save_debug(binary, "text_preprocess")
        
        return binary
    
    def _execute_ocr(
        self,
        processed: np.ndarray,
        config: str,
        parse_digits: bool = False,
        allow_suffix: bool = False
    ) -> OCRResult:
        """
        执行 OCR 识别
        
        Args:
            processed: 预处理后的图像
            config: Tesseract 配置
            parse_digits: 是否解析为数字
            allow_suffix: 是否允许 K/M/B 后缀
            
        Returns:
            OCRResult 识别结果
        """
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract 不可用")
            return OCRResult(success=False)
        
        try:
            # 获取详细输出（含置信度）
            data = pytesseract.image_to_data(
                processed,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # 提取文本和置信度
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                text = text.strip()
                if text:
                    conf = int(data['conf'][i])
                    if conf > 0:  # 过滤无效结果
                        texts.append(text)
                        confidences.append(conf)
            
            raw_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 解析数字
            value = None
            if parse_digits and raw_text:
                value = self._parse_number(raw_text, allow_suffix)
            
            success = len(raw_text) > 0
            
            return OCRResult(
                success=success,
                text=raw_text,
                value=value,
                confidence=avg_conf / 100.0,  # 转为 0-1
                raw_output=str(data)
            )
            
        except Exception as e:
            logger.error(f"OCR 执行失败: {e}")
            return OCRResult(success=False)
    
    def _parse_number(self, text: str, allow_suffix: bool = False) -> Optional[int]:
        """
        解析数字文本
        
        支持格式：
        - 纯数字: "1234"
        - 带后缀: "1K", "2M", "3B"
        - 带分隔符: "1,234", "1 234"
        
        Args:
            text: 输入文本
            allow_suffix: 是否允许 K/M/B 后缀
            
        Returns:
            解析的整数值，失败返回 None
        """
        # 清理文本
        text = text.strip().replace(",", "").replace(" ", "")
        
        if not text:
            return None
        
        # 处理后缀
        multiplier = 1
        if allow_suffix:
            text_upper = text.upper()
            if text_upper.endswith('K'):
                multiplier = 1000
                text = text[:-1]
            elif text_upper.endswith('M'):
                multiplier = 1000000
                text = text[:-1]
            elif text_upper.endswith('B'):
                multiplier = 1000000000
                text = text[:-1]
        
        # 提取数字部分
        digits = ""
        for char in text:
            if char.isdigit() or char == '.':
                digits += char
        
        if not digits:
            return None
        
        try:
            value = float(digits) * multiplier
            return int(value)
        except ValueError:
            return None
    
    def _save_debug(self, img: np.ndarray, prefix: str) -> None:
        """保存调试图像"""
        self._debug_counter += 1
        filename = f"debug_{prefix}_{self._debug_counter:04d}.png"
        try:
            cv2.imwrite(filename, img)
            logger.debug(f"保存调试图像: {filename}")
        except Exception as e:
            logger.debug(f"保存调试图像失败: {e}")
    
    def set_debug(self, enabled: bool) -> None:
        """启用/禁用调试模式"""
        self.debug = enabled
        logger.info(f"OCR 调试模式: {'启用' if enabled else '禁用'}")
    
    def __repr__(self) -> str:
        return f"OCREngine(upscale={self.upscale_factor}, debug={self.debug})"
