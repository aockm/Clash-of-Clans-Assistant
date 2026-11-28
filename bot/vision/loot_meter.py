import cv2
import numpy as np


class LootMeter:
    """读取主村庄右上角资源条的填充比例（视频坐标）。

    资源条右端固定、填充从右向左增长；数字文字把中间遮住，
    因此取条带上/下边缘行（文字不覆盖）判断填充到哪一列。
    """

    RIGHT = 1192          # 资源条右端（金币图标左侧）
    LEFT = 1017           # 资源条左端
    WIDTH = RIGHT - LEFT
    GOLD_ROWS = range(18, 30)
    ELIXIR_ROWS = range(54, 86)

    def ratios(self, frame: np.ndarray) -> tuple[float, float]:
        gold = self._ratio(frame, self.GOLD_ROWS, self._gold_mask)
        elixir = self._ratio(frame, self.ELIXIR_ROWS, self._elixir_mask)
        return gold, elixir

    def is_full(self, frame: np.ndarray, threshold: float = 0.95) -> bool:
        gold, elixir = self.ratios(frame)
        return gold >= threshold and elixir >= threshold

    def _ratio(self, frame: np.ndarray, rows, mask_fn) -> float:
        img = frame.astype(np.int16)
        best = 0
        for y in rows:
            if y >= img.shape[0]:
                continue
            row = img[y, self.LEFT:self.RIGHT]
            mask = mask_fn(row[:, 2], row[:, 1], row[:, 0])
            start = self._rightmost_run_start(mask)
            if start is None:
                continue
            length = self.WIDTH - start
            best = max(best, length)
        return best / self.WIDTH

    def _rightmost_run_start(self, mask: np.ndarray) -> int | None:
        """从右向左找连续填充段（允许少量空隙），返回段起点；无填充返回 None。"""
        if not mask[-1] and not mask[-3:].any():
            return None
        start, gap = self.WIDTH, 0
        for i in range(self.WIDTH - 1, -1, -1):
            if mask[i]:
                start, gap = i, 0
            else:
                gap += 1
                if gap > 4:
                    break
        return start

    @staticmethod
    def _gold_mask(r, g, b):
        # 高亮金黄填充（空格子被半透明压暗后偏灰/土黄，会被排除）
        return (b < 135) & (g > 185) & (r > 215)

    @staticmethod
    def _elixir_mask(r, g, b):
        # 紫色/品红填充（含较暗的深紫）
        return (b > 95) & (r > 95) & (g < 75)
