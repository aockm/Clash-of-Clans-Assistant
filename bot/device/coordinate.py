from dataclasses import dataclass


@dataclass
class CoordinateMapper:
    """视频帧坐标空间 ↔ 设备像素坐标空间的映射。"""

    video_size: tuple[int, int]
    device_size: tuple[int, int]

    @classmethod
    def from_device_size(cls, device_size: tuple[int, int],
                         max_size: int | None = None) -> "CoordinateMapper":
        """按 scrcpy --max-size 规则推算视频编码尺寸（等比缩放，长边不超过 max_size）。"""
        w, h = device_size
        if max_size and max(w, h) > max_size:
            scale = max_size / max(w, h)
            w, h = int(w * scale), int(h * scale)
        return cls(video_size=(w, h), device_size=device_size)

    @property
    def scale_x(self) -> float:
        return self.device_size[0] / self.video_size[0]

    @property
    def scale_y(self) -> float:
        return self.device_size[1] / self.video_size[1]

    def to_device(self, x: int, y: int) -> tuple[int, int]:
        return int(x * self.scale_x), int(y * self.scale_y)

    def to_video(self, x: int, y: int) -> tuple[int, int]:
        return int(x / self.scale_x), int(y / self.scale_y)
