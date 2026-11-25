"""交互式 ROI 框选 / 模板裁剪工具。

用法：
    python tools/crop_templates.py                     # 从 scrcpy 抓取当前画面
    python tools/crop_templates.py -i shot.png         # 使用已有截图
    python tools/crop_templates.py -i shot.png -o assets/templates

操作：
    鼠标拖拽框选 → 终端输入模板名 → 回车保存到输出目录
    框选窗口按 ESC 或直接回车（不拖拽）退出
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 状态分类器按这些名字查模板，命名对不上会一直识别为 UNKNOWN
SUGGESTED_NAMES = [
    "main_village_attack_btn",
    "attack_menu_find_match_btn",
    "army_setup_attack_btn",
    "find_match_next_btn",
    "find_match_end_battle_btn",
    "results_return_home",
    "battle_end_battle_btn",
    "popup_close",
]

WINDOW = "crop templates"


def capture_live_frame(serial: str | None, max_size: int) -> np.ndarray:
    """启动 scrcpy 抓一帧后关闭会话。"""
    from bot.device.frame_buffer import FrameBuffer
    from bot.device.scrcpy_session import ScrcpySession

    buffer = FrameBuffer(2)
    session = ScrcpySession(serial, max_size, 8_000_000, buffer)
    session.start()
    try:
        return session.wait_first_frame(timeout=30)
    finally:
        session.stop()


def draw_marks(img: np.ndarray, marks: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray:
    canvas = img.copy()
    for name, (x, y, w, h) in marks:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, name, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def select_roi(img: np.ndarray) -> tuple[int, int, int, int] | None:
    print("请在窗口中拖拽框选；按 ESC 或直接回车（不拖拽）退出")
    x, y, w, h = cv2.selectROI(WINDOW, img, showCrosshair=True, fromCenter=False)
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式框选并保存模板")
    parser.add_argument("-i", "--image", type=Path, help="已有截图路径；不填则从 scrcpy 实时抓取")
    parser.add_argument("-o", "--output", type=Path, default=Path("assets/templates"),
                        help="模板输出目录（默认 assets/templates）")
    parser.add_argument("--serial", default=None, help="设备序列号；不填取第一台")
    parser.add_argument("--max-size", type=int, default=1280, help="scrcpy 视频最大边长")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图片：{args.image}", file=sys.stderr)
            return 1
    else:
        print("正在从 scrcpy 抓取画面…")
        img = capture_live_frame(args.serial, args.max_size)

    print(f"画面尺寸：{img.shape[1]}x{img.shape[0]}")
    print("参考模板名（需与状态分类器一致）：")
    for n in SUGGESTED_NAMES:
        print(f"  - {n}")

    args.output.mkdir(parents=True, exist_ok=True)
    marks: list[tuple[str, tuple[int, int, int, int]]] = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            rect = select_roi(draw_marks(img, marks))
            if rect is None:
                break
            x, y, w, h = rect

            name = input("模板名称（留空跳过，q 退出）: ").strip()
            if name == "q":
                break
            if not name:
                continue
            if name.endswith(".png"):
                name = name[:-4]
            name = name.replace(" ", "_")

            crop = img[y:y + h, x:x + w]
            out_path = args.output / f"{name}.png"
            if not cv2.imwrite(str(out_path), crop):
                print(f"保存失败：{out_path}", file=sys.stderr)
                continue

            marks.append((name, rect))
            print(f"已保存 {out_path}  ROI=({x}, {y}, {w}, {h})")
    finally:
        cv2.destroyAllWindows()

    if marks:
        print("\n已生成的 ROI（可粘贴到 features 代码中）：")
        for name, (x, y, w, h) in marks:
            print(f"  {name}: ({x}, {y}, {w}, {h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
