import logging
import time
from pathlib import Path

RUN_LOG_DIR = Path("logs")
FILE_FORMAT = ("%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] "
               "%(funcName)s:%(lineno)d | %(message)s")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def start_run_log(console_level: int = logging.INFO) -> Path:
    """为本次运行新建一个日志文件，只记录本次运行（每次启动一个文件）。

    文件里固定记录 DEBUG 级全部细节（含各层内部判断），控制台仍按配置级别输出。
    """
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
    root.addHandler(handler)

    # 已有的控制台 handler 压低到配置级别，避免控制台被 DEBUG 刷满
    for existing in root.handlers:
        if existing is not handler and not isinstance(existing,
                                                      logging.FileHandler):
            existing.setLevel(console_level)

    # 第三方库的 DEBUG 只会淹没现场，压到 INFO；本项目各模块保持 DEBUG
    for noisy in ("PIL", "pytesseract", "adbutils", "redis", "urllib3",
                  "wetest"):
        logging.getLogger(noisy).setLevel(logging.INFO)
    return path
