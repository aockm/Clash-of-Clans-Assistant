# Clash-of-Clans-Assistant

基于 ADB 与计算机视觉的《部落冲突》自动化助手，无需 Root，兼容主流安卓真机与模拟器。

## 环境要求

- Python 3.10+
- Android 设备或模拟器（已开启 USB 调试 / ADB 调试）
- [adb](https://developer.android.com/tools/releases/platform-tools)（`adb devices` 能列出设备）
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - 必装：英文语言包 `eng`
  - 可选：中文语言包 `chi_sim`（用于识别建筑剩余升级时间，缺失时该功能自动跳过）
- Redis（可选，未启动时自动降级为进程内内存存储）

## 安装

```bash
cd Clash-of-Clans-Assistant
python3 -m venv .venv
source .venv/bin/activate
pip install adbutils opencv-python numpy pytesseract pyyaml redis av wetest-scrcpy
```

> 依赖清单见 `pyproject.toml`。`assets/scrcpy/scrcpy-server` 会在首次运行时自动补齐到 `wetest-scrcpy` 包目录，无需手动处理。

## 配置

编辑 `config/bot.yaml`：

| 配置项 | 说明 |
| --- | --- |
| `mode` | `run` 正常运行；`debug` 输出 DEBUG 日志并把画面逐帧存到 `screenshots/debug/` |
| `device.serial` | 设备序列号，`null` 表示自动取第一台设备 |
| `accounts.names` | 按顺序轮换的 Supercell ID 账号名 |
| `accounts.attacks_per_account` | 每个账号打满几场后切换 |
| `schedule` | 定时启动：`interval_minutes` 启动点间隔（从 0 点起算），`day_offset` 是否再偏移“当天日期 % 10”分钟 |
| `redis` | 账号状态存储，`enabled: false` 可关闭 |

## 启动

```bash
cd Clash-of-Clans-Assistant
source .venv/bin/activate
python main.py
```

- 首次启动立即执行一轮，之后按 `schedule` 配置的启动点自动进入下一轮。
- `debug` 模式：把 `config/bot.yaml` 顶部改为 `mode: debug`，日志更详细，画面截图保存在 `screenshots/debug/<时间戳>/`。

## 说明

- 游戏内建议使用横屏，首次运行前确认设备已连接：`adb devices`。
- 运行期间会轮换 `accounts.names` 中的账号；金币和圣水都达到 `accounts.loot_full_ratio`（默认 95%）时，该账号本轮不再进攻。
