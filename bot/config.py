from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DeviceConfig:
    serial: str | None = None
    scrcpy_max_size: int = 1280
    scrcpy_bit_rate: int = 8_000_000
    frame_buffer_size: int = 10


@dataclass
class SafetyConfig:
    click_jitter_px: int = 15
    action_delay_range: tuple[float, float] = (0.3, 0.8)
    confirm_frames: int = 2
    max_attacks_per_session: tuple[int, int] = (8, 12)


# 各大本营等级的默认资源门槛：金币或圣水任一达标即进攻
DEFAULT_LOOT_BY_TH: dict[int, int] = {
    1: 500, 2: 1_000, 3: 2_000, 4: 12_000, 5: 18_000, 6: 26_000,
    7: 38_000, 8: 55_000, 9: 75_000, 10: 100_000, 11: 130_000,
    12: 165_000, 13: 200_000, 14: 240_000, 15: 280_000, 16: 320_000,
}


@dataclass
class AttackConfig:
    town_hall_level: int = 10
    min_loot_gold: int | None = None
    min_loot_elixir: int | None = None
    max_next_attempts: int = 20

    def _default_loot(self) -> int:
        return DEFAULT_LOOT_BY_TH.get(self.town_hall_level, 100_000)

    @property
    def gold_threshold(self) -> int:
        return self.min_loot_gold or self._default_loot()

    @property
    def elixir_threshold(self) -> int:
        return self.min_loot_elixir or self._default_loot()


@dataclass
class DeployConfig:
    card_slot_start: tuple[int, int] = (250, 518)
    card_slot_pitch: int = 81
    card_slot_count: int = 11
    boundary_margin: int = 40
    max_rounds: int = 4
    map_bottom: int = 365
    max_recenter: int = 3


@dataclass
class UpgradeConfig:
    enabled: bool = False
    row_x: int = 1094            # 点击列表行的 x 坐标（行 y 由货币图标动态识别）
    cooldown: float = 20.0       # 两次升级尝试的最小间隔（秒）
    panel_timeout: float = 2.5   # 点“建造”后等待列表出现的时限
    confirm_timeout: float = 2.5  # 点列表行后等待确认框的时限
    max_failures: int = 3        # 连续失败次数上限，达到后本次会话不再升级
    nobuilder_cooldown: float = 600.0  # 工人全忙时暂停升级的时长（秒）


@dataclass
class BattleEndConfig:
    sample_interval: float = 3.0
    stable_samples: int = 3
    stall_seconds: int = 25
    history_size: int = 10


@dataclass
class AccountConfig:
    enabled: bool = False
    # 轮流使用的 Supercell ID 账号（按顺序），名称需与切换ID页显示一致
    names: list[str] = field(default_factory=list)
    attacks_per_account: int = 3   # 每个账号打完多少场后切下一个
    reward_taps: int = 5           # 切号加载页出现奖励宝箱时连续点击次数
    th_cache_hours: float = 168.0  # 大本营等级识别结果有效期（默认 7 天）
    loot_full_ratio: float = 0.95  # 资源条填充比例达到该值视为“资源已满”
    skip_valid_hours: float = 6.0  # “资源满且无可升级”状态的有效期，过期后重试该账号


@dataclass
class RedisConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    prefix: str = "coc"


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    battle_end: BattleEndConfig = field(default_factory=BattleEndConfig)
    upgrade: UpgradeConfig = field(default_factory=UpgradeConfig)
    accounts: AccountConfig = field(default_factory=AccountConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    templates_dir: Path = Path("assets/templates")
    log_level: str = "INFO"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config()
    if d := raw.get("device"):
        cfg.device = DeviceConfig(**d)
    if s := raw.get("safety"):
        s["action_delay_range"] = tuple(s["action_delay_range"])
        s["max_attacks_per_session"] = tuple(s["max_attacks_per_session"])
        cfg.safety = SafetyConfig(**s)
    if a := raw.get("attack"):
        cfg.attack = AttackConfig(**a)
    if d := raw.get("deploy"):
        if "card_slot_start" in d:
            d["card_slot_start"] = tuple(d["card_slot_start"])
        cfg.deploy = DeployConfig(**d)
    if b := raw.get("battle_end"):
        cfg.battle_end = BattleEndConfig(**b)
    if u := raw.get("upgrade"):
        cfg.upgrade = UpgradeConfig(**u)
    if a := raw.get("accounts"):
        cfg.accounts = AccountConfig(**a)
    if r := raw.get("redis"):
        cfg.redis = RedisConfig(**r)
    if p := raw.get("paths", {}).get("templates_dir"):
        cfg.templates_dir = Path(p)
    if l := raw.get("logging", {}).get("level"):
        cfg.log_level = l
    return cfg
