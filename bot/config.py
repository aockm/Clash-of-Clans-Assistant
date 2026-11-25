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


@dataclass
class AttackConfig:
    min_loot_gold: int = 300_000
    min_loot_elixir: int = 300_000
    max_next_attempts: int = 20


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
class BattleEndConfig:
    sample_interval: float = 3.0
    stable_samples: int = 3
    stall_seconds: int = 25
    history_size: int = 10


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    battle_end: BattleEndConfig = field(default_factory=BattleEndConfig)
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
    if p := raw.get("paths", {}).get("templates_dir"):
        cfg.templates_dir = Path(p)
    if l := raw.get("logging", {}).get("level"):
        cfg.log_level = l
    return cfg
