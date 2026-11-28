import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class AccountStore:
    """账号运行状态存储（Redis）：大本营等级、在建建筑完成时间戳、资源/升级状态。

    每个账号一个 Hash：<prefix>:account:<name>
      th          大本营等级
      buildings   在建建筑 JSON 列表，元素 {"finish_at": epoch 秒}
      loot_full   上次离开时资源是否已满（"1"/"0"）
      no_upgrade  上次离开时是否已无可升级建筑（"1"/"0"）
      updated_at  状态更新时间
    Redis 不可用时退回进程内字典，保证单次运行仍可工作。
    """

    def __init__(self, cfg) -> None:
        self._prefix = cfg.prefix
        self._redis = None
        self._mem: dict[str, dict[str, str]] = {}
        if not cfg.enabled:
            logger.info("Redis 已禁用，账号状态仅保存在内存")
            return
        try:
            import redis

            self._redis = redis.Redis(
                host=cfg.host, port=cfg.port, db=cfg.db,
                socket_connect_timeout=1.5, socket_timeout=1.5,
                decode_responses=True)
            self._redis.ping()
            logger.info("Redis 已连接：%s:%s/%s", cfg.host, cfg.port, cfg.db)
        except Exception as exc:
            self._redis = None
            logger.warning("Redis 不可用（%s），账号状态仅保存在内存", exc)

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _key(self, name: str) -> str:
        return f"{self._prefix}:account:{name}"

    def get(self, name: str) -> dict[str, Any] | None:
        try:
            raw = (self._redis.hgetall(self._key(name)) if self._redis
                   else self._mem.get(name))
        except Exception:
            logger.exception("读取账号状态失败：%s", name)
            return None
        if not raw:
            return None
        return {
            "th": int(raw["th"]) if raw.get("th") else None,
            "buildings": json.loads(raw.get("buildings") or "[]"),
            "loot_full": raw.get("loot_full") == "1",
            "no_upgrade": raw.get("no_upgrade") == "1",
            "updated_at": float(raw.get("updated_at") or 0),
        }

    def _write(self, name: str, fields: dict[str, Any]) -> None:
        data = {k: str(v) for k, v in fields.items()}
        data["updated_at"] = str(time.time())
        try:
            if self._redis:
                self._redis.hset(self._key(name), mapping=data)
            else:
                self._mem.setdefault(name, {}).update(data)
        except Exception:
            logger.exception("写入账号状态失败：%s", name)

    def set_th(self, name: str, th: int) -> None:
        self._write(name, {"th": th})

    def set_buildings(self, name: str, buildings: list[dict]) -> None:
        self._write(name, {"buildings": json.dumps(buildings, ensure_ascii=False)})

    def set_idle(self, name: str, loot_full: bool, no_upgrade: bool) -> None:
        self._write(name, {"loot_full": int(loot_full),
                           "no_upgrade": int(no_upgrade)})

    def should_skip(self, name: str, now: float | None = None,
                    valid_hours: float = 6.0) -> tuple[bool, str]:
        """资源已满且无可升级建筑时返回 True，表示本轮轮换跳过该账号。

        若已有建筑完成（finish_at <= now），说明空出了工人，值得重新访问。
        状态过期后也会重试，避免永久跳过。
        """
        now = now or time.time()
        rec = self.get(name)
        if rec is None:
            return False, "无历史状态"
        if not (rec["loot_full"] and rec["no_upgrade"]):
            return False, "资源未满或仍有可升级建筑"
        if now - rec["updated_at"] > valid_hours * 3600:
            return False, "状态已过期，重新检查"
        finished = [b for b in rec["buildings"]
                    if float(b.get("finish_at", 0)) <= now]
        if finished:
            return False, "有建筑已完成，可升级"
        return True, "资源已满且无可升级建筑，等待建筑完成"
