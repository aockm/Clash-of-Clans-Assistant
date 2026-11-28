import random
import time


class SessionControl:
    """会话时长/次数控制：达到随机上限后停止。"""

    def __init__(self, attack_range: tuple[int, int]) -> None:
        self._max_attacks = random.randint(*attack_range)
        self._attacks_done = 0
        self._stopped = False
        self._start = time.time()

    @property
    def max_attacks(self) -> int:
        return self._max_attacks

    def record_attack(self) -> None:
        self._attacks_done += 1

    def stop(self) -> None:
        """提前结束本会话（例如资源已满，不再进攻）。"""
        self._stopped = True

    def should_stop(self) -> bool:
        return self._stopped or self._attacks_done >= self._max_attacks
