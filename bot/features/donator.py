from bot.features.base import Feature


class Donator(Feature):
    """捐兵（第三阶段实现）：MainVillage → ClanChat → 识别请求 → 捐赠 → 返回。"""

    name = "donator"

    def run(self) -> bool:
        raise NotImplementedError("第三阶段实现：捐兵流程")
