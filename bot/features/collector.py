from bot.features.base import Feature


class Collector(Feature):
    """资源收集（视前面完成情况决定是否实现）：点击收集器气泡。"""

    name = "collector"

    def run(self) -> bool:
        raise NotImplementedError("视前面功能完成情况决定是否实现")
