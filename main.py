from bot.config import load_config
from bot.orchestrator import Orchestrator


def main() -> None:
    config = load_config("config/bot.yaml")
    Orchestrator(config).run()


if __name__ == "__main__":
    main()
