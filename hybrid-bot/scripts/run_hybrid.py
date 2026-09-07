import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.hybrid_bot import HybridBot

if __name__ == "__main__":
    bot = HybridBot()
    bot.run_check()