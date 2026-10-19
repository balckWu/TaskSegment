from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iris2d.runner import train_main

if __name__ == "__main__":
    train_main()
