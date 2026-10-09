"""Sequence training for MAC/MAG/MAL; select --architecture explicitly.

Shares the established dataset, TBPTT and checkpoint pipeline with RoboTTT.
"""

from .train_robottt import main

if __name__ == "__main__":
    main()
