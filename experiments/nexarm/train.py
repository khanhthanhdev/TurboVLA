"""Public NexArm training entry point for TurboVLA."""

import sys
from pathlib import Path

# Ensure starVLA runtime is discoverable
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "starvla_runtime"))

from starVLA.training.train_robotwin_clean_act_pi05_recipe import main

if __name__ == "__main__":
    main()
