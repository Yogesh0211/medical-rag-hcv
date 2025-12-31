from pathlib import Path
import os

# Set your default base directory (adjust in the notebook if needed).
# This is where your Drive folder "hcv_rag" should live when mounted in Colab.
DEFAULT_BASE_DIR = "/content/drive/MyDrive/hcv_rag"

def paths(base_dir: str = None):
    """Return a dict of canonical project paths under the given base dir."""
    base = Path(base_dir or DEFAULT_BASE_DIR)
    return {
        "BASE": base,
        "DATA": base / "data",
        "GUIDELINES": base / "data" / "guidelines",
        "ARTIFACTS": base / "artifacts",
        "OUTPUTS": base / "outputs",
    }

def ensure_dirs(base_dir: str = None):
    p = paths(base_dir)
    for k in ["DATA", "GUIDELINES", "ARTIFACTS", "OUTPUTS"]:
        p[k].mkdir(parents=True, exist_ok=True)
    return p
