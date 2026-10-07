from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "ultralytics==8.4.103",
            "timm==1.0.20",
            "pycocotools==2.0.11",
            "opencv-python-headless==4.11.0.86",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "filament_ensemble.run", *sys.argv[1:]],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )


if __name__ == "__main__":
    main()
