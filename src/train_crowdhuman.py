import subprocess
from pathlib import Path

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parent.parent
DATA = PROJECT_ROOT / "datasets" / "crowdhuman" / "crowdhuman.yaml"

commands = [
    # ---------------- YOLOv5 ----------------
    [
        "python", "train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "50",
        "--data", str(DATA),
        "--weights", "yolov5m.pt",
        "--device", "0",
        "--project", str(PROJECT_ROOT / "experiments" / "yolov5"),
        "--name", "crowdhuman_yolov5m",
        "--exist-ok",
    ],

    # ---------------- YOLOv8 ----------------
    [
        "yolo", "detect", "train",
        "model=yolov8m.pt",
        f"data={DATA}",
        "imgsz=640",
        "batch=16",
        "epochs=50",
        "device=0",
        f"project={PROJECT_ROOT / 'experiments' / 'yolov8'}",
        "name=crowdhuman_yolov8m",
        "exist_ok=True",
    ],

    # ---------------- YOLO11 ----------------
    [
        "yolo", "detect", "train",
        "model=yolo11m.pt",
        f"data={DATA}",
        "imgsz=640",
        "batch=16",
        "epochs=50",
        "device=0",
        f"project={PROJECT_ROOT / 'experiments' / 'yolo11'}",
        "name=crowdhuman_yolo11m",
        "exist_ok=True",
    ],
]

for cmd in commands:
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)