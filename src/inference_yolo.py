import subprocess
from pathlib import Path
import shutil

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

WEIGHTS_ROOT = PROJECT_ROOT / "weights"
OUTPUT_ROOT = PROJECT_ROOT / "inference"
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def get_yolov5_weight(dataset):
    path = WEIGHTS_ROOT / "yolov5" / f"{dataset}_yolov5m" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"YOLOv5 weight not found: {path}")
    return path

def get_yolo11_weight(dataset):
    path = WEIGHTS_ROOT / "yolo11" / f"{dataset}_yolo11m" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"YOLO11 weight not found: {path}")
    return path

def organize_output_images(output_dir):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for file in output_dir.iterdir():
        if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            shutil.move(str(file), images_dir / file.name)

DATASETS = {
    "bdd100k": PROJECT_ROOT / "datasets" / "bdd100k" / "images",
    "crowdhuman": PROJECT_ROOT / "datasets" / "crowdhuman" / "images",
    "pest24": PROJECT_ROOT / "datasets" / "pest24" / "images",
}


for dataset_name in DATASETS:
    print(f"\n=== YOLOv5 inference: {dataset_name} ===")

    weight = get_yolov5_weight(dataset_name)

    cmd = [
        "python", str(YOLOV5_ROOT / "detect.py"),
        "--weights", str(weight),
        "--source", str(DATASETS[dataset_name]),
        "--img", "640",
        "--conf-thres", "0.25",
        "--iou-thres", "0.7",
        "--save-txt",
        "--save-conf",
        "--project", str(OUTPUT_ROOT / "yolov5"),
        "--name", f"yolov5_{dataset_name}_distilbert",
        "--exist-ok"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    subprocess.run(cmd, check=True)

    organize_output_images(
        OUTPUT_ROOT / "yolov5" / f"yolov5_{dataset_name}_distilbert"
    )


for dataset_name in DATASETS:
    print(f"\n=== YOLO11 inference: {dataset_name} ===")

    weight = get_yolo11_weight(dataset_name)

    cmd = [
        "yolo", "detect", "predict",
        f"model={weight}",
        f"source={DATASETS[dataset_name]}",
        "imgsz=640",
        "conf=0.25",
        "iou=0.7",
        "save_txt=True",
        "save_conf=True",
        f"project={OUTPUT_ROOT / 'yolo11'}",
        f"name=yolo11_{dataset_name}_vlm",
        "exist_ok=True"
    ]

    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

    organize_output_images(
        OUTPUT_ROOT / "yolo11" / f"yolo11_{dataset_name}_vlm"
    )

print("\n✅ All inference finished")