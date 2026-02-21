import os
import json
from collections import defaultdict
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

INFERENCE_ROOT = PROJECT_ROOT / "inference"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


CLASS_NAME_MAP = {
    "bdd100k": {0: "vehicle"},
    "crowdhuman": {0: "person"},
    "pest24": {i: "insect" for i in range(24)}
}


def parse_yolo_label(label_path, dataset_name):
    counts = defaultdict(int)

    if dataset_name not in CLASS_NAME_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    class_map = CLASS_NAME_MAP[dataset_name]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            class_name = class_map.get(class_id, f"class{class_id}")
            counts[class_name] += 1

    return counts


def counts_to_prompt(counts):
    if not counts:
        return "no objects detected"

    parts = []
    for name, count in counts.items():
        if count == 1:
            parts.append(f"1 {name}")
        else:
            parts.append(f"{count} {name}s")

    return ", ".join(parts)


def process_inference_folder(infer_dir):
    labels_dir = infer_dir / "labels"

    if not labels_dir.exists():
        print(f"labels 폴더 없음 → skip: {infer_dir}")
        return

    folder_name = infer_dir.name.lower()
    path_str = str(infer_dir).lower()

    if "bdd100k" in folder_name:
        dataset_name = "bdd100k"
    elif "crowdhuman" in folder_name:
        dataset_name = "crowdhuman"
    elif "pest24" in folder_name:
        dataset_name = "pest24"
    else:
        print(f"dataset 식별 불가 → skip: {infer_dir}")
        return

    if "yolov5" in path_str:
        yolo_type = "yolov5"
    elif "yolo11" in path_str:
        yolo_type = "yolo11"
    elif "yolov8" in path_str:
        yolo_type = "yolov8"
    else:
        print(f"YOLO 타입 식별 불가 → skip: {infer_dir}")
        return

    print(f"\nProcessing: {infer_dir}")
    print(f"   dataset: {dataset_name}")
    print(f"   yolo: {yolo_type}")

    results = {}

    for file in labels_dir.iterdir():
        if file.suffix != ".txt":
            continue

        image_name = file.stem + ".jpg"

        counts = parse_yolo_label(file, dataset_name)
        prompt = counts_to_prompt(counts)

        results[image_name] = prompt

    save_dir = DATASETS_ROOT / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{yolo_type}_{dataset_name}_prompts.json"

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"saved → {save_path}")


def process_all_inference(base_dir):
    print("\n=== 전체 inference 처리 시작 ===")
    print(f"base: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        if "labels" in dirs:
            process_inference_folder(Path(root))

    print("\n전체 완료")


if __name__ == "__main__":
    process_all_inference(INFERENCE_ROOT)