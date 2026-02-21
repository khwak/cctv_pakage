import json
from pathlib import Path
import cv2
import argparse


PERSON_CLASSES = {"car", "truck", "bus"}
SPLITS = ["train", "val", "test"]


def convert_single_json(
    ann_path: Path,
    image_dir: Path,
    label_dir: Path,
):
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "r") as f:
        data = json.load(f)

    img_name = data.get("name")
    if img_name is None:
        return

    img_name = img_name + ".jpg"
    img_path = image_dir / img_name

    if not img_path.exists():
        return

    img = cv2.imread(str(img_path))
    if img is None:
        return

    img_h, img_w = img.shape[:2]
    yolo_lines = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):

            category = obj.get("category")
            if category not in PERSON_CLASSES:
                continue

            box2d = obj.get("box2d")
            if box2d is None:
                continue

            x1 = max(0, box2d["x1"])
            y1 = max(0, box2d["y1"])
            x2 = min(img_w, box2d["x2"])
            y2 = min(img_h, box2d["y2"])

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            xc = (x1 + w / 2) / img_w
            yc = (y1 + h / 2) / img_h
            bw = w / img_w
            bh = h / img_h

            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if yolo_lines:
        label_path = label_dir / f"{Path(img_name).stem}.txt"
        with open(label_path, "w") as lf:
            lf.write("\n".join(yolo_lines))


def convert_split(
    ann_root: Path,
    image_root: Path,
    label_root: Path,
    split: str,
):
    print(f"Processing {split}...")

    split_ann_dir = ann_root / split
    split_img_dir = image_root / split
    split_label_dir = label_root / split

    if not split_ann_dir.exists():
        print(f"Skip {split} (annotation dir not found)")
        return

    json_files = list(split_ann_dir.glob("*.json"))

    if not json_files:
        print(f"No json found in {split_ann_dir}")
        return

    for json_file in json_files:
        convert_single_json(
            ann_path=json_file,
            image_dir=split_img_dir,
            label_dir=split_label_dir,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_root", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--label_root", required=True, type=Path)
    args = parser.parse_args()

    for split in SPLITS:
        convert_split(
            ann_root=args.ann_root,
            image_root=args.image_root,
            label_root=args.label_root,
            split=split,
        )


if __name__ == "__main__":
    main()
