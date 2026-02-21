import json
from pathlib import Path
import cv2
import argparse

def convert_odgt(
    odgt_path: Path,
    image_dir: Path,
    label_dir: Path
):
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(odgt_path, "r") as f:
        for line in f:
            data = json.loads(line)
            img_id = data["ID"]

            img_path = image_dir / f"{img_id}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            yolo_lines = []

            for box in data.get("gtboxes", []):
                if box.get("tag") != "person":
                    continue

                if box.get("extra", {}).get("ignore", 0) == 1:
                    continue

                # full body box
                x, y, w, h = box["fbox"]

                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                if w <= 0 or h <= 0:
                    continue

                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                bw = w / img_w
                bh = h / img_h

                yolo_lines.append(
                    f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

            if yolo_lines:
                label_path = label_dir / f"{img_id}.txt"
                with open(label_path, "w") as lf:
                    lf.write("\n".join(yolo_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--odgt", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--label_root", required=True, type=Path)
    args = parser.parse_args()

    convert_odgt(
        odgt_path=args.odgt,
        image_dir=args.image_root / args.split,
        label_dir=args.label_root / args.split,
    )


if __name__ == "__main__":
    main()
