import os
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers.utils import logging
logging.disable_progress_bar()


CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))

YOLOV5_BASE = os.path.join(PROJECT_ROOT, "inference", "yolov5")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "outputs", "yolov5")

TARGET_DATASETS = ["bdd100k", "crowdhuman", "pest24"]


print("=== Loading Zero-shot Scene Classifier ===")

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

SCENE_LABELS = [
    "normal situation",
    "emergency situation",
    "crowded scene",
    "suspicious activity",
    "vehicle related scene",
    "pest detected"
]

print("Scene labels:", SCENE_LABELS)


def classify_prompt(text_prompt):
    input_text = f"CCTV scene with {text_prompt}"

    result = classifier(
        input_text,
        SCENE_LABELS,
        multi_label=False
    )

    return {
        "scene": result["labels"][0],
        "score": float(result["scores"][0])
    }


def get_dataset_name(folder_name):
    name = folder_name.lower()
    for ds in TARGET_DATASETS:
        if ds in name:
            return ds
    return None


def run_folder(infer_dir, dataset_name):
    prompts_path = os.path.join(infer_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        return

    folder_name = os.path.basename(infer_dir)

    save_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"{dataset_name}_distilbert_results.json"
    save_path = os.path.join(save_dir, save_name)

    print(f"\nProcessing: {infer_dir}")
    print(f"Saving to: {save_path}")

    with open(prompts_path) as f:
        prompts = json.load(f)

    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)
        print("Resume mode — existing results loaded")
    else:
        results = {}

    for img_name, yolo_prompt in tqdm(prompts.items()):
        if img_name in results:
            continue

        scene_result = classify_prompt(yolo_prompt)

        results[img_name] = {
            "yolo_objects": yolo_prompt,
            "scene": scene_result["scene"],
            "confidence": scene_result["score"]
        }

        if len(results) % 200 == 0:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ saved:", save_path)


def run_all(base_dir):
    print(f"\n=== Scanning {base_dir} ===")

    if not os.path.exists(base_dir):
        print("❌ inference directory not found:", base_dir)
        return

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)

        if not os.path.isdir(full_path):
            continue

        dataset_name = get_dataset_name(folder)
        if dataset_name is None:
            continue

        run_folder(full_path, dataset_name)


if __name__ == "__main__":
    print("\n=== Running Scene Classification ===")
    run_all(YOLOV5_BASE)
    print("\nAll scene classification finished")