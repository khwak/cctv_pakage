import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.utils import logging
logging.disable_progress_bar()


CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))

BASE_DIR = os.path.join(PROJECT_ROOT, "inference", "yolo11")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "outputs", "yolo11")

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
FIXED_RESOLUTION = (448, 448)

print(f"Loading {MODEL_NAME}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
).eval()


def run_folder(infer_dir):
    prompts_path = os.path.join(infer_dir, "prompts.json")
    images_dir = os.path.join(infer_dir, "images")

    if not os.path.exists(prompts_path):
        return

    folder_name = os.path.basename(infer_dir)

    # outputs/yolo11/<folder_name>/
    save_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "qwen_results.json")

    print(f"\nQwen processing: {infer_dir}")
    print(f"Saving to: {save_path}")

    with open(prompts_path) as f:
        prompts = json.load(f)

    results = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)
        print("Resume mode — existing results loaded")

    pbar = tqdm(prompts.items(), desc="Qwen Inference")

    for img_name, yolo_prompt in pbar:
        if img_name in results:
            continue

        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            image = raw_image.resize(FIXED_RESOLUTION, Image.LANCZOS)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Context: {yolo_prompt}. Describe the situation concisely."}
                ]
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            caption = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]

            results[img_name] = caption.strip()

        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            results[img_name] = "ERROR_SKIPPED"
            continue

        if len(results) % 20 == 0:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ saved: {save_path}")


def run_all():
    if not os.path.exists(BASE_DIR):
        print("❌ inference directory not found:", BASE_DIR)
        return

    for root, dirs, files in os.walk(BASE_DIR):
        if "prompts.json" in files:
            run_folder(root)


if __name__ == "__main__":
    run_all()