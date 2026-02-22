import os
import sys
import importlib.util
from unittest.mock import MagicMock
from transformers.utils import logging
logging.disable_progress_bar()

# flash_attn mock (Florence import 안정화)
mock_flash_attn = MagicMock()
mock_flash_attn.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
sys.modules["flash_attn"] = mock_flash_attn
sys.modules["flash_attn.flash_attn_interface"] = mock_flash_attn
sys.modules["flash_attn.bert_padding"] = mock_flash_attn

import json
import torch
from PIL import Image
from tqdm import tqdm
import transformers

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
transformers.utils.import_utils.is_flash_attn_available = lambda: False

from transformers import AutoProcessor, AutoModelForCausalLM


CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))

BASE_DIR = os.path.join(PROJECT_ROOT, "inference", "yolo11")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "outputs", "yolo11")

MODEL_NAME = "microsoft/Florence-2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Florence-2 on {device}...")


processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    attn_implementation="sdpa"
).to(device).eval()


def run_folder(infer_dir):
    prompts_path = os.path.join(infer_dir, "prompts.json")
    images_dir = os.path.join(infer_dir, "images")

    if not os.path.exists(prompts_path):
        return

    folder_name = os.path.basename(infer_dir)

    save_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "florence_results.json")

    print(f"\nFlorence processing: {infer_dir}")
    print(f"Saving to: {save_path}")

    with open(prompts_path) as f:
        prompts = json.load(f)

    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)
        print("Resume mode — existing results loaded")
    else:
        results = {}

    pbar = tqdm(prompts.items(), desc="Generating Captions")

    for img_name, yolo_prompt in pbar:
        if img_name in results:
            continue

        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            prompt = f"Describe the scene in detail. Context: {yolo_prompt}"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device, torch_dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=3
                )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            clean_caption = generated_text.replace(prompt, "").strip()
            results[img_name] = clean_caption

            if len(results) % 50 == 0:
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            continue

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