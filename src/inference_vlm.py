import subprocess
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

INFERENCE_ROOT = PROJECT_ROOT / "inference"

DISTILBERT_SCRIPT = PROJECT_ROOT / "src" / "run_distilbert.py"
FLORENCE_SCRIPT = PROJECT_ROOT / "src" / "run_florence.py"
QWEN_SCRIPT = PROJECT_ROOT / "src" / "run_qwen.py"


def run_script(script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    conda_env = "cctv_research"
    if script_path == QWEN_SCRIPT:
        conda_env = "cctv_qwen"

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(script_path),
    ]

    print(f"\nRunning ({conda_env}):", " ".join(cmd))
    subprocess.run(cmd, check=True)


print("=== VLM Inference Pipeline Start ===")
print(f"Using inference results from: {INFERENCE_ROOT}")


print("\n=== Step 1: DistilBERT ===")
run_script(DISTILBERT_SCRIPT)


print("\n=== Step 2: Florence-2 ===")
run_script(FLORENCE_SCRIPT)


print("\n=== Step 3: Qwen2.5-VL ===")
run_script(QWEN_SCRIPT)


print("\n✅ All VLM inference finished")