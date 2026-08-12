"""Modal entrypoints: volumes, HF secret, E4B smoke + train jobs."""
from __future__ import annotations

import modal

app = modal.App("structsvg-sft")

hf_secret = modal.Secret.from_name("huggingface-secret")

vol_hf = modal.Volume.from_name("structsvg-hf-cache", create_if_missing=True)
vol_data = modal.Volume.from_name("structsvg-data", create_if_missing=True)
vol_out = modal.Volume.from_name("structsvg-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libcairo2", "libgdk-pixbuf2.0-0", "libffi-dev", "shared-mime-info")
    .pip_install(
        "torch",
        "transformers>=4.52",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "huggingface_hub",
        "pillow",
        "lxml",
        "pyyaml",
        "cairosvg",
        "numpy",
        "tqdm",
        "jsonlines",
        "scipy",
        "scikit-learn",
        "matplotlib",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_CACHE": "/vol/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={"/vol/hf": vol_hf, "/vol/data": vol_data, "/vol/out": vol_out},
)
def smoke_e4b():
    """Load Gemma 4 E4B base + tiny forward check; profile VRAM."""
    import os

    import torch
    from transformers import AutoProcessor

    model_id = "google/gemma-4-E4B"
    assert os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), "HF token missing"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}
    mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    vol_out.commit()
    return {"ok": True, "model_id": model_id, "max_mem_gb": mem, "processor": type(processor).__name__}


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    secrets=[hf_secret],
    volumes={"/vol/hf": vol_hf, "/vol/data": vol_data, "/vol/out": vol_out},
)
def train_remote(config_name: str = "train_e4b_structsvg.yaml"):
    """
    Placeholder remote train launcher.
    Sync repo into image or mount code; for now instructs user to `modal run` after syncing.
    """
    return {
        "status": "ready_stub",
        "hint": f"Mount repo and run train.lora_sft --config configs/{config_name}",
        "config": config_name,
    }


@app.local_entrypoint()
def main(task: str = "smoke"):
    if task == "smoke":
        print(smoke_e4b.remote())
    elif task == "train":
        print(train_remote.remote())
    else:
        raise SystemExit(f"unknown task {task}")
