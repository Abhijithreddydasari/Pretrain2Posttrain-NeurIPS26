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
        "torchvision",
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
        "sentencepiece",
        "protobuf",
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
    """Load Gemma 4 E4B base with 4-bit BnB; profile VRAM.

    Transformers 5: pass BitsAndBytesConfig, not load_in_4bit= on from_pretrained.
    Use dtype= instead of torch_dtype=.
    """
    import os
    import traceback

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_id = "google/gemma-4-E4B"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "HF token missing (Modal secret huggingface-secret)"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = None
    loader = None
    errors: list[str] = []
    import transformers as tf

    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(tf, cls_name, None)
        if cls is None:
            errors.append(f"{cls_name} missing")
            continue
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            loader = cls_name
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")

    if model is None:
        return {
            "ok": False,
            "error": errors,
            "traceback": traceback.format_exc()[-2000:],
            "transformers": getattr(tf, "__version__", "?"),
        }

    n_params = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vol_out.commit()
    return {
        "ok": True,
        "model_id": model_id,
        "loader": loader,
        "processor": type(processor).__name__,
        "n_params": n_params,
        "max_mem_gb": round(mem, 2) if mem is not None else None,
        "gpu": gpu,
        "transformers": getattr(tf, "__version__", "?"),
    }


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
