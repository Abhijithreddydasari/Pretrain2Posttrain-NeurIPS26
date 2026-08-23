"""Modal entrypoints: volumes, HF secret, E4B smoke + train jobs."""
from __future__ import annotations

import modal

app = modal.App("structsvg-sft")

hf_secret = modal.Secret.from_name("huggingface-secret")

vol_hf = modal.Volume.from_name("structsvg-hf-cache", create_if_missing=True)
vol_data = modal.Volume.from_name("structsvg-data", create_if_missing=True)
vol_out = modal.Volume.from_name("structsvg-outputs", create_if_missing=True)

VOLUME_MOUNTS = {"/vol/hf": vol_hf, "/vol/out": vol_out}
DATA_ROOT = "/root/data"

_TRAIN_ENV = {
    "PYTHONPATH": "/root",
    "PYTHONUTF8": "1",
    "DATA_ROOT": DATA_ROOT,
    "HF_HOME": "/vol/hf",
    "HF_HUB_CACHE": "/vol/hf",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _local_repo_root():
    from pathlib import Path

    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "requirements.txt").is_file() and (parent / "train" / "lora_sft.py").is_file():
            return parent
    raise FileNotFoundError("repo root not found for Modal train image")


image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "libcairo2",
    "libgdk-pixbuf2.0-0",
    "libffi-dev",
    "shared-mime-info",
)

if modal.is_local():
    _repo = _local_repo_root()
    image = (
        image.pip_install_from_requirements(str(_repo / "requirements.txt"))
        .env(_TRAIN_ENV)
        .add_local_dir(str(_repo / "structsvg_lib"), remote_path="/root/structsvg_lib")
        .add_local_dir(str(_repo / "train"), remote_path="/root/train")
        .add_local_dir(str(_repo / "configs"), remote_path="/root/configs")
        .add_local_dir(str(_repo / "eval"), remote_path="/root/eval")
        .add_local_dir(str(_repo / "data" / "scripts"), remote_path="/root/data/scripts")
    )
    _broad = _repo / "data" / "processed" / "svg_diagrams"
    if _broad.exists() and (_broad / "train_manifest.jsonl").exists():
        image = image.add_local_dir(
            str(_broad),
            remote_path=f"{DATA_ROOT}/processed/svg_diagrams",
        )
    _vfig = _repo / "data" / "processed" / "vfig_bench"
    if _vfig.exists() and (_vfig / "id_manifest.jsonl").exists():
        image = image.add_local_dir(str(_vfig), remote_path=f"{DATA_ROOT}/processed/vfig_bench")
    _svgtest = _repo / "data" / "processed" / "svg_diagrams_test"
    if _svgtest.exists() and (_svgtest / "test_manifest.jsonl").exists():
        image = image.add_local_dir(str(_svgtest), remote_path=f"{DATA_ROOT}/processed/svg_diagrams_test")
else:
    image = image.env(_TRAIN_ENV)


def _resolve_train_config(cfg: dict, *, manifest_override: str | None = None) -> dict:
    from pathlib import Path

    if manifest_override:
        cfg.setdefault("data", {})["manifest"] = manifest_override
    else:
        local_manifest = Path(cfg["data"]["manifest"])
        vol_candidates = [
            Path(DATA_ROOT) / "processed" / "svg_diagrams" / "train_manifest.jsonl",
            Path(DATA_ROOT) / local_manifest.as_posix().lstrip("/"),
            Path("/vol/data") / local_manifest.as_posix().lstrip("/"),
            Path("/vol/data") / "processed" / "svg_diagrams" / "train_manifest.jsonl",
        ]
        for cand in vol_candidates:
            if cand.exists():
                cfg.setdefault("data", {})["manifest"] = str(cand)
                break
    return cfg


def _run_train_subprocess(
    cfg: dict,
    *,
    dry_run: bool = False,
    verify_loss_mask: bool = False,
    max_steps: int | None = None,
) -> dict:
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, "/root")
    cfg.setdefault("train", {})["output_dir"] = cfg["train"].get("output_dir", "/vol/out/e4b_broad")
    if not str(cfg["train"]["output_dir"]).startswith("/vol/"):
        cfg["train"]["output_dir"] = f"/vol/out/{Path(cfg['train']['output_dir']).name}"

    runtime_cfg = Path("/tmp/runtime_train.yaml")
    runtime_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd = [sys.executable, "-m", "train.lora_sft", "--config", str(runtime_cfg)]
    if dry_run:
        cmd.append("--dry-run")
    if verify_loss_mask:
        cmd.append("--verify-loss-mask")
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])

    print("[train]", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_chunks.append(line)
    proc.wait()
    stdout = "".join(stdout_chunks)
    vol_out.commit()
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "manifest": cfg["data"]["manifest"],
        "output_dir": cfg["train"]["output_dir"],
        "stdout_tail": stdout[-6000:],
        "stderr_tail": "",
    }


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60)
def upload_broad_status():
    """Check broad data presence (baked image or volume fallback)."""
    from pathlib import Path

    roots = [
        Path(DATA_ROOT) / "processed" / "svg_diagrams",
        Path("/vol/data") / "data" / "processed" / "svg_diagrams",
        Path("/vol/data") / "processed" / "svg_diagrams",
    ]
    for root in roots:
        man = root / "train_manifest.jsonl"
        pngs = list((root / "pngs").glob("*.png")) if (root / "pngs").exists() else []
        svgs = list((root / "svgs").glob("*.svg")) if (root / "svgs").exists() else []
        if man.exists():
            return {
                "manifest_exists": True,
                "manifest_path": str(man),
                "n_pngs": len(pngs),
                "n_svgs": len(svgs),
                "ok": len(pngs) >= 100 and len(svgs) >= 100,
                "root": str(root),
            }
    return {"ok": False, "manifest_exists": False}


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def smoke_e4b(*, bf16: bool = True):
    """Load Gemma 4 E4B base; default bf16."""
    import traceback

    import torch

    from train.model_load import load_vlm

    try:
        processor, model, loader = load_vlm(
            "google/gemma-4-E4B",
            load_in_4bit=not bf16,
            dtype_name="bfloat16",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        n_params = sum(p.numel() for p in model.parameters())
        mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        import transformers as tf

        return {
            "ok": True,
            "loader": loader,
            "processor": type(processor).__name__,
            "n_params": n_params,
            "max_mem_gb": round(mem, 2) if mem is not None else None,
            "gpu": gpu,
            "transformers": getattr(tf, "__version__", "?"),
            "precision": "bf16" if bf16 else "4bit",
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()[-2000:]}


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def probe_train(gpu: str = "L4", max_samples: int = 8, max_steps: int = 2):
    """2-step training probe; returns peak VRAM estimate."""
    import subprocess
    import sys
    from pathlib import Path

    import torch
    import yaml

    sys.path.insert(0, "/root")
    cfg_path = Path("/root/configs/train_e4b_broad.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = _resolve_train_config(cfg)
    cfg.setdefault("data", {})["max_samples"] = max_samples
    cfg.setdefault("train", {})["output_dir"] = "/vol/out/probe_train"
    cfg["train"]["logging_steps"] = 1

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = _run_train_subprocess(cfg, max_steps=max_steps)
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    result["peak_vram_gb"] = round(peak, 2) if peak is not None else None
    result["gpu"] = gpu
    result["recommended_gpu"] = (
        "L4" if peak and peak <= 20 else "A10" if peak and peak <= 24 else "L40S" if peak and peak <= 40 else "A100-40GB"
    )
    return result


def _train_impl(
    config_name: str,
    *,
    gpu: str = "L4",
    manifest_override: str | None = None,
    output_override: str | None = None,
    max_samples: int | None = None,
    dry_run: bool = False,
    verify_loss_mask: bool = False,
    max_steps: int | None = None,
):
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, "/root")
    cfg_path = Path("/root/configs") / config_name
    if not cfg_path.exists():
        return {"ok": False, "error": f"missing config {cfg_path}"}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = _resolve_train_config(cfg, manifest_override=manifest_override)
    if output_override:
        cfg.setdefault("train", {})["output_dir"] = output_override
    if max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = max_samples
    return _run_train_subprocess(
        cfg,
        dry_run=dry_run,
        verify_loss_mask=verify_loss_mask,
        max_steps=max_steps,
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def train_remote(
    config_name: str = "train_e4b_broad.yaml",
    *,
    manifest_override: str | None = None,
    output_override: str | None = None,
    max_samples: int | None = None,
    dry_run: bool = False,
    verify_loss_mask: bool = False,
    max_steps: int | None = None,
):
    return _train_impl(
        config_name,
        manifest_override=manifest_override,
        output_override=output_override,
        max_samples=max_samples,
        dry_run=dry_run,
        verify_loss_mask=verify_loss_mask,
        max_steps=max_steps,
    )


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def train_remote_a10(
    config_name: str = "train_e4b_broad.yaml",
    *,
    manifest_override: str | None = None,
    output_override: str | None = None,
    max_samples: int | None = None,
    dry_run: bool = False,
    verify_loss_mask: bool = False,
    max_steps: int | None = None,
):
    return _train_impl(
        config_name,
        manifest_override=manifest_override,
        output_override=output_override,
        max_samples=max_samples,
        dry_run=dry_run,
        verify_loss_mask=verify_loss_mask,
        max_steps=max_steps,
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def infer_remote(
    manifest_path: str,
    out_path: str,
    *,
    config_name: str = "model_e4b.yaml",
    adapter_path: str | None = None,
    protocol: str = "prompt",
    max_samples: int | None = None,
):
    """Generate SVG preds for a manifest (base or LoRA adapter)."""
    import json
    import sys
    from pathlib import Path

    import torch
    import yaml

    sys.path.insert(0, "/root")
    from peft import PeftModel

    from train.data_utils import load_manifest, resolve_image
    from train.model_load import load_vlm

    cfg = yaml.safe_load((Path("/root/configs") / config_name).read_text(encoding="utf-8"))
    man = Path(manifest_path)
    if not man.exists():
        man = Path(DATA_ROOT) / manifest_path.lstrip("/")
    rows = load_manifest(man, max_samples)
    prompt = cfg.get("prompt_template", "").strip()
    prefix = cfg.get("svg_prefix_scaffold", "")
    gen_cfg = cfg.get("generation", {})

    processor, model, loader = load_vlm(
        cfg["model_id"],
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        dtype_name=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=True,
    )
    if adapter_path:
        ap = Path(adapter_path)
        if not ap.exists():
            ap = Path(DATA_ROOT) / adapter_path.lstrip("/")
        model = PeftModel.from_pretrained(model, str(ap))
    model.eval()
    print(f"infer via {loader} adapter={adapter_path}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            image = resolve_image(r)
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if protocol == "svg_prefix":
                text = text + prefix
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
                    do_sample=gen_cfg.get("do_sample", False),
                    max_time=gen_cfg.get("max_time", 600),
                )
            new_ids = out_ids[0][input_len:]
            decoded = processor.decode(new_ids, skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "pred_text": decoded, "protocol": protocol}) + "\n")
            f.flush()
            print(f"generated {r['id']}", flush=True)
    vol_out.commit()
    return {"ok": True, "n": len(rows), "out": str(out)}


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
)
def sweep_remote(
    *,
    adapter_root: str = "/vol/out/e4b_broad",
    protocol: str = "prompt",
    pcts: str = "0,5,10,20,40,60,80,100",
    max_samples: int | None = None,
):
    import subprocess
    import sys

    sys.path.insert(0, "/root")
    cmd = [
        sys.executable,
        "-m",
        "eval.sweep_checkpoints",
        "--adapter-root",
        adapter_root,
        "--protocol",
        protocol,
        "--pcts",
        pcts,
        "--out-dir",
        "/vol/out/metrics/sweep",
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    print("[sweep]", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    stdout_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_chunks.append(line)
    proc.wait()
    vol_out.commit()
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout_tail": "".join(stdout_chunks)[-4000:]}


@app.local_entrypoint()
def main(
    task: str = "smoke",
    config: str = "train_e4b_broad.yaml",
    gpu: str = "l4",
    manifest: str = "",
    out: str = "",
    adapter: str = "",
    protocol: str = "prompt",
    max_samples: int = 0,
):
    ms = max_samples if max_samples > 0 else None
    if task == "smoke":
        print(smoke_e4b.remote(bf16=True))
    elif task == "upload_status":
        print(upload_broad_status.remote())
    elif task == "probe":
        print(probe_train.remote())
    elif task == "train_dry":
        print(train_remote.remote(config, dry_run=True))
    elif task == "verify_mask":
        print(train_remote.remote(config, verify_loss_mask=True, max_samples=8))
    elif task == "train":
        fn = train_remote_a10.remote if gpu.lower() in {"a10", "a10g"} else train_remote.remote
        print(fn(config))
    elif task == "infer":
        if not manifest or not out:
            raise SystemExit("infer requires --manifest and --out")
        ap = adapter or None
        print(infer_remote.remote(manifest, out, adapter_path=ap, protocol=protocol, max_samples=ms))
    elif task == "sweep":
        print(sweep_remote.remote(protocol=protocol, max_samples=ms))
    else:
        raise SystemExit(f"unknown task {task}")
