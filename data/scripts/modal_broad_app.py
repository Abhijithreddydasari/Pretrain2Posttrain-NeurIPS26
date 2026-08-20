"""Modal entrypoints for the broad 2k coreset data pipeline.

Usage (from repo root):
  modal run data/scripts/modal_broad_app.py --stage check
  modal run data/scripts/modal_broad_app.py --stage all
  modal run data/scripts/modal_broad_app.py --stage scan
  modal run data/scripts/modal_broad_app.py --stage all --pilot

Requires Modal secret: huggingface-secret (HF_TOKEN).
Data persists on volume structsvg-data mounted at /root/data/processed/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = "structsvg-broad"
# Match local layout so parquet paths (data/processed/broad/...) resolve under /root.
BROAD_ROOT = Path("/root/data/processed/broad")
FIG_ROOT = BROAD_ROOT / "figures"

hf_secret = modal.Secret.from_name("huggingface-secret")
vol_data = modal.Volume.from_name("structsvg-data", create_if_missing=True)


def _local_repo_root() -> Path:
    """Repo checkout root (must contain requirements-broad-modal.txt)."""
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "requirements-broad-modal.txt").is_file():
            return parent
    raise FileNotFoundError(
        "requirements-broad-modal.txt not found — create it at repo root before modal run"
    )


_BROAD_ENV = {
    "PYTHONPATH": "/root",
    "HF_HOME": "/tmp/hf",
    "HF_HUB_CACHE": "/tmp/hf",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "BROAD_TQDM": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Base image steps run at deploy time (local). On Modal workers, is_local() is False and
# the deployed image snapshot is used — do not reference checkout-only paths there.
broad_image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git",
    "fontconfig",
    "fonts-dejavu-core",
    "fonts-liberation",
    "fonts-freefont-ttf",
)

if modal.is_local():
    _repo_root = _local_repo_root()
    # env before add_local_* (Modal forbids build steps after local mounts)
    broad_image = (
        broad_image
        .pip_install_from_requirements(str(_repo_root / "requirements-broad-modal.txt"))
        .env(_BROAD_ENV)
        .add_local_dir(str(_repo_root / "structsvg_lib"), remote_path="/root/structsvg_lib")
        .add_local_dir(str(_repo_root / "data" / "scripts"), remote_path="/root/data/scripts")
    )
else:
    broad_image = broad_image.env(_BROAD_ENV)

VOLUME_MOUNTS = {"/root/data/processed": vol_data}

app = modal.App(APP_NAME)


def _setup(*, require_hf: bool = False) -> None:
    sys.path.insert(0, "/root")
    from data.scripts.broad_io import configure_broad_logging

    configure_broad_logging()
    if require_hf:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN missing — create Modal secret huggingface-secret with HF_TOKEN")


def _commit() -> None:
    vol_data.commit()


def _print_gate(stage: str, result: dict) -> None:
    status = "PASS" if result.get("ok") else "FAIL"
    print(f"[gate:{stage}] {status} {json.dumps(result, default=str)}")


@app.function(
    image=broad_image,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
    cpu=2,
    memory=4096,
    timeout=30 * 60,
)
def stage_test_hashes() -> dict:
    _setup(require_hf=True)
    from data.scripts.broad_checks import check_test_hashes
    from data.scripts.build_test_hashes import build_test_hashes

    BROAD_ROOT.mkdir(parents=True, exist_ok=True)
    print("[stage:test_hashes] building test split hashes...")
    build_test_hashes(BROAD_ROOT / "test_hashes.jsonl")
    result = check_test_hashes(BROAD_ROOT)
    _print_gate("test_hashes", result)
    _commit()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "test_hashes gate failed"))
    return result


@app.function(
    image=broad_image,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
    cpu=8,
    memory=16384,
    timeout=24 * 60 * 60,
)
def stage_scan(*, pilot: bool = False, max_rows: int | None = None, workers: int = 8) -> dict:
    _setup(require_hf=True)
    from data.scripts.broad_checks import check_scan
    from data.scripts.broad_scan_pool import scan_pool

    BROAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[stage:scan] pilot={pilot} max_rows={max_rows} workers={workers} out={BROAD_ROOT}")
    stats = scan_pool(BROAD_ROOT, pilot=pilot, max_rows=max_rows, workers=workers)
    result = check_scan(BROAD_ROOT, pilot=pilot)
    result["scan_stats"] = stats
    _print_gate("scan", result)
    _commit()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "scan gate failed"))
    return result


@app.function(
    image=broad_image,
    secrets=[hf_secret],
    volumes=VOLUME_MOUNTS,
    gpu="L4",
    memory=16384,
    timeout=4 * 60 * 60,
)
def stage_embed(*, pilot: bool = False, fresh: bool = False, batch_size: int = 128) -> dict:
    _setup(require_hf=True)
    from data.scripts.broad_checks import check_embed
    from data.scripts.broad_embed import embed_pool

    print(f"[stage:embed] pilot={pilot} fresh={fresh} batch_size={batch_size} out={BROAD_ROOT}")
    stats = embed_pool(
        BROAD_ROOT,
        pilot=pilot,
        fresh=fresh,
        batch_size=batch_size,
        preload=False,
        io_workers=16,
    )
    result = check_embed(BROAD_ROOT)
    result["embed_stats"] = stats
    _print_gate("embed", result)
    _commit()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "embed gate failed"))
    return result


@app.function(
    image=broad_image,
    volumes=VOLUME_MOUNTS,
    cpu=4,
    memory=8192,
    timeout=4 * 60 * 60,
)
def stage_select(*, pilot: bool = False, target_n: int = 2000) -> dict:
    _setup()
    from data.scripts.broad_checks import check_select
    from data.scripts.broad_select_coreset import select_coreset

    print(f"[stage:select] pilot={pilot} target_n={target_n} out={BROAD_ROOT}")
    stats = select_coreset(BROAD_ROOT, pilot=pilot, target_n=target_n)
    result = check_select(BROAD_ROOT, pilot=pilot, target_n=target_n)
    result["selection_stats"] = stats
    _print_gate("select", result)
    _commit()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "select gate failed"))
    return result


@app.function(
    image=broad_image,
    volumes=VOLUME_MOUNTS,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
)
def stage_visualize(*, pilot: bool = False) -> dict:
    _setup()
    from data.scripts.broad_checks import check_visualize
    from data.scripts.broad_visualize import visualize

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[stage:visualize] pilot={pilot} fig_dir={FIG_ROOT}")
    stats = visualize(BROAD_ROOT, pilot=pilot, fig_dir=FIG_ROOT)
    result = check_visualize(BROAD_ROOT, fig_dir=FIG_ROOT)
    result["viz_stats"] = stats
    _print_gate("visualize", result)
    _commit()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "visualize gate failed"))
    return result


@app.function(
    image=broad_image,
    volumes=VOLUME_MOUNTS,
    cpu=1,
    memory=2048,
    timeout=10 * 60,
)
def stage_check(*, pilot: bool = False, target_n: int = 2000) -> dict:
    _setup()
    from data.scripts.broad_checks import run_checks

    result = run_checks(BROAD_ROOT, stage="all", pilot=pilot, target_n=target_n, fig_dir=FIG_ROOT)
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    stage: str = "all",
    pilot: bool = False,
    fresh_embed: bool = False,
    target_n: int = 2000,
    max_rows: int | None = None,
    workers: int = 8,
    embed_batch_size: int = 128,
):
    """Run broad pipeline stages on Modal with per-stage gates."""
    stages = {
        "test_hashes": lambda: stage_test_hashes.remote(),
        "scan": lambda: stage_scan.remote(pilot=pilot, max_rows=max_rows, workers=workers),
        "embed": lambda: stage_embed.remote(pilot=pilot, fresh=fresh_embed, batch_size=embed_batch_size),
        "select": lambda: stage_select.remote(pilot=pilot, target_n=target_n),
        "visualize": lambda: stage_visualize.remote(pilot=pilot),
        "check": lambda: stage_check.remote(pilot=pilot, target_n=target_n),
    }

    if stage == "all":
        order = ["test_hashes", "scan", "embed", "select", "visualize"]
        results = {}
        for name in order:
            print(f"\n=== running {name} ===")
            results[name] = stages[name]()
        print("\n=== final check ===")
        results["check"] = stages["check"]()
        print(json.dumps(results, indent=2, default=str))
        return

    if stage not in stages:
        raise SystemExit(f"unknown stage {stage}; choose from {sorted(stages)} or all")
    print(stages[stage]())
