"""Checkpoint scheduling at training-percent milestones."""
from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import TrainerCallback


def estimate_total_steps(
    n_samples: int,
    *,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps * world_size)
    steps_per_epoch = max(1, math.ceil(n_samples / effective_batch))
    return max(1, int(steps_per_epoch * num_train_epochs))


def pct_to_steps(checkpoints_pct: list[int | float], total_steps: int) -> dict[int, int]:
    """Map checkpoint pct → global step (1-indexed training steps)."""
    out: dict[int, int] = {}
    for pct in checkpoints_pct:
        p = int(pct)
        if p <= 0:
            out[p] = 0
        elif p >= 100:
            out[p] = total_steps
        else:
            out[p] = max(1, round(total_steps * p / 100))
    return out


class TrainMetricsCallback(TrainerCallback):
    """Append training metrics to disk for loss-curve plotting."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.trainer = None

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        entry = {"step": int(state.global_step), **logs}
        with (self.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_log.json").write_text(
            json.dumps(state.log_history, indent=2),
            encoding="utf-8",
        )
        if self.trainer is not None:
            self.trainer.save_state()
        print(f"[train] saved train_log.json ({len(state.log_history)} points)", flush=True)


class CheckpointPctCallback(TrainerCallback):
    """Save LoRA adapter snapshots at configured pct milestones."""

    def __init__(
        self,
        *,
        checkpoints_pct: list[int | float],
        total_steps: int,
        output_dir: str | Path,
        trainer=None,
    ):
        self.output_dir = Path(output_dir)
        self.total_steps = total_steps
        self.step_map = pct_to_steps(checkpoints_pct, total_steps)
        self.pending = {step: pct for pct, step in self.step_map.items() if pct > 0 and step > 0}
        self.manifest: list[dict] = []
        self.trainer = trainer

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer

    def _save(self, pct: int, step: int, *, tag: str) -> None:
        if self.trainer is None:
            return
        ckpt_dir = self.output_dir / f"checkpoint_pct_{pct:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(ckpt_dir))
        entry = {
            "pct": pct,
            "step": step,
            "total_steps": self.total_steps,
            "path": str(ckpt_dir),
            "tag": tag,
        }
        self.manifest.append(entry)
        self._write_manifest()
        print(f"[train] saved checkpoint {pct}% at step {step} → {ckpt_dir}", flush=True)

    def _write_manifest(self) -> None:
        path = self.output_dir / "checkpoint_manifest.json"
        path.write_text(json.dumps({"checkpoints": self.manifest}, indent=2), encoding="utf-8")

    def on_train_begin(self, args, state, control, **kwargs):
        print("[train] train loop starting (saving checkpoint 0% if scheduled)...", flush=True)
        if 0 in self.step_map:
            self._save(0, 0, tag="base_plus_init_lora")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.pending:
            pct = self.pending.pop(step)
            self._save(pct, step, tag="scheduled")

    def on_train_end(self, args, state, control, **kwargs):
        if 100 in self.step_map:
            final_step = int(state.global_step)
            if not any(e.get("pct") == 100 for e in self.manifest):
                self._save(100, final_step, tag="final")


class SampleProgressCallback(TrainerCallback):
    """Print one status line every `interval` training examples (image→SVG rows seen)."""

    def __init__(self, *, n_samples: int, num_epochs: float, interval: int = 100):
        self.n_samples = n_samples
        self.num_epochs = num_epochs
        self.interval = max(1, interval)
        self.total_examples = max(1, int(n_samples * num_epochs))
        self._next_milestone = self.interval

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) == 1:
            loss = state.log_history[-1].get("loss") if state.log_history else None
            print(f"[train] step 1 complete loss={loss}", flush=True)
        effective = max(
            1,
            args.per_device_train_batch_size * args.gradient_accumulation_steps,
        )
        seen = int(state.global_step) * effective
        if seen < self._next_milestone:
            return
        loss = state.log_history[-1].get("loss") if state.log_history else None
        epoch = state.log_history[-1].get("epoch") if state.log_history else "?"
        print(
            f"[train] examples {seen}/{self.total_examples} "
            f"step {state.global_step} epoch {epoch} loss={loss}",
            flush=True,
        )
        while self._next_milestone <= seen:
            self._next_milestone += self.interval
