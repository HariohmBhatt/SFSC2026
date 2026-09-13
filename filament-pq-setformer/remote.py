"""Execute the PQ-SetFormer E1 training smoke on a Kaggle P100."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import core
import data
import experiment_logging
import model as model_module
import numpy as np
import torch
from config import ExperimentConfig
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

WORK = Path("/kaggle/working/pq-setformer")
TASK = json.loads(Path("/kaggle/working/task-config.json").read_text())
AMP_INITIAL_SCALE = 16.0
MAX_AMP_OVERFLOWS = 4


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Record resolved package and pinned-upstream provenance in every report."""
    names = (
        "torch",
        "torchvision",
        "detectron2",
        "maskdino",
        "numpy",
        "scipy",
        "pycocotools",
        "wandb",
    )
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "vendored-unpackaged"
    lock = json.loads(Path("/kaggle/working/vendor-lock.json").read_text())
    versions["detectron2_commit"] = str(lock["detectron2"]["commit"])
    versions["maskdino_commit"] = str(lock["maskdino"]["commit"])
    return versions


def verify_source_snapshot() -> None:
    source = {name: (Path("/kaggle/working") / name).read_text() for name in TASK["source_files"]}
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != TASK["source_sha256"]:
        raise ValueError("Extracted source snapshot hash mismatch")
    lock = source.get("vendor-lock.json", "").encode()
    if hashlib.sha256(lock).hexdigest() != TASK["vendor_lock_sha256"]:
        raise ValueError("Extracted vendor lock hash mismatch")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def gpu_preflight() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    architectures = torch.cuda.get_arch_list()
    if "P100" not in name or capability != (6, 0) or "sm_60" not in architectures:
        raise RuntimeError(
            f"Invalid GPU runtime: name={name}, capability={capability}, arches={architectures}"
        )
    device = torch.device("cuda:0")
    tensor = torch.randn(512, 512, device=device, requires_grad=True)
    loss = (tensor @ tensor).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(tensor.grad).all():
        raise RuntimeError("Real FP32 CUDA backward was non-finite")
    del tensor, loss
    torch.cuda.empty_cache()
    from maskdino.modeling.pixel_decoder.ops.functions import MSDeformAttnFunction
    from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
        ms_deform_attn_core_pytorch,
    )

    shapes = torch.tensor(((2, 2), (1, 1)), dtype=torch.long, device=device)
    level_starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.rand(1, 5, 2, 4, device=device, requires_grad=True)
    locations = torch.rand(1, 3, 2, 2, 2, 2, device=device, requires_grad=True)
    attention_logits = torch.rand(1, 3, 2, 2, 2, device=device, requires_grad=True)
    attention = attention_logits.flatten(-2).softmax(-1).view_as(attention_logits)
    output = MSDeformAttnFunction.apply(
        value,
        shapes,
        level_starts,
        locations,
        attention,
        2,
    )
    reference = ms_deform_attn_core_pytorch(value, shapes, locations, attention)
    if not torch.allclose(output, reference, rtol=1e-2, atol=1e-3):
        raise RuntimeError("MaskDINO deformable-attention CUDA output failed parity")
    custom_loss = output.square().mean()
    custom_loss.backward()
    custom_tensors = (output, value.grad, locations.grad, attention_logits.grad)
    if any(tensor is None or not torch.isfinite(tensor).all() for tensor in custom_tensors):
        raise RuntimeError("MaskDINO deformable-attention FP32 forward/backward failed")
    del value, locations, attention_logits, attention, output, reference, custom_loss
    torch.cuda.empty_cache()
    return {
        "name": name,
        "capability": list(capability),
        "architectures": architectures,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "custom_op_forward_backward": True,
    }


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def _loss_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {f"train/{name}": float(value.detach()) for name, value in losses.items()}
    aliases = {
        "loss_ce": "train/loss_cls",
        "loss_mask": "train/loss_mask_bce",
        "loss_dice": "train/loss_dice",
        "loss_bbox": "train/loss_box_l1",
        "loss_giou": "train/loss_giou",
    }
    for source, target in aliases.items():
        values = [
            float(value.detach()) for name, value in losses.items() if name.startswith(source)
        ]
        if values:
            result[target] = float(sum(values))
    return result


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Report named optimizer buckets without depending on group ordering."""
    values = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
    decoder = [value for name, value in values.items() if name.startswith("decoder-")]
    backbone = [value for name, value in values.items() if name.startswith("backbone-")]
    if not decoder or not backbone or len(set(decoder)) != 1 or len(set(backbone)) != 1:
        raise RuntimeError(
            "Optimizer groups do not expose one decoder and one backbone learning rate"
        )
    return {"optim/lr_decoder": decoder[0], "optim/lr_backbone": backbone[0]}


def train_smoke(
    network: model_module.PQSetFormer,
    loader: DataLoader,
    experiment: ExperimentConfig,
    logger: experiment_logging.ExperimentLogger,
    microbatches: int,
) -> tuple[dict, torch.optim.Optimizer, object]:
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(
        model_module.optimizer_groups(network, experiment),
        weight_decay=experiment.training.weight_decay,
    )
    if microbatches % experiment.training.accumulation:
        raise ValueError("Training microbatches must divide exactly into optimizer steps")
    target_steps = microbatches // experiment.training.accumulation
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, target_steps, experiment.training.warmup_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=experiment.training.amp,
        init_scale=AMP_INITIAL_SCALE,
        growth_interval=1000,
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    optimizer_step = 0
    amp_overflows = 0
    attempted_microbatches = 0
    minimum_loss = math.inf
    maximum_loss = -math.inf
    last_loss = None
    last_record = None
    optimizer_started = time.monotonic()
    for microbatch, batch in enumerate(loader, 1):
        attempted_microbatches = microbatch
        last_record = {
            "filename": batch[0]["filename"],
            "annotation_record": batch[0]["annotation_record"],
            "presentation": batch[0]["presentation"],
        }
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=experiment.training.amp,
        ):
            losses = network.train_batch(batch)
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at {last_record}: {losses}")
        scalar_loss = float(total_loss.detach())
        minimum_loss = min(minimum_loss, scalar_loss)
        maximum_loss = max(maximum_loss, scalar_loss)
        last_loss = scalar_loss
        scaler.scale(total_loss / experiment.training.accumulation).backward()
        if microbatch % experiment.training.accumulation:
            continue

        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(network.parameters(), experiment.training.gradient_clip)
        if not torch.isfinite(grad_norm):
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            amp_overflows += 1
            logger.log(
                "amp_overflow",
                step=optimizer_step,
                **{
                    "amp/old_scale": float(old_scale),
                    "amp/new_scale": float(new_scale),
                    "amp/skipped_step": 1,
                    "microbatch": microbatch,
                },
            )
            if new_scale >= old_scale:
                raise FloatingPointError("GradScaler did not back off after non-finite gradients")
            if amp_overflows > MAX_AMP_OVERFLOWS:
                raise FloatingPointError(f"Repeated AMP overflows at {last_record}")
            continue
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise FloatingPointError(f"AMP skipped an optimizer update at {last_record}")
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_step += 1
        if not model_module.finite_parameters(network):
            raise FloatingPointError(f"Model parameters became non-finite at {last_record}")

        elapsed = time.monotonic() - optimizer_started
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        metrics = {
            **_loss_metrics(losses),
            "train/loss_total": scalar_loss,
            **optimizer_learning_rates(optimizer),
            "optim/grad_norm": float(grad_norm),
            "amp/enabled": experiment.training.amp,
            "amp/scale": float(scaler.get_scale()),
            "amp/skipped_step": 0,
            "perf/step_ms": elapsed * 1000,
            "perf/images_per_second": experiment.training.accumulation / max(elapsed, 1e-6),
            "memory/allocated_gb": memory_allocated,
            "memory/reserved_gb": memory_reserved,
            "microbatch": microbatch,
            "epoch_equivalent": microbatch / len(loader.dataset.filenames),
        }
        logger.log("optimizer_step", step=optimizer_step, **metrics)
        write_json(
            WORK / "progress.json",
            {
                "status": "training",
                "run_id": TASK["run_id"],
                "optimizer_step": optimizer_step,
                "microbatch": microbatch,
                "loss": scalar_loss,
                "last_record": last_record,
            },
        )
        optimizer_started = time.monotonic()
        if optimizer_step == target_steps:
            break

    if optimizer_step != target_steps or last_loss is None:
        raise RuntimeError("Training did not execute the declared optimizer-step count")
    return (
        {
            "microbatches": microbatches,
            "attempted_microbatches": attempted_microbatches,
            "optimizer_steps": optimizer_step,
            "amp_overflows": amp_overflows,
            "minimum_loss": minimum_loss,
            "maximum_loss": maximum_loss,
            "last_loss": last_loss,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            "last_record": last_record,
        },
        optimizer,
        scaler,
    )


def infer_smoke(
    network: model_module.PQSetFormer,
    annotation_index: data.AnnotationIndex,
    image_directory: Path,
    names: list[str],
    experiment: ExperimentConfig,
    checkpoint_sha256: str,
) -> tuple[dict, dict]:
    cache = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    per_file = {}
    for presentation, filename in enumerate(names):
        item = data.build_training_item(
            index=annotation_index,
            image_directory=image_directory,
            filenames=[filename],
            presentation=presentation,
            input_size=experiment.model.input_size,
            seed=experiment.training.seed,
            augment=False,
        )
        raw = network.predict(
            [item],
            [(2048, 2048)],
            presence_floor=experiment.assembly.presence_floor,
            mask_threshold=experiment.assembly.mask_threshold,
        )[0]
        candidates = [
            core.Candidate(
                row["query_id"],
                row["mask"],
                row["presence"],
                row["predicted_iou"],
                row["mask_probability"],
                presence_logit=row["presence_logit"],
                source=row["source"],
            )
            for row in raw
        ]
        masks = core.assemble_instances(candidates, recipe=experiment.assembly)
        records = [
            list(data.rasterize_record(record, native_size=2048, output_size=2048))
            for record in annotation_index.records_by_file[filename]
        ]
        metrics = core.evaluate_run({filename: masks}, {filename: records})
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_iou += metrics["sum_iou"]
        per_file[filename] = metrics["files"][filename]
        file_denominator = metrics["tp"] + 0.5 * (metrics["fp"] + metrics["fn"])
        file_iou = metrics["sum_iou"]
        per_file[filename].update(
            pq=file_iou / file_denominator if file_denominator else 0.0,
            sq=file_iou / metrics["tp"] if metrics["tp"] else 0.0,
            rq=metrics["tp"] / file_denominator if file_denominator else 0.0,
        )
        rows = [
            core.serialize_candidate(
                candidate,
                maximum_probability_bytes=experiment.model.candidate_probability_cache_bytes,
            )
            for candidate in candidates
        ]
        for row, overlap in zip(rows, core.candidate_overlap_summaries(candidates), strict=True):
            row.update(
                filename=filename,
                checkpoint_sha256=checkpoint_sha256,
                **overlap,
            )
            row["row_sha256"] = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        cache[filename] = rows
        del item, raw, candidates, masks, records
    if not any(cache.values()):
        raise RuntimeError("Stage inference collapsed to an all-empty candidate set")
    denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    metrics = {
        "pq": total_iou / denominator if denominator else 0.0,
        "sq": total_iou / total_tp if total_tp else 0.0,
        "rq": total_tp / denominator if denominator else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "sum_iou": total_iou,
        "files": per_file,
    }
    return cache, metrics


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / "progress.json", {"status": "starting", "run_id": TASK["run_id"]})
    verify_source_snapshot()
    experiment = ExperimentConfig.from_dict(TASK["experiment"])
    if experiment.sha256 != TASK["config_sha256"]:
        raise ValueError("Experiment config hash mismatch")
    run_kind = TASK["run_kind"]
    expected_microbatches = {
        "smoke": experiment.training.smoke_microbatches,
        "preflight": experiment.training.preflight_microbatches,
        "pilot": experiment.training.target_microbatches,
    }.get(run_kind)
    if (
        run_kind not in {"smoke", "preflight", "pilot"}
        or TASK["training_microbatches"] != expected_microbatches
    ):
        raise ValueError("Frozen run kind and training schedule disagree")
    seed_everything(experiment.training.seed)
    gpu = gpu_preflight()
    training_root = data.discover_dataset_root(Path("/kaggle/input"))
    observed_dataset = {
        "source": TASK["dataset"]["source"],
        **data.training_dataset_snapshot(training_root),
    }
    if observed_dataset != TASK["dataset"]:
        raise ValueError("Private training dataset snapshot mismatch")
    annotation_path = training_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    annotation_index = data.load_annotation_index(
        annotation_path, set(TASK["training_names"] + TASK["inference_names"])
    )
    if annotation_index.sha256 != TASK["dataset"]["annotation_sha256"]:
        raise ValueError("Private dataset annotation index hash mismatch")
    if annotation_index.maximum_instances > experiment.model.audited_max_instances:
        raise ValueError("Runtime annotation count exceeds the frozen query-capacity audit")

    logger_context = {
        "experiment_run_id": TASK["run_id"],
        "stage": TASK["stage"],
        "execution_source_sha256": TASK["source_sha256"],
        "project": experiment.logging.project,
        "job_type": "train",
        "tags": ["E1", run_kind, str(experiment.model.input_size), gpu["name"]],
        "config": {**experiment.to_dict(), "task": TASK},
        "wandb_version": experiment.logging.wandb_version,
    }
    with experiment_logging.ExperimentLogger.start(logger_context, WORK / "logs") as logger:
        dataset = data.VirtualTrainingDataset(
            index=annotation_index,
            image_directory=training_root / "train_images",
            filenames=TASK["training_names"],
            presentations=expected_microbatches
            + experiment.training.accumulation * MAX_AMP_OVERFLOWS,
            input_size=experiment.model.input_size,
            seed=experiment.training.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=experiment.training.physical_batch,
            shuffle=False,
            num_workers=experiment.training.workers,
            pin_memory=True,
            persistent_workers=experiment.training.workers > 0,
            prefetch_factor=2 if experiment.training.workers > 0 else None,
            collate_fn=data.instance_collate,
        )
        network = model_module.PQSetFormer(experiment).to(torch.device("cuda:0"))
        initial_fingerprint = hashlib.sha256(
            next(network.parameters()).detach().cpu().numpy().tobytes()
        ).hexdigest()
        training, optimizer, scaler = train_smoke(
            network, loader, experiment, logger, expected_microbatches
        )
        if training["peak_reserved_gb"] > experiment.model.peak_vram_gb:
            raise RuntimeError(
                f"Peak reserved memory {training['peak_reserved_gb']:.3f} GB exceeds "
                f"the {experiment.model.peak_vram_gb:.3f} GB limit"
            )
        final_fingerprint = hashlib.sha256(
            next(network.parameters()).detach().cpu().numpy().tobytes()
        ).hexdigest()
        if initial_fingerprint == final_fingerprint:
            raise RuntimeError("Optimizer did not change the model fingerprint")

        checkpoint = WORK / f"{run_kind}-checkpoint.pt"
        torch.save(
            {
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "experiment": experiment.to_dict(),
                "config_sha256": experiment.sha256,
                "source_sha256": TASK["source_sha256"],
                "vendor_lock_sha256": TASK["vendor_lock_sha256"],
                "git_sha": TASK["git_sha"],
                "membership_sha256": TASK["membership_sha256"],
                "dataset": TASK["dataset"],
                "stage": TASK["stage"],
                "run_kind": run_kind,
                "training_microbatches": expected_microbatches,
                "training": training,
            },
            checkpoint,
        )
        del network, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded = model_module.PQSetFormer(experiment).to(torch.device("cuda:0"))
        reloaded.load_state_dict(saved["model"], strict=True)
        if not model_module.finite_parameters(reloaded):
            raise RuntimeError("Reloaded checkpoint contains non-finite parameters")
        del saved
        torch.cuda.empty_cache()

        cache, metrics = infer_smoke(
            reloaded,
            annotation_index,
            training_root / "train_images",
            TASK["inference_names"],
            experiment,
            digest(checkpoint),
        )
        cache_path = WORK / f"{run_kind}-candidates.json"
        write_json(cache_path, cache)
        report = {
            "status": "complete",
            "run_id": TASK["run_id"],
            "stage": TASK["stage"],
            "run_kind": run_kind,
            "training_microbatches": expected_microbatches,
            "source_sha256": TASK["source_sha256"],
            "vendor_lock_sha256": TASK["vendor_lock_sha256"],
            "git_sha": TASK["git_sha"],
            "membership_sha256": TASK["membership_sha256"],
            "config_sha256": experiment.sha256,
            "dataset": TASK["dataset"],
            "runtime_versions": runtime_versions(),
            "annotation_sha256": annotation_index.sha256,
            "gpu": gpu,
            "training": training,
            "memory_limit_gb": experiment.model.peak_vram_gb,
            "inference": metrics,
            "candidate_images": len(cache),
            "candidate_count": sum(map(len, cache.values())),
            "checkpoint_sha256": digest(checkpoint),
            "candidates_sha256": digest(cache_path),
            "model_parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
            "initialization": reloaded.initialization_provenance,
            "quality_head": True,
            "quality_kind": "conditional_mask_iou",
            "wandb_run_id": logger.manifest["wandb_run_id"],
        }
        report_path = WORK / f"{run_kind}-report.json"
        write_json(report_path, report)
        logger.log(
            f"{run_kind}_evaluation",
            **{
                "eval/global_pq": metrics["pq"],
                "eval/sq": metrics["sq"],
                "eval/rq": metrics["rq"],
                "eval/tp": metrics["tp"],
                "eval/fp": metrics["fp"],
                "eval/fn": metrics["fn"],
                "eval/sum_matched_iou": metrics["sum_iou"],
            },
        )
        logger.log_table(
            f"{run_kind}_images",
            [
                {
                    "filename": name,
                    "candidates": len(rows),
                    **metrics["files"][name],
                }
                for name, rows in sorted(cache.items())
            ],
        )
        logger.log_artifact(
            f"{run_kind}-model",
            [checkpoint],
            {"config_sha256": experiment.sha256, "source_sha256": TASK["source_sha256"]},
        )
        logger.log_artifact(
            f"{run_kind}-evaluation",
            [cache_path, report_path],
            {"membership_sha256": TASK["membership_sha256"]},
        )
    wandb_bundles = list((WORK / "logs/wandb").rglob("run-*.wandb"))
    if len(wandb_bundles) != 1:
        raise RuntimeError(f"Expected one offline W&B bundle, found {len(wandb_bundles)}")
    report["wandb_bundle_sha256"] = digest(wandb_bundles[0])
    write_json(report_path, report)
    write_json(WORK / "progress.json", {"status": "complete", "run_id": TASK["run_id"]})


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "offline")
    main()
