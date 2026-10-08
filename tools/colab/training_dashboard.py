#!/usr/bin/env python3
"""Render an Ultralytics training log as a local, updating terminal dashboard.

The dashboard only reads a local log file. It does not connect to Colab, open an
SSH session, or write to the training directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

DEFAULT_LOG = Path("outputs/ultimate-ensemble-v1/training-live.log")
DEFAULT_CONFIG = Path("filament_ensemble/config.json")
PROGRESS_PREFIX = "SFSC_PROGRESS "
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
TRAINER_NAME_RE = re.compile(r"(?:^|,\s*)name=([^,\s]+)")
TRAIN_RE = re.compile(
    r"^\s*(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"(?P<gpu>[\d.]+G)\s+"
    r"(?P<box>[-+\d.eE]+)\s+"
    r"(?P<seg>[-+\d.eE]+)\s+"
    r"(?P<cls>[-+\d.eE]+)\s+"
    r"(?P<dfl>[-+\d.eE]+)\s+"
    r"(?P<sem>[-+\d.eE]+)\s+"
    r"(?P<instances>\d+)\s+"
    r"(?P<size>\d+):\s+"
    r"(?P<percent>\d+)%.*?"
    r"(?P<batch>\d+)/(?P<batches>\d+)"
    r"(?:\s+(?P<rate>[\d.]+)it/s)?"
    r"(?:\s+(?P<elapsed>[^\s<]+)(?:<(?P<remaining>\S+))?)?\s*$"
)
VALIDATION_RE = re.compile(
    r"^\s*all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+"
    r"(?P<box_p>[-+\d.eE]+)\s+(?P<box_r>[-+\d.eE]+)\s+"
    r"(?P<box_map50>[-+\d.eE]+)\s+(?P<box_map>[-+\d.eE]+)\s+"
    r"(?P<mask_p>[-+\d.eE]+)\s+(?P<mask_r>[-+\d.eE]+)\s+"
    r"(?P<mask_map50>[-+\d.eE]+)\s+(?P<mask_map>[-+\d.eE]+)\s*$"
)
VALIDATION_TIMING_RE = re.compile(
    r"^\s*Class\s+Images.*?:\s+100%.*?\s(?P<elapsed>(?:\d+:)?\d+(?:\.\d+)?s)\s*$"
)
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


@dataclass(frozen=True)
class TrainPoint:
    epoch: int
    epochs: int
    gpu: str
    box: float
    seg: float
    cls: float
    dfl: float
    sem: float
    instances: int
    size: int
    percent: int
    batch: int
    batches: int
    rate: float | None
    elapsed: str | None
    remaining: str | None


@dataclass(frozen=True)
class ValidationPoint:
    epoch: int
    images: int
    instances: int
    box_p: float
    box_r: float
    box_map50: float
    box_map: float
    mask_p: float
    mask_r: float
    mask_map50: float
    mask_map: float


@dataclass
class TrainingRun:
    name: str
    points: dict[int, TrainPoint] = field(default_factory=dict)
    validations: dict[int, ValidationPoint] = field(default_factory=dict)
    validation_durations: list[float] = field(default_factory=list)
    current: TrainPoint | None = None
    last_completed_epoch: int | None = None


@dataclass(frozen=True)
class DashboardState:
    runs: tuple[TrainingRun, ...]
    events: tuple[dict[str, object], ...]
    modified_at: float
    byte_count: int

    @property
    def current_run(self) -> TrainingRun | None:
        return self.runs[-1] if self.runs else None


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, value: str, code: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.enabled else value

    def bold(self, value: str) -> str:
        return self.paint(value, "1")

    def dim(self, value: str) -> str:
        return self.paint(value, "2")

    def green(self, value: str) -> str:
        return self.paint(value, "32")

    def yellow(self, value: str) -> str:
        return self.paint(value, "33")

    def cyan(self, value: str) -> str:
        return self.paint(value, "36")

    def red(self, value: str) -> str:
        return self.paint(value, "31")


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\x00", "")


def train_point(match: re.Match[str]) -> TrainPoint:
    values = match.groupdict()
    return TrainPoint(
        epoch=int(values["epoch"]),
        epochs=int(values["epochs"]),
        gpu=values["gpu"],
        box=float(values["box"]),
        seg=float(values["seg"]),
        cls=float(values["cls"]),
        dfl=float(values["dfl"]),
        sem=float(values["sem"]),
        instances=int(values["instances"]),
        size=int(values["size"]),
        percent=int(values["percent"]),
        batch=int(values["batch"]),
        batches=int(values["batches"]),
        rate=float(values["rate"]) if values["rate"] else None,
        elapsed=values["elapsed"],
        remaining=values["remaining"],
    )


def validation_point(match: re.Match[str], epoch: int) -> ValidationPoint:
    values = match.groupdict()
    return ValidationPoint(
        epoch=epoch,
        images=int(values["images"]),
        instances=int(values["instances"]),
        box_p=float(values["box_p"]),
        box_r=float(values["box_r"]),
        box_map50=float(values["box_map50"]),
        box_map=float(values["box_map"]),
        mask_p=float(values["mask_p"]),
        mask_r=float(values["mask_r"]),
        mask_map50=float(values["mask_map50"]),
        mask_map=float(values["mask_map"]),
    )


def duration_seconds(value: str) -> float:
    value = value.removesuffix("s")
    parts = value.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0
    return 0.0


def parse_log(path: Path) -> DashboardState:
    stat = path.stat()
    content = path.read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    runs: list[TrainingRun] = []
    events: list[dict[str, object]] = []
    pending_name: str | None = None

    for raw_line in content.splitlines():
        line = clean_line(raw_line)
        if line.startswith(PROGRESS_PREFIX):
            try:
                event = json.loads(line.removeprefix(PROGRESS_PREFIX))
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(event, dict):
                    events.append(event)
            continue
        if line.startswith("{") and '"event": "validation_image"' in line:
            try:
                legacy = json.loads(line)
            except json.JSONDecodeError:
                pass
            else:
                events.append(
                    {
                        **legacy,
                        "event": "progress",
                        "phase": "validate",
                        "current": legacy.get("index", 0),
                    }
                )
            continue
        if "engine/trainer:" in line:
            match = TRAINER_NAME_RE.search(line)
            if match:
                pending_name = match.group(1)
            continue

        match = TRAIN_RE.match(line)
        if match:
            point = train_point(match)
            needs_run = not runs
            if runs and runs[-1].current is not None:
                previous = runs[-1].current
                needs_run = point.epoch < previous.epoch or (
                    pending_name is not None and pending_name != runs[-1].name
                )
            if needs_run:
                runs.append(TrainingRun(pending_name or f"YOLO training run {len(runs) + 1}"))
            run = runs[-1]
            run.current = point
            run.points[point.epoch] = point
            if point.percent == 100 or point.batch >= point.batches:
                run.last_completed_epoch = point.epoch
            pending_name = None
            continue

        match = VALIDATION_RE.match(line)
        if match and runs:
            run = runs[-1]
            epoch = run.last_completed_epoch
            if epoch is not None:
                run.validations[epoch] = validation_point(match, epoch)
            continue

        match = VALIDATION_TIMING_RE.match(line)
        if match and runs:
            elapsed = duration_seconds(match.group("elapsed"))
            if elapsed > 0:
                runs[-1].validation_durations.append(elapsed)

    return DashboardState(tuple(runs), tuple(events), stat.st_mtime, stat.st_size)


def compact_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def precise_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def item_eta(run: TrainingRun) -> tuple[float, float] | None:
    current = run.current
    if current is None or not current.rate:
        return None
    training_epoch = current.batches / current.rate
    validation = (
        sum(run.validation_durations[-10:]) / len(run.validation_durations[-10:])
        if run.validation_durations
        else max(5.0, training_epoch * 0.06)
    )
    checkpoint_overhead = 4.0
    current_training = max(current.batches - current.batch, 0) / current.rate
    epochs_after_current = max(current.epochs - current.epoch, 0)
    remaining = current_training + (epochs_after_current + 1) * (
        validation + checkpoint_overhead
    )
    remaining += epochs_after_current * training_epoch
    full_item = current.epochs * (training_epoch + validation + checkpoint_overhead)
    return remaining, full_item


def progress_bar(fraction: float, width: int) -> str:
    width = max(10, width)
    fraction = min(1.0, max(0.0, fraction))
    eighths = int(round(fraction * width * 8))
    full, partial = divmod(eighths, 8)
    blocks = "▏▎▍▌▋▊▉"
    result = "█" * min(full, width)
    if full < width and partial:
        result += blocks[partial - 1]
    return result.ljust(width, "·")


def sampled(values: Sequence[float], width: int) -> list[float]:
    if len(values) <= width:
        return list(values)
    if width <= 1:
        return [values[-1]]
    return [values[round(index * (len(values) - 1) / (width - 1))] for index in range(width)]


def sparkline(
    values: Sequence[float],
    width: int,
    axis: tuple[float, float] | None = None,
) -> str:
    if not values:
        return "(waiting for a completed epoch)"
    rows = sampled(values, max(1, width))
    low, high = axis or (min(rows), max(rows))
    if high == low:
        return "▄" * len(rows)
    glyphs = "▁▂▃▄▅▆▇█"
    return "".join(
        glyphs[round(min(1.0, max(0.0, (value - low) / (high - low))) * 7)]
        for value in rows
    )


def trend(values: Sequence[float]) -> str:
    if len(values) < 2:
        return "   n/a"
    delta = values[-1] - values[0]
    arrow = "↓" if delta < 0 else "↑" if delta > 0 else "→"
    return f"{arrow}{delta:+.4f}"


def metric_row(
    label: str,
    values: Sequence[float],
    graph_width: int,
    palette: Palette,
    color: str,
    axis: tuple[float, float] | None = None,
) -> str:
    if not values:
        return f"  {label:<10} {'--':>8}  {palette.dim('(waiting for data)')}"
    low, high = axis or (min(values), max(values))
    plot = palette.paint(sparkline(values, graph_width, (low, high)), color)
    y_axis = palette.dim(f"y[{low:.3f},{high:.3f}]")
    return f"  {label:<14} {values[-1]:>8.4f}  {plot}  {y_axis}  {trend(values):>9}"


def completed(run: TrainingRun) -> bool:
    point = run.current
    return bool(point and point.epoch >= point.epochs and point.batch >= point.batches)


def latest_phase_event_with(
    events: Sequence[dict[str, object]], phase: str, field: str
) -> dict[str, object]:
    return next(
        (
            event
            for event in reversed(events)
            if event.get("phase") == phase and event.get(field) is not None
        ),
        {},
    )


def active_phase(events: Sequence[dict[str, object]]) -> str | None:
    active: str | None = None
    for event in events:
        kind = event.get("event")
        phase = event.get("phase")
        if kind == "phase_start" and isinstance(phase, str):
            active = phase
        elif kind == "phase_complete" and phase == active:
            active = None
    return active


def current_pipeline_events(
    events: Sequence[dict[str, object]],
) -> Sequence[dict[str, object]]:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("event") == "pipeline_start":
            return events[index:]
    return events


def structured_eta_rows(
    events: Sequence[dict[str, object]],
    progress: dict[str, object],
    config: dict[str, object],
) -> list[str]:
    phase = progress.get("phase")
    unit = progress.get("unit")
    if phase == "validate":
        starts = [
            event
            for event in events
            if event.get("phase") == "validate"
            and event.get("event") == "phase_start"
            and isinstance(event.get("time"), str)
        ]
        completed = [
            event
            for event in events
            if event.get("phase") == "validate"
            and event.get("event") == "progress"
            and isinstance(event.get("time"), str)
        ]
        if not starts or not completed:
            return []
        previous = datetime.fromisoformat(str(starts[-1]["time"]))
        durations = []
        for event in completed:
            finished = datetime.fromisoformat(str(event["time"]))
            duration = (finished - previous).total_seconds()
            if duration >= 2.0:  # Ignore near-instant Drive cache hits.
                durations.append(duration)
            previous = finished
        if not durations:
            return []
        recent = durations[-8:]
        image_seconds = sum(recent) / len(recent)
        now = datetime.now(previous.tzinfo or UTC)
        elapsed = max((now - previous).total_seconds(), 0.0)
        current = int(progress.get("current", 0))
        total = int(progress.get("total", 0))
        remaining = max(image_seconds - elapsed, 0.0) + max(total - current - 1, 0) * image_seconds
        finish = datetime.fromtimestamp(time.time() + remaining, IST)
        return [
            f"  Exact-PQ ETA {precise_duration(remaining)}  •  "
            f"estimated finish {finish:%H:%M:%S IST}  •  "
            f"{precise_duration(image_seconds)}/image"
        ]
    if phase != "refiners" or not isinstance(unit, str):
        return []
    starts = [
        event
        for event in events
        if event.get("phase") == phase
        and event.get("unit") == unit
        and event.get("event") == "unit_start"
    ]
    completed_epochs = [
        event
        for event in events
        if event.get("phase") == phase
        and event.get("unit") == unit
        and event.get("event") == "epoch_complete"
        and isinstance(event.get("time"), str)
    ]
    if not starts or not completed_epochs or not isinstance(starts[-1].get("time"), str):
        return []
    previous = datetime.fromisoformat(str(starts[-1]["time"]))
    durations = []
    for event in completed_epochs:
        finished = datetime.fromisoformat(str(event["time"]))
        durations.append((finished - previous).total_seconds())
        previous = finished
    recent = durations[-8:]
    epoch_seconds = sum(recent) / len(recent)
    now = datetime.now(previous.tzinfo or UTC)
    elapsed = max((now - previous).total_seconds(), 0.0)
    epoch = int(progress.get("epoch", 0))
    epochs = int(progress.get("epochs", config.get("refiner_epochs", 0)))
    remaining = max(epoch_seconds - elapsed, 0.0) + max(epochs - epoch, 0) * epoch_seconds
    unit_index = int(progress.get("unit_index", 1))
    unit_total = int(
        progress.get(
            "unit_total",
            len(config.get("refiner_backbones", [])) * int(config.get("refiner_folds", 0)),
        )
    )
    stage_remaining = remaining + max(unit_total - unit_index, 0) * epochs * epoch_seconds
    item_finish = datetime.fromtimestamp(time.time() + remaining, IST)
    stage_finish = datetime.fromtimestamp(time.time() + stage_remaining, IST)
    return [
        f"  Current refiner ETA {precise_duration(remaining)}  •  "
        f"estimated finish {item_finish:%H:%M:%S IST}",
        f"  Refiner stage ETA   {precise_duration(stage_remaining)}  •  "
        f"estimated finish {stage_finish:%H:%M:%S IST}",
    ]


def pipeline_lines(
    state: DashboardState,
    config: dict[str, object],
    width: int,
    palette: Palette,
) -> list[str]:
    events = current_pipeline_events(state.events)
    detector_total = len(config.get("detectors", []))
    prompt_total = int(config.get("prompt_oof_folds", 0))
    refiner_total = len(config.get("refiner_backbones", [])) * int(
        config.get("refiner_folds", 0)
    )
    semantic_total = int(config.get("semantic_epochs", 0))
    production = [run for run in state.runs if run.name.startswith("detector-")]
    prompts = [run for run in state.runs if run.name.startswith("fold")]
    detector_done = min(detector_total, sum(completed(run) for run in production))
    prompt_done = min(prompt_total, sum(completed(run) for run in prompts))

    complete_phases = {
        str(event.get("phase"))
        for event in events
        if event.get("event") == "phase_complete"
    }
    if "detectors" in complete_phases:
        detector_done = detector_total
    prompt_units = {
        str(event.get("unit"))
        for event in events
        if event.get("phase") == "prompts" and event.get("event") == "unit_complete"
    }
    prompt_done = min(prompt_total, max(prompt_done, len(prompt_units)))
    if "prompts" in complete_phases:
        prompt_done = prompt_total
    refiner_units = {
        str(event.get("unit"))
        for event in events
        if event.get("phase") == "refiners" and event.get("event") == "unit_complete"
    }
    semantic_event = latest_phase_event_with(events, "semantic", "epoch")
    validation_event = latest_phase_event_with(events, "validate", "current")
    semantic_epoch = int(semantic_event.get("epoch", 0))
    validation_current = int(validation_event.get("current", 0))
    validation_total = int(validation_event.get("total", 0)) or int(
        config.get("_validation_total", 0)
    )

    structured_active = active_phase(events)
    if structured_active == "validate" or "validate" in complete_phases:
        detector_done = detector_total
        prompt_done = prompt_total
        semantic_epoch = semantic_total
        complete_phases.update({"detectors", "prompts", "refiners", "semantic"})
    inferred_active = structured_active
    if inferred_active is None:
        if validation_event:
            inferred_active = "validate"
        elif detector_done < detector_total:
            inferred_active = "detectors"
        elif prompt_done < prompt_total:
            inferred_active = "prompts"
        elif prompt_total and prompt_done >= prompt_total:
            inferred_active = "refiners"

    def marker(phase: str, done: bool) -> str:
        if done:
            return palette.green("✓")
        if inferred_active == phase:
            return palette.yellow("▶")
        return palette.dim("○")

    rows = [palette.bold("PIPELINE PROGRESS")]
    rows.append(
        f"  {marker('detectors', detector_done >= detector_total > 0)} "
        f"Production detectors  {detector_done}/{detector_total} complete"
    )
    rows.append(
        f"  {marker('prompts', prompt_done >= prompt_total > 0 or 'prompts' in complete_phases)} "
        f"OOF prompt detectors  {prompt_done}/{prompt_total} complete"
    )
    rows.append(
        f"  {marker('refiners', 'refiners' in complete_phases)} "
        f"Prompt refiners       "
        f"{refiner_total if 'refiners' in complete_phases else len(refiner_units)}/"
        f"{refiner_total} complete"
    )
    rows.append(
        f"  {marker('semantic', 'semantic' in complete_phases)} "
        f"Semantic proposal     epoch {semantic_epoch}/{semantic_total}"
    )
    rows.append(
        f"  {marker('validate', 'validate' in complete_phases)} "
        f"Exact-PQ validation   image {validation_current}/{validation_total or '?'}"
    )
    failure = next(
        (event for event in reversed(events) if event.get("event") == "pipeline_failed"),
        None,
    )
    if failure:
        rows.append(palette.red(f"  Pipeline failure: {failure.get('error', 'unknown error')}"))

    progress = next(
        (
            event
            for event in reversed(events)
            if event.get("event") == "progress"
            and isinstance(event.get("current"), (int, float))
            and isinstance(event.get("total"), (int, float))
        ),
        None,
    )
    if progress and progress.get("phase") == structured_active:
        current = float(progress["current"])
        total = max(float(progress["total"]), 1.0)
        fraction = current / total
        bar_width = max(16, min(48, width - 38))
        detail = str(progress.get("unit") or progress.get("filename") or "")
        rows.extend(
            [
                "",
                (
                    f"  {palette.cyan(progress_bar(fraction, bar_width))} "
                    f"{fraction * 100:6.2f}%  {detail}"
                ),
            ]
        )
        epoch = progress.get("epoch")
        epochs = progress.get("epochs")
        step = progress.get("step")
        steps = progress.get("steps")
        details = []
        if epoch is not None and epochs is not None:
            details.append(f"epoch {epoch}/{epochs}")
        if step is not None and steps is not None:
            details.append(f"step {step}/{steps}")
        if progress.get("train_loss") is not None:
            details.append(f"train loss {float(progress['train_loss']):.4f}")
        if details:
            rows.append("  " + "  •  ".join(details))
        rows.extend(structured_eta_rows(events, progress, config))
    return rows


def structured_metric_lines(
    state: DashboardState, width: int, palette: Palette
) -> list[str]:
    events = current_pipeline_events(state.events)
    phase = active_phase(events)
    if phase not in {"refiners", "semantic"}:
        return []
    epoch_events = [
        event
        for event in events
        if event.get("phase") == phase and event.get("event") == "epoch_complete"
    ]
    if not epoch_events:
        return []
    latest_unit = epoch_events[-1].get("unit")
    if latest_unit is not None:
        epoch_events = [event for event in epoch_events if event.get("unit") == latest_unit]
    graph_width = max(12, min(48, width - 58))
    train_losses = [float(event["train_loss"]) for event in epoch_events]
    validation_losses = [
        float(event["validation_loss"])
        for event in epoch_events
        if event.get("validation_loss") is not None
    ]
    rows = [
        "",
        palette.bold("CURRENT STAGE METRICS")
        + palette.dim(f"  {phase} • {latest_unit or 'full-disk'}"),
        metric_row("train loss", train_losses, graph_width, palette, "35"),
    ]
    if validation_losses:
        rows.append(metric_row("validation loss", validation_losses, graph_width, palette, "36"))
    return rows


def render(
    state: DashboardState,
    path: Path,
    width: int,
    color: bool,
    config: dict[str, object],
    run_number: int | None = None,
) -> str:
    palette = Palette(color)
    width = max(72, width)
    graph_width = max(12, min(48, width - 58))
    divider = "─" * min(width, 110)
    age = time.time() - state.modified_at
    freshness = palette.green("LIVE") if age < 15 else palette.red(f"STALE {compact_age(age)}")
    lines = [
        palette.bold("SFSC TRAINING DASHBOARD") + f"  {freshness}",
        palette.dim(f"local source: {path}  •  {state.byte_count / 1_048_576:.1f} MiB"),
        divider,
    ]
    lines.extend(pipeline_lines(state, config, width, palette))
    lines.extend(structured_metric_lines(state, width, palette))
    lines.append(divider)

    if run_number is not None and not 1 <= run_number <= len(state.runs):
        lines.append(
            palette.yellow(
                f"Requested run {run_number}, but the log currently contains "
                f"{len(state.runs)} run(s)."
            )
        )
        return "\n".join(lines)
    selected_run = run_number or len(state.runs)
    run = state.runs[selected_run - 1] if state.runs else None
    if run is None or run.current is None:
        lines.extend(
            [
                palette.yellow("Waiting for an Ultralytics training row in the log."),
                "The file is readable, but no epoch progress has been detected yet.",
            ]
        )
        return "\n".join(lines)

    current = run.current
    epoch_fraction = current.batch / max(current.batches, 1)
    overall = ((current.epoch - 1) + epoch_fraction) / max(current.epochs, 1)
    bar_width = max(20, min(62, width - 34))
    pipeline_phase = active_phase(current_pipeline_events(state.events))
    eta = None if pipeline_phase == "validate" else item_eta(run)
    eta_lines: list[str] = []
    if eta is not None:
        remaining, full_item = eta
        item_finish = datetime.fromtimestamp(time.time() + remaining, IST)
        eta_lines.append(
            f"Current item ETA {precise_duration(remaining)}  •  "
            f"estimated finish {item_finish:%H:%M:%S IST}"
        )
        if run.name.startswith("fold") and selected_run == len(state.runs):
            events = current_pipeline_events(state.events)
            completed_prompts = {
                str(event.get("unit"))
                for event in events
                if event.get("phase") == "prompts"
                and event.get("event") == "unit_complete"
            }
            prompt_total = int(config.get("prompt_oof_folds", 0))
            later_folds = max(prompt_total - len(completed_prompts) - 1, 0)
            stage_remaining = remaining + later_folds * full_item
            stage_finish = datetime.fromtimestamp(time.time() + stage_remaining, IST)
            eta_lines.append(
                f"OOF stage ETA    {precise_duration(stage_remaining)}  •  "
                f"estimated finish {stage_finish:%H:%M:%S IST}"
            )
    lines.extend(
        [
            f"Run {selected_run}/{len(state.runs)}  {palette.bold(run.name)}",
            (
                f"{palette.cyan(progress_bar(overall, bar_width))} "
                f"{overall * 100:6.2f}%"
            ),
            (
                f"Epoch {current.epoch}/{current.epochs}  •  batch "
                f"{current.batch}/{current.batches} ({current.percent}%)  •  "
                f"{current.rate or 0:.1f} it/s  •  epoch ETA {current.remaining or '--'}"
            ),
            *eta_lines,
            (
                f"GPU {current.gpu}  •  image {current.size}²  •  "
                f"instances in batch {current.instances}  •  log age {compact_age(age)}"
            ),
            divider,
            palette.bold("LOSS HISTORY")
            + palette.dim(
                "  latest running average • observed y-axis • change since first local epoch"
            ),
        ]
    )

    epochs = sorted(run.points)
    loss_specs = [
        ("box", [run.points[epoch].box for epoch in epochs], "36"),
        ("segmentation", [run.points[epoch].seg for epoch in epochs], "35"),
        ("classification", [run.points[epoch].cls for epoch in epochs], "33"),
        ("DFL", [run.points[epoch].dfl for epoch in epochs], "34"),
        ("semantic", [run.points[epoch].sem for epoch in epochs], "32"),
    ]
    for label, values, metric_color in loss_specs:
        lines.append(metric_row(label, values, graph_width, palette, metric_color))
    lines.append(
        palette.dim(
            f"  epochs represented: {epochs[0]}–{epochs[-1]} "
            f"({len(epochs)} points; current epoch included)"
        )
    )

    lines.extend(
        [
            divider,
            palette.bold("VALIDATION HISTORY")
            + palette.dim("  observed y-axis for readable trend detail"),
        ]
    )
    validation_epochs = sorted(run.validations)
    map50 = [run.validations[epoch].mask_map50 for epoch in validation_epochs]
    map5095 = [run.validations[epoch].mask_map for epoch in validation_epochs]
    precision = [run.validations[epoch].mask_p for epoch in validation_epochs]
    recall = [run.validations[epoch].mask_r for epoch in validation_epochs]
    lines.append(metric_row("mask mAP50", map50, graph_width, palette, "32"))
    lines.append(metric_row("mask mAP50-95", map5095, graph_width, palette, "32"))
    lines.append(metric_row("precision", precision, graph_width, palette, "36"))
    lines.append(metric_row("recall", recall, graph_width, palette, "36"))
    if validation_epochs:
        lines.append(
            palette.dim(
                f"  validation epochs: {validation_epochs[0]}–{validation_epochs[-1]} "
                f"({len(validation_epochs)} points)"
            )
        )

    lines.extend(
        [
            divider,
            palette.dim(
                f"Runs found: {len(state.runs)}  •  refreshes locally  •  "
                "use --run N for history  •  Ctrl+C exits dashboard only"
            ),
        ]
    )
    return "\n".join(lines)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a single-screen dashboard for a local Ultralytics training log."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="local log file to read")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="pipeline configuration used for expected stage counts",
    )
    parser.add_argument(
        "--refresh", type=float, default=1.0, help="screen refresh interval in seconds (default: 1)"
    )
    parser.add_argument("--once", action="store_true", help="render once and exit")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument(
        "--run",
        type=int,
        help="show a specific detected model run (default: latest run)",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    path = args.log.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if args.refresh <= 0:
        raise SystemExit("--refresh must be greater than zero")
    if not path.is_file():
        raise SystemExit(f"Log file does not exist: {path}")
    if not config_path.is_file():
        raise SystemExit(f"Pipeline config does not exist: {config_path}")
    config = json.loads(config_path.read_text())
    split_manifest = config_path.parent.parent / str(config.get("split_manifest", ""))
    if split_manifest.is_file():
        with split_manifest.open(newline="") as stream:
            config["_validation_total"] = sum(
                row.get("split") == "validation" for row in csv.DictReader(stream)
            )

    interactive = sys.stdout.isatty() and not args.once
    color = sys.stdout.isatty() and not args.no_color
    if interactive:
        sys.stdout.write("\033[?1049h\033[?25l")
        sys.stdout.flush()

    def restore_terminal() -> None:
        if interactive:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()

    def handle_sigterm(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        while True:
            try:
                state = parse_log(path)
                terminal_width = shutil.get_terminal_size((110, 30)).columns
                screen = render(state, path, terminal_width, color, config, args.run)
            except (OSError, UnicodeError) as error:
                screen = f"Could not read {path}: {error}"
            if interactive:
                sys.stdout.write("\033[H\033[2J" + screen)
                sys.stdout.flush()
            else:
                print(screen)
            if args.once or not interactive:
                return 0
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0
    finally:
        restore_terminal()


if __name__ == "__main__":
    raise SystemExit(main())
