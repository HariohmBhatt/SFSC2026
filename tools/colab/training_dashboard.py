#!/usr/bin/env python3
"""Render an Ultralytics training log as a local, updating terminal dashboard.

The dashboard only reads a local log file. It does not connect to Colab, open an
SSH session, or write to the training directory.
"""

from __future__ import annotations

import argparse
import re
import shutil
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG = Path("outputs/ultimate-ensemble-v1/training-live.log")
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
    current: TrainPoint | None = None
    last_completed_epoch: int | None = None


@dataclass(frozen=True)
class DashboardState:
    runs: tuple[TrainingRun, ...]
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
        box_p=float(values["box_p"]),
        box_r=float(values["box_r"]),
        box_map50=float(values["box_map50"]),
        box_map=float(values["box_map"]),
        mask_p=float(values["mask_p"]),
        mask_r=float(values["mask_r"]),
        mask_map50=float(values["mask_map50"]),
        mask_map=float(values["mask_map"]),
    )


def parse_log(path: Path) -> DashboardState:
    stat = path.stat()
    content = path.read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    runs: list[TrainingRun] = []
    pending_name: str | None = None

    for raw_line in content.splitlines():
        line = clean_line(raw_line)
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

    return DashboardState(tuple(runs), stat.st_mtime, stat.st_size)


def compact_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


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


def render(
    state: DashboardState, path: Path, width: int, color: bool, run_number: int | None = None
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
    if args.refresh <= 0:
        raise SystemExit("--refresh must be greater than zero")
    if not path.is_file():
        raise SystemExit(f"Log file does not exist: {path}")

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
                screen = render(state, path, terminal_width, color, args.run)
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
