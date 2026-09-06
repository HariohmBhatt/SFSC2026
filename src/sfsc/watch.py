"""Track public competition activity: leaderboard, notebooks, topics, and methods.

Fetches public Kaggle state through the authenticated Kaggle CLI, keeps a
versioned local state store, pulls notebook sources for new or changed kernels,
extracts method tags and validity warnings, and renders a self-contained HTML
dashboard. The tool never mutates competition state; it only reads.
"""

from __future__ import annotations

import argparse
import csv
import functools
import http.server
import io
import json
import re
import subprocess
import threading
import time
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_COMPETITION = "filament-segmentation-2026"
DEFAULT_OUT = Path("reports/competition_watch")
DEFAULT_SEARCH = "solar filament"
CLI_TIMEOUT = 180.0
MAX_STORED_SOURCE_BYTES = 256_000
SNAPSHOT_CAP = 120

AUDIT_NOTES = [
    {
        "date": "2026-09-05",
        "text": (
            "The two top-voted baseline U-Net notebooks use a filename-keyed lookup "
            "(annotations.get(filename)) against the COCO-style annotation JSON. It never matches "
            "(0/1154 records), so they train on all-empty masks and submit one empty RLE per test "
            "image. On our 106-image validation split that submission path scores PQ = 0.0000 "
            "(0 TP, 1237 FN). Their 0.9/'1st' leaderboard scores ride the public LB empty/missing "
            "prediction defect, not segmentation quality."
        ),
    },
    {
        "date": "2026-09-05",
        "text": (
            "Fususu's I079 Mask R-CNN + crop U-Net refiner handoff (MD5-locked weights verified) is "
            "a real, well-engineered inference pipeline. Reproduced locally on CPU over our fixed "
            "106-image validation split: proper one-to-one PQ = 0.409 (SQ 0.664, RQ 0.616; "
            "774 TP / 503 FP / 463 FN) versus the claimed public 0.70. Detection, not mask shape, "
            "is the bottleneck."
        ),
    },
    {
        "date": "2026-09-05",
        "text": (
            "The organizer self-evaluation notebook counts every (gt, pred) pair with IoU > 0.5 as "
            "a true positive - no one-to-one assignment. Overlapping duplicate predictions can "
            "inflate the official metric. Empty submissions score 0.0 under that notebook's formula "
            "yet 1.00 on the live LB, so the live evaluator diverges from the shared notebook."
        ),
    },
]

TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "u-net": (r"u[-_ ]?net",),
    "resunet": (r"res\s?u[-_ ]?net",),
    "attention u-net": (r"attention\s?u[-_ ]?net",),
    "nnu-net": (r"nn[-_ ]?u[-_ ]?net",),
    "mask r-cnn": (r"mask[-_ ]?r[-_ ]?cnn",),
    "faster r-cnn": (r"faster[-_ ]?r[-_ ]?cnn",),
    "yolo": (r"\byolo\d?\b",),
    "detr": (r"\bdetr\b",),
    "segformer": (r"segformer",),
    "deeplab": (r"deeplabv?3?\+?",),
    "hrnet": (r"\bhrnet\b",),
    "swin": (r"\bswin\b",),
    "segment-anything": (r"segment[- ]anything|\bsam\b",),
    "fpn": (r"\bfpn\b",),
    "smp": (r"segmentation[_-]?models[_-]?pytorch|\bsmp\b",),
    "timm": (r"\btimm\b",),
    "monai": (r"\bmonai\b",),
    "pytorch": (r"import torch|from torch",),
    "tensorflow": (r"tensorflow|\btf\.keras",),
    "detectron2": (r"detectron2",),
    "mmdetection": (r"\bmmdet",),
    "ultralytics": (r"ultralytics",),
    "dice loss": (r"dice[-_ ]?loss",),
    "bce loss": (r"BCEWithLogits|binary[_ ]cross[_ ]entropy|BCELoss",),
    "focal loss": (r"focal",),
    "tversky": (r"tversky",),
    "clDice": (r"cldice",),
    "lovasz": (r"lov[aá]sz|lovasz",),
    "tta": (r"\btta\b|test[- ]time[- ]aug",),
    "watershed": (r"watershed",),
    "connected components": (r"connectedComponents|connected_components",),
    "tiling": (r"\btil(ing|es?)\b|sliding[-_ ]window",),
    "resnet backbone": (r"resnet\d+",),
    "efficientnet": (r"efficientnet",),
    "imagenet weights": (r"imagenet|pretrained",),
    "coco rle": (r"pycocotools|\brle\b",),
    "nms/wbf": (r"\bnms\b|\bwbf\b|non[_ ]?max",),
    "ensemble": (r"ensemble|average.*model|model.*averag",),
    "pseudo-labels": (r"pseudo[-_ ]?label",),
    "detector->refiner": (r"refin",),
    "spine metadata": (r"\bspine\b",),
    "annotator sets": (r"annotator|annotation[_ ]set|magfilo",),
    "morphology": (r"morpholog|erode|dilat",),
    "hungarian matching": (r"linear_sum_assignment|hungarian",),
    "instance seeds": (r"seed|marker[-_ ]controlled",),
}

FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "broken-annotation-lookup": (
        r"(annotations?[a-z_]*|ann[a-z_]*)\.get\(\s*(filename|file_name|image_name|fname|img_name)",
        r"\.get\(\s*(filename|file_name|image_name)\s*(,|\))",
    ),
    "empty-submission-path": (
        r"filament_count\s*==\s*0",
        r"empty_(mask|rle)",
        r"np\.zeros(_like)?\(\s*(binary_)?mask",
    ),
    "lb-defect-mentions": (
        r"malformed",
        r"(lb|leaderboard)[-_ ]?(defect|bug|exploit|probe|issue)",
        r"all[- ]empty",
        r"empty[- ]prediction",
        r"missing[- ]prediction",
    ),
}

FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "epochs": (r"epochs['\"]?\s*[=:]\s*(\d+)",),
    "image_size": (
        r"(?:img_size|image_size|IMG_SIZE|input_size)['\"]?\s*[=:]?\s*[\[(]?\s*(\d{3,4})\b",
        r"\bResize\(\s*(\d{3,4})",
    ),
    "batch_size": (r"batch[_ ]?size['\"]?\s*[=:]\s*(\d+)",),
    "learning_rate": (r"(?:\blr\b|learning_rate)['\"]?\s*[=:]\s*([0-9][0-9.eE-]*)",),
    "threshold": (r"threshold['\"]?\s*[=:]\s*(0?\.\d+)",),
    "backbone": (r"encoder_name\s*=\s*['\"]([\w-]+)", r"backbone\s*[=:]\s*['\"]([\w-]+)"),
    "model": (r"\b(smp\.\w+|maskrcnn_\w+|fasterrcnn_\w+|retinanet_\w+|YOLO\w*|MasksAndBoxes\w*)",),
}

EXCERPT_PATTERN = re.compile(
    r"(smp\.|maskrcnn_|fasterrcnn_|UNet|Unet|unet\(|DiceLoss|BCEWithLogits|criterion|"
    r"threshold|watershed|connectedComponents|TTA|tta|fillPoly|annotations|linear_sum_assignment|"
    r"refin|yolo|focal|tversky|cldice)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cli(arguments: list[str]) -> str:
    command = ["kaggle", *arguments]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"kaggle {' '.join(arguments)} failed: {message[:300]}")
    return completed.stdout


def parse_cli_csv(text: str) -> list[dict[str, str]]:
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("Next Page Token")
    ]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return [dict(row) for row in reader]


def parse_topic_messages(text: str) -> list[dict[str, str]]:
    rows = [
        row for row in csv.reader(io.StringIO(text)) if row and any(cell.strip() for cell in row)
    ]
    header_index = None
    for index, row in enumerate(rows):
        if len(row) > 2 and row[0].strip() == "id" and "content" in row:
            header_index = index
            break
    if header_index is None:
        return []
    header = [cell.strip() for cell in rows[header_index]]
    messages = []
    for row in rows[header_index + 1 :]:
        if len(row) != len(header):
            continue
        record = dict(zip(header, row, strict=True))
        if record.get("content"):
            messages.append(record)
    return messages


def load_json_output(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        if start < 0:
            raise
        return json.loads(stripped[start:])


def fetch_leaderboard(competition: str, pages: int = 3) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_teams: set[str] = set()
    token: str | None = None
    for _ in range(pages):
        arguments = [
            "competitions",
            "leaderboard",
            competition,
            "--show",
            "--csv",
            "--page-size",
            "200",
        ]
        if token:
            arguments.extend(["--page-token", token])
        output = run_cli(arguments)
        match = re.search(r"^Next Page Token = (\S+)", output, re.MULTILINE)
        token = match.group(1) if match else None
        new_rows = 0
        for row in parse_cli_csv(output):
            try:
                if row["teamName"] in seen_teams:
                    continue
                seen_teams.add(row["teamName"])
                parsed.append(
                    {
                        "team": row["teamName"],
                        "score": float(row["score"]),
                        "submitted": row["submissionDate"],
                    }
                )
                new_rows += 1
            except (KeyError, ValueError):
                continue
        if not token or not new_rows:
            break
    return parsed


def fetch_kernels(competition: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for sort_by in ("voteCount", "dateRun"):
        try:
            payload = load_json_output(
                run_cli(
                    [
                        "kernels",
                        "list",
                        "--competition",
                        competition,
                        "--sort-by",
                        sort_by,
                        "--page-size",
                        "100",
                        "--format",
                        "json",
                    ]
                )
            )
        except (RuntimeError, ValueError):
            continue
        for row in payload:
            reference = row.get("ref")
            if not reference:
                continue
            existing = merged.get(reference)
            votes = int(row.get("totalVotes") or 0)
            last_run = row.get("lastRunTime") or ""
            if existing is None or votes >= int(existing.get("votes") or 0):
                merged[reference] = {
                    "ref": reference,
                    "title": row.get("title") or reference,
                    "author": row.get("author") or reference.split("/")[0],
                    "votes": votes,
                    "last_run": last_run,
                }
    return sorted(merged.values(), key=lambda row: (-row["votes"], row["ref"]))


def fetch_topics(competition: str) -> list[dict[str, Any]]:
    rows = parse_cli_csv(run_cli(["competitions", "topics", "list", competition, "--csv"]))
    topics = []
    for row in rows:
        try:
            topics.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "author": row["authorName"],
                    "comments": int(row["commentCount"] or 0),
                    "votes": int(row["votes"] or 0),
                    "posted": row["postDate"],
                }
            )
        except (KeyError, ValueError):
            continue
    return topics


def fetch_search_models(search: str) -> list[dict[str, Any]]:
    try:
        payload = load_json_output(
            run_cli(["models", "list", "--search", search, "--format", "json"])
        )
    except (RuntimeError, ValueError):
        return []
    return [
        {
            "ref": row.get("ref") or "",
            "title": row.get("title") or "",
            "subtitle": row.get("subtitle") or "",
            "author": row.get("author") or "",
        }
        for row in payload
        if row.get("ref")
    ]


def fetch_search_datasets(search: str) -> list[dict[str, Any]]:
    try:
        rows = parse_cli_csv(run_cli(["datasets", "list", "--search", search, "--csv"]))
    except RuntimeError:
        return []
    parsed = []
    for row in rows:
        parsed.append(
            {
                "ref": row.get("ref") or "",
                "title": row.get("title") or "",
                "size": row.get("size") or "",
                "updated": row.get("lastUpdated") or "",
                "downloads": row.get("downloadCount") or "",
                "votes": row.get("voteCount") or "",
            }
        )
    return parsed


def pull_kernel_source(reference: str, target_dir: Path) -> str | None:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_cli(["kernels", "pull", reference, "-p", str(target_dir)])
    except RuntimeError:
        return None
    for pattern in ("*.ipynb", "*.py", "*.r", "*.md"):
        for path in sorted(target_dir.glob(pattern)):
            try:
                return path.read_text(errors="replace")
            except OSError:
                continue
    return None


def notebook_code_cells(source: str) -> str:
    if not source.lstrip().startswith("{"):
        return source
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return source
    cells = []
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            text = "\n".join(f"# {line}" for line in text.splitlines())
        cells.append(text)
    return "\n".join(cells)


def match_patterns(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) is not None for pattern in patterns)


def extract_tags(text: str) -> list[str]:
    found = [tag for tag, patterns in TAG_PATTERNS.items() if match_patterns(patterns, text)]
    return sorted(found)


def extract_facts(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for name, patterns in FACT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                facts[name] = match.group(1)
                break
    return facts


def extract_flags(text: str) -> list[str]:
    return [flag for flag, patterns in FLAG_PATTERNS.items() if match_patterns(patterns, text)]


def extract_excerpts(text: str, limit: int = 10) -> list[dict[str, Any]]:
    excerpts = []
    for index, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or len(stripped) > 240:
            continue
        if EXCERPT_PATTERN.search(stripped):
            excerpts.append({"line": index, "text": stripped})
            if len(excerpts) >= limit:
                break
    return excerpts


def extract_external_inputs(text: str, competition: str) -> list[str]:
    slugs = set(re.findall(r"/kaggle/input/([A-Za-z0-9_.-]+)/", text))
    return sorted(slug for slug in slugs if competition.lower() not in slug.lower())


def analyze_source(source: str, competition: str) -> dict[str, Any]:
    text = notebook_code_cells(source)
    return {
        "lines": text.count("\n") + 1,
        "tags": extract_tags(text),
        "facts": extract_facts(text),
        "flags": extract_flags(text),
        "excerpts": extract_excerpts(text),
        "external_inputs": extract_external_inputs(text, competition),
        "bytes": len(text),
    }


def load_store(store_path: Path) -> dict[str, Any]:
    if store_path.exists():
        try:
            return json.loads(store_path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "version": 1,
        "competition": DEFAULT_COMPETITION,
        "updated_at": None,
        "snapshots": [],
        "leaderboard": {"fetched_at": None, "rows": []},
        "kernels": {},
        "topics": {},
        "models": [],
        "datasets": [],
    }


def save_store(store: dict[str, Any], store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=1))


def merge_kernels(
    store: dict[str, Any], fetched: list[dict[str, Any]], timestamp: str
) -> tuple[list[dict[str, Any]], list[str]]:
    kernels = store["kernels"]
    new_refs: list[str] = []
    for row in fetched:
        reference = row["ref"]
        record = kernels.get(reference)
        if record is None:
            new_refs.append(reference)
            record = {
                **row,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "votes_history": [[timestamp, row["votes"]]],
                "analysis": None,
                "pulled_at": None,
                "source_signature": None,
                "pull_error": False,
            }
            kernels[reference] = record
        else:
            record["last_seen"] = timestamp
            record["title"] = row["title"]
            record["author"] = row["author"]
            record["last_run"] = row["last_run"]
            history = record.setdefault("votes_history", [])
            if not history or history[-1][1] != row["votes"]:
                history.append([timestamp, row["votes"]])
                if len(history) > SNAPSHOT_CAP:
                    del history[: len(history) - SNAPSHOT_CAP]
            record["votes"] = row["votes"]
    for record in kernels.values():
        record["is_new"] = record["first_seen"] == timestamp
    return fetched, new_refs


def select_pull_candidates(
    store: dict[str, Any], fetched: list[dict[str, Any]], budget: int
) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for row in fetched:
        record = store["kernels"][row["ref"]]
        signature = f"{row['votes']}:{row['last_run']}"
        if record["pull_error"]:
            continue
        if record["analysis"] is None or record["source_signature"] != signature:
            candidates.append((-row["votes"], row["ref"], signature))
    candidates.sort()
    selected = [reference for _, reference, _ in candidates[:budget]]
    signatures = {reference: signature for _, reference, signature in candidates[:budget]}
    store["_pending_signatures"] = signatures
    return selected


def refresh_kernel_analysis(
    store: dict[str, Any],
    reference: str,
    cache_dir: Path,
    competition: str,
) -> None:
    record = store["kernels"][reference]
    slug = reference.replace("/", "__")
    source = pull_kernel_source(reference, cache_dir / slug)
    if source is None:
        record["pull_error"] = True
        record["analysis"] = record.get("analysis") or {
            "lines": 0,
            "tags": [],
            "facts": {},
            "flags": ["pull-failed"],
            "excerpts": [],
            "external_inputs": [],
            "bytes": 0,
        }
        return
    if len(source) > MAX_STORED_SOURCE_BYTES:
        source = source[:MAX_STORED_SOURCE_BYTES]
    (cache_dir / slug / "source.txt").write_text(source, errors="replace")
    record["analysis"] = analyze_source(source, competition)
    record["pull_error"] = False
    record["pulled_at"] = utc_now()
    record["source_signature"] = store.get("_pending_signatures", {}).get(reference)


def refresh_topics(
    store: dict[str, Any], topics: list[dict[str, Any]], timestamp: str, depth: int
) -> None:
    by_id = {str(topic["id"]): topic for topic in topics}
    recent = sorted(topics, key=lambda topic: topic["posted"], reverse=True)[:depth]
    active = sorted(topics, key=lambda topic: -(topic["votes"] + topic["comments"]))[:2]
    wanted = {str(topic["id"]) for topic in [*recent, *active]}
    stored = store["topics"]
    for topic_id, topic in by_id.items():
        existing = stored.get(topic_id) or {}
        existing.update(topic)
        existing.setdefault("messages", [])
        stored[topic_id] = existing
    for topic_id in wanted:
        previous = stored[topic_id]
        if previous.get("messages") and previous.get("fetched_at"):
            continue
        try:
            messages = parse_topic_messages(
                run_cli(["competitions", "topics", "show", topic_id, "--csv"])
            )
        except RuntimeError:
            continue
        trimmed = [
            {
                "author": message.get("authorName", ""),
                "posted": message.get("postDate", ""),
                "votes": message.get("votes", ""),
                "content": re.sub(r"<[^>]+>", " ", message.get("content", ""))[:600],
            }
            for message in messages[-8:]
        ]
        previous["messages"] = trimmed
        previous["fetched_at"] = timestamp
    stale = set(stored) - set(by_id)
    for topic_id in stale:
        del stored[topic_id]


def update_snapshots(store: dict[str, Any], timestamp: str, new_count: int) -> None:
    rows = store["leaderboard"]["rows"]
    top_score = rows[0]["score"] if rows else None
    snapshot = {
        "ts": timestamp,
        "top_score": top_score,
        "kernel_count": len(store["kernels"]),
        "new_count": new_count,
        "lb_rows": len(rows),
    }
    snapshots = store["snapshots"]
    if not snapshots or snapshots[-1]["ts"] != timestamp:
        snapshots.append(snapshot)
    if len(snapshots) > SNAPSHOT_CAP:
        del snapshots[: len(snapshots) - SNAPSHOT_CAP]


def render(store: dict[str, Any], html_path: Path) -> None:
    payload = {
        "competition": store["competition"],
        "updated_at": store["updated_at"],
        "audit_notes": AUDIT_NOTES,
        "snapshots": store["snapshots"],
        "leaderboard": store["leaderboard"]["rows"],
        "kernels": sorted(
            (
                {
                    "ref": reference,
                    "title": record["title"],
                    "author": record["author"],
                    "votes": record["votes"],
                    "votes_delta": votes_delta(record),
                    "first_seen": record["first_seen"],
                    "last_seen": record["last_seen"],
                    "last_run": record.get("last_run") or "",
                    "is_new": record.get("is_new", False),
                    "pull_error": record.get("pull_error", False),
                    **(
                        record.get("analysis")
                        or {
                            "lines": 0,
                            "tags": [],
                            "facts": {},
                            "flags": [],
                            "excerpts": [],
                            "external_inputs": [],
                            "bytes": 0,
                        }
                    ),
                }
                for reference, record in store["kernels"].items()
            ),
            key=lambda row: (-row["votes"], row["ref"]),
        ),
        "topics": sorted(
            store["topics"].values(),
            key=lambda topic: topic.get("posted", ""),
            reverse=True,
        ),
        "models": store["models"],
        "datasets": store["datasets"],
    }
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(payload))


def votes_delta(record: dict[str, Any]) -> int:
    history = record.get("votes_history") or []
    if len(history) < 2:
        return 0
    return int(history[-1][1]) - int(history[-2][1])


FLAG_DESCRIPTIONS = {
    "broken-annotation-lookup": (
        "Looks annotations up by filename on the COCO-style JSON - that never matches, so masks "
        "are all-empty (verified locally)."
    ),
    "empty-submission-path": (
        "Submission fallback encodes an empty mask when nothing is detected - rides the public LB "
        "empty/missing-prediction defect."
    ),
    "lb-defect-mentions": (
        "Source mentions the public LB empty-prediction defect, malformed submissions, or probes."
    ),
    "pull-failed": "Source could not be pulled (deleted, private, or CLI error).",
}


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data_json).replace(
        "__FLAG_DESCRIPTIONS__", json.dumps(FLAG_DESCRIPTIONS).replace("</", "<\\/")
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SFSC 2026 Competition Watch</title>
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --border:#2d3748;
  --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --good:#3fb950;
  --warn:#d29922; --bad:#f85149; --chip:#21262d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:22px;margin:0 0 2px}
h2{font-size:16px;margin:28px 0 10px;color:var(--accent)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.muted{color:var(--muted)}
.small{font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .num{font-size:24px;font-weight:700}
.card .lbl{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.4px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;overflow:hidden}
th{background:var(--panel2);text-align:left;padding:8px 10px;font-size:12px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0;cursor:pointer;white-space:nowrap}
td{padding:8px 10px;border-top:1px solid var(--border);vertical-align:top}
tr:hover td{background:#182030}
.bar{height:8px;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:4px;min-width:2px}
.chip{display:inline-block;background:var(--chip);border:1px solid var(--border);color:#a5d6ff;
  border-radius:999px;padding:1px 9px;margin:1px 2px;font-size:11.5px;cursor:pointer;white-space:nowrap}
.chip:hover{border-color:var(--accent)}
.badge{display:inline-block;border-radius:5px;padding:1px 7px;font-size:11px;font-weight:600;
  margin:1px 2px;white-space:nowrap}
.b-new{background:#12351d;color:#3fb950;border:1px solid #2ea043}
.b-red{background:#3d1114;color:#f85149;border:1px solid #f85149}
.b-amber{background:#332208;color:#d29922;border:1px solid #d29922}
.b-blue{background:#0d2a45;color:#58a6ff;border:1px solid #1f6feb}
.b-gray{background:var(--chip);color:var(--muted);border:1px solid var(--border)}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 14px;align-items:center}
input[type=search],select{background:var(--panel2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:7px 10px;font-size:13px;outline:none}
input[type=search]{min-width:280px}
.tagcloud{line-height:2.1}
details{display:inline}
summary{cursor:pointer;color:var(--accent);font-size:12px;user-select:none;white-space:nowrap}
pre{background:#0a0e14;border:1px solid var(--border);border-radius:8px;padding:10px 12px;
  overflow:auto;max-height:340px;font-size:12px;line-height:1.5;margin:6px 0}
pre .ln{color:#4a5b70;display:inline-block;width:38px;user-select:none}
.note{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--warn);
  border-radius:8px;padding:10px 14px;margin:10px 0}
.facts{color:var(--muted);font-size:12px;margin-top:2px}
.facts b{color:#a5d6ff;font-weight:600}
.score-cell{font-variant-numeric:tabular-nums;font-weight:600}
.msg{border-top:1px dashed var(--border);padding:6px 0}
.msg .who{color:#a5d6ff;font-weight:600}
.sortable thead th.asc::after{content:" ↑"}
.sortable thead th.desc::after{content:" ↓"}
</style>
</head>
<body>
<div class="wrap">
  <h1>SFSC 2026 — Competition Watch</h1>
  <div class="muted small" id="meta"></div>

  <div class="cards" id="cards"></div>

  <div class="note"><b>Local audit notes</b><div id="audit"></div></div>

  <h2>Public leaderboard</h2>
  <div class="muted small">Public scores only; the LB had a known empty/missing-prediction defect era.</div>
  <div style="max-height:420px;overflow:auto;margin-top:8px">
    <table class="sortable" id="lb"></table>
  </div>

  <h2>Method feed — competition notebooks</h2>
  <div class="controls">
    <input id="q" type="search" placeholder="filter title / author / tag / code excerpt…">
    <select id="sortSel">
      <option value="votes">sort: votes</option>
      <option value="delta">sort: vote delta</option>
      <option value="last_run">sort: last run (newest)</option>
      <option value="first_seen">sort: first seen (newest)</option>
    </select>
    <label class="small muted"><input type="checkbox" id="newOnly"> new only</label>
    <label class="small muted"><input type="checkbox" id="flagOnly"> issues only</label>
    <span id="feedCount" class="small muted"></span>
  </div>
  <div class="tagcloud" id="tagcloud"></div>
  <table class="sortable" id="feed"></table>

  <h2>Discussion topics</h2>
  <table id="topics"></table>

  <h2>Models</h2>
  <table id="models"></table>

  <h2>Datasets</h2>
  <table id="datasets"></table>

  <h2>Run history</h2>
  <table id="history"></table>
</div>

<script>
const DATA = __DATA__;
const FLAG_DESCRIPTIONS = __FLAG_DESCRIPTIONS__;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const kaggleLink = (kind, ref) => {
  const [owner, slug] = ref.split("/");
  return "https://www.kaggle.com/" + kind + "/" + owner + "/" + slug;
};
const fmtDate = s => (s || "").replace("T", " ").replace(/:\d\dZ$/, "Z");

document.getElementById("meta").textContent =
  "competition: " + DATA.competition + " · state updated: " + (DATA.updated_at || "never") +
  " · " + DATA.kernels.length + " notebooks tracked";

const stats = [
  ["Notebooks", DATA.kernels.length],
  ["New this run", DATA.kernels.filter(k => k.is_new).length],
  ["LB top score", DATA.leaderboard.length ? DATA.leaderboard[0].score : "—"],
  ["LB rows", DATA.leaderboard.length],
  ["Topics", DATA.topics.length],
  ["Models", DATA.models.length],
  ["Datasets", DATA.datasets.length],
];
document.getElementById("cards").innerHTML = stats.map(([lbl, num]) =>
  `<div class="card"><div class="num">${esc(num)}</div><div class="lbl">${esc(lbl)}</div></div>`).join("");

document.getElementById("audit").innerHTML = DATA.audit_notes.map(n =>
  `<div class="msg"><span class="who">${esc(n.date)}</span> — ${esc(n.text)}</div>`).join("");

const lbMax = DATA.leaderboard.length ? DATA.leaderboard[0].score : 1;
const lbRows = DATA.leaderboard.map((r, i) =>
  `<tr><td class="muted">${i + 1}</td><td>${esc(r.team)}</td>
   <td class="score-cell" style="width:280px"><div style="display:flex;gap:8px;align-items:center">
   <span>${r.score.toFixed(2)}</span><div class="bar" style="width:${(r.score / lbMax * 100).toFixed(1)}%"></div></div></td>
   <td class="muted small">${esc(r.submitted)}</td></tr>`).join("");
document.getElementById("lb").innerHTML =
  "<thead><tr><th>#</th><th>team</th><th>score</th><th>submitted</th></tr></thead><tbody>" +
  (lbRows || '<tr><td colspan="4" class="muted">no leaderboard rows</td></tr>') + "</tbody>";

const allTags = {};
DATA.kernels.forEach(k => (k.tags || []).forEach(t => allTags[t] = (allTags[t] || 0) + 1));
document.getElementById("tagcloud").innerHTML = Object.entries(allTags)
  .sort((a, b) => b[1] - a[1])
  .map(([t, n]) => `<span class="chip" onclick="toggleTag('${esc(t)}')">${esc(t)} <b>${n}</b></span>`)
  .join(" ");

let activeTag = null;
window.toggleTag = t => { activeTag = (activeTag === t) ? null : t; renderFeed(); };

function flagBadge(f) {
  const cls = f === "lb-defect-mentions" ? "b-amber" : f === "pull-failed" ? "b-gray" : "b-red";
  const desc = FLAG_DESCRIPTIONS[f] || f;
  return `<span class="badge ${cls}" title="${esc(desc)}">${esc(f)}</span>`;
}

function detailsHtml(k) {
  const facts = Object.entries(k.facts || {}).map(([k2, v]) => `<b>${esc(k2)}</b> ${esc(v)}`).join(" · ");
  const ext = (k.external_inputs || []).length
    ? `<div class="facts"><b>extra inputs</b> ${esc(k.external_inputs.join(", "))}</div>` : "";
  const code = (k.excerpts || []).map(e =>
    `<div><span class="ln">${e.line}</span>${esc(e.text)}</div>`).join("");
  return `<div style="margin-top:6px">
    ${facts ? `<div class="facts">${facts}</div>` : ""}${ext}
    ${code ? `<pre>${code}</pre>` : '<div class="muted small">no excerpts</div>'}
    <div class="small muted">source cached · first seen ${esc(fmtDate(k.first_seen))} · last run ${esc(fmtDate(k.last_run))}</div>
  </div>`;
}

function renderFeed() {
  const q = document.getElementById("q").value.toLowerCase();
  const sortSel = document.getElementById("sortSel").value;
  const newOnly = document.getElementById("newOnly").checked;
  const flagOnly = document.getElementById("flagOnly").checked;
  let rows = DATA.kernels.filter(k => {
    if (newOnly && !k.is_new) return false;
    if (flagOnly && !(k.flags || []).length) return false;
    if (activeTag && !(k.tags || []).includes(activeTag)) return false;
    if (!q) return true;
    const hay = [k.title, k.author, k.ref, ...(k.tags || []),
      ...(k.excerpts || []).map(e => e.text)].join(" ").toLowerCase();
    return hay.includes(q);
  });
  const cmp = {
    votes: (a, b) => b.votes - a.votes,
    delta: (a, b) => (b.votes_delta || 0) - (a.votes_delta || 0),
    last_run: (a, b) => (b.last_run || "").localeCompare(a.last_run || ""),
    first_seen: (a, b) => (b.first_seen || "").localeCompare(a.first_seen || ""),
  }[sortSel];
  rows = [...rows].sort(cmp);
  document.getElementById("feedCount").textContent = rows.length + " / " + DATA.kernels.length + " shown";
  const body = rows.map(k => `<tr>
    <td style="white-space:nowrap">
      ${k.is_new ? '<span class="badge b-new">NEW</span>' : ""}
      ${k.votes_delta ? `<span class="badge b-blue">▲${k.votes_delta}</span>` : ""}
    </td>
    <td>
      <a href="${kaggleLink("code", k.ref)}" target="_blank">${esc(k.title)}</a>
      <div class="facts">by ${esc(k.author)} · ${k.votes} vote${k.votes === 1 ? "" : "s"} · last run ${esc(fmtDate(k.last_run))}</div>
      <div>${(k.tags || []).map(t => `<span class="chip">${esc(t)}</span>`).join("")}</div>
      ${(k.flags || []).length ? "<div>" + k.flags.map(flagBadge).join("") + "</div>" : ""}
      <details><summary>method details</summary>${detailsHtml(k)}</details>
    </td>
  </tr>`).join("");
  document.getElementById("feed").innerHTML =
    "<thead><tr><th>signal</th><th>notebook</th></tr></thead><tbody>" +
    (body || '<tr><td colspan="2" class="muted">no matches</td></tr>') + "</tbody>";
}

["q", "sortSel", "newOnly", "flagOnly"].forEach(id =>
  document.getElementById(id).addEventListener("input", renderFeed));
renderFeed();

document.getElementById("topics").innerHTML =
  "<thead><tr><th>topic</th><th>activity</th><th>latest messages</th></tr></thead><tbody>" +
  DATA.topics.map(t => `<tr><td>
      <a href="https://www.kaggle.com/competitions/${esc(DATA.competition)}/discussion/${esc(t.id)}" target="_blank">${esc(t.title)}</a>
      <div class="facts">${esc(t.author)} · posted ${esc(fmtDate(t.posted))}</div></td>
    <td class="small">${t.comments} comments · ${t.votes} votes</td>
    <td class="small">${(t.messages || []).slice(-3).reverse().map(m =>
      `<div class="msg"><span class="who">${esc(m.author)}</span> <span class="muted">${esc(fmtDate(m.posted))}</span><br>${esc(m.content)}</div>`).join("") ||
      '<span class="muted">—</span>'}</td></tr>`).join("") + "</tbody>";

const simpleTable = (id, rows) => {
  document.getElementById(id).innerHTML =
    "<tbody>" + (rows || '<tr><td class="muted">none found</td></tr>') + "</tbody>";
};
simpleTable("models", DATA.models.map(m =>
  `<tr><td><a href="${kaggleLink("models", m.ref)}" target="_blank">${esc(m.title)}</a>
   <div class="facts">${esc(m.subtitle)}</div></td>
   <td style="white-space:nowrap" class="small">${esc(m.author)}</td></tr>`).join(""));

simpleTable("datasets", DATA.datasets.map(d =>
  `<tr><td><a href="https://www.kaggle.com/datasets/${esc(d.ref)}" target="_blank">${esc(d.title)}</a></td>
   <td class="small muted" style="white-space:nowrap">${esc(d.size)} · ${esc(d.downloads)} dl · upd ${esc(fmtDate(d.updated))}</td></tr>`).join(""));

document.getElementById("history").innerHTML =
  "<thead><tr><th>run (UTC)</th><th>LB top</th><th>notebooks tracked</th><th>new</th></tr></thead><tbody>" +
  DATA.snapshots.slice(-30).reverse().map(s =>
    `<tr><td>${esc(fmtDate(s.ts))}</td><td>${s.top_score ?? "—"}</td>
     <td>${s.kernel_count}</td><td>${s.new_count}</td></tr>`).join("") + "</tbody>";
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--search", default=DEFAULT_SEARCH)
    parser.add_argument("--max-pulls", type=int, default=30)
    parser.add_argument("--topic-depth", type=int, default=6)
    parser.add_argument("--skip-fetch", action="store_true", help="re-render from stored state")
    parser.add_argument("--serve", type=int, default=0, help="serve the dashboard on this port")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address for --serve (use 0.0.0.0 for LAN access)",
    )
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    return parser


def serve(out_dir: Path, port: int, host: str = "127.0.0.1") -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"serving {out_dir} at http://{host}:{port}/index.html")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


def main() -> None:
    arguments = build_parser().parse_args()
    out_dir: Path = arguments.out
    cache_dir = out_dir / "cache"
    store_path = out_dir / "store.json"
    html_path = out_dir / "index.html"
    store = load_store(store_path)
    store["competition"] = arguments.competition

    if not arguments.skip_fetch:
        timestamp = utc_now()
        print(f"[{timestamp}] fetching leaderboard…")
        try:
            rows = fetch_leaderboard(arguments.competition)
            store["leaderboard"] = {"fetched_at": timestamp, "rows": rows}
        except (RuntimeError, KeyError, ValueError) as error:
            print(f"  leaderboard fetch failed: {error}")

        print("fetching kernels…")
        try:
            fetched = fetch_kernels(arguments.competition)
        except (RuntimeError, ValueError) as error:
            print(f"  kernel fetch failed: {error}")
            fetched = []
        _, new_refs = merge_kernels(store, fetched, timestamp)
        print(f"  {len(fetched)} listed, {len(new_refs)} new")

        candidates = select_pull_candidates(store, fetched, arguments.max_pulls)
        print(f"pulling sources for {len(candidates)} notebooks…")
        for index, reference in enumerate(candidates, 1):
            try:
                refresh_kernel_analysis(store, reference, cache_dir, arguments.competition)
            except (RuntimeError, OSError) as error:
                store["kernels"][reference]["pull_error"] = True
                print(f"  [{index}/{len(candidates)}] {reference}: {error}")
            else:
                tags = store["kernels"][reference]["analysis"]["tags"]
                print(f"  [{index}/{len(candidates)}] {reference}: {len(tags)} tags")
        store.pop("_pending_signatures", None)

        print("fetching topics…")
        try:
            topics = fetch_topics(arguments.competition)
            refresh_topics(store, topics, timestamp, arguments.topic_depth)
            print(f"  {len(topics)} topics, messages for top {arguments.topic_depth}")
        except RuntimeError as error:
            print(f"  topic fetch failed: {error}")

        print("fetching models and datasets…")
        store["models"] = fetch_search_models(arguments.search)
        store["datasets"] = fetch_search_datasets(arguments.search)

        update_snapshots(store, timestamp, len(new_refs))
        store["updated_at"] = timestamp
        save_store(store, store_path)
        print("state saved.")

    render(store, html_path)
    print(f"dashboard written to {html_path}")

    if arguments.serve:
        serve(out_dir, arguments.serve, arguments.host)
    elif arguments.open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
