"""CPU contracts for blocked splits, calibration, assembly, and PQ reporting."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

from sfsc.evaluation import PQResult, pairwise_iou, panoptic_quality

FEATURE_NAMES = (
    "presence_logit",
    "predicted_conditional_iou",
    "log_native_area",
    "mask_entropy",
    "maximum_iou_with_stronger_query",
    "maximum_containment_with_stronger_query",
    "component_count_capped",
)
CANDIDATE_SCHEMA_VERSION = 2
PILOT_SALTS = {
    1: ("native-v4:9", "native-v4:q:8"),
    2: ("pq-setformer:pilot2:34", "pq-setformer:pilot2:q:0"),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _day(filename: str) -> int:
    try:
        return datetime.strptime(filename[:14], "%Y%m%d%H%M%S").date().toordinal()
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid timestamped filename: {filename!r}") from error


@dataclass(frozen=True)
class ExperimentSplit:
    pilot: int
    block_days: int
    purge_days: int
    role_salt: str
    calibration_salt: str
    members: dict[str, list[str]]
    assessment_blocks: tuple[int, ...]

    @property
    def fitting(self) -> list[str]:
        return self.members["fitting"]

    @property
    def quality_fit(self) -> list[str]:
        return self.members["quality_fit"]

    @property
    def quality_calibration(self) -> list[str]:
        return self.members["quality_calibration"]

    @property
    def assessment(self) -> list[str]:
        return self.members["assessment"]

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["assessment_blocks"] = list(self.assessment_blocks)
        value["sha256"] = self.sha256
        return value


def make_experiment_split(
    filenames: Sequence[str],
    *,
    pilot: int,
    block_days: int = 60,
    purge_days: int = 7,
) -> ExperimentSplit:
    """Create one predeclared, cross-site date-blocked R/Q-fit/Q-cal/H split."""
    if pilot not in PILOT_SALTS:
        raise ValueError(f"pilot must be one of {sorted(PILOT_SALTS)}")
    names = sorted(filenames)
    if len(names) != len(set(names)):
        raise ValueError("Split filenames must be unique")
    if not names or block_days <= 0 or purge_days < 0:
        raise ValueError("Split requires filenames and positive block settings")

    role_salt, calibration_salt = PILOT_SALTS[pilot]
    days = {name: _day(name) for name in names}
    blocks = {name: day // block_days for name, day in days.items()}
    roles = {}
    for name in names:
        bucket = int(hashlib.sha256(f"{role_salt}:{blocks[name]}".encode()).hexdigest()[:8], 16) % 5
        roles[name] = "R" if bucket < 3 else "Q" if bucket == 3 else "H"

    assessment = [name for name in names if roles[name] == "H"]
    quality = [
        name
        for name in names
        if roles[name] == "Q"
        and all(abs(days[name] - days[other]) > purge_days for other in assessment)
    ]
    fitting = [
        name
        for name in names
        if roles[name] == "R"
        and all(
            abs(days[name] - days[other]) > purge_days
            for other in names
            if roles[other] in {"Q", "H"}
        )
    ]
    quality_calibration = [
        name
        for name in quality
        if int(
            hashlib.sha256(f"{calibration_salt}:{blocks[name]}".encode()).hexdigest()[:8],
            16,
        )
        % 3
        == 0
    ]
    quality_fit = sorted(set(quality) - set(quality_calibration))
    selected = set(fitting) | set(quality_fit) | set(quality_calibration) | set(assessment)
    members = {
        "fitting": sorted(fitting),
        "quality_fit": quality_fit,
        "quality_calibration": sorted(quality_calibration),
        "assessment": sorted(assessment),
        "purged": sorted(set(names) - selected),
    }
    if any(not values for role, values in members.items() if role != "purged"):
        raise ValueError("Blocked split produced an empty required role")
    return ExperimentSplit(
        pilot=pilot,
        block_days=block_days,
        purge_days=purge_days,
        role_salt=role_salt,
        calibration_salt=calibration_salt,
        members=members,
        assessment_blocks=tuple(sorted({blocks[name] for name in assessment})),
    )


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2 or not array.any():
        raise ValueError("Cannot encode an empty or non-two-dimensional mask")
    encoded = coco_mask.encode(np.asfortranarray(array))
    return {"size": list(array.shape), "counts": encoded["counts"].decode("ascii")}


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    value = {"size": list(rle["size"]), "counts": str(rle["counts"]).encode("ascii")}
    result = coco_mask.decode(value).astype(bool)
    if result.ndim != 2 or not result.any():
        raise ValueError("Decoded RLE is empty or malformed")
    return result


@dataclass(frozen=True)
class Candidate:
    query_id: str
    mask: np.ndarray
    presence: float
    predicted_iou: float
    mask_probability: np.ndarray | None = None
    presence_logit: float | None = None
    source: str = "identity"

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("Candidate mask must be non-empty and two-dimensional")
        if not 0 <= self.presence <= 1 or not 0 <= self.predicted_iou <= 1:
            raise ValueError("Candidate scores must be probabilities")
        if self.presence_logit is not None and not math.isfinite(self.presence_logit):
            raise ValueError("presence_logit must be finite when present")
        if not self.source:
            raise ValueError("Candidate source provenance is required")
        object.__setattr__(self, "mask", mask)
        if self.mask_probability is not None:
            probability = np.asarray(self.mask_probability, dtype=np.float32)
            if (
                probability.shape != mask.shape
                or not np.isfinite(probability).all()
                or probability.min() < 0
                or probability.max() > 1
            ):
                raise ValueError("mask_probability must be finite, aligned, and in [0, 1]")
            object.__setattr__(self, "mask_probability", probability)


def candidate_cache_hash(cache: Mapping[str, Sequence[Candidate]]) -> str:
    """Hash a candidate cache by content, including probability provenance."""
    serial = {}
    for filename, candidates in sorted(cache.items()):
        serial[filename] = [
            {
                "query_id": candidate.query_id,
                "mask": encode_rle(candidate.mask),
                "presence": candidate.presence,
                "presence_logit": candidate.presence_logit,
                "predicted_iou": candidate.predicted_iou,
                "source": candidate.source,
                "probability_sha256": (
                    None
                    if candidate.mask_probability is None
                    else hashlib.sha256(candidate.mask_probability.tobytes()).hexdigest()
                ),
            }
            for candidate in candidates
        ]
    return _digest(serial)


def serialize_candidate(
    candidate: Candidate, *, maximum_probability_bytes: int = 262_144, probability_size: int = 256
) -> dict[str, Any]:
    """Serialize a native candidate without silently dropping ownership data.

    The full-resolution boolean RLE is the official provisional mask.  Dense
    probabilities are quantized and compressed only when their bounded payload
    fits; callers must deliberately choose a smaller cached representation if it
    does not.  This makes a cache incapable of pretending it supports a later
    mask-threshold search when it does not.
    """
    rows, columns = np.where(candidate.mask)
    row: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "query_id": candidate.query_id,
        "presence": candidate.presence,
        "presence_logit": candidate.presence_logit,
        "predicted_iou": candidate.predicted_iou,
        "source": candidate.source,
        "native_area": int(candidate.mask.sum()),
        "component_count": component_count(candidate.mask),
        "bbox_xyxy": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ],
        "rle": encode_rle(candidate.mask),
    }
    if candidate.mask_probability is None:
        raise ValueError("Candidate v2 cache requires retained mask probabilities")
    probability = candidate.mask_probability
    if probability_size <= 0:
        raise ValueError("Candidate probability cache size must be positive")
    height, width = probability.shape
    scale = min(1.0, probability_size / max(height, width))
    cached_shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    if cached_shape != probability.shape:
        probability = np.asarray(
            Image.fromarray(probability).resize(
                (cached_shape[1], cached_shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    quantized = np.rint(probability * 255).astype(np.uint8).tobytes()
    encoded = zlib.compress(quantized, level=9)
    if len(encoded) > maximum_probability_bytes:
        raise ValueError(
            "Candidate probability payload exceeds the declared cache bound; "
            "do not write a threshold-untunable cache"
        )
    row.update(
        probability_codec="q8-zlib-v1",
        probability_shape=list(cached_shape),
        probability_native_shape=list(candidate.mask_probability.shape),
        probability_payload=base64.b64encode(encoded).decode("ascii"),
        probability_payload_sha256=hashlib.sha256(quantized).hexdigest(),
        mask_entropy=float(_mask_entropy(candidate.mask_probability)),
    )
    return row


def deserialize_candidate(row: Mapping[str, Any]) -> Candidate:
    """Read only the versioned, ownership-capable candidate-cache schema."""
    if row.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("Candidate cache schema mismatch")
    probability = None
    if row.get("probability_codec") is not None:
        if row.get("probability_codec") != "q8-zlib-v1":
            raise ValueError("Unsupported candidate probability codec")
        shape = tuple(row.get("probability_shape", ()))
        if len(shape) != 2 or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError("Candidate probability shape is invalid")
        try:
            raw = zlib.decompress(base64.b64decode(str(row["probability_payload"]), validate=True))
        except (KeyError, ValueError, zlib.error) as error:
            raise ValueError("Candidate probability payload is malformed") from error
        if len(raw) != math.prod(shape):
            raise ValueError("Candidate probability payload length is invalid")
        if row.get("probability_payload_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Candidate probability payload hash is invalid")
        probability = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255
        native_shape = tuple(row.get("probability_native_shape", shape))
        if len(native_shape) != 2 or any(
            not isinstance(dim, int) or dim <= 0 for dim in native_shape
        ):
            raise ValueError("Candidate native probability shape is invalid")
        if native_shape != shape:
            probability = np.asarray(
                Image.fromarray(probability).resize(
                    (native_shape[1], native_shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
    if probability is None:
        raise ValueError("Candidate v2 cache requires a probability payload")
    return Candidate(
        str(row["query_id"]),
        decode_rle(row["rle"]),
        float(row["presence"]),
        float(row["predicted_iou"]),
        probability,
        presence_logit=(
            None if row.get("presence_logit") is None else float(row["presence_logit"])
        ),
        source=str(row.get("source", "")),
    )


@dataclass(frozen=True)
class MatchTargets:
    match_probability: np.ndarray
    matched_iou: np.ndarray


def official_match_targets(
    candidates: Sequence[Candidate], truth_records: Sequence[Sequence[np.ndarray]]
) -> MatchTargets:
    """Label candidates through complete one-to-one PQ matching for every record."""
    if not truth_records:
        raise ValueError("At least one complete annotation record is required")
    matched = np.zeros(len(candidates), dtype=np.float64)
    matched_iou = np.zeros(len(candidates), dtype=np.float64)
    predictions = [candidate.mask for candidate in candidates]
    for truths in truth_records:
        if not truths or not predictions:
            continue
        ious = pairwise_iou(list(truths), predictions)
        eligible = np.where(ious > 0.5, ious, 0.0)
        truth_indices, prediction_indices = linear_sum_assignment(eligible, maximize=True)
        for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True):
            value = eligible[truth_index, prediction_index]
            if value > 0:
                matched[prediction_index] += 1
                matched_iou[prediction_index] += value
    divisor = len(truth_records)
    return MatchTargets(matched / divisor, matched_iou / divisor)


def component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))[1])


def _mask_entropy(probability: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1 - 1e-6)
    return float(np.mean(-clipped * np.log(clipped) - (1 - clipped) * np.log(1 - clipped)))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def _containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    return intersection / max(min(int(first.sum()), int(second.sum())), 1)


def candidate_overlap_summaries(candidates: Sequence[Candidate]) -> list[dict[str, float]]:
    """Native overlap/containment summaries against strictly stronger queries."""
    result = []
    for index, candidate in enumerate(candidates):
        stronger = [
            other
            for other_index, other in enumerate(candidates)
            if (other.presence * other.predicted_iou, -other_index)
            > (candidate.presence * candidate.predicted_iou, -index)
        ]
        result.append(
            {
                "maximum_iou_with_stronger_query": max(
                    (_mask_iou(candidate.mask, other.mask) for other in stronger), default=0.0
                ),
                "maximum_containment_with_stronger_query": max(
                    (_containment(candidate.mask, other.mask) for other in stronger), default=0.0
                ),
            }
        )
    return result


def candidate_features(candidates: Sequence[Candidate]) -> np.ndarray:
    """Return the frozen, low-capacity calibrator feature matrix."""
    rows = []
    for index, candidate in enumerate(candidates):
        stronger = [
            other
            for other_index, other in enumerate(candidates)
            if (other.presence * other.predicted_iou, -other_index)
            > (candidate.presence * candidate.predicted_iou, -index)
        ]
        maximum_iou = max((_mask_iou(candidate.mask, other.mask) for other in stronger), default=0)
        maximum_containment = max(
            (_containment(candidate.mask, other.mask) for other in stronger), default=0
        )
        probability = (
            candidate.mask_probability
            if candidate.mask_probability is not None
            else candidate.mask.astype(np.float32)
        )
        entropy = _mask_entropy(probability)
        presence = min(max(candidate.presence, 1e-6), 1 - 1e-6)
        presence_logit = (
            candidate.presence_logit
            if candidate.presence_logit is not None
            else math.log(presence / (1 - presence))
        )
        rows.append(
            [
                presence_logit,
                candidate.predicted_iou,
                math.log1p(int(candidate.mask.sum())),
                entropy,
                maximum_iou,
                maximum_containment,
                min(component_count(candidate.mask), 20),
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape((-1, len(FEATURE_NAMES)))


@dataclass(frozen=True)
class Calibrator:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    regularization: float
    examples: int
    membership_sha256: str = ""
    candidate_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "regularization": self.regularization,
            "examples": self.examples,
            "membership_sha256": self.membership_sha256,
            "candidate_sha256": self.candidate_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Calibrator:
        result = cls(
            feature_names=tuple(value["feature_names"]),
            mean=tuple(map(float, value["mean"])),
            scale=tuple(map(float, value["scale"])),
            coefficients=tuple(map(float, value["coefficients"])),
            regularization=float(value["regularization"]),
            examples=int(value["examples"]),
            membership_sha256=str(value.get("membership_sha256", "")),
            candidate_sha256=str(value.get("candidate_sha256", "")),
        )
        if result.feature_names != FEATURE_NAMES:
            raise ValueError("Calibrator feature schema mismatch")
        if len(result.coefficients) != len(FEATURE_NAMES) + 1:
            raise ValueError("Calibrator coefficient shape mismatch")
        if bool(result.membership_sha256) != bool(result.candidate_sha256):
            raise ValueError("Calibrator provenance is incomplete")
        return result

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def predict(
        self, candidates: Sequence[Candidate], features: np.ndarray | None = None
    ) -> np.ndarray:
        matrix = candidate_features(candidates) if features is None else np.asarray(features)
        if matrix.shape != (len(candidates), len(self.feature_names)):
            raise ValueError("Calibrator input shape mismatch")
        normalized = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = self.coefficients[0] + normalized @ np.asarray(self.coefficients[1:])
        return expit(logits)


def fit_calibrator(
    candidates_by_file: Mapping[str, Sequence[Candidate]],
    truths_by_file: Mapping[str, Sequence[Sequence[np.ndarray]]],
    *,
    regularization: float = 0.02,
) -> Calibrator:
    if set(candidates_by_file) != set(truths_by_file):
        raise ValueError("Candidate and truth filename coverage differs")
    matrices = []
    targets = []
    for filename in sorted(candidates_by_file):
        candidates = candidates_by_file[filename]
        if not candidates:
            continue
        matrices.append(candidate_features(candidates))
        targets.append(
            official_match_targets(candidates, truths_by_file[filename]).match_probability
        )
    if not matrices:
        raise ValueError("No calibration candidates")
    matrix = np.concatenate(matrices)
    target = np.concatenate(targets)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 0.05)
    normalized = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = normalized @ beta
        penalty = regularization * np.dot(beta[1:], beta[1:]) / 2
        loss = np.mean(np.logaddexp(0, logits) - target * logits) + penalty
        gradient = normalized.T @ (expit(logits) - target) / len(target)
        gradient[1:] += regularization * beta[1:]
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(normalized.shape[1]), jac=True, method="L-BFGS-B")
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f"Calibration failed: {fitted.message}")
    membership_sha256 = _digest(sorted(candidates_by_file))
    return Calibrator(
        feature_names=FEATURE_NAMES,
        mean=tuple(mean),
        scale=tuple(scale),
        coefficients=tuple(fitted.x),
        regularization=regularization,
        examples=len(target),
        membership_sha256=membership_sha256,
        candidate_sha256=candidate_cache_hash(candidates_by_file),
    )


def fit_and_select_q_recipe(
    quality_fit_candidates: Mapping[str, Sequence[Candidate]],
    quality_fit_truths: Mapping[str, Sequence[Sequence[np.ndarray]]],
    quality_cal_candidates: Mapping[str, Sequence[Candidate]],
    quality_cal_truths: Mapping[str, Sequence[Sequence[np.ndarray]]],
    recipes: Sequence[Any],
) -> tuple[Calibrator, Any, dict[str, Any]]:
    """Fit solely on Q-fit and select one declared recipe solely on Q-cal.

    H is deliberately absent from this public seam.  Empty grids, overlapping
    members, or candidates without retained probabilities fail closed.
    """
    q_fit = set(quality_fit_candidates)
    q_cal = set(quality_cal_candidates)
    if q_fit & q_cal:
        raise ValueError("Q-fit and Q-cal candidate memberships must be disjoint")
    if set(quality_fit_truths) != q_fit or set(quality_cal_truths) != q_cal:
        raise ValueError("Q-fit/Q-cal truth coverage is incomplete")
    if not recipes:
        raise ValueError("Q-cal recipe selection requires a declared bounded grid")
    if any(
        candidate.mask_probability is None
        for cache in (*quality_fit_candidates.values(), *quality_cal_candidates.values())
        for candidate in cache
    ):
        raise ValueError("Q calibration requires ownership-capable candidate probabilities")
    calibrator = fit_calibrator(quality_fit_candidates, quality_fit_truths)
    scored: list[tuple[float, int, Any, dict[str, Any]]] = []
    for index, recipe in enumerate(recipes):
        predictions = {
            filename: assemble_instances(candidates, recipe=recipe, calibrator=calibrator)
            for filename, candidates in quality_cal_candidates.items()
        }
        report = evaluate_run(predictions, quality_cal_truths)
        scored.append((float(report["pq"]), index, recipe, report))
    _, _, recipe, report = max(scored, key=lambda item: (item[0], -item[1]))
    return (
        calibrator,
        recipe,
        {
            "selection_split": "Q-cal",
            "quality_fit_membership_sha256": _digest(sorted(q_fit)),
            "quality_cal_membership_sha256": _digest(sorted(q_cal)),
            "quality_cal_candidate_sha256": candidate_cache_hash(quality_cal_candidates),
            "selected_recipe_index": next(
                index for _, index, value, _ in scored if value == recipe
            ),
            "metrics": report,
        },
    )


def assemble_instances(
    candidates: Sequence[Candidate],
    *,
    recipe: Any,
    calibrator: Calibrator | None = None,
) -> list[np.ndarray]:
    """Select, geometrically deduplicate, and assign every pixel to at most one query."""
    if not candidates:
        return []
    probabilities = (
        calibrator.predict(candidates)
        if calibrator is not None
        else np.asarray([candidate.presence for candidate in candidates])
    )
    utilities = probabilities * np.asarray([candidate.predicted_iou for candidate in candidates])
    ordered = sorted(
        zip(utilities, candidates, strict=True),
        key=lambda item: (-float(item[0]), item[1].query_id),
    )
    retained: list[tuple[float, Candidate, np.ndarray]] = []
    area_floor = max(int(recipe.minimum_area), 1)
    for utility, candidate in ordered:
        probability = (
            candidate.mask_probability
            if candidate.mask_probability is not None
            else candidate.mask.astype(np.float32)
        )
        effective_mask = probability >= recipe.mask_threshold
        # Only numerically empty candidates are removed before ownership.  The
        # selected area threshold applies to the post-ownership remainder.
        if utility < recipe.utility_threshold or not effective_mask.any():
            continue
        if any(
            _mask_iou(effective_mask, previous_mask) >= recipe.duplicate_iou
            or _containment(effective_mask, previous_mask) >= recipe.containment
            for _, _, previous_mask in retained
        ):
            continue
        retained.append((float(utility), candidate, effective_mask))
    if not retained:
        return []

    shape = retained[0][1].mask.shape
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int16)
    for index, (utility, candidate, eligible) in enumerate(retained):
        probability = (
            candidate.mask_probability
            if candidate.mask_probability is not None
            else candidate.mask.astype(np.float32)
        )
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        score = logit + recipe.ownership_gamma * math.log(max(utility, 1e-6))
        stronger = eligible & (score > best_score)
        best_score[stronger] = score[stronger]
        owner[stronger] = index
    result = []
    for index in range(len(retained)):
        mask = owner == index
        if int(mask.sum()) >= area_floor:
            result.append(mask)
    return result


def evaluate_run(
    predictions: Mapping[str, Sequence[np.ndarray]],
    truths: Mapping[str, Sequence[Sequence[np.ndarray]]],
) -> dict[str, Any]:
    if set(predictions) != set(truths):
        raise ValueError("Prediction and truth filename coverage differs")
    total = PQResult()
    per_file = {}
    for filename in sorted(truths):
        file_total = PQResult()
        masks = list(predictions[filename])
        for record in truths[filename]:
            file_total += panoptic_quality(list(record), masks)
        total += file_total
        per_file[filename] = asdict(file_total)
    return {
        "pq": total.pq,
        "sq": total.segmentation_quality,
        "rq": total.recognition_quality,
        "tp": total.true_positives,
        "fp": total.false_positives,
        "fn": total.false_negatives,
        "sum_iou": total.matched_iou,
        "files": per_file,
    }


def _recall_at_match_threshold(
    predictions: Sequence[np.ndarray], truths: Sequence[Sequence[np.ndarray]]
) -> tuple[int, int, int, int]:
    """Count recalled truths plus split/merge errors over complete records."""
    recalled = total = fragments = merges = 0
    for record in truths:
        total += len(record)
        if not record or not predictions:
            continue
        ious = pairwise_iou(list(record), list(predictions))
        recalled += int((ious > 0.5).any(axis=1).sum())
        fragments += int(((ious > 0.1).sum(axis=1) > 1).sum())
        merges += int(((ious > 0.1).sum(axis=0) > 1).sum())
    return recalled, total, fragments, merges


def candidate_diagnostics(
    candidates_by_file: Mapping[str, Sequence[Candidate]],
    truths_by_file: Mapping[str, Sequence[Sequence[np.ndarray]]],
    *,
    final_predictions: Mapping[str, Sequence[np.ndarray]] | None = None,
    duplicate_suppressions: int = 0,
    ownership_damaged_matches: int = 0,
) -> dict[str, Any]:
    """CPU-feasible evidence required before a candidate set can be promoted."""
    if set(candidates_by_file) != set(truths_by_file):
        raise ValueError("Candidate diagnostic coverage differs from truth coverage")
    if final_predictions is not None and set(final_predictions) != set(truths_by_file):
        raise ValueError("Final diagnostic coverage differs from truth coverage")
    raw_recalled = raw_total = final_recalled = final_total = fragments = merges = 0
    count_error = 0
    for filename in sorted(truths_by_file):
        raw, total, raw_fragments, raw_merges = _recall_at_match_threshold(
            [candidate.mask for candidate in candidates_by_file[filename]], truths_by_file[filename]
        )
        raw_recalled += raw
        raw_total += total
        fragments += raw_fragments
        merges += raw_merges
        if final_predictions is not None:
            recalled, count, _, _ = _recall_at_match_threshold(
                final_predictions[filename], truths_by_file[filename]
            )
            final_recalled += recalled
            final_total += count
            count_error += sum(
                abs(len(final_predictions[filename]) - len(record))
                for record in truths_by_file[filename]
            )
    result: dict[str, Any] = {
        "raw_candidate_recall": raw_recalled / raw_total if raw_total else 0.0,
        "raw_candidate_recalled": raw_recalled,
        "truth_instances": raw_total,
        "duplicate_suppressions": int(duplicate_suppressions),
        "ownership_damaged_matches": int(ownership_damaged_matches),
        "fragmentation": fragments,
        "merges": merges,
        "instance_count_absolute_error": count_error,
    }
    if final_predictions is not None:
        result.update(
            final_recall=final_recalled / final_total if final_total else 0.0,
            final_recalled=final_recalled,
        )
    return result


def paired_block_bootstrap(
    challenger_files: Mapping[str, Mapping[str, float]],
    control_files: Mapping[str, Mapping[str, float]],
    *,
    blocks: Sequence[int],
    replicates: int = 2_000,
    seed: int = 20260910,
) -> dict[str, float]:
    """Shared date-block bootstrap of additive PQ sufficient statistics.

    Each mapping value must provide ``tp``, ``fp``, ``fn`` and ``sum_iou``.
    A comparison absent from either side is intentionally an error, rather than
    producing a scientifically meaningless confidence interval.
    """
    if not challenger_files or set(challenger_files) != set(control_files):
        raise ValueError("A matched control with identical per-file evidence is required")
    if replicates < 1 or not blocks:
        raise ValueError("Bootstrap needs positive replicates and assessment blocks")
    by_block: dict[int, list[str]] = {int(block): [] for block in blocks}
    for filename in challenger_files:
        block = _day(filename) // 60
        if block not in by_block:
            raise ValueError("Per-file evidence lies outside frozen assessment blocks")
        by_block[block].append(filename)
    if any(not names for names in by_block.values()):
        raise ValueError("Every assessment block needs matched control evidence")

    def pq(rows: Mapping[str, Mapping[str, float]], sampled: Sequence[int]) -> float:
        totals = {key: 0.0 for key in ("tp", "fp", "fn", "sum_iou")}
        for block in sampled:
            for filename in by_block[block]:
                for key in totals:
                    value = rows[filename].get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Bootstrap evidence contains a non-finite PQ statistic")
                    totals[key] += value
        denominator = totals["tp"] + 0.5 * (totals["fp"] + totals["fn"])
        return totals["sum_iou"] / denominator if denominator else 0.0

    rng = np.random.default_rng(seed)
    observed = pq(challenger_files, list(by_block)) - pq(control_files, list(by_block))
    draws = np.asarray(
        [
            pq(challenger_files, sampled := rng.choice(list(by_block), len(by_block), replace=True))
            - pq(control_files, sampled)
            for _ in range(replicates)
        ]
    )
    return {
        "pq_delta": float(observed),
        "bootstrap_lower_95": float(np.quantile(draws, 0.025)),
        "bootstrap_upper_95": float(np.quantile(draws, 0.975)),
        "replicates": float(replicates),
    }
