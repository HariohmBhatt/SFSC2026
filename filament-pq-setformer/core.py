"""CPU contracts for blocked splits, calibration, assembly, and PQ reporting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
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
    source: str = "identity"

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("Candidate mask must be non-empty and two-dimensional")
        if not 0 <= self.presence <= 1 or not 0 <= self.predicted_iou <= 1:
            raise ValueError("Candidate scores must be probabilities")
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


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def _containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    return intersection / max(min(int(first.sum()), int(second.sum())), 1)


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
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        entropy = float(np.mean(-clipped * np.log(clipped) - (1 - clipped) * np.log(1 - clipped)))
        presence = min(max(candidate.presence, 1e-6), 1 - 1e-6)
        rows.append(
            [
                math.log(presence / (1 - presence)),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "regularization": self.regularization,
            "examples": self.examples,
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
        )
        if result.feature_names != FEATURE_NAMES:
            raise ValueError("Calibrator feature schema mismatch")
        if len(result.coefficients) != len(FEATURE_NAMES) + 1:
            raise ValueError("Calibrator coefficient shape mismatch")
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
    return Calibrator(
        feature_names=FEATURE_NAMES,
        mean=tuple(mean),
        scale=tuple(scale),
        coefficients=tuple(fitted.x),
        regularization=regularization,
        examples=len(target),
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
    retained: list[tuple[float, Candidate]] = []
    area_floor = max(int(recipe.minimum_area), 1)
    for utility, candidate in ordered:
        if utility < recipe.utility_threshold or int(candidate.mask.sum()) < area_floor:
            continue
        if any(
            _mask_iou(candidate.mask, previous.mask) >= recipe.duplicate_iou
            or _containment(candidate.mask, previous.mask) >= recipe.containment
            for _, previous in retained
        ):
            continue
        retained.append((float(utility), candidate))
    if not retained:
        return []

    shape = retained[0][1].mask.shape
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int16)
    for index, (utility, candidate) in enumerate(retained):
        probability = (
            candidate.mask_probability
            if candidate.mask_probability is not None
            else candidate.mask.astype(np.float32)
        )
        eligible = probability >= recipe.mask_threshold
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
