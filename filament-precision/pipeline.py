"""Filament ResUNet: native-resolution three-head instance segmentation on P100."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from geometry import (
    blend_window,
    input_channels,
    instances,
    pq_score,
    pq_totals,
    read_image,
    sample_crop,
    stratified_validation,
    tile_starts,
)


def log(output: Path, event: str, **fields) -> None:
    record = {"event": event, "unix_time": time.time(), **fields}
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    with (output / "training-log.jsonl").open("a") as stream:
        stream.write(line + "\n")
    temporary = output / "progress.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(output / "progress.json")


def build_model(base: int):
    import tensorflow as tf

    layers = tf.keras.layers

    def block(x, channels):
        skip = layers.Conv2D(channels, 1, padding="same", use_bias=False)(x)
        for _ in range(2):
            x = layers.Conv2D(
                channels,
                3,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(x)
            x = layers.GroupNormalization(groups=8, epsilon=1e-5)(x)
            x = layers.Activation("relu")(x)
        return layers.Activation("relu")(layers.Add()([x, skip]))

    inputs = layers.Input((None, None, 2), name="halpha_and_local_contrast")
    x = inputs
    skips = []
    for width in [base, base * 2, base * 4, base * 8]:
        x = block(x, width)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = block(x, base * 16)
    x = layers.SpatialDropout2D(0.15)(x)
    for width, skip in zip([base * 8, base * 4, base * 2, base], reversed(skips), strict=True):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = block(x, width)
    logits = layers.Conv2D(
        3,
        1,
        dtype="float32",
        name="foreground_interior_boundary",
        bias_initializer=tf.keras.initializers.Constant([-4.0, -4.0, -4.0]),
    )(x)
    return tf.keras.Model(inputs, logits, name="filament_resunet")


def segmentation_loss(target, logits, positive_weights):
    import tensorflow as tf

    prediction = tf.math.sigmoid(logits)
    axes = (1, 2)
    overlap = tf.reduce_sum(target * prediction, axis=axes)
    dice = 1 - (2 * overlap + 1) / (tf.reduce_sum(target + prediction, axis=axes) + 1)
    positive_weight = tf.constant(positive_weights, dtype=tf.float32)
    bce = tf.nn.weighted_cross_entropy_with_logits(target, logits, positive_weight)
    bce = tf.reduce_mean(bce, axis=axes)
    return tf.reduce_mean(tf.reduce_sum((bce + dice) * tf.constant([1.0, 0.5, 0.25]), axis=-1))


def train_stream(context):
    import tensorflow as tf

    config = context["config"]
    names = context["train_filenames"]
    assert names, "Training requires allowed filenames"
    assert set(names) <= set(
        context["dataset"].filenames(
            *(("train", "validation") if context["phase"] == "full_train" else ("train",))
        )
    )
    rng = np.random.default_rng(context["seed"])
    side, batch_size = config["crop_size"], config["batch_size"]

    def generator():
        while True:
            batch = [
                sample_crop(
                    context["dataset"],
                    names[int(rng.integers(len(names)))],
                    config,
                    rng,
                )
                for _ in range(batch_size)
            ]
            yield np.stack([x for x, _ in batch]), np.stack([y for _, y in batch])

    stream = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((batch_size, side, side, 2), tf.float32),
            tf.TensorSpec((batch_size, side, side, 3), tf.float32),
        ),
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return iter(stream.with_options(options).prefetch(2))


def infer(model, gray: np.ndarray, config: dict) -> np.ndarray:
    import tensorflow as tf

    side = config["crop_size"]
    original = gray.shape
    padded = np.pad(
        gray,
        ((0, max(0, side - gray.shape[0])), (0, max(0, side - gray.shape[1]))),
        mode="reflect",
    )
    # Compute the contrast channel globally so it is identical in overlapping tiles.
    channels = input_channels(padded)
    coordinates = list(
        itertools.product(
            tile_starts(padded.shape[0], side, config["inference_stride"]),
            tile_starts(padded.shape[1], side, config["inference_stride"]),
        )
    )
    total = np.zeros((*padded.shape, 3), dtype=np.float32)
    coverage = np.zeros((*padded.shape, 1), dtype=np.float32)
    window = blend_window(side)
    batch_size = config["batch_size"]
    for start in range(0, len(coordinates), batch_size):
        locations = coordinates[start : start + batch_size]
        batch = np.stack([channels[y : y + side, x : x + side] for y, x in locations])
        prediction = tf.math.sigmoid(model(batch, training=False)).numpy()
        if config["tta_flips"]:
            mirrored = tf.math.sigmoid(model(batch[:, :, ::-1, :], training=False)).numpy()
            prediction = (prediction + mirrored[:, :, ::-1, :]) * 0.5
        for (y, x), probability in zip(locations, prediction, strict=True):
            total[y : y + side, x : x + side] += probability * window
            coverage[y : y + side, x : x + side] += window
    assert np.all(coverage > 0), "Uncovered inference pixels"
    return (total / coverage)[: original[0], : original[1]]


def measure(model, dataset, filenames: list[str], config: dict, parameters: dict):
    totals = np.zeros(4, dtype=np.float64)
    for name in filenames:
        probability = infer(model, read_image(str(dataset.image_path(name))), config)
        labels = instances(probability, parameters)
        for record in dataset.records[name]:
            totals += pq_totals(dataset.truth(record), labels)
    return pq_score(totals), totals.tolist()


def calibrate(model, context, checkpoint: Path, config: dict, tag="calibration") -> dict:
    dataset, output = context["dataset"], context["output_dir"]
    filenames = stratified_validation(dataset, config["calibration_images_per_site"])
    combinations = config["postprocess_grid"]
    totals = np.zeros((len(combinations), 4), dtype=np.float64)
    for index, name in enumerate(filenames):
        probabilities = infer(model, read_image(str(dataset.image_path(name))), config)
        truths = [dataset.truth(record) for record in dataset.records[name]]
        for combination, parameters in enumerate(combinations):
            labels = instances(probabilities, parameters)
            for truth in truths:
                totals[combination] += pq_totals(truth, labels)
        log(
            output,
            "calibration_image",
            index=index + 1,
            images=len(filenames),
            filename=name,
        )
    scores = [pq_score(t) for t in totals]
    best = int(np.argmax(scores))
    report = {
        "filenames": filenames,
        "selected": combinations[best],
        "pq": scores[best],
        "grid": [
            dict(parameters=p, pq=s, totals=t.tolist())
            for p, s, t in zip(combinations, scores, totals, strict=True)
        ],
    }
    (checkpoint / f"{tag}.json").write_text(json.dumps(report, indent=2))
    log(output, "calibration_complete", tag=tag, pq=scores[best], parameters=combinations[best])
    return combinations[best]


def run(context):
    import tensorflow as tf

    config = dict(context["config"])
    output = context["output_dir"]
    phase = context["phase"]
    assert phase == "evaluate", "This research experiment is validation-only"
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    if phase in {"submit", "internal_test"}:
        saved = json.loads((context["checkpoint"] / "recipe.json").read_text())
        config = saved["config"]
    elif phase == "full_train":
        # Read the frozen evaluation recipe; start full-training weights from scratch.
        saved = json.loads((context["checkpoint"] / "recipe.json").read_text())
        config = saved["config"]
    model = build_model(config["base_filters"])
    if phase == "evaluate" and config.get("initial_checkpoint"):
        assert context["checkpoint"] is not None, "Train-only upstream checkpoint required"
        model.load_weights(context["checkpoint"] / "best.weights.h5")
        log(output, "warm_start_loaded", checkpoint=str(context["checkpoint"]))
    log(
        output,
        "model_created",
        parameters=model.count_params(),
        phase=phase,
        train_images=len(context["train_filenames"]),
        input_shape=[config["crop_size"]] * 2,
        batch_size=config["batch_size"],
    )
    if phase in {"evaluate", "full_train"}:
        context = {**context, "config": config}
        total_steps = config["epochs"] * config["steps_per_epoch"]
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            config["learning_rate"],
            total_steps,
            alpha=config["minimum_learning_rate"] / config["learning_rate"],
        )
        optimizer = tf.keras.optimizers.AdamW(
            schedule, weight_decay=config["weight_decay"], global_clipnorm=1.0
        )

        @tf.function(reduce_retracing=True, jit_compile=False)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = segmentation_loss(y, model(x, training=True), config["positive_weights"])
            tf.debugging.assert_all_finite(loss, "Non-finite training loss")
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
            return loss

        batches = train_stream(context)
        validation = (
            stratified_validation(context["dataset"], config["checkpoint_images_per_site"])
            if phase == "evaluate"
            else []
        )
        started = time.monotonic()
        best_pq, best_epoch = -1.0, 0
        parameters = config["default_postprocess"]
        if phase == "evaluate":
            parameters = calibrate(model, context, checkpoint, config, "baseline-calibration")
            best_pq, totals = measure(model, context["dataset"], validation, config, parameters)
            model.save_weights(checkpoint / "initial.weights.h5")
            model.save_weights(checkpoint / "best.weights.h5")
            (checkpoint / "best.json").write_text(
                json.dumps({"epoch": 0, "pq": best_pq, "totals": totals}, indent=2)
            )
            log(output, "baseline_calibrated", pq=best_pq, totals=totals, parameters=parameters)
        training_epochs = saved["best_epoch"] if phase == "full_train" else config["epochs"]
        for epoch in range(1, training_epochs + 1):
            epoch_started = time.monotonic()
            losses = []
            for step in range(1, config["steps_per_epoch"] + 1):
                batch_x, batch_y = next(batches)
                loss = float(train_step(batch_x, batch_y).numpy())
                losses.append(loss)
                if step == 1 or step % 50 == 0:
                    memory = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
                    log(
                        output,
                        "training_step",
                        epoch=epoch,
                        step=step,
                        loss=loss,
                        peak_allocator_gb=memory,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    if memory > 14:
                        raise RuntimeError("Model exceeds the fixed 14 GB allocator envelope")
            model.save_weights(checkpoint / "last.weights.h5")
            ran_out = time.monotonic() - started >= config["training_time_limit_hours"] * 3600
            should_validate = epoch == 1 or epoch % config["validation_every"] == 0 or ran_out
            score = None
            if validation and (should_validate or epoch == training_epochs):
                score, totals = measure(model, context["dataset"], validation, config, parameters)
                log(
                    output,
                    "checkpoint_validation",
                    epoch=epoch,
                    pq=score,
                    totals=totals,
                    filenames=validation,
                )
                if score > best_pq:
                    best_pq, best_epoch = score, epoch
                    model.save_weights(checkpoint / "best.weights.h5")
                    (checkpoint / "best.json").write_text(
                        json.dumps({"epoch": epoch, "pq": score, "totals": totals}, indent=2)
                    )
            if not validation:
                best_epoch = epoch
                model.save_weights(checkpoint / "best.weights.h5")
            log(
                output,
                "epoch_complete",
                epoch=epoch,
                mean_loss=float(np.mean(losses)),
                seconds=time.monotonic() - epoch_started,
                validation_pq=score,
                best_epoch=best_epoch,
            )
            if ran_out:
                log(output, "training_time_limit", completed_epochs=epoch)
                if phase == "full_train" and epoch < training_epochs:
                    raise RuntimeError("Full training did not complete the frozen number of epochs")
                break
        assert best_epoch >= 0, "No valid checkpoint selected"
        model.load_weights(checkpoint / "best.weights.h5")
        if phase == "evaluate":
            parameters = calibrate(model, context, checkpoint, config)
            # The last checkpoint may need different probability thresholds after
            # changing BCE weights. Compare it with the best fixed-threshold model.
            selected = json.loads((checkpoint / "calibration.json").read_text())
            model.load_weights(checkpoint / "last.weights.h5")
            last_parameters = calibrate(model, context, checkpoint, config, "last-calibration")
            last = json.loads((checkpoint / "last-calibration.json").read_text())
            if last["pq"] > selected["pq"]:
                best_epoch, best_pq, parameters = epoch, last["pq"], last_parameters
                model.save_weights(checkpoint / "best.weights.h5")
                (checkpoint / "calibration.json").write_text(json.dumps(last, indent=2))
            else:
                best_pq = selected["pq"]
                model.load_weights(checkpoint / "best.weights.h5")
            (checkpoint / "best.json").write_text(
                json.dumps(
                    {"epoch": best_epoch, "pq": best_pq, "selection": "calibration"}, indent=2
                )
            )
        else:
            parameters = saved["postprocess"]
        recipe = {
            "config": config,
            "best_epoch": best_epoch,
            "postprocess": parameters,
            "checkpoint_validation_pq": best_pq if phase == "evaluate" else None,
        }
        (checkpoint / "recipe.json").write_text(json.dumps(recipe, indent=2))
        log(
            output,
            "training_complete",
            best_epoch=best_epoch,
            seconds=time.monotonic() - started,
            recipe=recipe,
        )
        if phase == "full_train":
            return {"checkpoint": "checkpoint"}
    else:
        model.load_weights(context["checkpoint"] / "best.weights.h5")
        parameters = saved["postprocess"]
    folder = output / "predictions"
    folder.mkdir(exist_ok=True)
    inference_started = time.monotonic()
    for index, name in enumerate(context["predict_filenames"]):
        image = read_image(str(context["dataset"].image_path(name, competition=phase == "submit")))
        probability = infer(model, image, config)
        if config.get("save_validation_probabilities"):
            probability_folder = output / "probabilities"
            probability_folder.mkdir(exist_ok=True)
            np.save(probability_folder / f"{Path(name).stem}.npy", probability.astype(np.float16))
        labels = instances(probability, parameters)
        np.save(
            folder / f"{Path(name).stem}.npy",
            labels.astype(np.int32),
            allow_pickle=False,
        )
        log(
            output,
            "prediction_saved",
            filename=name,
            index=index + 1,
            images=len(context["predict_filenames"]),
            instances=int(labels.max(initial=0)),
        )
    log(output, "inference_complete", seconds=time.monotonic() - inference_started)
    if phase == "evaluate":
        # Fair control: same native checkpoint, validation-calibrated extraction,
        # full 106-image metric; never touch the already-used internal holdout.
        from sfsc.dataset import evaluate

        model.load_weights(checkpoint / "initial.weights.h5")
        baseline_parameters = json.loads((checkpoint / "baseline-calibration.json").read_text())[
            "selected"
        ]
        baseline_folder = output / "baseline-predictions"
        baseline_folder.mkdir(exist_ok=True)
        for name in context["predict_filenames"]:
            probability = infer(model, read_image(str(context["dataset"].image_path(name))), config)
            labels = instances(probability, baseline_parameters)
            np.save(baseline_folder / f"{Path(name).stem}.npy", labels.astype(np.int32))
        baseline_metrics = evaluate(context["dataset"], baseline_folder, "validation")
        (output / "baseline-metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
        log(output, "baseline_full_validation", pq=baseline_metrics["pq"])
    result = {"predictions": "predictions"}
    if phase == "evaluate":
        result["checkpoint"] = "checkpoint"
    return result
