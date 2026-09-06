"""Complete the frozen precision stage on 601 images, then generate checked RLEs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import precision_model as model_code
from plan import training_plan

from sfsc.dataset import submission
from sfsc.system import digest


def run(context):
    import tensorflow as tf

    if context["phase"] != "full_train":
        raise ValueError("This runner performs frozen full training only")
    output = context["output_dir"]
    bundle = Path(__file__).parent
    precision = json.loads((bundle / "precision-recipe.json").read_text())
    expected_baseline = json.loads((bundle / "baseline-recipe.json").read_text())
    baseline = json.loads((context["checkpoint"] / "recipe.json").read_text())
    if baseline != expected_baseline:
        raise ValueError("Mounted baseline recipe differs from the frozen source")
    plan = training_plan(
        precision, baseline, context["train_filenames"], context["dataset"].filenames("test")
    )
    config = plan["config"]
    if config != context["config"]:
        raise ValueError("Runtime config differs from the frozen precision recipe")
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(exist_ok=False)
    model = model_code.build_model(config["base_filters"])
    model.load_weights(context["checkpoint"] / "best.weights.h5")
    model_code.log(
        output,
        "full_train_started",
        train_images=len(context["train_filenames"]),
        baseline_epochs=60,
        precision_epochs=plan["epochs"],
        optimizer_steps=plan["optimizer_steps"],
        schedule_steps=plan["schedule_steps"],
        parameters=model.count_params(),
        upstream=str(context["checkpoint"]),
        postprocess=plan["postprocess"],
    )
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        config["learning_rate"],
        plan["schedule_steps"],
        alpha=config["minimum_learning_rate"] / config["learning_rate"],
    )
    optimizer = tf.keras.optimizers.AdamW(
        schedule, weight_decay=config["weight_decay"], global_clipnorm=1.0
    )

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = model_code.segmentation_loss(
                y, model(x, training=True), config["positive_weights"]
            )
        tf.debugging.assert_all_finite(loss, "Non-finite full-training loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Non-finite full-training gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        return loss

    batches = model_code.train_stream(context)
    started = time.monotonic()
    for epoch in range(1, plan["epochs"] + 1):
        epoch_start = time.monotonic()
        losses = []
        for step in range(1, plan["steps_per_epoch"] + 1):
            x, y = next(batches)
            loss = float(train_step(x, y).numpy())
            losses.append(loss)
            if step == 1 or step % 50 == 0:
                peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
                model_code.log(
                    output,
                    "training_step",
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    peak_allocator_gb=peak,
                    elapsed_seconds=time.monotonic() - started,
                )
                if peak > 14:
                    raise RuntimeError("Full training exceeds the 14 GB allocator envelope")
        model.save_weights(checkpoint / "last.weights.h5")
        model_code.log(
            output,
            "epoch_complete",
            epoch=epoch,
            mean_loss=float(np.mean(losses)),
            seconds=time.monotonic() - epoch_start,
        )
        if time.monotonic() - started > config["training_time_limit_hours"] * 3600:
            raise RuntimeError("Frozen training budget exceeded; no partial model submission")
    if int(optimizer.iterations.numpy()) != plan["optimizer_steps"]:
        raise RuntimeError("Incomplete frozen optimizer schedule")
    model.save_weights(checkpoint / "best.weights.h5")
    recipe = {
        "config": config,
        "best_epoch": plan["epochs"],
        "postprocess": plan["postprocess"],
        "checkpoint_validation_pq": None,
        "training_scope": "601 train+validation; 106 internal-test images excluded",
        "baseline_epochs": 60,
        "precision_optimizer_steps": plan["optimizer_steps"],
        "precision_schedule_steps": plan["schedule_steps"],
    }
    (checkpoint / "recipe.json").write_text(json.dumps(recipe, indent=2))
    (checkpoint / "best.json").write_text(
        json.dumps(
            {"epoch": plan["epochs"], "pq": None, "selection": "frozen development recipe"},
            indent=2,
        )
    )
    model_code.log(output, "training_complete", recipe=recipe)

    predictions = output / "predictions"
    predictions.mkdir(exist_ok=False)
    images = sorted((context["dataset"].root / "test/test_images").glob("*.jpeg"))
    if len(images) != 180:
        raise RuntimeError("Expected exactly 180 competition test images")
    inference_start = time.monotonic()
    for index, path in enumerate(images, 1):
        probability = model_code.infer(model, model_code.read_image(str(path)), config)
        labels = model_code.instances(probability, plan["postprocess"])
        np.save(predictions / f"{path.stem}.npy", labels.astype(np.int32), allow_pickle=False)
        if index == 1 or index % 10 == 0:
            model_code.log(
                output,
                "test_prediction",
                index=index,
                images=len(images),
                elapsed_seconds=time.monotonic() - inference_start,
            )
    report = submission(context["dataset"].root, predictions, output / "submission.csv")
    report.update(
        status="completed",
        submission_sha256=digest(output / "submission.csv"),
        weights_sha256=digest(checkpoint / "best.weights.h5"),
        postprocess=plan["postprocess"],
        precision_epochs=plan["epochs"],
        train_images=601,
        inference_seconds=time.monotonic() - inference_start,
    )
    (output / "submission-report.json").write_text(json.dumps(report, indent=2))
    model_code.log(
        output,
        "submission_generated",
        instances=report["instances"],
        empty_images=report["empty_images"],
        sha256=report["submission_sha256"],
    )
    return {"checkpoint": "checkpoint"}
