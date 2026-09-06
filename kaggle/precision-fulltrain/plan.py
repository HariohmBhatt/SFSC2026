"""Validate the two frozen training stages without importing a GPU framework."""


def training_plan(precision, baseline, train_names, heldout_names):
    if baseline["best_epoch"] != 60 or baseline["config"]["epochs"] != 60:
        raise ValueError("Upstream must be the completed 60-epoch baseline")
    config = dict(precision["config"])
    epochs = precision["best_epoch"]
    if not 0 < epochs <= config["epochs"]:
        raise ValueError("Invalid frozen precision checkpoint epoch")
    if len(train_names) != 601 or len(set(train_names)) != 601:
        raise ValueError("Full training requires exactly 601 unique development images")
    if set(train_names) & set(heldout_names):
        raise ValueError("Internal holdout cannot enter full training")
    for key in ["base_filters", "crop_size", "batch_size"]:
        if config[key] != baseline["config"][key]:
            raise ValueError(f"Model geometry differs from baseline: {key}")
    return {
        "config": config,
        "epochs": epochs,
        "steps_per_epoch": config["steps_per_epoch"],
        # A selected epoch is a prefix of the original schedule. Compressing the
        # 40-epoch cosine curve into 15 epochs would be a different recipe.
        "schedule_steps": config["epochs"] * config["steps_per_epoch"],
        "optimizer_steps": epochs * config["steps_per_epoch"],
        "postprocess": dict(precision["postprocess"]),
    }
