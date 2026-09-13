"""Static contract tests for Kaggle-only model and training code."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "filament-pq-setformer"


def definitions(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {
        node.name: {
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    return functions, classes


def test_model_exposes_one_small_train_and_predict_interface() -> None:
    functions, classes = definitions(HERE / "model.py")

    assert {"build_maskdino_config", "optimizer_groups", "finite_parameters"} <= functions
    assert {"train_batch", "predict"} <= classes["PQSetFormer"]


def test_model_keeps_device_selection_outside_the_module() -> None:
    source = (HERE / "model.py").read_text()

    assert ".cuda(" not in source
    assert "torch.cuda.set_device" not in source
    assert "NUM_OBJECT_QUERIES" in source
    assert "NUM_CLASSES = 1" in source
    assert 'META_ARCHITECTURE = "MaskDINO"' in source


def test_remote_training_uses_safe_amp_accumulation_and_gradient_clipping() -> None:
    source = (HERE / "remote.py").read_text()

    assert "GradScaler" in source
    assert ".unscale_(optimizer)" in source
    assert "clip_grad_norm_" in source
    assert "finite_parameters" in source
    assert "optimizer_step" in source
    assert "ExperimentLogger.start" in source
    assert "MSDeformAttnFunction.apply" in source
    assert "custom_op_forward_backward" in source
    assert 'run_kind not in {"smoke", "preflight", "pilot"}' in source
    assert "experiment.training.preflight_microbatches" in source


def test_remote_evaluation_logs_exact_per_file_pq_breakdown() -> None:
    source = (HERE / "remote.py").read_text()

    assert "per_file[filename].update(" in source
    assert "pq=file_iou / file_denominator" in source
    assert '**metrics["files"][name]' in source


def test_remote_releases_training_model_before_checkpoint_reload() -> None:
    source = (HERE / "remote.py").read_text()

    release = source.index("del network, optimizer")
    reload_model = source.index("reloaded = model_module.PQSetFormer")
    assert release < reload_model
    assert source.index("torch.cuda.empty_cache()", release) < reload_model


def test_remote_amp_overflow_backs_off_and_still_requires_five_updates() -> None:
    source = (HERE / "remote.py").read_text()

    assert "AMP_INITIAL_SCALE = 16.0" in source
    assert '"amp_overflow"' in source
    assert "if amp_overflows > MAX_AMP_OVERFLOWS" in source
    assert "if optimizer_step == target_steps:" in source
    assert '"attempted_microbatches"' in source


def test_quality_adapter_is_packaged_and_wrapper_uses_its_real_features() -> None:
    model = (HERE / "model.py").read_text()
    controller = (HERE / "run.py").read_text()

    assert "_conditional_iou_loss" in model
    assert "self.base.criterion.matcher" in model
    assert "quality_targets = torch.zeros_like(quality_logits)" in model
    assert "pred_query_features" in model
    assert "pixel_features" in model
    assert "mask_confidence_proxy" not in model
    assert "PQ-SetFormer E2 adapter" in controller
    assert "pred_query_features" in controller


def test_quality_adapter_uses_b_first_decoder_features() -> None:
    """The DINO decoder already returns hs as [B, seq, C]; never transpose again.

    The original ``hs[-1].transpose(0, 1)`` reordered batch and query axes and
    crashed the conditional-IoU head with "Expected size 92 but got size 1".
    """
    controller = (HERE / "run.py").read_text()

    assert "pq_query_features = hs[-1]" in controller
    assert "pq_query_features = hs[-1].transpose" not in controller
    assert "pq_query_features[:, mask_dict['pad_size']:, :]" in controller
