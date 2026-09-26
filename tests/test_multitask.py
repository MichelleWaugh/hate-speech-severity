
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast

from hsd.models.multitask import ModelConfig, XLMRMultiTask


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    encoder_config = XLMRobertaConfig(
        vocab_size=250002,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    encoder = XLMRobertaModel(encoder_config)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    output_dir = tmp_path_factory.mktemp("tiny_xlmr")
    encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def build_tiny_config(model_dir: Path, target_head_detach: bool) -> ModelConfig:
    return ModelConfig(
        model_name=str(model_dir),
        hidden_dropout=0.1,
        target_head_detach=target_head_detach,
        num_classes=4,
        num_severity_levels=4,
        num_target_labels=8,
    )


def test_forward_returns_expected_shapes(tiny_encoder_dir: Path) -> None:
    config = build_tiny_config(tiny_encoder_dir, target_head_detach=False)
    model = XLMRMultiTask(config)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs["class_logits"].shape == (2, 4)
    assert outputs["severity_logits"].shape == (2, 4)
    assert outputs["target_logits"].shape == (2, 8)


def test_target_head_detach_blocks_encoder_gradient(tiny_encoder_dir: Path) -> None:
    config = build_tiny_config(tiny_encoder_dir, target_head_detach=True)
    model = XLMRMultiTask(config)
    model.train()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = outputs["target_logits"].sum()
    loss.backward()
    encoder_grad_norms = [
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert sum(encoder_grad_norms) == 0.0


def test_target_head_not_detached_allows_encoder_gradient(tiny_encoder_dir: Path) -> None:
    config = build_tiny_config(tiny_encoder_dir, target_head_detach=False)
    model = XLMRMultiTask(config)
    model.train()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = outputs["target_logits"].sum()
    loss.backward()
    encoder_grad_norms = [
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert sum(encoder_grad_norms) > 0.0


def test_save_and_load_reproduces_logits(tmp_path: Path, tiny_encoder_dir: Path) -> None:
    config = build_tiny_config(tiny_encoder_dir, target_head_detach=False)
    model = XLMRMultiTask(config)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        original_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    output_dir = tmp_path / "best"
    model.save_pretrained(output_dir)
    reloaded_model = XLMRMultiTask.from_pretrained(output_dir)

    with torch.no_grad():
        reloaded_outputs = reloaded_model(input_ids=input_ids, attention_mask=attention_mask)

    for key in ("class_logits", "severity_logits", "target_logits"):
        assert torch.allclose(original_outputs[key], reloaded_outputs[key], atol=1e-5)
