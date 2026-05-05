"""Drop-in wrapper for AMD Quark quantization with Growt auditing."""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from growt_client import (
    AuditResult,
    GrowtClient,
    MetricsResult,
    QuantizationAuditResult,
    format_audit_report,
    format_quantization_report,
)
from growt_quark.extractor import extract_features

logger = logging.getLogger("growt_quark")


def growt_quantize(
    model: torch.nn.Module,
    config: Any,
    calibration_data: DataLoader,
    labels: Optional[list] = None,
    val_accuracy: Optional[float] = None,
    fail_on_red_flag: bool = True,
    api_url: str = "https://api.transferoracle.ai",
    api_key: Optional[str] = None,
    max_samples: int = 5000,
) -> tuple[torch.nn.Module, AuditResult]:
    """Drop-in audit wrapper for AMD Quark quantization.

    Usage:
        from quark.torch import ModelQuantizer
        from quark.torch.quantization.config.config import QConfig
        from growt_quark import growt_quantize

        # Instead of: quantized = quantizer.quantize_model(model, dataloader)
        quantized, audit = growt_quantize(model, quant_config, calib_loader)
    """
    from quark.torch import ModelQuantizer

    client = GrowtClient(api_url=api_url, api_key=api_key)

    # 1. Deep-copy original (CPU to save VRAM)
    logger.info("[Growt] Preserving original model...")
    original_model = copy.deepcopy(model).cpu()

    # 2. Quantize with AMD Quark
    logger.info("[Growt] Running AMD Quark quantization...")
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model, calibration_data)

    # 3. Extract features from ORIGINAL on calibration data
    logger.info("[Growt] Extracting original model features...")
    device = next(quantized_model.parameters()).device
    original_model = original_model.to(device)
    features_original, extracted_labels = extract_features(original_model, calibration_data, max_samples=max_samples)
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Extract features from QUANTIZED on SAME data
    logger.info("[Growt] Extracting quantized model features...")
    features_quantized, _ = extract_features(quantized_model, calibration_data, max_samples=max_samples)

    final_labels = labels if labels is not None else extracted_labels.tolist()

    # 5. Growt audit
    logger.info("[Growt] Running structural transfer audit...")
    audit = client.audit_transfer(
        features_train=features_original.tolist(),
        labels_train=final_labels,
        features_deploy=features_quantized.tolist(),
        val_accuracy=val_accuracy,
    )

    # 6. SQNR metrics
    metrics = client.metrics_compare(
        features_reference=features_original.tolist(),
        features_compare=features_quantized.tolist(),
        labels_reference=final_labels,
    )

    # 7. Rich report
    print(format_audit_report(audit, metrics, title="GROWT AMD QUARK QUANTIZATION AUDIT"))

    if fail_on_red_flag and audit.diagnosis == "RED_FLAG":
        raise RuntimeError(f"[Growt] Quantization RED_FLAG — unsafe to deploy.\n{audit.report}")

    return quantized_model, audit


def growt_quantize_compare(
    model: torch.nn.Module,
    variants: dict[str, Any],
    calibration_data: DataLoader,
    labels: Optional[list] = None,
    api_url: str = "https://api.transferoracle.ai",
    api_key: Optional[str] = None,
    max_samples: int = 5000,
) -> QuantizationAuditResult:
    """Compare multiple AMD Quark quantization configs."""
    from quark.torch import ModelQuantizer

    client = GrowtClient(api_url=api_url, api_key=api_key)

    features_ref, extracted_labels = extract_features(model, calibration_data, max_samples=max_samples)
    final_labels = labels if labels is not None else extracted_labels.tolist()

    variant_features: dict[str, list[list[float]]] = {}
    metrics_per: dict[str, MetricsResult] = {}

    for name, config in variants.items():
        logger.info("[Growt] Quantizing variant '%s'...", name)
        variant_model = copy.deepcopy(model)
        quantizer = ModelQuantizer(config)
        quantizer.quantize_model(variant_model, calibration_data)

        feats, _ = extract_features(variant_model, calibration_data, max_samples=max_samples)
        variant_features[name] = feats.tolist()
        metrics_per[name] = client.metrics_compare(features_ref.tolist(), variant_features[name], final_labels)

        del variant_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = client.audit_quantization(features_ref.tolist(), final_labels, variant_features)
    print(format_quantization_report(result, metrics_per))
    return result
