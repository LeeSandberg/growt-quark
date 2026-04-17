# growt-quark

**Growt safety audit for [AMD Quark](https://github.com/amd/Quark) quantization** — know what you lost.

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

> "Rate-distortion says you WILL lose something. Growt tells you WHAT."

## What is this?

Drop-in audit wrapper for AMD Quark's `ModelQuantizer.quantize_model()`. Compares model structure before and after quantization, reports per-class coverage, SQNR, and flags degradation.

## Install

```bash
pip install growt-quark
```

## Quick Start

```python
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from growt_quark import growt_quantize

# Instead of: quantized = quantizer.quantize_model(model, dataloader)
quantized, audit = growt_quantize(
    model,
    config=quant_config,
    calibration_data=calib_loader,
    api_key="your-growt-api-key",
)
# Prints: diagnosis, SQNR, per-class coverage
```

## Multi-Variant Comparison

```python
from growt_quark import growt_quantize_compare

result = growt_quantize_compare(
    model,
    variants={"INT4": int4_config, "FP8": fp8_config},
    calibration_data=calib_loader,
)
```

## How It Works

1. Deep-copies your model before quantization
2. Runs `ModelQuantizer.quantize_model()` on the copy
3. Extracts features from BOTH models on SAME data
4. Calls Growt API to compare original vs quantized
5. Reports per-class coverage, SQNR, and flags degradation

## License

[MPL-2.0](LICENSE)

## Status & Contributing

This is an early release to get the integration started. The code works but is not battle-tested in production yet. We welcome contributions:

- Bug fixes and improvements — PRs welcome
- New features and endpoint integrations
- Better error handling and edge cases
- Documentation improvements
- Test coverage

Open an issue or submit a PR on GitHub. All contributions must be compatible with the MPL-2.0 license.


## Related

- [Documentation](https://transferoracle.ai/growt/docs) — API reference, all plugins, tiers
- [growt-client](https://github.com/LeeSandberg/growt-client) — Python client (shared by all plugins)
- [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) — NVIDIA ModelOpt
- [growt-quark](https://github.com/LeeSandberg/growt-quark) — AMD Quark
- [growt-nemo](https://github.com/LeeSandberg/growt-nemo) — NeMo / PyTorch Lightning
- [growt-vllm](https://github.com/LeeSandberg/growt-vllm) — vLLM (NVIDIA + AMD)
- [growt-triton](https://github.com/LeeSandberg/growt-triton) — Triton Inference Server
- [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) — TensorRT validator
- [growt-tao](https://github.com/LeeSandberg/growt-tao) — TAO Toolkit

