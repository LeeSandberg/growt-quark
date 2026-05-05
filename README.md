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

## All Growt Plugins

Open-source plugins and SDKs for the [Transfer Oracle](https://transferoracle.ai) structural AI auditing API.
Plugin code is MPL-2.0; API access is commercial and requires an [API key](https://transferoracle.ai/growt/plugins).

| Plugin | Platform | What it does |
|--------|----------|-------------|
| [growt-client](https://github.com/LeeSandberg/growt-client) | Core | Python client library |
| [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) | NVIDIA | ModelOpt quantization audit |
| [growt-quark](https://github.com/LeeSandberg/growt-quark) | AMD | Quark quantization audit |
| [growt-nemo](https://github.com/LeeSandberg/growt-nemo) | NVIDIA | NeMo / PyTorch Lightning callback |
| [growt-vllm](https://github.com/LeeSandberg/growt-vllm) | NVIDIA + AMD | vLLM inference monitor |
| [growt-triton](https://github.com/LeeSandberg/growt-triton) | NVIDIA | Triton Inference Server monitor |
| [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) | NVIDIA | TensorRT engine validator |
| [growt-tao](https://github.com/LeeSandberg/growt-tao) | NVIDIA | TAO Toolkit pipeline |
| [mlflow-growt](https://github.com/LeeSandberg/mlflow-growt) | MLflow | Evaluator + deployment gate |
| [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) | HuggingFace | TrainerCallback + Model Card |
| [growt-wandb](https://github.com/LeeSandberg/growt-wandb) | W&B | Callback + artifact + registry gate |
| [growt-airflow](https://github.com/LeeSandberg/growt-airflow) | Airflow | Pre-deployment audit operator |
| [growt-kubeflow](https://github.com/LeeSandberg/growt-kubeflow) | Kubeflow | Pipeline validation component |
| [growt-kserve](https://github.com/LeeSandberg/growt-kserve) | KServe | Pre-serve validation transformer |
| [growt-dagster](https://github.com/LeeSandberg/growt-dagster) | Dagster | Asset + resource for audit |
| [growt-dvc](https://github.com/LeeSandberg/growt-dvc) | DVC | Pre-push model validation |
| [growt-bentoml](https://github.com/LeeSandberg/growt-bentoml) | BentoML | Pre-serve validation hook |
| [growt-argo](https://github.com/LeeSandberg/growt-argo) | Argo | Workflow validation template |
| [growt-prefect](https://github.com/LeeSandberg/growt-prefect) | Prefect | Task + block for audit |
| [growt-clearml](https://github.com/LeeSandberg/growt-clearml) | ClearML | Pipeline step + callback |
| [growt-docker](https://github.com/LeeSandberg/growt-docker) | Docker/OCI | Audit metadata in containers |

**Links:** [API Docs](https://transferoracle.ai/growt/docs) · [Get API Key](https://transferoracle.ai/growt/plugins) · [LLM Benchmark](https://transferoracle.ai/growt/llm-benchmark) · [All Benchmarks](https://transferoracle.ai/benchmarks)
