# Python GPU Templates (CUDA + Apple MPS)

This project is a starter template for running Python compute code on GPU when available.

Supported template modules:
- Tensor alias (`tensor`) -> TensorFlow backend (CUDA or MPS)
- PyTorch (`torch`) -> CUDA or MPS
- TensorFlow (`tensorflow`) -> CUDA or Metal (MPS via `tensorflow-metal`)
- Transformers (`transformers`) -> CUDA or MPS through PyTorch backend
- CuPy (`cupy`) -> CUDA
- cuML (`cuml`) -> CUDA

## 1) Create environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install base dependencies

```bash
pip install -r requirements.txt
```

## 3) Install framework packages

Install only what you need.

### PyTorch

See official selector for your OS/CUDA build: https://pytorch.org/get-started/locally/

To find version of cuda:
```bash
nvcc --version
```

### TensorFlow

- Linux/Windows CUDA setup: install `tensorflow` (with proper NVIDIA stack) https://www.tensorflow.org/install/pip
- macOS Apple Silicon: https://developer.apple.com/metal/tensorflow-plugin/

```bash
xcode-select --install
pip install tensorflow-macos tensorflow-metal
```

### Transformers

Transformers uses a backend framework for execution. In this template, the demo uses the
PyTorch backend, so install both:

```bash
pip install transformers torch
```

For TensorFlow + Hugging Face TF-model inference compatibility, this project pins:
- `transformers>=4.41,<5`
- `huggingface-hub>=0.23,<1.0`
- `tf-keras>=2.16,<2.21` (required for Transformers TF inference with Keras 3)

Reason: Transformers v5 removed TensorFlow auto-model APIs like `TFAutoModel`.

Important compatibility note for macOS Apple Silicon:
- Use `tensorflow-macos` with `tensorflow-metal`.
- `tensorflow-metal` wheels are not available for every Python release.
- If `pip install tensorflow-metal` says "No matching distribution found", use Python 3.11 or 3.12 in a fresh virtual environment.
- If TensorFlow import fails with `libmetal_plugin.dylib` errors, reinstall with a known-good pair such as `tensorflow-macos==2.16.2 tensorflow-metal==1.1.0`.

Example (using Python 3.11):

```bash
# install python 3.11 if needed
brew install python@3.11

# create a clean env with 3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# then install TensorFlow + Metal plugin
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0
```

TensorFlow template note:
- `gpu_templates/tensorflow_template.py` now also runs a tiny Hugging Face check.
- It pulls `hf-internal-testing/tiny-random-distilbert` from the Hugging Face Hub and runs a quick sample-data check.
- Install `transformers` to enable this step:

```bash
pip install transformers
```

### CuPy

Use wheel matching your CUDA version, for example:

```bash
pip install cupy-cuda12x
```

### cuML (RAPIDS)

Install with the RAPIDS instructions for your CUDA/Python combination:
https://docs.rapids.ai/install
for v12
```bash
pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```
## 4) Run device checks

```bash
python -m gpu_templates.run_all
```

You will get per-framework status plus a tiny compute demo for each available package.

## 5) Project layout

- `gpu_templates/pytorch_template.py`
- `gpu_templates/tensorflow_template.py`
- `gpu_templates/tensor_template.py`
- `gpu_templates/cupy_template.py`
- `gpu_templates/cuml_template.py`
- `gpu_templates/transformers_template.py`
- `gpu_templates/run_all.py`

Use these files as starting points for your real model or tensor pipelines.
