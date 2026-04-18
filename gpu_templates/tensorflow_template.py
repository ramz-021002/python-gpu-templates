from __future__ import annotations

import importlib
import os
import platform
from typing import Any, Dict


def _run_hf_tensorflow_demo(tf: Any) -> str:
    model_id = "hf-internal-testing/tiny-random-distilbert"

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return f"HF step skipped (install huggingface stack): {exc}"

    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        return f"HF step skipped (install transformers): {exc}"

    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        return "HF step skipped (AutoTokenizer missing in transformers build)"

    try:
        local_model_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "*.json",
                "vocab.txt",
                "merges.txt",
                "tokenizer.model",
                "special_tokens_map.json",
                "tf_model.h5",
                "model.safetensors",
                "pytorch_model.bin",
            ],
        )
        tokenizer = auto_tokenizer.from_pretrained(local_model_dir)
        tokenized_np = tokenizer(
            [
                "GPU templates are useful for quick setup.",
                "This is a TensorFlow Hugging Face inference check.",
            ],
            return_tensors="np",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: tf.convert_to_tensor(value)
            for key, value in tokenized_np.items()
        }
    except Exception as exc:
        return f"HF step failed (download/tokenization issue): {exc}"

    tf_auto_model = getattr(transformers, "TFAutoModel", None)
    if tf_auto_model is None:
        token_mean = float(tf.reduce_mean(tf.cast(inputs["input_ids"], tf.float32)).numpy())
        return (
            f"HF model '{model_id}' pulled and tokenized into TensorFlow tensors "
            f"(TFAutoModel unavailable in transformers {transformers.__version__}; "
            f"use transformers<5 for TF auto-model inference), "
            f"token_mean={token_mean:.6f}"
        )

    try:
        model = tf_auto_model.from_pretrained(local_model_dir)
        outputs = model(**inputs)
        hidden_mean = float(tf.reduce_mean(outputs.last_hidden_state).numpy())
    except Exception as exc:
        return f"HF model pulled but TF inference failed: {exc}"

    return f"HF model '{model_id}' pulled and inference ok, mean={hidden_mean:.6f}"


def run_tensorflow_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "tensorflow",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

    # Reduce TensorFlow startup log noise unless user explicitly overrides.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    try:
        import tensorflow as tf
    except Exception as exc:
        result["message"] = f"TensorFlow not installed or failed to import: {exc}"
        return result

    result["available"] = True

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        detail_text = " ".join(
            str(value).lower()
            for value in [gpus[0].name, details.get("device_name", "")]
        )

        # TensorFlow on Apple Silicon commonly surfaces METAL as a generic GPU device.
        if "metal" in detail_text or (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ):
            chosen_device = "mps"
        else:
            chosen_device = "cuda"
        with tf.device("/GPU:0"):
            x = tf.random.normal((1024, 1024))
            y = tf.random.normal((1024, 1024))
            z = tf.matmul(x, y)
    else:
        chosen_device = "cpu"
        with tf.device("/CPU:0"):
            x = tf.random.normal((1024, 1024))
            y = tf.random.normal((1024, 1024))
            z = tf.matmul(x, y)

    mean_value = float(tf.reduce_mean(z).numpy())
    hf_message = _run_hf_tensorflow_demo(tf)

    result["device"] = chosen_device
    result["ok"] = True
    result["message"] = f"Matmul successful, mean={mean_value:.6f}; {hf_message}"
    return result


if __name__ == "__main__":
    print(run_tensorflow_demo())
