from __future__ import annotations

import platform
from typing import Any, Dict


def run_tensorflow_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "tensorflow",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

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

    result["device"] = chosen_device
    result["ok"] = True
    result["message"] = f"Matmul successful, mean={mean_value:.6f}"
    return result


if __name__ == "__main__":
    print(run_tensorflow_demo())
