from __future__ import annotations

from typing import Any, Dict


def run_cupy_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "cupy",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

    try:
        import cupy as cp
    except Exception as exc:
        result["message"] = f"CuPy not installed or failed to import: {exc}"
        return result

    result["available"] = True
    
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        result["message"] = f"CUDA runtime unavailable for CuPy: {exc}"
        return result

    if device_count <= 0:
        result["message"] = "No CUDA GPU visible to CuPy"
        return result

    x = cp.random.randn(1024, 1024)
    y = cp.random.randn(1024, 1024)
    z = x @ y

    result["device"] = "cuda"
    result["ok"] = True
    result["message"] = f"Matmul successful, mean={float(z.mean().get()):.6f}"
    return result


if __name__ == "__main__":
    print(run_cupy_demo())
