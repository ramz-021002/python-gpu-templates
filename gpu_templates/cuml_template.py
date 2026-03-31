from __future__ import annotations

from typing import Any, Dict


def run_cuml_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "cuml",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

    try:
        import cupy as cp
        from cuml.linear_model import LinearRegression
    except Exception as exc:
        result["message"] = f"cuML not installed or failed to import: {exc}"
        return result

    result["available"] = True

    try:
        # cuML runs on CUDA devices only.
        x = cp.asarray([[1.0], [2.0], [3.0], [4.0]])
        y = cp.asarray([3.0, 5.0, 7.0, 9.0])
        model = LinearRegression()
        model.fit(x, y)
        pred = model.predict(cp.asarray([[5.0]]))
    except Exception as exc:
        result["message"] = f"cuML runtime error (likely CUDA stack issue): {exc}"
        return result

    result["device"] = "cuda"
    result["ok"] = True
    result["message"] = (
        f"Linear regression successful, pred@5={float(pred.get()[0]):.6f}"
    )
    return result


if __name__ == "__main__":
    print(run_cuml_demo())
