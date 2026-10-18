"""QAT smoke test. Runs in a subprocess with TF_USE_LEGACY_KERAS=1, because importing
tfmot switches tf.keras to tf_keras for the whole process and would leak into the other
(Keras 3) tests.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow_model_optimization") is None
    or importlib.util.find_spec("tf_keras") is None,
    reason="tensorflow-model-optimization / tf_keras not installed",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SMOKE_SCRIPT = r"""
import json, sys
import numpy as np
import tensorflow as tf
import tf_keras
from ecg import evaluate, qat
from ecg.model import build_multiscale_bn_folded_model, build_multiscale_ptq_model

rng = np.random.default_rng(0)
X_spec = rng.random((16, 33, 12, 1), dtype=np.float32)
X_rr = rng.uniform(0.2, 2.0, size=(16, 5)).astype(np.float32)

result = {}
try:
    qat.quantize_aware(build_multiscale_ptq_model((33, 12, 1)))
    result["baseline_quantizes"] = True
except RuntimeError:
    result["baseline_quantizes"] = False

qm = qat.quantize_aware(build_multiscale_bn_folded_model((33, 12, 1), keras_mod=tf_keras))
qm.compile(optimizer="adam", loss="binary_crossentropy")
qm.fit([X_spec, X_rr], np.zeros(16), epochs=1, verbose=0)

tflite_bytes = qat.convert_qat_to_int8_tflite(qm)
path = sys.argv[1]
with open(path, "wb") as f:
    f.write(tflite_bytes)

interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
result["ops"] = [op["op_name"] for op in interpreter._get_ops_details()]
result["io_dtypes"] = [d["dtype"].__name__ for d in interpreter.get_input_details() + interpreter.get_output_details()]
probs = evaluate.predict_tflite(path, X_spec, X_rr)
result["n_probs"] = int(len(probs))
result["probs_in_range"] = bool(((probs >= 0) & (probs <= 1)).all())
print("RESULT " + json.dumps(result))
"""


def test_qat_quantize_convert_and_predict(tmp_path):
    env = dict(os.environ, TF_USE_LEGACY_KERAS="1", PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT, str(tmp_path / "qat.tflite")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    result = json.loads(lines[-1][len("RESULT "):])

    # The baseline (BN after ReLU6 / after Concatenate) is not QAT-compatible.
    assert result["baseline_quantizes"] is False
    # BN folded into the convs, all INT8 I/O, and evaluate.predict_tflite can bind the
    # QAT model's inputs (whose names don't contain rr_input/spectrogram_input).
    assert "MUL" not in result["ops"] and "ADD" not in result["ops"]
    assert "DEQUANTIZE" not in result["ops"]  # no float fallback anywhere in the graph
    assert result["io_dtypes"] == ["int8", "int8", "int8"]
    assert result["n_probs"] == 16 and result["probs_in_range"]
