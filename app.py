from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
import os
import time

app = Flask(__name__)

# ── Config ──
HF_REPO_ID = os.environ.get("HF_REPO_ID", "verisimb/testindobertjudol").strip()
_raw_token = os.environ.get("HF_TOKEN", "").strip()
HF_TOKEN = _raw_token or None  # None = akses anonim (repo public)

if HF_TOKEN is None:
    print(
        "Peringatan: HF_TOKEN kosong — OK untuk model Hugging Face public. "
        "Repo privat wajib set HF_TOKEN di Coolify.",
        flush=True,
    )

print(f"HF_REPO_ID: {HF_REPO_ID}", flush=True)
print("Loading tokenizer dari Hugging Face…", flush=True)

try:
    tokenizer = BertTokenizer.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
except Exception as e:
    import traceback
    print(f"❌ Gagal memuat tokenizer: {e}", flush=True)
    traceback.print_exc()
    raise

print("Mengunduh / memuat ONNX model…", flush=True)
try:
    onnx_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="model.onnx",
        token=HF_TOKEN,
    )
except Exception as e:
    import traceback
    print(f"❌ Gagal mengunduh model.onnx: {e}", flush=True)
    traceback.print_exc()
    raise

print(f"Loading ONNX model dari: {onnx_path}", flush=True)


def _default_intra_threads() -> int:
    """CPU untuk matmul BERT: default menyamai jumlah core (dibatasi) — bukan 2 seperti sebelumnya."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    return max(1, min(n, 8))


_intra = int(os.environ.get("ONNX_NUM_THREADS", str(_default_intra_threads())))
_inter = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
print(
    f"ONNX Runtime threads: intra_op={_intra}, inter_op={_inter} "
    f"(override via ONNX_NUM_THREADS / ONNX_INTER_OP_THREADS)",
    flush=True,
)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = _intra
sess_options.inter_op_num_threads = _inter
# Graf BERT kebanyakan berantai; paralelisme utama dari intra_op (MatMul), inter_op=1 mengurangi overhead.
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True

try:
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
except Exception as e:
    import traceback
    print(f"❌ Gagal memuat sesi ONNX: {e}", flush=True)
    traceback.print_exc()
    raise

print("✅ ONNX model berhasil dimuat!", flush=True)


def prediksi(teks: str):
    start = time.time()
    enc = tokenizer(
        teks,
        return_tensors="np",
        max_length=128,
        truncation=True,
        padding=True,
    )
    inputs = {
        k: v
        for k, v in enc.items()
        if k in ("input_ids", "attention_mask", "token_type_ids")
    }
    logits = session.run(None, inputs)[0]
    # Softmax numerik stabil
    logits = logits[0]
    m = np.max(logits)
    e = np.exp(logits - m)
    probs = e / e.sum()
    pred = int(np.argmax(probs))
    elapsed = round((time.time() - start) * 1000, 1)
    return {
        "label": "judi" if pred == 1 else "bukan_judi",
        "label_text": "🎰 Judi Online" if pred == 1 else "✅ Bukan Judi Online",
        "confidence": round(float(probs[pred]) * 100, 2),
        "prob_judi": round(float(probs[1]) * 100, 2),
        "prob_bukan_judi": round(float(probs[0]) * 100, 2),
        "ms": elapsed,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prediksi", methods=["POST"])
def api_prediksi():
    data = request.get_json()
    if not data or "teks" not in data:
        return jsonify({"error": "Field 'teks' wajib diisi"}), 400
    teks = data["teks"].strip()
    if not teks:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400
    if len(teks) > 1000:
        return jsonify({"error": "Teks maksimal 1000 karakter"}), 400
    return jsonify(prediksi(teks))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    data = request.get_json()
    if not data or "teks_list" not in data:
        return jsonify({"error": "Field 'teks_list' wajib diisi"}), 400
    teks_list = data["teks_list"]
    if len(teks_list) > 50:
        return jsonify({"error": "Maksimal 50 teks per batch"}), 400
    hasil = [{"teks": t, **prediksi(t)} for t in teks_list]
    return jsonify({"results": hasil, "total": len(hasil)})


@app.route("/api/health")
def health():
    return jsonify(
        {"status": "ok", "model": HF_REPO_ID, "runtime": "onnx", "device": "cpu"}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
