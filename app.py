from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
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

_MAX_SEQ = int(os.environ.get("MAX_SEQ_LENGTH", "128"))
_pad_mode = os.environ.get("TOKENIZER_PAD_MODE", "max_length").strip().lower()
# max_length = bentuk [1, MAX_SEQ] statis (biasanya lebih ramah ORT). longest = pad ke panjang urutan (lebih sedikit FLOPs teks pendek).
_TOKENIZER_PAD = _pad_mode if _pad_mode in ("max_length", "longest") else "max_length"

if HF_TOKEN is None:
    print(
        "Peringatan: HF_TOKEN kosong — OK untuk model Hugging Face public. "
        "Repo privat wajib set HF_TOKEN di Coolify.",
        flush=True,
    )

print(f"HF_REPO_ID: {HF_REPO_ID}", flush=True)
print(f"Tokenizer: pad={_TOKENIZER_PAD}, max_length={_MAX_SEQ}", flush=True)
print("Loading tokenizer dari Hugging Face…", flush=True)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID, token=HF_TOKEN, use_fast=True
    )
except (ValueError, TypeError, OSError):
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=HF_TOKEN)

_tok_cls = type(tokenizer).__name__
print(f"Tokenizer class: {_tok_cls}", flush=True)

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
    """CPU untuk matmul BERT: default menyamai jumlah core (dibatasi)."""
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

_ORT_INPUT_ORDER = [inp.name for inp in session.get_inputs()]
print(f"Input ONNX (urutan): {_ORT_INPUT_ORDER}", flush=True)


def _feeds_from_enc(enc: dict) -> dict:
    """Tensor sesuai urutan graf; salin hanya jika tidak C-contiguous."""
    out = {}
    for name in _ORT_INPUT_ORDER:
        if name not in enc:
            continue
        x = enc[name]
        out[name] = x if x.flags.c_contiguous else np.ascontiguousarray(x)
    return out


def prediksi(teks: str, profile: bool = False):
    t0_ns = time.perf_counter_ns()
    enc = tokenizer(
        teks,
        return_tensors="np",
        max_length=_MAX_SEQ,
        truncation=True,
        padding=_TOKENIZER_PAD,
    )
    t1_ns = time.perf_counter_ns()
    feeds = _feeds_from_enc(enc)
    if len(feeds) != len(_ORT_INPUT_ORDER):
        missing = set(_ORT_INPUT_ORDER) - set(feeds)
        raise RuntimeError(f"Input ONNX hilang dari tokenizer: {missing}")
    t2_ns = time.perf_counter_ns()
    logits = session.run(None, feeds)[0]
    t3_ns = time.perf_counter_ns()
    logits = logits[0]
    m = float(np.max(logits))
    e = np.exp(logits - m)
    probs = e / e.sum()
    pred = int(np.argmax(probs))
    t4_ns = time.perf_counter_ns()

    def _ms(a: int, b: int) -> float:
        return round((b - a) / 1_000_000, 2)

    elapsed = _ms(t0_ns, t4_ns)
    result = {
        "label": "judi" if pred == 1 else "bukan_judi",
        "label_text": "🎰 Judi Online" if pred == 1 else "✅ Bukan Judi Online",
        "confidence": round(float(probs[pred]) * 100, 2),
        "prob_judi": round(float(probs[1]) * 100, 2),
        "prob_bukan_judi": round(float(probs[0]) * 100, 2),
        "ms": elapsed,
    }
    if profile:
        result["_profile"] = {
            "tokenizer_ms": _ms(t0_ns, t1_ns),
            "feeds_ms": _ms(t1_ns, t2_ns),
            "onnx_ms": _ms(t2_ns, t3_ns),
            "post_ms": _ms(t3_ns, t4_ns),
        }
    return result


try:
    prediksi("warmup satu kali untuk cache allocator")
except Exception as _warm_e:
    print(f"Peringatan warmup: {_warm_e}", flush=True)


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
    prof = request.headers.get("X-Inference-Profile", "").strip() == "1"
    return jsonify(prediksi(teks, profile=prof))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    data = request.get_json()
    if not data or "teks_list" not in data:
        return jsonify({"error": "Field 'teks_list' wajib diisi"}), 400
    teks_list = data["teks_list"]
    if len(teks_list) > 50:
        return jsonify({"error": "Maksimal 50 teks per batch"}), 400
    prof = request.headers.get("X-Inference-Profile", "").strip() == "1"
    hasil = [{"teks": t, **prediksi(t, profile=prof)} for t in teks_list]
    return jsonify({"results": hasil, "total": len(hasil)})


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": HF_REPO_ID,
            "runtime": "onnx",
            "device": "cpu",
            "tokenizer_pad": _TOKENIZER_PAD,
            "max_seq_length": _MAX_SEQ,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
