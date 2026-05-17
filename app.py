from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import os
import time

app = Flask(__name__)

# ── Config ──
HF_REPO_ID = os.environ.get("HF_REPO_ID", "verisimb/indobert2").strip()
_raw_token = os.environ.get("HF_TOKEN", "").strip()
HF_TOKEN = _raw_token or None  # None = akses anonim (repo public)

if HF_TOKEN is None:
    print(
        "Peringatan: HF_TOKEN kosong — OK untuk model Hugging Face public. "
        "Repo privat wajib set HF_TOKEN di Coolify.",
        flush=True,
    )

print(f"HF_REPO_ID: {HF_REPO_ID}", flush=True)


# ── Threading PyTorch (CPU) ──
def _default_intra_threads() -> int:
    """CPU untuk matmul BERT: default menyamai jumlah core (dibatasi)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    return max(1, min(n, 8))


_intra = int(os.environ.get("TORCH_NUM_THREADS", str(_default_intra_threads())))
_inter = int(os.environ.get("TORCH_INTER_OP_THREADS", "1"))
torch.set_num_threads(_intra)
try:
    torch.set_num_interop_threads(_inter)
except RuntimeError:
    # set_num_interop_threads hanya boleh dipanggil sebelum operasi paralel pertama
    pass
print(
    f"PyTorch threads: intra_op={_intra}, inter_op={_inter} "
    f"(override via TORCH_NUM_THREADS / TORCH_INTER_OP_THREADS)",
    flush=True,
)


# ── Load tokenizer & model ──
print("Loading tokenizer dari Hugging Face…", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
    print(f"Tokenizer: {type(tokenizer).__name__}", flush=True)
except Exception as e:
    import traceback
    print(f"❌ Gagal memuat tokenizer: {e}", flush=True)
    traceback.print_exc()
    raise

print("Loading model PyTorch dari Hugging Face…", flush=True)
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID,
        token=HF_TOKEN,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optional: dynamic INT8 quantization untuk hemat RAM ~4x (FP32 → INT8)
    # Hanya CPU. Aktifkan dengan TORCH_QUANTIZE=1 di env.
    if device.type == "cpu" and os.environ.get("TORCH_QUANTIZE", "0") == "1":
        print("Menerapkan dynamic INT8 quantization (Linear)…", flush=True)
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        print("Quantization selesai.", flush=True)

    print(f"Model: {type(model).__name__} | device: {device}", flush=True)
except Exception as e:
    import traceback
    print(f"❌ Gagal memuat model: {e}", flush=True)
    traceback.print_exc()
    raise

print("Model berhasil dimuat!", flush=True)


@torch.inference_mode()
def prediksi(teks: str):
    start = time.time()
    enc = tokenizer(
        teks,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred = int(np.argmax(probs))
    elapsed = round((time.time() - start) * 1000, 1)
    return {
        "label": "judi" if pred == 1 else "bukan_judi",
        "label_text": "Judi Online" if pred == 1 else "Bukan Judi Online",
        "confidence": round(float(probs[pred]) * 100, 2),
        "prob_judi": round(float(probs[1]) * 100, 2),
        "prob_bukan_judi": round(float(probs[0]) * 100, 2),
        "ms": elapsed,
    }


# Warmup biar request pertama tidak lambat (alokasi tensor + JIT).
# Catatan: jangan jalankan di master saat --preload + fork() di Linux,
# karena OpenMP/MKL threadpool ter-init sebelum fork → worker deadlock.
# Sekarang gunicorn jalan tanpa --preload, jadi import-time warmup aman:
# app.py diimpor di setiap worker, OpenMP init di proses worker itu sendiri.
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
        {
            "status": "ok",
            "model": HF_REPO_ID,
            "runtime": "pytorch",
            "device": str(device),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
