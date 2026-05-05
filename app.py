from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import os
import time

app = Flask(__name__)

# ── Config ──
# Ganti dengan repo ID HuggingFace kamu, contoh: "verysimbolon/indobert-judionline"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "username/indobert-judionline")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")   # HuggingFace token (wajib untuk repo privat)
DEVICE     = torch.device("cpu")               # Oracle ARM CPU

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN tidak ditemukan! Set environment variable HF_TOKEN.")

print(f"Loading model dari HuggingFace: {HF_REPO_ID}")
print(f"Device: {DEVICE}")

tokenizer = BertTokenizer.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
model     = BertForSequenceClassification.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
model     = model.to(DEVICE)
model.eval()

print("✅ Model berhasil dimuat!")

def prediksi(teks: str):
    start = time.time()
    enc = tokenizer(
        teks,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
    elapsed = round((time.time() - start) * 1000, 1)
    pred = int(np.argmax(probs))
    return {
        "label": "judi" if pred == 1 else "bukan_judi",
        "label_text": "🎰 Judi Online" if pred == 1 else "✅ Bukan Judi Online",
        "confidence": round(float(probs[pred]) * 100, 2),
        "prob_judi": round(float(probs[1]) * 100, 2),
        "prob_bukan_judi": round(float(probs[0]) * 100, 2),
        "ms": elapsed
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
    hasil = prediksi(teks)
    return jsonify(hasil)

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
    return jsonify({"status": "ok", "model": HF_REPO_ID, "device": str(DEVICE)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
