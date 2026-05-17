"""Gunicorn config untuk judi-detector.

Tujuan: pastikan setiap worker melakukan warmup model sendiri setelah fork,
agar request pertama tidak lambat (dan tidak kena --timeout) karena MKL/MKLDNN
lazy-init di process worker yang baru.
"""

from __future__ import annotations


def post_fork(server, worker):
    """Dijalankan di setiap worker child setelah fork dari master.

    Karena `--preload` memuat model di master, worker mewarisi memori via COW.
    Tetapi beberapa primitif BLAS/MKL melakukan inisialisasi lazy di proses
    tempat mereka pertama kali dipanggil. Kita paksa satu inferensi dummy.
    """
    try:
        # Import lokal: app.py sudah punya `prediksi` siap pakai.
        from app import prediksi

        prediksi("warmup worker post_fork")
        worker.log.info("Warmup worker selesai (pid=%s).", worker.pid)
    except Exception as e:  # noqa: BLE001
        worker.log.warning("Warmup worker gagal (pid=%s): %s", worker.pid, e)
