# LLM from Scratch

Implementasi kecil GPT-style language model dari dasar menggunakan PyTorch. Repository ini memisahkan kode inti dari notebook agar eksperimen bisa dijalankan ulang, dites, dan dikembangkan sebagai project Python.

## Status

Repository sebelumnya berisi notebook eksplorasi. Struktur saat ini menambahkan module Python, test dasar, dependency file, config, dan dokumentasi agar lebih siap untuk portfolio engineering.

## Struktur

```text
.
├── configs/                 # Konfigurasi eksperimen
├── data/
│   └── sample/              # Dataset kecil yang aman untuk demo/test
├── notebooks/               # Notebook eksplorasi asli
├── src/llm_from_scratch/    # Kode reusable
├── tests/                   # Unit tests
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Check

```bash
pytest
python -m llm_from_scratch.generate --prompt "Every effort" --max-new-tokens 8
```

Tokenizer default mencoba memakai GPT-2 BPE dari `tiktoken`. Jika file encoding belum tersedia dan environment tidak punya akses network, kode otomatis memakai byte-level tokenizer lokal agar test dan demo tetap berjalan.

## Training Demo

```bash
python -m llm_from_scratch.train --data-path data/sample/tiny_corpus.txt --epochs 1
```

Training demo memakai konfigurasi sangat kecil agar bisa berjalan cepat di CPU. Untuk eksperimen serius, naikkan ukuran model dan dataset secara bertahap.

## Catatan Dataset

Repository ini masih memiliki file teks Harry Potter dari versi awal. File tersebut berisiko lisensi/copyright dan sebaiknya tidak dipakai sebagai dataset publik portfolio. Gunakan dataset public-domain atau dataset internal yang jelas izinnya. Folder `data/raw/` dan `data/processed/` sudah di-ignore untuk dataset lokal.

## Gap yang Sudah Ditutup

- Struktur project Python ditambahkan.
- Dependency dan config dasar ditambahkan.
- Kode tokenizer, dataset, model, training, dan generation diekstrak ke `src/`.
- Test dasar ditambahkan untuk shape dan alur minimal.
- Notebook dipindahkan ke `notebooks/`.

## Remaining Technical Debt

- Bersihkan output notebook agar diff lebih kecil.
- Hapus atau keluarkan file copyrighted dari Git history sebelum publish.
- Tambah checkpointing, evaluation metrics, dan config YAML parser bila eksperimen membesar.
- Tambah CI untuk menjalankan `pytest`.
