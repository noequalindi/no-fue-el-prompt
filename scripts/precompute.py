#!/usr/bin/env python3
"""Pre-descarga el modelo pelado (DistilBERT) y precalcula sus embeddings.

Todo esto es forward-pass puro (sin gradientes), así que se puede calcular
una sola vez antes de la charla. En vivo, cargar estos .npz es instantáneo
y no depende de la red.

Los modelos quedan en ./models (lo configura `biaslab/__init__.py`).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# biaslab va antes que transformers: así los modelos se bajan a ./models.
import biaslab  # noqa: E402
from biaslab.data import load_bug, load_winomt  # noqa: E402
from biaslab.features import embed_texts, load_encoder  # noqa: E402
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

# Modelos del Acto 2 usados solo para la demo "pre-entrenado vs. des-sesgado"
# (fill-mask + mini-benchmark): los bajamos acá para no depender de la red
# durante la charla. No hace falta cachear embeddings de estos, se calculan
# rápido en vivo (gold_BUG es chico).
EXTRA_MODELS = [
    "distilbert-base-cased",
    "aieng-lab/distilbert-base-cased-gradiend-gender-debiased",
]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Descargando/cacheando distilbert-base-uncased (modelo pelado, una sola vez)...")
    tokenizer, model = load_encoder()

    for name in EXTRA_MODELS:
        print(f"Descargando/cacheando {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModelForMaskedLM.from_pretrained(name)
        AutoModel.from_pretrained(name)

    datasets = {
        "gold": load_bug(DATA_DIR / "bug" / "gold_BUG.csv"),
        "balanced": load_bug(DATA_DIR / "bug" / "balanced_BUG_sample.csv"),
        "winomt_pro": load_winomt(DATA_DIR / "winomt" / "en_pro.txt", "pro"),
        "winomt_anti": load_winomt(DATA_DIR / "winomt" / "en_anti.txt", "anti"),
    }

    for name, df in datasets.items():
        out_path = CACHE_DIR / f"embeddings_{name}.npz"
        print(f"Embeddings para '{name}' ({len(df)} oraciones)...")
        vectors = embed_texts(df["text"].tolist(), tokenizer, model)
        np.savez_compressed(
            out_path,
            vectors=vectors,
            label=df["label"].to_numpy(),
            gender=df["gender"].to_numpy(),
            profession=df["profession"].to_numpy(),
            stereotype=df["stereotype"].to_numpy(),
        )
        print(f"  guardado en {out_path}")

    print("Listo. Embeddings cacheados en", CACHE_DIR)
    print("Modelos descargados en", biaslab.MODELS_DIR)


if __name__ == "__main__":
    main()
