"""Bias Lab.

Al importar este paquete, los modelos de Hugging Face se descargan y se leen
desde ./models (dentro del proyecto) en vez de ~/.cache/huggingface. Por eso
hay que importar `biaslab` ANTES que `transformers` / `huggingface_hub`.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR / "hub")
