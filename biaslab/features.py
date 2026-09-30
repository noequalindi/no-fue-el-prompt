"""Embeddings 'congelados' de un encoder pre-entrenado (DistilBERT).

Nada de esto entrena nada: es pura extracción de features con el modelo
tal cual salió del pre-entrenamiento (transfer learning sin fine-tuning).
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"


def load_encoder(model_name: str = MODEL_NAME, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    texts,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 64,
) -> np.ndarray:
    """Un vector por texto: mean pooling de la última capa oculta."""
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        vectors = (summed / counts).cpu().numpy()
        all_vectors.append(vectors)
    return np.concatenate(all_vectors, axis=0)
