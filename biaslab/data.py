"""Carga y preparación de datos para Bias Lab (BUG + WinoMT).

La tarea del taller es: predecir el género de la persona a partir del
contexto de la oración. Para que el modelo no haga trampa, tapamos tanto
la palabra de profesión (que a veces se repite en el texto) como el o los
pronombres (que directamente delatan el género, sería la respuesta escrita
en el enunciado).
"""
import ast
import re
from pathlib import Path

import pandas as pd

MASK_PERSON = "[PERSON]"
MASK_PRONOUN = "[PRONOUN]"

LABEL_MAP = {"male": 0, "female": 1}
LABEL_NAMES = {0: "male", 1: "female"}

_PRONOUN_PATTERN = re.compile(
    r"\b(he|she|him|her|his|hers|himself|herself)\b", re.IGNORECASE
)

# Algunas oraciones de gold_BUG llegaron con mojibake: UTF-8 (p.ej. "•", "“", "—")
# leído como cp1252 en algún paso previo del pipeline original ("â€¢", "â€œ", "â€”").
_MOJIBAKE_FALLBACK = {
    "â€œ": "“", "â€\x9d": "”", "â€": "”",
    "â€™": "’", "â€˜": "‘",
    "â€“": "–", "â€”": "—", "â€¢": "•",
}


def _fix_mojibake(text: str) -> str:
    try:
        text = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for bad, good in _MOJIBAKE_FALLBACK.items():
        text = text.replace(bad, good)
    return text


def _mask_pronouns(text: str) -> str:
    return _PRONOUN_PATTERN.sub(MASK_PRONOUN, text)


def _mask_profession_token(tokens: list[str], profession_idx: int) -> str:
    tokens = list(tokens)
    if 0 <= profession_idx < len(tokens):
        tokens[profession_idx] = MASK_PERSON
    return " ".join(tokens)


def load_bug(csv_path) -> pd.DataFrame:
    """Carga un CSV de BUG (gold_BUG.csv, balanced_BUG.csv o su muestra).

    Devuelve columnas: text (enmascarado), original, label, gender,
    profession, stereotype.
    """
    df = pd.read_csv(csv_path)
    df["gender"] = df["predicted gender"].astype(str).str.lower()
    df = df[df["gender"].isin(LABEL_MAP)].copy()
    df["label"] = df["gender"].map(LABEL_MAP)

    texts = []
    for _, row in df.iterrows():
        tokens = [_fix_mojibake(t) for t in ast.literal_eval(row["tokens"])]
        masked = _mask_profession_token(tokens, int(row["profession_first_index"]))
        texts.append(_mask_pronouns(masked))
    df["text"] = texts
    df["original"] = df["sentence_text"].map(_fix_mojibake)
    # "Doctor" (inicio de oración) y "doctor" (en medio) son la misma profesión
    df["profession"] = df["profession"].astype(str).str.lower()
    df["stereotype"] = df["stereotype"].astype(int)

    keep = ["text", "original", "label", "gender", "profession", "stereotype"]
    return df[keep].reset_index(drop=True)


def load_winomt(txt_path, source_label: str) -> pd.DataFrame:
    """Carga un archivo de WinoMT (en_pro.txt / en_anti.txt).

    Formato de cada línea: gender \\t word_index \\t sentence \\t profession
    `source_label` es "pro" (estereotípico) o "anti" (anti-estereotípico).
    """
    rows = []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            gender, idx, sentence, profession = line.split("\t")
            words = sentence.split(" ")
            idx = int(idx)
            if 0 <= idx < len(words):
                words[idx] = MASK_PERSON
            text = _mask_pronouns(" ".join(words))
            rows.append(
                {
                    "text": text,
                    "original": sentence,
                    "label": LABEL_MAP[gender.lower()],
                    "gender": gender.lower(),
                    "profession": profession.lower(),
                    "stereotype": 1 if source_label == "pro" else -1,
                }
            )
    return pd.DataFrame(rows)
