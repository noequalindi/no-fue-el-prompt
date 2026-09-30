#!/usr/bin/env python3
"""Descarga y prepara BUG y WinoMT en ./data. Se corre una sola vez (setup.sh).

Requiere conexión a internet. Por eso este script se corre ANTES de la
charla, no durante.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

BUG_URL = "https://github.com/SLAB-NLP/BUG/raw/main/data.tar.gz"
WINOMT_BASE = "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender/master/data/aggregates"

# Tamaño de la muestra de balanced_BUG que usamos para entrenar en vivo.
# balanced_BUG.csv tiene 25504 filas; con 6000 alcanza para un benchmark
# sólido y entra cómodo en el tiempo de la charla.
SAMPLE_SIZE = 6000
SAMPLE_SEED = 42


def download(url: str, timeout: int = 30) -> bytes:
    print(f"  descargando {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def fetch_bug(bug_dir: Path) -> None:
    raw = download(BUG_URL)
    wanted = {"gold_BUG.csv", "balanced_BUG.csv"}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name in wanted:
                member.name = name
                tar.extract(member, path=bug_dir)

    balanced_path = bug_dir / "balanced_BUG.csv"
    full = pd.read_csv(balanced_path)
    sample = full.sample(n=min(SAMPLE_SIZE, len(full)), random_state=SAMPLE_SEED)
    sample = sample.reset_index(drop=True)
    sample.to_csv(bug_dir / "balanced_BUG_sample.csv", index=False)
    print(f"  balanced_BUG_sample.csv: {len(sample)} filas (de {len(full)})")


def fetch_winomt(winomt_dir: Path) -> None:
    for split in ("en_pro.txt", "en_anti.txt"):
        content = download(f"{WINOMT_BASE}/{split}")
        (winomt_dir / split).write_bytes(content)


def main() -> None:
    bug_dir = DATA_DIR / "bug"
    winomt_dir = DATA_DIR / "winomt"
    bug_dir.mkdir(parents=True, exist_ok=True)
    winomt_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] BUG dataset (SLAB-NLP/BUG)")
    fetch_bug(bug_dir)

    print("[2/2] WinoMT / mt_gender (gabrielStanovsky/mt_gender)")
    fetch_winomt(winomt_dir)

    print("Listo. Datos en:", DATA_DIR)


if __name__ == "__main__":
    main()
