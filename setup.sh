#!/usr/bin/env bash
# Setup completo de Bias Lab. Corré esto ANTES de la charla (necesita internet).
# Instala los paquetes (install.sh), baja los datasets a ./data, los modelos
# a ./models y precalcula los embeddings en ./cache.
# Uso: ./setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

./install.sh

# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "== Bias Lab: datos y modelos =="

echo "[1/2] Descargando datasets (BUG + WinoMT) a ./data..."
python scripts/download_data.py

echo "[2/2] Descargando los modelos pre-entrenados a ./models (distilbert-base-uncased"
echo "      y las variantes vanilla/des-sesgada) y precalculando embeddings en ./cache,"
echo "      para que la charla no dependa de la red..."
python scripts/precompute.py

echo ""
echo "Listo. Para arrancar el notebook:"
echo "  source .venv/bin/activate"
echo "  jupyter notebook notebooks/bias_lab.ipynb"
