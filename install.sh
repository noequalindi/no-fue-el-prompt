#!/usr/bin/env bash
# Instala todo lo necesario para correr los notebooks de Bias Lab:
# entorno virtual (.venv), dependencias de requirements.txt (incluye Jupyter)
# y un kernel de Jupyter llamado "Bias Lab".
#
# Uso: ./install.sh
# Si tenés varias versiones de Python: PYTHON_BIN=python3.12 ./install.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "No encuentro '$PYTHON_BIN'. Instalá Python 3.9 o superior (https://www.python.org/downloads/)."
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Hace falta Python 3.9 o superior. Tenés: $("$PYTHON_BIN" --version)"
    exit 1
fi

echo "== Bias Lab: instalación de paquetes =="
echo "Usando $("$PYTHON_BIN" --version)"

if [ ! -d ".venv" ]; then
    echo "[1/3] Creando entorno virtual (.venv)..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "[1/3] Entorno virtual ya existe, lo reuso."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] Instalando dependencias + Jupyter (puede tardar varios minutos)..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

echo "[3/3] Registrando el kernel de Jupyter 'Bias Lab'..."
python -m ipykernel install --user --name bias-lab --display-name "Bias Lab" >/dev/null

echo ""
echo "Paquetes instalados. Siguiente paso (necesita internet):"
echo "  ./setup.sh     # baja datos y modelos a esta carpeta"
