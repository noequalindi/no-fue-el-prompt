# No fue el Prompt — Bias Lab

Mini-proyecto hands-on para una charla sobre sesgos en IA: *"Data Science &
benchmarks para dummies"*. Arranca con análisis de datos, sigue con cómo un
modelo de lenguaje convierte texto en vectores y termina comparando un modelo
pre-entrenado contra uno des-sesgado, con benchmarks cortados por género y
estereotipo.

La pregunta que guía todo el taller: **¿puede un modelo aprender que ciertas
profesiones "tienen género"?**

Las slides de la charla están en
[`Intro a la IA - Data Science.pdf`](Intro%20a%20la%20IA%20-%20Data%20Science.pdf).

## Requisitos

- **Python 3.9 o superior** ([python.org/downloads](https://www.python.org/downloads/)).
  Para ver tu versión: `python3 --version`.
- **~3 GB libres**: ~1.5 GB de dependencias (sobre todo PyTorch) y ~800 MB de
  modelos.
- **Internet**, solo para la instalación. Después todo corre offline.
- macOS o Linux. En Windows, ver [Instalación en Windows](#instalación-en-windows).

## Instalación

Primero clonar el repo:

```bash
git clone <url-del-repo>
cd no_fue_el_prompt
```

Después, un solo comando instala y descarga todo:

```bash
./setup.sh
```

`setup.sh` hace, en orden:

1. **Instala los paquetes** (llama a `install.sh`):
   - crea un entorno virtual en `.venv/`;
   - instala todo lo de `requirements.txt`: pandas, scikit-learn, matplotlib,
     PyTorch, transformers, **Jupyter** e ipywidgets;
   - registra un kernel de Jupyter llamado **"Bias Lab"**.
2. **Descarga los datasets** BUG y WinoMT a `data/`.
3. **Descarga los modelos** a `models/` (dentro del proyecto, no en
   `~/.cache`):
   - `distilbert-base-uncased`, el "modelo pelado";
   - `distilbert-base-cased`, el vanilla;
   - `aieng-lab/distilbert-base-cased-gradiend-gender-debiased`, el des-sesgado.
4. **Precalcula los embeddings** de todos los datasets en `cache/`.

Tarda varios minutos, según la conexión. Si se corta, se puede volver a
correr: reusa lo que ya descargó.

Si solo querés los paquetes, sin datos ni modelos:

```bash
./install.sh
```

Si tenés varias versiones de Python, podés elegir cuál usar:

```bash
PYTHON_BIN=python3.12 ./setup.sh
```

### Instalación en Windows

Los scripts `.sh` no corren en PowerShell. Tenés dos opciones: usar WSL, o
hacer los mismos pasos a mano desde la carpeta del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name bias-lab --display-name "Bias Lab"
python scripts\download_data.py
python scripts\precompute.py
```

## Abrir el notebook

```bash
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
jupyter notebook notebooks/bias_lab.ipynb
```

Si usás **VS Code**, abrí el notebook y elegí como kernel `.venv` o
**"Bias Lab"** (arriba a la derecha, "Select Kernel").

El notebook tiene 3 partes y un cierre:

| Parte | Pregunta |
|---|---|
| 1. Datos | ¿Qué realidad le estamos mostrando al modelo? (EDA) |
| 2. El modelo por dentro | ¿Cómo convierte DistilBERT una oración en vectores de 768 números? Tokens, capas, atención. Encoder (entiende/clasifica) vs. generativo |
| 3. Tres benchmarks | Vanilla vs. des-sesgado: fill-mask con 3 oraciones, clasificador (regresión logística) sobre embeddings, fill-mask sobre el 87% de `gold` |
| Cierre | ¿Y si reentrenáramos las últimas capas? (solo se explica, no se ejecuta) |

Lo único que se entrena es el Benchmark 2: una regresión logística de 769
pesos sobre los vectores. Los encoders no se tocan.

La tarea de clasificación es **predecir el género de la persona a partir del
contexto**. Se tapan tanto la profesión como los pronombres, porque si no el
modelo "haría trampa" leyendo la respuesta en la propia oración.

`notebooks/bias_lab_con_entrenamiento.ipynb` es la versión extendida para el
workshop: suma TF-IDF, fine-tuning, sesgo deliberado y validación con WinoMT.

## Dónde quedan los modelos

Todos los modelos de Hugging Face se guardan en `models/`, dentro del
proyecto. Esto lo configura [`biaslab/__init__.py`](biaslab/__init__.py), que
define `HF_HOME` y `HF_HUB_CACHE` apenas se importa `biaslab`. Así los alumnos
no llenan su `~/.cache` y, para borrar todo, alcanza con borrar la carpeta del
proyecto.

Para que funcione, **hay que importar `biaslab` antes que `transformers` o
`huggingface_hub`**. Los notebooks y los scripts ya lo hacen en su primera
celda o línea. Si agregás un notebook o script nuevo, respetá ese orden.

## Estructura

```
Intro a la IA - Data Science.pdf   slides de la charla
biaslab/                  librería chica compartida por los scripts y el notebook
  __init__.py               apunta la caché de Hugging Face a ./models
  data.py                   carga y enmascarado de BUG y WinoMT
  features.py               embeddings congelados con DistilBERT
  evaluation.py             benchmark "sliced" (por género/estereotipo/profesión)
scripts/
  download_data.py          baja BUG y WinoMT a ./data
  precompute.py             baja los modelos a ./models y cachea embeddings en ./cache
notebooks/
  bias_lab.ipynb            el notebook de la charla
  bias_lab_con_entrenamiento.ipynb   versión extendida (workshop)
slides/                   material de apoyo de la intro (pptx, docx, html)
install.sh                instala los paquetes (venv + requirements + kernel)
setup.sh                  install.sh + datos + modelos + embeddings
requirements.txt          dependencias de Python
```

`data/`, `models/`, `cache/` y `.venv/` no se versionan (ver `.gitignore`):
los genera `setup.sh`.

## Problemas comunes

- **`permission denied: ./setup.sh`**: ejecutá `chmod +x setup.sh install.sh`.
- **`No encuentro la carpeta 'biaslab'`** en el notebook: abrí Jupyter desde la
  carpeta del proyecto, o desde `notebooks/`.
- **`ModuleNotFoundError`** en el notebook: el kernel no es el del proyecto.
  Elegí "Bias Lab" o `.venv` como kernel.
- **El notebook intenta bajar algo de internet**: no se corrió `setup.sh`, o
  se cortó a la mitad. Volvé a correrlo.

## Créditos

- **BUG**: Levy et al., *"Collecting a Large-Scale Gender Bias Dataset for
  Coreference Resolution and Machine Translation"*: [SLAB-NLP/BUG](https://github.com/SLAB-NLP/BUG)
- **WinoMT**: Stanovsky et al., *"Evaluating Gender Bias in Machine
  Translation"*: [gabrielStanovsky/mt_gender](https://github.com/gabrielStanovsky/mt_gender)
- **Modelo des-sesgado**: [aieng-lab/distilbert-base-cased-gradiend-gender-debiased](https://huggingface.co/aieng-lab/distilbert-base-cased-gradiend-gender-debiased)
