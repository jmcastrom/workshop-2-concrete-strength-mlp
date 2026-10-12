# Workshop 2 — Red neuronal profunda (MLP) desde cero en PyTorch: resistencia del concreto

Trabajo académico del curso **ST1613 Applied Machine Learning** (Workshop 2). Se construye, entrena y evalúa una
**red neuronal *fully connected* (MLP) implementada desde cero en PyTorch** para predecir la **resistencia a la
compresión del concreto** (`csMPa`, en MPa) a partir de la dosificación de la mezcla y de su edad. El problema
corresponde a la competencia de Kaggle
[Concrete Strength Regression](https://www.kaggle.com/competitions/concrete-strength-regression).

**Integrantes:** Equipo 3 — Jorge Andrés Durán y Juan Miguel Castro.

**Tecnologías:** Python 3.12 · PyTorch · pandas · NumPy · scikit-learn (preprocesamiento y *baselines*) ·
SciPy · Matplotlib · Jupyter.

## Contenido

- [Estructura del repositorio](#estructura-del-repositorio)
- [Resumen de resultados](#resumen-de-resultados)
- [El problema y los datos](#el-problema-y-los-datos)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Ejecución local (Windows)](#ejecución-local-windows)
- [Ejecución en Google Colab](#ejecución-en-google-colab)
- [Submission de Kaggle](#submission-de-kaggle)
- [Reproducibilidad y limitaciones](#reproducibilidad-y-limitaciones)
- [Uso académico](#uso-académico)

## Estructura del repositorio

```text
workshop-2-concrete-strength-mlp/
├── README.md
├── Taller2_Regresion_Concreto_MLP.ipynb   # notebook principal, ejecutado y con sus salidas
├── submission.csv                         # predicciones del modelo final sobre data/test.csv
├── requirements.txt                       # dependencias con las versiones validadas
├── .gitignore
├── .gitattributes                         # conserva los CSV byte a byte (sin conversión de finales de línea)
└── data/
    ├── train.csv                          # datos etiquetados (originales, sin modificar)
    ├── test.csv                           # datos a predecir (originales, sin modificar)
    └── sample_submission.csv              # plantilla de Kaggle (original, sin modificar)
```

| Archivo | Propósito |
|---|---|
| `Taller2_Regresion_Concreto_MLP.ipynb` | Notebook del taller, **ejecutado** de principio a fin con un kernel nuevo. Conserva sus salidas, tablas y figuras reales. |
| `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` | Datos originales de la competencia, sin modificar. |
| `submission.csv` | Predicciones del modelo final sobre `data/test.csv`. El notebook la genera en la raíz del repositorio. |
| `requirements.txt` | Dependencias con las versiones con las que se validó el notebook. |
| `.gitignore` | Excluye entornos virtuales, cachés, checkpoints, secretos y archivos temporales. |
| `.gitattributes` | Evita que Git convierta los finales de línea (los CSV usan CRLF) para que los datos se publiquen idénticos. |

Los notebooks de referencia de la clase (`01_mnist.ipynb`, `02_asl.ipynb`), que el enunciado propone como punto de
partida, no forman parte de este repositorio: son material del curso y el notebook del taller no los necesita para
ejecutarse.

## Resumen de resultados

| | MSE de validación cruzada (MPa²) | R² |
|---|---|---|
| Predecir la media (*baseline*) | 280.44 | −0.013 |
| Regresión lineal con variables transformadas (*baseline*) | 51.77 | 0.812 |
| **MLP seleccionada** (`log1p(age)` + *weight decay* 10⁻³) | **23.87 ± 0.66** | **0.913** |

Las cifras son medias (± desviación estándar) de una **validación cruzada de 5 folds** dentro de `data/train.csv`.
**No son el puntaje de Kaggle** ni garantizan el desempeño sobre datos nuevos. El puntaje público todavía está
[pendiente](#submission-de-kaggle).

## El problema y los datos

El objetivo es predecir `csMPa` (resistencia a la compresión, en MPa), una variable continua: es un problema de
**regresión**. La métrica oficial de la competencia es el **MSE** (MPa²); menor es mejor.

| Archivo | Dimensiones | Papel |
|---|---|---|
| `data/train.csv` | 721 filas × 10 columnas | Datos etiquetados: predictores + `csMPa`. Se usan para todo el entrenamiento y la validación. |
| `data/test.csv` | 309 filas × 9 columnas | Mismos predictores, sin `csMPa`. Sólo se usa para generar las predicciones finales. |
| `data/sample_submission.csv` | 10 filas × 2 columnas | Plantilla de Kaggle: define las columnas `Row ID,csMPa`. Sus 10 filas tienen identificadores de *train*, así que **no** fija el número de filas de la submission (ver [Submission de Kaggle](#submission-de-kaggle)). |

| Columna | Descripción | Unidad |
|---|---|---|
| `Row ID` | Identificador de la fila (0–720 en train, 721–1029 en test) | — |
| `cement` | Cemento | kg/m³ de mezcla |
| `slag` | Escoria de alto horno | kg/m³ |
| `flyash` | Ceniza volante | kg/m³ |
| `water` | Agua | kg/m³ |
| `superplasticizer` | Superplastificante | kg/m³ |
| `coarseaggregate` | Agregado grueso | kg/m³ |
| `fineaggregate` | Agregado fino | kg/m³ |
| `age` | Edad de la probeta al ensayo | días (1–365) |
| `csMPa` | **Objetivo:** resistencia a la compresión (sólo en train) | MPa |

**`Row ID` no se usa como predictor.** Es un identificador secuencial sin relación con la resistencia (correlación
≈ −0.02 en train); se conserva sólo para identificar las predicciones de test.

Características relevantes encontradas en el EDA:
- Todas las variables son numéricas, sin valores faltantes ni infinitos, y con escalas muy distintas.
- `slag`, `flyash` y `superplasticizer` tienen muchos ceros: ≈ 45 %, 54 % y 35 % de las filas.
- `age` es muy asimétrica (≈ 3.3) y sólo toma 14 valores distintos.
- Hay **16 filas duplicadas exactas** y **7 grupos** con los mismos predictores pero distinta resistencia (hasta
  33 MPa de diferencia).
- Train y test tienen distribuciones parecidas, y ≈ 70 % de las filas de test usa una mezcla que también aparece en
  train.

## Metodología

Todo lo descrito aquí está implementado en
[`Taller2_Regresion_Concreto_MLP.ipynb`](Taller2_Regresion_Concreto_MLP.ipynb).

1. **Análisis exploratorio.** Dimensiones, tipos y estadísticas; faltantes y duplicados; distribuciones del objetivo
   y de los predictores; correlaciones de Pearson y Spearman; dispersión frente al objetivo; relaciones de dominio
   (edad, agua/material cementante, presencia de componentes); comparación train/test con la prueba de
   Kolmogorov–Smirnov.
2. **Limpieza.** Se eliminan **sólo** las filas idénticas en predictores y objetivo (16; quedan 705), antes de
   cualquier partición. Las réplicas con distinta resistencia se conservan como ruido experimental.
3. **Variables derivadas candidatas,** definidas con una única función determinista:
   - `log1p(age)`;
   - material cementante total;
   - relaciones agua/cementante y agua/cemento;
   - indicadores de presencia de componentes.
4. **Preprocesamiento sin fuga de datos.** Un `Pipeline` de scikit-learn aplica las variables derivadas y un
   `StandardScaler`. El objetivo también se estandariza para entrenar y se invierte antes de cualquier métrica. Los
   escaladores se ajustan **sólo** con los datos de entrenamiento de cada partición, y una celda lo verifica
   numéricamente.
5. **PyTorch desde cero.**
   - `Dataset` propio con tensores `float32` y objetivo `(N, 1)`.
   - `DataLoader` con `shuffle=True` en entrenamiento y `shuffle=False` en validación.
   - MLP `nn.Module` con **3 capas ocultas** (64-64-32, ReLU) y **una salida lineal**.
   - Pérdida `nn.MSELoss()` y optimizador **Adam**.
   - Funciones propias `train()` y `validate()`, con `model.eval()` y `torch.no_grad()` en validación.
   - **Early stopping** con restauración de los mejores pesos.
6. **Evaluación.**
   - Una partición 80/20 para el modelo base y sus curvas de aprendizaje.
   - **Validación cruzada K-Fold (K = 5)** para comparar y seleccionar. En cada fold se ajusta todo desde cero, y la
     época de parada se elige con un 15 % interno, de modo que el fold de validación sólo mide.
   - Métricas en MPa: **MSE** (oficial), RMSE, MAE y R².
7. **Baselines:** media, regresión lineal con las 8 variables originales y regresión lineal con las variables
   transformadas.
8. **Experimentos.**
   - **Etapa A, representación (E0–E3):** variables originales → `log1p(age)` → + derivadas → + indicadores.
   - **Etapa B, red y entrenamiento (E4–E9), cambiando un solo factor:** anchura de la red, dropout, *weight decay*,
     learning rate y batch size.
9. **Selección:** menor MSE de validación cruzada, con una **regla de 1 error estándar pareado** que prefiere la opción
   más simple cuando la diferencia está dentro del ruido entre folds.
10. **Modelo final y submission.**
    - Reentrenamiento con todo `data/train.csv` durante un número de épocas derivado de los folds (mediana de los
      pasos de optimización óptimos).
    - Predicción sobre `data/test.csv` con el mismo preprocesamiento.
    - Escritura y validación de `submission.csv` en la raíz del repositorio.

## Resultados

Validación cruzada de 5 folds sobre las 705 filas únicas de `data/train.csv`. El MSE se reporta como media ±
desviación estándar entre folds; RMSE, MAE y R² son medias entre folds.

| Configuración | MSE (MPa²) | RMSE (MPa) | MAE (MPa) | R² |
|---|---|---|---|---|
| B0 · Media | 280.44 ± 35.07 | 16.72 | 13.66 | −0.013 |
| B1 · Regresión lineal (8 originales) | 108.31 ± 9.74 | 10.40 | 8.18 | 0.602 |
| B2 · Regresión lineal (`log1p(age)` + derivadas) | 51.77 ± 5.82 | 7.19 | 5.51 | 0.812 |
| E0 · MLP base (8 originales) | 30.86 ± 3.78 | 5.55 | 4.00 | 0.885 |
| E1 · MLP + `log1p(age)` | 25.03 ± 0.91 | 5.00 | 3.59 | 0.908 |
| E2 · E1 + variables derivadas | 25.82 ± 3.16 | 5.07 | 3.50 | 0.906 |
| E3 · E2 + indicadores de presencia | 25.04 ± 2.82 | 5.00 | 3.40 | 0.909 |
| E4 · E1 + red 32-32-16 | 27.44 ± 5.27 | 5.22 | 3.76 | 0.900 |
| E5 · E1 + red 128-128-64 | 26.17 ± 3.45 | 5.11 | 3.40 | 0.904 |
| E6 · E1 + dropout 0.1 | 25.57 ± 2.31 | 5.05 | 3.58 | 0.907 |
| **E7 · E1 + *weight decay* 10⁻³ (seleccionada)** | **23.87 ± 0.66** | **4.88** | **3.42** | **0.913** |
| E8 · E1 + learning rate 3·10⁻³ | 24.40 ± 1.47 | 4.94 | 3.41 | 0.910 |
| E9 · E1 + batch 64 | 24.38 ± 1.15 | 4.94 | 3.47 | 0.911 |

**Qué funcionó.** Sólo dos decisiones mejoraron por encima del ruido entre folds y en los 5 folds:
- representar la edad como `log1p(age)` (−5.83 MPa² frente a E0);
- un *weight decay* de 10⁻³ (−1.17 MPa² frente a E1), que redujo el sobreajuste observado en las curvas del modelo
  base.

La MLP seleccionada supera a todos los *baselines* en los 5 folds, con un 54 % menos de MSE que la regresión lineal
con las mismas variables transformadas: su ventaja viene de la no linealidad, no sólo del preprocesamiento.

**Resultados desfavorables, reportados tal cual.**
- Las variables derivadas (agua/cementante, cementante total, agua/cemento), aunque son las más correlacionadas con
  la resistencia, **empeoraron ligeramente** la red (+0.79 MPa²). Los indicadores de presencia no aportaron frente a
  E1.
- Ni una red más ancha ni una más estrecha mejoraron, y el dropout no ayudó.
- El error se duplica en las filas con réplicas contradictorias (MAE 6.87 frente a 3.33 MPa): un piso de ruido propio
  de los datos.

**Validación cruzada frente a evaluación externa.** La validación cruzada es una estimación **interna** y algo
optimista, porque la configuración se eligió entre 10 redes con esos mismos folds. La única estimación independiente
es el puntaje de Kaggle sobre `data/test.csv`, que todavía no se ha obtenido.

## Ejecución local (Windows)

Probado en **Windows con Python 3.12.6**, en CPU, desde la raíz del repositorio clonado. En **PowerShell**:

```powershell
git clone https://github.com/jmcastrom/workshop-2-concrete-strength-mlp.git
cd workshop-2-concrete-strength-mlp
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
code .
```

En VS Code (con las extensiones de Python y Jupyter):
1. Abrir `Taller2_Regresion_Concreto_MLP.ipynb`.
2. **Select Kernel → Python Environments → `.venv`**.
3. **Restart → Run All.** El notebook lee los CSV desde `data/` y mide e imprime sus tiempos. En la máquina de
   validación (CPU), cada ejecución completa tardó entre 2 y 4 minutos (116–206 s en cinco ejecuciones).
4. Al terminar, `submission.csv` queda en la **raíz del repositorio**.

Para ejecutarlo sin abrir VS Code (por ejemplo, para comprobar que corre de principio a fin), desde la raíz del
repositorio:

```powershell
python -m pip install nbclient
python -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('Taller2_Regresion_Concreto_MLP.ipynb', as_version=4); NotebookClient(nb, timeout=None, kernel_name='python3').execute(); nbformat.write(nb, 'ejecutado.ipynb'); print('OK')"
```

Este comando guarda la copia ejecutada como `ejecutado.ipynb` (no se versiona; puede borrarse) y regenera
`submission.csv`.

**Linux/macOS (no probado):** los pasos equivalentes son `python3.12 -m venv .venv`, `source .venv/bin/activate` y
`python -m pip install -r requirements.txt`. En Linux, `torch` desde PyPI descarga por defecto una versión con soporte
CUDA; para una instalación sólo CPU puede usarse el índice oficial de PyTorch
(`--index-url https://download.pytorch.org/whl/cpu`).

> Las cifras del notebook se obtuvieron en CPU. En otro hardware (p. ej., GPU) o con otras versiones de las librerías,
> los resultados pueden variar ligeramente.

## Ejecución en Google Colab

La compatibilidad con Colab y Google Drive **está implementada en el código, pero no se ha probado en una sesión real
de Colab**. Lo único probado es la ejecución local en Windows.

1. Abrir el notebook en Colab con
   [este enlace](https://colab.research.google.com/github/jmcastrom/workshop-2-concrete-strength-mlp/blob/main/Taller2_Regresion_Concreto_MLP.ipynb),
   o con **Archivo → Abrir cuaderno → GitHub** y la URL del repositorio. Abierto así, el notebook no trae la carpeta
   `data/`: la celda 1.1 montará Drive y, si no encuentra la carpeta del proyecto, abrirá un selector para subir los
   tres CSV.
2. Alternativa: subir la carpeta completa del repositorio (con `data/`) a *Mi unidad* de Google Drive y abrir el
   notebook desde allí.
3. *(Opcional)* **Entorno de ejecución → Cambiar tipo de entorno → GPU**; el código también funciona en CPU.
4. **Entorno de ejecución → Reiniciar y ejecutar todo.**

La celda **1.1** localiza los CSV sola. Recorre estas carpetas raíz y, en cada una, busca primero `data/` y luego la
propia raíz:
1. Carpeta actual, `workshop-2-concrete-strength-mlp/`, `taller2-machine-learning-main/` y `/content`.
2. Sólo en Colab: monta Google Drive y busca `CARPETA_DRIVE` y cualquier carpeta con uno de esos dos nombres en
   *Mi unidad* o un nivel más abajo.
3. Sólo en Colab: si tampoco los encuentra, abre un selector para subirlos.

| Variable (celda 1.1) | Valor por defecto | Uso |
|---|---|---|
| `CARPETA_DRIVE` | `/content/drive/MyDrive/workshop-2-concrete-strength-mlp` | Única ruta editable: carpeta del proyecto en Drive. Cambiarla si se subió con otro nombre o en otra ubicación. |
| `ARCHIVOS` | `["train.csv", "test.csv", "sample_submission.csv"]` | Archivos que deben estar juntos en una misma carpeta (normalmente `data/`). |
| `NOMBRES_PROYECTO` | `["workshop-2-concrete-strength-mlp", "taller2-machine-learning-main"]` | Nombres de carpeta del proyecto que se buscan automáticamente. |

`USAR_DRIVE` ya no existe en el código: Drive sólo se monta en Colab y sólo si los CSV no se encontraron antes. En
local nunca se intenta montar Drive. `submission.csv` se escribe siempre en la raíz del proyecto (la carpeta que
contiene `data/`), o en `/content` si los CSV se subieron sueltos.

## Submission de Kaggle

- **Archivo:** [`submission.csv`](submission.csv), en la raíz del repositorio, generado por el notebook con el modelo
  seleccionado (E7).
- **Formato:** columnas `Row ID,csMPa`, las de la plantilla, con 309 filas, una por cada fila de `data/test.csv`, en el
  mismo orden (`Row ID` 721–1029). Predicciones en MPa, finitas, entre 4.84 y 73.91.
- **Validación en el notebook:** la celda 10.3 valida la submission en memoria, la escribe, la relee desde el disco y
  comprueba columnas, número de filas, identificadores, orden, valores finitos y que el contenido sea exactamente el
  de las predicciones del modelo.
- **Comprobación rápida desde la terminal** (con el entorno activado, desde la raíz del repositorio):

  ```powershell
  python -c "import pandas as pd, numpy as np; s = pd.read_csv('submission.csv'); t = pd.read_csv('data/test.csv'); print(s.shape, list(s.columns), 'IDs = test:', bool((s['Row ID'].values == t['Row ID'].values).all()), 'finitas:', bool(np.isfinite(s['csMPa']).all()))"
  ```

  Resultado esperado: `(309, 2) ['Row ID', 'csMPa'] IDs = test: True finitas: True`.
- **Formato oficial:** **no se ha confirmado** en la página de Kaggle. La plantilla local sólo tiene 10 filas con
  identificadores de train, así que el número de filas y los identificadores esperados deben comprobarse en la
  competencia antes de enviar. Por eso `FORMATO_KAGGLE_CONFIRMADO = False` en el notebook.
- **Puntaje público:** **pendiente.** El archivo aún no se ha enviado a Kaggle. Este repositorio no contiene ningún
  puntaje ni captura del leaderboard.

**Pasos para enviarla a Kaggle:**
1. Abrir la [competencia](https://www.kaggle.com/competitions/concrete-strength-regression), iniciar sesión y, si se
   pide, aceptar sus reglas.
2. En **Overview → Evaluation** y en **Data**, confirmar que se esperan las columnas `Row ID,csMPa` y 309 filas con
   `Row ID` del 721 al 1029. Si no coincide, no enviar el archivo.
3. Pulsar **Submit Prediction** (o la opción equivalente de la interfaz actual), seleccionar `submission.csv` de la raíz
   del repositorio y enviarlo.
4. En la pestaña **Submissions**, esperar a que termine la evaluación y anotar el **Public Score** (MSE en MPa²).

## Reproducibilidad y limitaciones

**Reproducibilidad.**
- Todas las particiones, inicializaciones y órdenes de *batches* derivan de `SEED = 42`.
- En CPU, con las versiones de `requirements.txt`, varias ejecuciones completas independientes produjeron métricas
  idénticas y el mismo `submission.csv`, byte a byte.
- El notebook incluye un `MODO_RAPIDO` para una prueba de humo de pocos minutos; sus resultados **no** son válidos y,
  en ese modo, el notebook no escribe `submission.csv`.

**Limitaciones.**
- **Muestra pequeña:** 705 filas únicas; cada fold de validación tiene 141.
- **Ruido irreducible:** algunas mezclas idénticas, ensayadas a la misma edad, tienen resistencias muy distintas.
- **Optimismo por selección:** la estimación de validación cruzada es algo optimista, porque se eligió entre 10
  configuraciones con los mismos folds.
- **Early stopping con 85 filas por fold:** la época óptima es ruidosa, y el número de épocas del modelo final es una
  aproximación.
- **Sobreajuste residual:** existe incluso con *weight decay*.
- **Validación por filas:** la validación aleatoria por filas reproduce la relación train/test de Kaggle, pero no mide
  la generalización a dosificaciones totalmente nuevas.
- **Pruebas por plataforma:** sólo se probó en Windows (CPU); Linux, macOS y Colab no se han probado.

## Uso académico

Repositorio de un trabajo académico, publicado para su revisión. Los datos de `data/` provienen de la competencia de
Kaggle indicada arriba, y su uso está sujeto a las reglas de esa competencia. El repositorio no declara una licencia
propia.
