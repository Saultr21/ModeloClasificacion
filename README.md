# Sistema de Clasificación de Documentos con ML

Sistema completo para procesar PDFs con OCR, extraer texto y entrenar modelos de clasificación de documentos usando Machine Learning. Ideal como plantilla para proyectos similares.

## 📁 Estructura del Proyecto

```
Modelo/
├── config.py                    # Configuración centralizada
├── .env                         # Variables de entorno
├── pyproject.toml              # Dependencias (gestión con uv)
├── README.md                    # Este archivo
│
├── scripts/                     # Scripts ejecutables
│   ├── procesar_pdfs.py        # Extrae texto de PDFs (híbrido: OCR + texto embedido)
│   └── mover_txts.py           # Mueve TXTs a carpeta organizada
│
├── entrenamiento/              # Entrenamiento del modelo ML
│   ├── entrenar_modelo.ipynb   # Notebook de entrenamiento (TF-IDF + SVM)
│   └── model/                  # Modelos entrenados (generado automáticamente)
│       ├── ClasificadorDocumentos.pkl
│       └── info_modelo.pkl
│
└── datos/                      # Datos del proyecto
    ├── documentos-original/    # PDFs organizados por clase
    │   ├── clase1/
    │   │   ├── doc1.pdf
    │   │   └── doc2.pdf
    │   ├── clase2/
    │   └── ...
    │
    └── documentos-txt/         # TXTs extraídos organizados por clase
        ├── clase1/
        │   ├── doc1.txt
        │   └── doc2.txt
        └── clase2/
```

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

Este proyecto usa [uv](https://github.com/astral-sh/uv) para gestión de dependencias:

```powershell
# Instalar uv (si no lo tienes)
pip install uv

# Instalar dependencias del proyecto
uv sync

# Para incluir dependencias de desarrollo (Jupyter)
uv sync --extra dev
```

### 2. Configurar el Entorno

El archivo `.env` contiene todas las configuraciones. Las más importantes:

```env
# Carpetas principales
DOCUMENTOS_ORIGINAL_DIR=datos/documentos-original
DOCUMENTOS_TXT_DIR=datos/documentos-txt

# OCR
OCR_USE_GPU=true          # true para GPU, false para CPU
OCR_LANG=es               # Idioma del OCR
OCR_DPI=200               # Calidad de OCR (mayor = mejor calidad, más lento)

# Procesamiento
MAX_WORKERS=4             # Hilos paralelos para procesar PDFs
```

### 3. Preparar los Datos

Organiza tus PDFs por clase en `datos/documentos-original/`:

```
datos/documentos-original/
├── facturas/
│   ├── factura001.pdf
│   └── factura002.pdf
├── contratos/
│   ├── contrato001.pdf
│   └── contrato002.pdf
└── recibos/
    └── recibo001.pdf
```

### 4. Procesar PDFs → Extraer Texto

```powershell
uv run python scripts/procesar_pdfs.py
```

Esto extrae el texto de cada PDF usando:
- **Extracción directa** para páginas con texto embedido
- **OCR (PaddleOCR)** para documentos escaneados o imágenes

Los archivos `.txt` se guardan junto a los PDFs originales.

### 5. Mover TXTs a Carpeta Organizada

```powershell
uv run python scripts/mover_txts.py
```

Mueve los `.txt` a `datos/documentos-txt/` manteniendo la estructura de clases.

### 6. Entrenar el Modelo

Abre el notebook de entrenamiento:

```powershell
uv run jupyter notebook entrenamiento/entrenar_modelo.ipynb
```

O con Jupyter Lab:

```powershell
uv run jupyter lab
```

**El notebook incluye:**
- ✅ Carga automática de datos desde `datos/documentos-txt/`
- ✅ División en train/validation/test con estratificación
- ✅ Entrenamiento con TF-IDF + SVM
- ✅ Optimización de hiperparámetros con GridSearchCV
- ✅ Visualización de métricas y matriz de confusión
- ✅ Guardado automático del modelo en `entrenamiento/model/`
- ✅ Predicción interactiva con nuevos documentos

## 🔧 Configuración Avanzada

### Parámetros de OCR

Edita `.env` para ajustar el comportamiento del OCR:

```env
# Confianza mínima para aceptar texto OCR (0.0 - 1.0)
OCR_CONFIDENCE_THRESHOLD=0.5

# DPI para OCR de alta calidad
OCR_DPI_HIGH_QUALITY=250

# Umbral para detectar imágenes que requieren OCR (píxeles)
IMAGE_PIXEL_THRESHOLD=200000

# Caracteres mínimos para considerar texto embedido
TEXT_CHAR_THRESHOLD=100
```

### Limpieza de Texto

El sistema elimina automáticamente firmas digitales y códigos de verificación:

```env
LIMPIAR_FIRMAS_DIGITALES=true
LIMPIAR_CODIGOS_VERIFICACION=true
```

## 📊 Flujo de Trabajo Completo

```
1. Organizar PDFs por clase
   └─> datos/documentos-original/clase1/*.pdf

2. Extraer texto
   └─> python scripts/procesar_pdfs.py
       └─> Genera *.txt junto a cada PDF

3. Mover TXTs
   └─> python scripts/mover_txts.py
       └─> datos/documentos-txt/clase1/*.txt

4. Entrenar modelo
   └─> Ejecutar notebook: entrenar_modelo.ipynb
       └─> Genera modelo en entrenamiento/model/

5. Predecir nuevos documentos
   └─> Usar última celda del notebook
```

## 🎯 Usar Como Plantilla

Para adaptar este proyecto a un nuevo conjunto de datos:

1. **Limpiar datos anteriores:**
   ```powershell
   Remove-Item -Recurse datos/documentos-original/*
   Remove-Item -Recurse datos/documentos-txt/*
   Remove-Item -Recurse entrenamiento/model/*
   ```

2. **Agregar nuevos PDFs** en `datos/documentos-original/`, organizados por clase

3. **Seguir el flujo de trabajo** desde el paso 2

4. **Ajustar configuración** en `.env` si es necesario (OCR, DPI, etc.)

## 🛠️ Comandos Útiles

```powershell
# Ver configuración actual
uv run python config.py

# Instalar solo dependencias base (sin dev)
uv sync --no-dev

# Actualizar dependencias
uv sync --upgrade

# Ejecutar scripts
uv run python scripts/procesar_pdfs.py
uv run python scripts/mover_txts.py

# Jupyter
uv run jupyter notebook
uv run jupyter lab
```

## 📝 Notas Técnicas

### OCR con PaddleOCR

- **GPU**: Requiere CUDA 11.x instalado
- **CPU**: Funciona sin CUDA pero más lento
- **Idiomas**: Configurable en `.env` (`OCR_LANG=es`)

### Procesamiento Híbrido

El script `procesar_pdfs.py` detecta automáticamente:
- Páginas escaneadas → usa OCR
- Páginas con texto embedido → extracción directa
- Ahorra tiempo procesando solo lo necesario

### Modelo ML

- **Algoritmo**: TF-IDF + SVM lineal
- **Optimización**: GridSearchCV con validación cruzada
- **Métricas**: Accuracy, F1-macro, matriz de confusión
- **Formato**: Guardado con `joblib` (`.pkl`)

## ⚠️ Solución de Problemas

### Error: "No se encontraron PDFs"

Verifica que los PDFs estén en `datos/documentos-original/clase1/`, no en la raíz.

### Error: "Modelo no existe"

Ejecuta el notebook de entrenamiento completo antes de intentar predicciones.

### OCR muy lento

- Reduce `OCR_DPI` en `.env` (ej: 150)
- Reduce `MAX_WORKERS` si tienes poca RAM

### Error de CUDA/GPU

Si PaddleOCR da error con GPU:
```env
OCR_USE_GPU=false
```

## 📦 Dependencias Principales

- **PaddleOCR**: OCR con deep learning
- **scikit-learn**: Modelo de clasificación (TF-IDF + SVM)
- **PyMuPDF**: Procesamiento de PDFs
- **OpenCV**: Procesamiento de imágenes
- **pandas**: Manipulación de datos
- **joblib**: Persistencia del modelo

