# 🚀 Guía de Instalación con GPU (NVIDIA)

Esta guía explica cómo configurar el proyecto para usar **aceleración GPU** con PaddleOCR en Windows.

## 📋 Requisitos Previos

### Hardware Necesario
- ✅ **GPU NVIDIA** compatible con CUDA (serie GTX/RTX)
- ✅ Mínimo 4GB de VRAM recomendado
- ✅ Driver NVIDIA actualizado

### Software Requerido
- ✅ **CUDA Toolkit 12.x o 11.x** instalado
- ✅ **Python 3.10+**
- ✅ **uv** (gestor de paquetes)

---

## 🔍 Verificar GPU y CUDA

### 1. Verificar GPU NVIDIA
```powershell
nvidia-smi
```

Deberías ver información de tu GPU (nombre, memoria, driver version).

### 2. Verificar CUDA Toolkit
```powershell
nvcc --version
```

Deberías ver la versión de CUDA instalada (ej: CUDA 12.8).

---

## 📦 Instalación del Proyecto

### 1. Clonar e Instalar Dependencias

```powershell
# Clonar el repositorio (si aplica)
cd ruta/al/proyecto

# Instalar todas las dependencias (incluidas las GPU)
uv sync
```

**Nota**: El `pyproject.toml` ya incluye las dependencias GPU correctas:
- `paddlepaddle-gpu==2.6.2` (compilado para CUDA 11)
- `nvidia-cudnn-cu11==8.9.5.29` (cuDNN 8 para CUDA 11)
- `nvidia-cublas-cu11==11.11.3.6` (cuBLAS para CUDA 11)
- `nvidia-cuda-nvrtc-cu11==11.8.89` (CUDA Runtime)

### 2. Activar GPU en Configuración

Edita el archivo `.env`:

```properties
# Usar GPU (true/false)
OCR_USE_GPU=true
```

---

## 🔧 Instalación Manual (si es necesario)

Si necesitas instalar las dependencias GPU manualmente:

```powershell
# Desinstalar versiones CPU/incorrectas (si existen)
uv pip uninstall paddlepaddle
uv pip uninstall nvidia-cudnn-cu12
uv pip uninstall nvidia-cublas-cu12

# Instalar PaddlePaddle GPU
uv pip install paddlepaddle-gpu==2.6.2

# Instalar dependencias GPU para CUDA 11
uv pip install nvidia-cudnn-cu11==8.9.5.29
uv pip install nvidia-cublas-cu11==11.11.3.6
uv pip install nvidia-cuda-nvrtc-cu11==11.8.89
```

---

## ✅ Verificar Instalación GPU

Ejecuta este comando para verificar que PaddleOCR puede usar la GPU:

```powershell
python -c "import paddle; print(f'CUDA disponible: {paddle.is_compiled_with_cuda()}'); print(f'GPU count: {paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0}')"
```

Deberías ver:
```
CUDA disponible: True
GPU count: 1
```

---

## 🎯 Prueba de PaddleOCR con GPU

```powershell
python -c "import sys; sys.path.insert(0, '.'); from utils.pdf_ocr_paddleocr import ExtractorFacturas; ext = ExtractorFacturas(); print('Inicializando...'); result = ext.inicializar_ocr(); print(f'GPU OK: {result}')"
```

Deberías ver:
```
✅ PaddleOCR inicializado correctamente (GPU)
GPU OK: True
```

---

## 🐛 Solución de Problemas

### Problema: "cudnn64_8.dll not found"

**Causa**: cuDNN no está instalado o no está en el PATH.

**Solución**:
```powershell
uv pip install nvidia-cudnn-cu11==8.9.5.29
```

### Problema: "cublas64_11.dll not found"

**Causa**: cuBLAS no está instalado.

**Solución**:
```powershell
uv pip install nvidia-cublas-cu11==11.11.3.6
```

### Problema: GPU no se detecta

**Causa**: Driver NVIDIA desactualizado o CUDA no instalado.

**Solución**:
1. Actualizar driver NVIDIA desde https://www.nvidia.com/drivers
2. Instalar CUDA Toolkit desde https://developer.nvidia.com/cuda-downloads

### Problema: Versiones incompatibles

**Importante**: PaddlePaddle 2.6.2 GPU está compilado para **CUDA 11**, NO CUDA 12.

Por eso usamos:
- ✅ `nvidia-cudnn-cu11` (NO `nvidia-cudnn-cu12`)
- ✅ `nvidia-cublas-cu11` (NO `nvidia-cublas-cu12`)

Aunque tu sistema tenga CUDA 12, las bibliotecas de CUDA 11 son **compatibles hacia adelante**.

---

## 🔄 Cambiar entre GPU y CPU

### Usar GPU (Recomendado si tienes NVIDIA)
En `.env`:
```properties
OCR_USE_GPU=true
```

**Ventajas**:
- ⚡ **3-10x más rápido**
- ✅ Mejor para procesar muchos documentos

### Usar CPU (Fallback)
En `.env`:
```properties
OCR_USE_GPU=false
```

**Cuándo usar CPU**:
- ❌ No tienes GPU NVIDIA
- ❌ Tienes GPU AMD/Intel (no compatible con CUDA)
- ❌ Problemas de compatibilidad

---

## 📊 Comparativa de Rendimiento

| Modo | Tiempo por página | Aceleración |
|------|------------------|-------------|
| CPU  | ~2-5 segundos    | 1x          |
| GPU  | ~0.3-1 segundo   | **3-10x**   |

---

## 🔍 Información Técnica

### Arquitectura de Dependencias GPU

```
PaddleOCR 2.8.1
    ├── PaddlePaddle-GPU 2.6.2 (CUDA 11)
    │   ├── nvidia-cudnn-cu11==8.9.5.29
    │   ├── nvidia-cublas-cu11==11.11.3.6
    │   └── nvidia-cuda-nvrtc-cu11==11.8.89
    └── OpenCV 4.10.0
```

### Parches Implementados

El proyecto incluye **parches automáticos** en el código:

1. **PATH de DLLs**: Agrega automáticamente las rutas de cuDNN y cuBLAS al PATH
2. **Caracteres especiales**: Maneja nombres de usuario con acentos (ej: "Sánche")
3. **Directorio seguro**: Usa `C:\PaddleOCR_Safe` para evitar problemas de permisos

Estos parches están en:
- `utils/pdf_ocr_paddleocr.py`
- `utils/procesar_lote_pdfs.py`

---

## 📝 Comandos Útiles

```powershell
# Ver versión de PaddlePaddle
python -c "import paddle; print(paddle.__version__)"

# Ver versiones de paquetes NVIDIA
uv pip list | Select-String "nvidia"

# Verificar uso de GPU en tiempo real
nvidia-smi -l 1  # Actualiza cada segundo

# Reinstalar todo desde pyproject.toml
uv sync --reinstall
```

---

## 💡 Notas Importantes

1. **Compatible con CUDA 12**: Aunque PaddlePaddle 2.6.2 está compilado para CUDA 11, funciona correctamente con CUDA 12 instalado en el sistema.

2. **Primer inicio lento**: La primera vez que uses PaddleOCR descargará modelos (~20MB). Posteriores ejecuciones serán más rápidas.

3. **VRAM requerida**: PaddleOCR usa ~2-3GB de VRAM. Si tienes poca memoria, considera usar CPU.

4. **Drivers actualizados**: Mantén los drivers NVIDIA actualizados para mejor compatibilidad.

---

## 🆘 Soporte

Si tienes problemas:

1. Verifica que `nvidia-smi` funcione
2. Verifica que `nvcc --version` muestre CUDA
3. Revisa el archivo `.env` tenga `OCR_USE_GPU=true`
4. Intenta con CPU temporalmente (`OCR_USE_GPU=false`)
5. Consulta los logs para más detalles

---

**Última actualización**: Octubre 2025  
