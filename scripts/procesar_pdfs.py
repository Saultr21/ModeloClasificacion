"""
Procesador batch de PDFs - Extrae texto de PDFs con enfoque híbrido:
- Páginas con imágenes: OCR con PaddleOCR
- Páginas sin imágenes: Extracción directa de texto embedido
"""

import os
import sys
import time
import cv2
import numpy as np
import fitz
from pathlib import Path

# Cargar configuración desde .env
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def _resolver_ppocr_home() -> Path:
    """Obtiene la ruta del caché de PaddleOCR respetando PPOCR_HOME."""
    env_path = os.getenv("PPOCR_HOME")
    if env_path:
        return Path(env_path)
    return Path.home() / ".paddleocr"

try:
    # AGREGAR cuDNN al PATH antes de importar PaddleOCR (CRÍTICO para GPU)
    import os
    import sys
    
    # Obtener la ruta de site-packages del entorno virtual actual
    dll_paths = []
    for path in sys.path:
        if 'site-packages' in path and os.path.exists(path):
            cudnn_bin_path = os.path.join(path, 'nvidia', 'cudnn', 'bin')
            cublas_bin_path = os.path.join(path, 'nvidia', 'cublas', 'bin')
            cuda_nvrtc_bin_path = os.path.join(path, 'nvidia', 'cuda_nvrtc', 'bin')
            
            # Agregar todas las rutas de DLLs de NVIDIA al PATH
            paths_to_add = []
            if os.path.exists(cudnn_bin_path):
                paths_to_add.append(cudnn_bin_path)
                dll_paths.append(cudnn_bin_path)
            if os.path.exists(cublas_bin_path):
                paths_to_add.append(cublas_bin_path)
                dll_paths.append(cublas_bin_path)
            if os.path.exists(cuda_nvrtc_bin_path):
                paths_to_add.append(cuda_nvrtc_bin_path)
                dll_paths.append(cuda_nvrtc_bin_path)
            
            if paths_to_add:
                current_path = os.environ.get('PATH', '')
                new_path = os.pathsep.join(paths_to_add) + os.pathsep + current_path
                os.environ['PATH'] = new_path
                break

    # En Windows agregar también los directorios directamente al cargador de DLL
    if dll_paths and hasattr(os, 'add_dll_directory'):
        for dll_path in dll_paths:
            try:
                os.add_dll_directory(dll_path)
            except (FileNotFoundError, OSError):
                pass
    
    # Parche para caracteres especiales en nombre de usuario (igual que en pdf_ocr_paddleocr.py)
    import tempfile
    
    # Crear directorio alternativo ANTES de importar PaddleOCR
    safe_paddle_dir = r"C:\PaddleOCR_Safe"
    os.makedirs(safe_paddle_dir, exist_ok=True)
    
    # Monkey patch para forzar directorio alternativo
    original_expanduser = os.path.expanduser
    def patched_expanduser(path):
        if path == '~' or path.startswith('~/'):
            # Reemplazar ~ con directorio seguro en lugar del usuario problemático
            return path.replace('~', safe_paddle_dir)
        return original_expanduser(path)
    
    # Aplicar el parche antes de importar PaddleOCR
    os.path.expanduser = patched_expanduser
    os.expanduser = patched_expanduser
    
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    
    # Verificar disponibilidad de CUDA/GPU
    try:
        import paddle
        if config.OCR_USE_GPU and not paddle.is_compiled_with_cuda():
            print("ADVERTENCIA: GPU solicitada pero CUDA no disponible. Usando CPU.")
            # No modificamos config.OCR_USE_GPU aquí, se manejará en la inicialización
    except Exception as e:
        print(f"No se pudo verificar CUDA: {e}")
    
    # Restaurar función original después de importar
    os.path.expanduser = original_expanduser
    os.expanduser = original_expanduser
    
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Error: PaddleOCR no disponible")

class ProcesadorBatchPDFs:
    def __init__(self):
        self.ocr = None
        self.stats = {
            'total': 0,
            'exitosos': 0,
            'errores': 0,
            'paginas_ocr': 0,
            'paginas_texto': 0
        }
        self._ppocr_cache = _resolver_ppocr_home()
        self._ocr_reintento = False
        
    def inicializar_ocr(self):
        """Inicializa PaddleOCR una sola vez usando configuración de .env con parche para caracteres especiales"""
        if not PADDLEOCR_AVAILABLE:
            return False
        try:
            # Aplicar parche temporal durante la inicialización (igual que en pdf_ocr_paddleocr.py)
            import os
            original_expanduser = os.path.expanduser
            safe_paddle_dir = r"C:\PaddleOCR_Safe"
            
            def patched_expanduser(path):
                if path == '~' or path.startswith('~/'):
                    return path.replace('~', safe_paddle_dir)
                return original_expanduser(path)
            
            # Activar parche
            os.path.expanduser = patched_expanduser
            os.expanduser = patched_expanduser
            
            try:
                self.ocr = PaddleOCR(
                    lang=config.OCR_LANG,
                    use_angle_cls=config.OCR_USE_ANGLE_CLS,
                    use_gpu=config.OCR_USE_GPU,
                    show_log=False
                )
                self._ocr_reintento = False
                print(f"✅ PaddleOCR inicializado correctamente en {safe_paddle_dir}")
                return True
            finally:
                # Restaurar función original
                os.path.expanduser = original_expanduser
                os.expanduser = original_expanduser
                
        except Exception as e:
            print(f"Error inicializando PaddleOCR: {e}")

            mensaje = str(e).lower()
            # Manejar pesos corruptos descargados a medias
            if ("unexpected end of data" in mensaje or "tar file" in mensaje or "bad crc" in mensaje) and not self._ocr_reintento:
                self._ocr_reintento = True
                self._limpiar_cache_ppocr()
                print("Intentando descargar modelos de PaddleOCR nuevamente...")
                return self.inicializar_ocr()
            return False

    def _limpiar_cache_ppocr(self):
        """Elimina el caché de PaddleOCR para forzar una descarga limpia."""
        try:
            if self._ppocr_cache.exists():
                print(f"Eliminando caché corrupto de PaddleOCR en {self._ppocr_cache}")
                import shutil
                shutil.rmtree(self._ppocr_cache)
        except Exception as e:
            print(f"No se pudo limpiar el caché de PaddleOCR: {e}")
    
    def detectar_paginas_con_imagenes(self, pdf_path, umbral_pixels=None, umbral_texto=None):
        """
        Detecta qué páginas necesitan OCR basándose en:
        - Tamaño de imágenes (logos pequeños vs documentos escaneados)
        - Cantidad de texto extraíble directamente
        """
        # Usar valores de configuración si no se especifican
        umbral_pixels = umbral_pixels or config.IMAGE_PIXEL_THRESHOLD
        umbral_texto = umbral_texto or config.TEXT_CHAR_THRESHOLD
        
        doc = fitz.open(pdf_path)
        paginas_info = []
        
        for i, page in enumerate(doc):
            imgs = page.get_images(full=True)
            texto = page.get_text()
            texto_len = len(texto.strip())
            
            necesita_ocr = False
            
            if imgs:
                for img in imgs:
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        width = base_image["width"]
                        height = base_image["height"]
                        size_pixels = width * height
                        
                        if size_pixels > umbral_pixels:
                            necesita_ocr = True
                            break
                    except:
                        pass
            
            if not necesita_ocr and texto_len < umbral_texto:
                necesita_ocr = True
            
            paginas_info.append({
                'num': i,
                'necesita_ocr': necesita_ocr,
                'num_imagenes': len(imgs),
                'caracteres_texto': texto_len
            })
        
        doc.close()
        return paginas_info
    
    def extraer_texto_embedido(self, pdf_path, num_pagina):
        """Extrae texto embedido directamente del PDF"""
        doc = fitz.open(pdf_path)
        page = doc[num_pagina]
        texto = page.get_text()
        doc.close()
        return texto.strip()
    
    def convertir_pagina_a_imagen(self, pdf_path, num_pagina, dpi=None):
        """Convierte una página de PDF a imagen para OCR usando DPI de configuración"""
        dpi = dpi or config.OCR_DPI_HIGH_QUALITY
        doc = fitz.open(pdf_path)
        page = doc[num_pagina]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img_data = pix.tobytes("ppm")
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
        doc.close()
        
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def mejorar_imagen(self, image):
        """Preprocesa imagen para mejor OCR - configuración optimizada"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def extraer_texto_ocr(self, pdf_path, num_pagina):
        """Extrae texto de una página usando OCR"""
        if not self.ocr:
            if not self.inicializar_ocr():
                return ""
        
        try:
            img = self.convertir_pagina_a_imagen(pdf_path, num_pagina)
            img_mejorada = self.mejorar_imagen(img)
            
            temp_path = f"temp_ocr_{num_pagina}.jpg"
            cv2.imwrite(temp_path, img_mejorada)
            
            result = self.ocr.ocr(temp_path)
            
            os.remove(temp_path)
            
            if not result or not result[0]:
                return ""
            
            elementos = []
            for detection in result[0]:
                text_info = detection[1]
                text = text_info[0]
                confidence = text_info[1]
                
                # Usar umbral de confianza de configuración
                if confidence > config.OCR_CONFIDENCE_THRESHOLD:
                    bbox = detection[0]
                    bbox_np = np.array(bbox)
                    elementos.append({
                        'texto': text,
                        'x': np.mean(bbox_np[:, 0]),
                        'y': np.mean(bbox_np[:, 1])
                    })
            
            elementos.sort(key=lambda x: (x['y'], x['x']))
            
            filas = []
            # Usar tolerancia de configuración
            tolerancia_y = config.OCR_ROW_TOLERANCE_Y
            
            for elemento in elementos:
                fila_encontrada = False
                for fila in filas:
                    if abs(elemento['y'] - fila[0]['y']) <= tolerancia_y:
                        fila.append(elemento)
                        fila_encontrada = True
                        break
                if not fila_encontrada:
                    filas.append([elemento])
            
            for fila in filas:
                fila.sort(key=lambda x: x['x'])
            
            texto_final = []
            for fila in filas:
                contenido_fila = [elem['texto'].strip() for elem in fila if elem.get('texto')]
                contenido_fila = [c for c in contenido_fila if c]
                if contenido_fila:
                    texto_final.append(' | '.join(contenido_fila))
            
            return '\n'.join(texto_final)
            
        except Exception as e:
            print(f"Error en OCR página {num_pagina}: {e}")
            return ""
    
    def procesar_pdf(self, pdf_path):
        """Procesa un PDF completo con estrategia híbrida"""
        try:
            print(f"Procesando: {pdf_path}")
            
            paginas_info = self.detectar_paginas_con_imagenes(pdf_path)
            
            print(f"Total páginas detectadas: {len(paginas_info)}")
            
            texto_completo = []
            
            for info in paginas_info:
                num_pag = info['num']
                
                if info['necesita_ocr']:
                    self.stats['paginas_ocr'] += 1
                    print(f"Página {num_pag}: OCR (img={info['num_imagenes']}, chars={info['caracteres_texto']})")
                    texto = self.extraer_texto_ocr(pdf_path, num_pag)
                else:
                    self.stats['paginas_texto'] += 1
                    print(f"Página {num_pag}: Texto ({info['caracteres_texto']} chars)")
                    texto = self.extraer_texto_embedido(pdf_path, num_pag)
                
                texto_completo.append(f"PÁGINA {num_pag}\n{texto if texto else '(página vacía)'}\n")
            
            resultado = '\n'.join(texto_completo)
            
            output_path = os.path.splitext(pdf_path)[0] + '.txt'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(resultado)
            
            self.stats['exitosos'] += 1
            print(f"  OK: {output_path}")
            
        except Exception as e:
            self.stats['errores'] += 1
            print(f"  Error: {e}")
    
    def procesar_directorio(self, directorio_base):
        """Procesa recursivamente todos los PDFs en el directorio"""
        print(f"Escaneando directorio: {directorio_base}")
        
        pdfs = []
        for root, dirs, files in os.walk(directorio_base):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdfs.append(pdf_path)
        
        self.stats['total'] = len(pdfs)
        print(f"Total PDFs encontrados: {self.stats['total']}")
        
        inicio = time.time()
        
        for i, pdf_path in enumerate(pdfs, 1):
            print(f"\n[{i}/{self.stats['total']}] ", end='')
            self.procesar_pdf(pdf_path)
        
        tiempo_total = time.time() - inicio
        
        print("\n" + "="*60)
        print("RESUMEN")
        print("="*60)
        print(f"Total procesados: {self.stats['total']}")
        print(f"Exitosos: {self.stats['exitosos']}")
        print(f"Errores: {self.stats['errores']}")
        print(f"Páginas con OCR: {self.stats['paginas_ocr']}")
        print(f"Páginas con texto: {self.stats['paginas_texto']}")
        print(f"Tiempo total: {tiempo_total:.2f}s")
        print(f"Tiempo promedio: {tiempo_total/self.stats['total']:.2f}s por PDF")

def main():
    """
    Procesa todos los PDFs en la carpeta de documentos originales.
    Los TXTs se guardan en la misma ubicación que los PDFs.
    """
    directorio_base = str(config.DOCUMENTOS_ORIGINAL_DIR)
    
    if not os.path.exists(directorio_base):
        print(f"Error: El directorio {directorio_base} no existe.")
        print(f"Crea la estructura de carpetas con tus PDFs organizados por clase:")
        print(f"  {directorio_base}/")
        print(f"    ├── clase1/")
        print(f"    │   ├── documento1.pdf")
        print(f"    │   └── documento2.pdf")
        print(f"    ├── clase2/")
        print(f"    └── ...")
        return
    
    procesador = ProcesadorBatchPDFs()
    
    print("Procesando PDFs del directorio de documentos originales")
    print(f"Ruta: {directorio_base}\n")
    
    inicio = time.time()
    
    # Buscar todos los PDFs recursivamente
    todos_pdfs = []
    for root, dirs, files in os.walk(directorio_base):
        for file in files:
            if file.lower().endswith('.pdf'):
                todos_pdfs.append(os.path.join(root, file))
    
    if not todos_pdfs:
        print(f"Error: No se encontraron PDFs en {directorio_base}")
        return
    
    procesador.stats['total'] = len(todos_pdfs)
    print(f"Total de PDFs encontrados: {procesador.stats['total']}\n")
    
    for i, pdf_path in enumerate(todos_pdfs, 1):
        print(f"\n[{i}/{procesador.stats['total']}] ", end='')
        procesador.procesar_pdf(pdf_path)
    
    tiempo_total = time.time() - inicio
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"Total procesados: {procesador.stats['total']}")
    print(f"Exitosos: {procesador.stats['exitosos']}")
    print(f"Errores: {procesador.stats['errores']}")
    print(f"Páginas con OCR: {procesador.stats['paginas_ocr']}")
    print(f"Páginas con texto: {procesador.stats['paginas_texto']}")
    print(f"Tiempo total: {tiempo_total:.2f}s")
    if procesador.stats['total'] > 0:
        print(f"Tiempo promedio: {tiempo_total/procesador.stats['total']:.2f}s por PDF")

if __name__ == "__main__":
    main()
