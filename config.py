"""
Módulo de configuración centralizada
Carga variables de entorno desde .env para toda la aplicación
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
# Buscar .env en la raíz del proyecto
ROOT_DIR = Path(__file__).parent
ENV_PATH = ROOT_DIR / '.env'

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"Configuracion cargada desde: {ENV_PATH}")
else:
    print(f"Archivo .env no encontrado en {ENV_PATH}")
    print("   Usando valores por defecto")


class Config:
    """Clase de configuración centralizada"""
    
    # ============================================
    # RUTAS PRINCIPALES
    # ============================================
    ROOT_DIR = ROOT_DIR
    
    # Rutas de datos
    DATOS_DIR = ROOT_DIR / os.getenv('DATOS_DIR', 'datos')
    DOCUMENTOS_ORIGINAL_DIR = ROOT_DIR / os.getenv('DOCUMENTOS_ORIGINAL_DIR', 'datos/documentos-original')
    DOCUMENTOS_TXT_DIR = ROOT_DIR / os.getenv('DOCUMENTOS_TXT_DIR', 'datos/documentos-txt')
    
    # Rutas de entrenamiento
    ENTRENAMIENTO_DIR = ROOT_DIR / os.getenv('ENTRENAMIENTO_DIR', 'entrenamiento')
    MODELO_DIR = ROOT_DIR / os.getenv('MODELO_DIR', 'entrenamiento/model')
    MODELO_PKL_PATH = ROOT_DIR / os.getenv('MODELO_PKL_PATH', 'entrenamiento/model/ClasificadorDocumentos.pkl')
    MODELO_INFO_PATH = ROOT_DIR / os.getenv('MODELO_INFO_PATH', 'entrenamiento/model/info_modelo.pkl')
    
    # Archivos de log
    LOG_FILE_BATCH = os.getenv('LOG_FILE_BATCH', 'procesador_batch.log')
    
    # ============================================
    # PARÁMETROS DE OCR
    # ============================================
    OCR_LANG = os.getenv('OCR_LANG', 'es')
    OCR_USE_GPU = os.getenv('OCR_USE_GPU', 'false').lower() == 'true'
    OCR_USE_ANGLE_CLS = os.getenv('OCR_USE_ANGLE_CLS', 'true').lower() == 'true'
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.5'))
    OCR_DPI = int(os.getenv('OCR_DPI', '200'))
    OCR_DPI_HIGH_QUALITY = int(os.getenv('OCR_DPI_HIGH_QUALITY', '250'))
    OCR_ROW_TOLERANCE_Y = int(os.getenv('OCR_ROW_TOLERANCE_Y', '30'))
    
    # ============================================
    # DETECCIÓN DE IMÁGENES
    # ============================================
    IMAGE_PIXEL_THRESHOLD = int(os.getenv('IMAGE_PIXEL_THRESHOLD', '200000'))
    TEXT_CHAR_THRESHOLD = int(os.getenv('TEXT_CHAR_THRESHOLD', '100'))
    MAX_PAGES_TO_CHECK = int(os.getenv('MAX_PAGES_TO_CHECK', '3'))
    
    # ============================================
    # PROCESAMIENTO BATCH
    # ============================================
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    
    # ============================================
    # LOGGING
    # ============================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
    
    # ============================================
    # EXTENSIONES
    # ============================================
    EXTENSIONES_PDF = os.getenv('EXTENSIONES_PDF', '.pdf,.PDF').split(',')
    EXTENSIONES_IMAGEN = os.getenv('EXTENSIONES_IMAGEN', '.jpg,.jpeg,.png,.tiff,.bmp').split(',')
    
    @classmethod
    def print_config(cls):
        """Imprime la configuración actual"""
        print("\n" + "="*60)
        print("CONFIGURACION ACTUAL")
        print("="*60)
        print(f"Carpeta raiz: {cls.ROOT_DIR}")
        print(f"Datos: {cls.DATOS_DIR}")
        print(f"PDFs originales: {cls.DOCUMENTOS_ORIGINAL_DIR}")
        print(f"TXTs extraidos: {cls.DOCUMENTOS_TXT_DIR}")
        print(f"Modelo: {cls.MODELO_DIR}")
        print(f"OCR Idioma: {cls.OCR_LANG}")
        print(f"OCR GPU: {cls.OCR_USE_GPU}")
        print(f"DPI: {cls.OCR_DPI} (Alta calidad: {cls.OCR_DPI_HIGH_QUALITY})")
        print(f"Confianza minima OCR: {cls.OCR_CONFIDENCE_THRESHOLD}")
        print(f"Paginas a revisar: {cls.MAX_PAGES_TO_CHECK}")
        print(f"Workers paralelos: {cls.MAX_WORKERS}")
        print(f"Nivel de log: {cls.LOG_LEVEL}")
        print("="*60 + "\n")


# Instancia global de configuración
config = Config()


if __name__ == "__main__":
    # Para probar la configuración
    config.print_config()
