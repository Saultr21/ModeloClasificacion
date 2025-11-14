"""
Script para mover archivos TXT desde la carpeta de documentos originales
a la carpeta de documentos-txt, manteniendo la estructura de clases.

Uso:
    python scripts/mover_txts.py
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def mover_txts(origen=None, destino=None, verbose=True):
    """Mueve archivos TXT manteniendo estructura de carpetas"""
    src = Path(origen) if origen else config.DOCUMENTOS_ORIGINAL_DIR
    dst = Path(destino) if destino else config.DOCUMENTOS_TXT_DIR
    
    src = src.resolve()
    dst = dst.resolve()
    
    if not src.exists():
        print(f"Error: {src} no existe")
        return {'error': 'origen_no_existe'}
    
    dst.mkdir(parents=True, exist_ok=True)
    
    stats = {'movidos': 0, 'omitidos': 0, 'errores': 0}
    
    if verbose:
        print("=" * 60)
        print("MOVER ARCHIVOS TXT")
        print(f"Origen:  {src}")
        print(f"Destino: {dst}")
        print("=" * 60)
    
    for root, _, files in os.walk(src):
        for filename in files:
            if not filename.lower().endswith('.txt'):
                continue
            
            archivo_origen = Path(root) / filename
            ruta_relativa = Path(root).relative_to(src)
            
            if ruta_relativa == Path('.'):
                if verbose:
                    print(f"Omitido: {filename}")
                stats['omitidos'] += 1
                continue
            
            clase = ruta_relativa.parts[0]
            carpeta_destino = dst / clase
            carpeta_destino.mkdir(parents=True, exist_ok=True)
            
            archivo_destino = carpeta_destino / filename
            
            # Si el archivo ya existe, agregar sufijo numérico
            if archivo_destino.exists():
                base = archivo_destino.stem
                ext = archivo_destino.suffix
                contador = 1
                while archivo_destino.exists():
                    archivo_destino = carpeta_destino / f"{base}_{contador}{ext}"
                    contador += 1
            
            try:
                shutil.move(str(archivo_origen), str(archivo_destino))
                if verbose:
                    print(f"Movido: {clase}/{filename}")
                stats['movidos'] += 1
            except Exception as e:
                if verbose:
                    print(f"Error: {filename} - {e}")
                stats['errores'] += 1
    
    if verbose:
        print("=" * 60)
        print(f"Archivos movidos: {stats['movidos']}")
        print(f"Archivos omitidos: {stats['omitidos']}")
        print(f"Errores: {stats['errores']}")
        print("=" * 60)
    
    return stats


def main():
    origen = sys.argv[1] if len(sys.argv) > 1 else None
    destino = sys.argv[2] if len(sys.argv) > 2 else None
    mover_txts(origen, destino)


if __name__ == "__main__":
    main()
