import easyocr
from PIL import Image
import re
import json
import os

# 1. Configuración del Lector OCR (español)
reader = easyocr.Reader(['es'], gpu=False)  # GPU=False para usar solo CPU

def procesar_imagen(ruta_imagen):
    """Extrae texto de imágenes médicas y estructura datos para EPI-12"""
    try:
        # 2. Extracción de texto con EasyOCR
        resultados = reader.readtext(ruta_imagen, detail=0, paragraph=True)
        texto_completo = "\n".join(resultados)
        
        print("\nTexto crudo extraído:")
        print(texto_completo)
        
        # 3. Extracción estructurada de datos médicos
        datos_epi12 = {
            "paciente": extraer_paciente(texto_completo),
            "diagnostico": extraer_diagnostico(texto_completo),
            "fecha": extraer_fecha(texto_completo),
            "medico": extraer_medico(texto_completo),
            "texto_original": texto_completo
        }
        
        return datos_epi12
    
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None

# Funciones de extracción específicas (ajusta según tus formatos)
def extraer_paciente(texto):
    """Extrae datos del paciente usando expresiones regulares"""
    patrones = {
        "nombre": r"(Nombre|Paciente):?\s*([A-Za-zÁÉÍÓÚáéíóúñ\s]+)",
        "cedula": r"(Cédula|CI|Identificación):?\s*([VEve\d-]+)",
        "edad": r"Edad:?\s*(\d+)",
        "sexo": r"Sexo:?\s*([MFmf])"
    }
    
    datos = {}
    for campo, patron in patrones.items():
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            datos[campo] = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
    
    return datos

def extraer_diagnostico(texto):
    """Extrae información de diagnóstico"""
    match = re.search(r"Diagnóstico:?\s*(.*?)(?=\n|$)", texto, re.IGNORECASE)
    return match.group(1).strip() if match else "No especificado"

def extraer_fecha(texto):
    """Extrae fechas en formato común"""
    match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{2,4})", texto)
    return match.group(1) if match else "Fecha no encontrada"

def extraer_medico(texto):
    """Extrae nombre del médico"""
    match = re.search(r"(Médico|Dr|Dra)[.:]?\s*([A-Za-zÁÉÍÓÚáéíóúñ\s]+)", texto, re.IGNORECASE)
    return match.group(2).strip() if match else "No especificado"

# 4. Ejemplo de uso
if __name__ == "__main__":
    # Procesar imagen de ejemplo
    imagen_ejemplo = "nota_medica_ejemplo.jpg"  # Cambia por tu imagen
    
    if os.path.exists(imagen_ejemplo):
        datos = procesar_imagen(imagen_ejemplo)
        print("\nDatos estructurados para EPI-12:")
        print(json.dumps(datos, indent=2, ensure_ascii=False))
        
        # Guardar resultados
        with open('epi12_datos.json', 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
    else:
        print(f"Error: El archivo {imagen_ejemplo} no existe")
        print("Creando imagen de ejemplo...")
        # Puedes agregar aquí código para crear una imagen de prueba
