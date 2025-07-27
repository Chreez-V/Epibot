# Versión optimizada y corregida
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
import time

class EstadisticaEPI12(BaseModel):
    enfermedad: str
    codigo: str
    total_hombres: int = 0
    total_mujeres: int = 0

def main():
    print("Iniciando procesamiento optimizado...")
    start = time.time()
    
    # 1. Carga eficiente del PDF
    loader = PyPDFLoader("PacientesFicticios.pdf")
    pages = loader.load_and_split()
    print(f"PDF cargado en {time.time()-start:.2f}s")

    # 2. Procesamiento directo sin embeddings costosos
    prompt_template = """
    Analiza estos registros médicos y devuelve solo un JSON con conteos por género para cada enfermedad.
    Ejemplo: {"enfermedad": "Diarrea", "codigo": "A09", "hombres": 3, "mujeres": 2}
    
    Registros:
    {texto}
    """
    
    llm = Ollama(model="tinyllama")
    resultados = []
    
    # Procesar en lotes pequeños
    for i, page in enumerate(pages[:3]):  # Solo 3 páginas para prueba
        print(f"Procesando página {i+1}...")
        prompt = prompt_template.format(texto=page.page_content)
        try:
            response = llm.invoke(prompt)
            if "{" in response and "}" in response:
                data = json.loads(response)
                resultados.append(data)
                print(f"Página {i+1} procesada")
        except Exception as e:
            print(f"Error en página {i+1}: {str(e)}")
    
    # Guardar resultados
    with open('resultados_rapidos.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\nProceso completado en {time.time()-start:.2f} segundos")
    print(f"Resultados guardados en 'resultados_rapidos.json'")

if __name__ == "__main__":
    main()
