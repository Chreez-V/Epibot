# Importaciones
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import time
import getpass
import shutil

# Configurar API Key de Gemini
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Inicializar modelos de Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Carga de PDF
loader = PyPDFLoader("PacientesFicticios.pdf")
pages = loader.load()

# División de texto en chunks optimizados
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
    separators=["\nnombre paciente:", "\nApellido paciente:", "\nCédula paciente:", "\n\n"]
)
chunks = text_splitter.split_documents(pages)

# Eliminar directorio de persistencia existente si existe
persist_directory = "./epi12_vector_db_new"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Crear base de datos vectorial
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Modelos Pydantic
class RangoEdad(BaseModel):
    hombres: int = Field(default=0)
    mujeres: int = Field(default=0)

class EstadisticaEPI12(BaseModel):
    enfermedad: str  # Aquí incluiremos el código CIE-10 como parte del texto si es necesario
    menores_1_ano: RangoEdad
    entre_1_4_anos: RangoEdad
    entre_5_6_anos: RangoEdad
    entre_7_9_anos: RangoEdad
    entre_10_14_anos: RangoEdad
    entre_15_19_anos: RangoEdad
    entre_20_24_anos: RangoEdad
    entre_25_44_anos: RangoEdad
    entre_45_59_anos: RangoEdad
    entre_60_64_anos: RangoEdad
    mayores_65_anos: RangoEdad
    total_hombres: int
    total_mujeres: int
    total_general: int

class ResultadoEPI12(BaseModel):
    enfermedades: List[EstadisticaEPI12]

# Plantilla de prompt mejorada (sin el campo codigo_cie10)
EPI12_PROMPT_TEMPLATE = """
Eres un asistente médico especializado en estadísticas. Analiza estos registros y devuelve SOLO un JSON válido:

{{
  "enfermedades": [
    {{
      "enfermedad": "Nombre (puede incluir código CIE-10 si es relevante)",
      "menores_1_ano": {{ "hombres": 0, "mujeres": 0 }},
      "entre_1_4_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_5_6_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_7_9_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_10_14_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_15_19_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_20_24_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_25_44_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_45_59_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_60_64_anos": {{ "hombres": 0, "mujeres": 0 }},
      "mayores_65_anos": {{ "hombres": 0, "mujeres": 0 }},
      "total_hombres": 0,
      "total_mujeres": 0,
      "total_general": 0
    }}
  ]
}}

Registros:
{context}
"""

# Configuración de la cadena
prompt = ChatPromptTemplate.from_template(EPI12_PROMPT_TEMPLATE)

def procesar_con_gemini(context: str) -> Optional[Dict[str, Any]]:
    """Envía un prompt a Gemini y procesa la respuesta"""
    response = None
    try:
        response = llm.invoke(prompt.format(context=context))
        
        if not response or not hasattr(response, 'content'):
            print("Error: Respuesta inválida de Gemini")
            return None
            
        # Limpieza de la respuesta
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()  # Remover markdown code block
        
        data = json.loads(content)
        
        # Limpieza adicional: eliminar campos codigo_cie10 si existen
        if "enfermedades" in data:
            for enfermedad in data["enfermedades"]:
                if "codigo_cie10" in enfermedad:
                    # Opcional: puedes agregar el código al nombre de la enfermedad
                    if enfermedad["codigo_cie10"]:
                        enfermedad["enfermedad"] = f"{enfermedad['enfermedad']} ({enfermedad['codigo_cie10']})"
                    del enfermedad["codigo_cie10"]
        
        return data
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        print(f"Respuesta cruda: {response.content if response else 'No response'}")
        return None
    except Exception as e:
        print(f"Error al procesar respuesta: {e}")
        print(f"Respuesta cruda: {response.content if response else 'No response'}")
        return None

def generar_estadisticas():
    print("\nIniciando procesamiento con Gemini...")
    start_time = time.time()
    
    resultados = []
    batch_size = 2  # Más pequeño para evitar timeouts
    delay = 1  # Delay entre requests
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        context = "\n---\n".join(doc.page_content for doc in batch)
        
        print(f"\nProcesando lote {(i//batch_size)+1}/{(len(chunks)//batch_size)+1}")
        data = procesar_con_gemini(context)
        
        if data and "enfermedades" in data:
            # Filtrar enfermedades vacías o con nombres None
            enfermedades_validas = [e for e in data["enfermedades"] if e.get("enfermedad")]
            resultados.extend(enfermedades_validas)
            print(f"✅ Lote completado - {len(enfermedades_validas)} enfermedades")
        else:
            print("❌ Error en lote, continuando...")
        
        time.sleep(delay)  # Respeta rate limits
    
    if resultados:
        try:
            estadisticas = ResultadoEPI12(enfermedades=resultados)
            with open('estadisticas_epi12.json', 'w') as f:
                json.dump(estadisticas.model_dump(), f, indent=2)
            
            print(f"\n✅ Proceso completado en {time.time()-start_time:.2f} segundos")
            print("Resultados guardados en 'estadisticas_epi12.json'")
            return estadisticas
        except Exception as e:
            print(f"Error al validar los resultados: {e}")
            return None
    
    print("\n❌ No se generaron resultados válidos")
    return None

if __name__ == "__main__":
    # Verificar conexión con Gemini primero
    try:
        test_response = llm.invoke("Hola")
        if not test_response or not test_response.content:
            raise ConnectionError("No se recibió respuesta de Gemini")
        print("✅ Conexión con Gemini establecida correctamente")
    except Exception as e:
        print(f"Error de conexión con Gemini: {e}")
        exit(1)
    
    stats = generar_estadisticas()
    
    if stats and stats.enfermedades:
        print("\nMuestra de resultados:")
        for i, enf in enumerate(stats.enfermedades[:3]):
            print(f"\nEnfermedad {i+1}: {enf.enfermedad}")
            print(f"Total casos: {enf.total_general} (H: {enf.total_hombres}, M: {enf.total_mujeres})")
    
    # Ejemplo de búsqueda semántica
    if len(chunks) > 0:
        similar_docs = vector_db.similarity_search("Pacientes con fiebre", k=1)
        print("\nDocumento más relevante para 'Pacientes con fiebre':")
        print(similar_docs[0].page_content[:200] + "...")
