# Importaciones actualizadas y corregidas
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistanceEvalChain
)
from pydantic import BaseModel, Field
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from typing import List
import json

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

# Configuración de embeddings
embeddings = OllamaEmbeddings(
    model="tinyllama",
    base_url='http://localhost:11434'
)

# Crear base de datos vectorial
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./epi12_vector_db"
)

# Definición de esquemas Pydantic actualizados
class RangoEdad(BaseModel):
    hombres: int = Field(default=0)
    mujeres: int = Field(default=0)

class EstadisticaEPI12(BaseModel):
    enfermedad: str
    codigo_cie10: str
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

# Plantilla de prompt mejorada
EPI12_PROMPT_TEMPLATE = """
Eres un sistema especializado en procesar reportes médicos para el formato EPI-12 de Venezuela.
Analiza los siguientes registros de pacientes y genera estadísticas por rangos de edad y género.

Registros:
{context}

Instrucciones:
1. Identifica todos los casos por enfermedad (nombre y código CIE-10)
2. Clasifica cada caso en los rangos de edad del EPI-12
3. Calcula totales por género y generales
4. Devuelve solo un JSON válido con la estructura EPI-12

Formato de salida requerido (JSON):
{{
  "enfermedades": [
    {{
      "enfermedad": "NombreEnfermedad",
      "codigo_cie10": "Código",
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
"""

# Configuración corregida de la cadena RAG
prompt = ChatPromptTemplate.from_template(EPI12_PROMPT_TEMPLATE)
llm = Ollama(model="tinyllama", temperature=0)

# Creamos una cadena personalizada
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

qa_chain = RetrievalQA(
    combine_documents_chain=stuff_chain,
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    output_key="result"
)

def evaluate_embeddings():
    # Configuración correcta del evaluador de distancia
    embedding_distance = EmbeddingDistance.COSINE  # Usamos distancia coseno
    
    # Creación del evaluador
    eval_chain = EmbeddingDistanceEvalChain(
        embeddings=embeddings,
        distance_metric=embedding_distance
    )
    
    # Evaluación correctamente implementada
    resultado = eval_chain.evaluate_strings(
        prediction="Paciente femenino de 5 años con diarrea",
        reference="Diarreas (A08-A09)"
    )
    
    print(f"\nDistancia coseno entre embeddings: {resultado['score']:.4f}")
    # Nota: Menor distancia = más similares (0 idénticos, 1 totalmente diferentes)

# Procesamiento completo corregido
def generar_estadisticas():
    query = "Genera estadísticas EPI-12 para todos los pacientes"
    result = qa_chain({"query": query})
    
    try:
        # Parseamos el resultado a nuestro modelo Pydantic
        datos = json.loads(result["result"])
        estadisticas = ResultadoEPI12(**datos)
        
        with open('estadisticas_epi12.json', 'w') as f:
            # Solución al warning de dict() deprecado
            json.dump(estadisticas.model_dump(), f, indent=2)
        
        print("\nEstadísticas generadas en 'estadisticas_epi12.json'")
        return estadisticas
    except Exception as e:
        print(f"Error al procesar resultados: {e}")
        print(f"Respuesta cruda del modelo: {result['result']}")
        return None

# Ejecución principal
if __name__ == "__main__":
    evaluate_embeddings()
    stats = generar_estadisticas()
    
    if stats:
        print("\nMuestra de estadísticas generadas:")
        for i, enf in enumerate(stats.enfermedades[:3]):
            print(f"\nEnfermedad {i+1}: {enf.enfermedad} ({enf.codigo_cie10})")
            print(f"Total casos: {enf.total_general} (H: {enf.total_hombres}, M: {enf.total_mujeres})")
    
    # Ejemplo de consulta semántica
    similar_docs = vector_db.similarity_search(
        "Pacientes con enfermedades respiratorias", 
        k=2
    )
    print("\nDocumentos similares para 'enfermedades respiratorias':")
    for i, doc in enumerate(similar_docs, 1):
        print(f"\nDocumento {i}:\n{doc.page_content[:200]}...")
