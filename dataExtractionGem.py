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

# Agrega al inicio del archivo
from typing import Optional
from langchain_community.vectorstores import Chroma

# Reemplaza la variable global por una clase administradora
class VectorDBManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._vector_db = None
            cls._instance._current_file = None
        return cls._instance
    
    @property
    def vector_db(self) -> Optional[Chroma]:
        return self._vector_db
    
    @vector_db.setter
    def vector_db(self, db: Chroma):
        self._vector_db = db
    
    @property
    def current_file(self) -> Optional[str]:
        return self._current_file
    
    @current_file.setter
    def current_file(self, file_path: str):
        self._current_file = file_path

# Inicializa el manager global
vector_db_manager = VectorDBManager()

# Configurar API Key de Gemini
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Inicializar modelos de Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Modelos Pydantic
class RangoEdad(BaseModel):
    hombres: int = Field(default=0)
    mujeres: int = Field(default=0)

class EstadisticaEPI12(BaseModel):
    orden: int
    enfermedad: str
    menores_1_ano: RangoEdad
    entre_1_4_anos: RangoEdad
    entre_5_6_anos: RangoEdad
    entre_7_9_anos: RangoEdad
    entre_10_11_anos: RangoEdad
    entre_12_14_anos: RangoEdad
    entre_15_19_anos: RangoEdad
    entre_20_24_anos: RangoEdad
    entre_25_44_anos: RangoEdad
    entre_45_59_anos: RangoEdad
    entre_60_64_anos: RangoEdad
    mayores_65_anos: RangoEdad
    edad_ignorada: RangoEdad
    total_hombres: int
    total_mujeres: int
    total_general: int

class ResultadoEPI12(BaseModel):
    enfermedades: List[EstadisticaEPI12]

# Lista de las 52 enfermedades oficiales del EPI-12
ENFERMEDADES_OFICIALES = [
    "Cólera (A00)",
    "Diarreas (A08-A09)",
    "Amibiasis (A06)",
    "Fiebre Tifoidea (A01.0)",
    "ETA Nº de Brotes",
    "Casos Asociados a Brotes de ETA",
    "Hepatitis Aguda Tipo A (B15)",
    "Tuberculosis (A15-A19)",
    "Influenza (J10-J11) Enfermedad Tipo Influenza",
    "Sífilis Congénita (A50)",
    "Infección Asintomática VIH (Z21)",
    "Enfermedad VIH/SIDA (B20-B24)",
    "Tosferina (A37)",
    "Parotiditis (B26)",
    "Tétanos Neonatal (A33)",
    "Tétanos Obstétrico (A34)",
    "Tétanos (otros) (A35)",
    "Difteria (A36)",
    "Sarampión Sospecha (B05)",
    "Rubéola (B06)",
    "Dengue sin signos de alarma (A90)",
    "Dengue con signos de alarma (A90)",
    "Dengue grave (A91)",
    "Chikungunya (A92.0)",
    "Zika (U06)",
    "Encefalitis Equina Venezolana (A92.2)",
    "Fiebre Amarilla (A95)",
    "Leishmaniasis Visceral (B55.0)",
    "Leishmaniasis Cutánea (B55.1)",
    "Leishmaniasis Mucocutánea (B55.2)",
    "Leishmaniasis no Específica (B55.9)",
    "Enfermedad de Chagas Aguda (B57.0-B57.1)",
    "Enfermedad de Chagas Crónica (B57.2-B57.5)",
    "Rabia Humana (A82)",
    "Fiebre Hemorrágica Venezolana (A96.8)",
    "Leptospirosis (A27)",
    "Meningitis Viral (A87)",
    "Meningitis Bacteriana (G00)",
    "Meningitis Meningocóccica (A39.0)",
    "Enfermedad Meningococcica (A39.9)",
    "Varicela (B01)",
    "Hepatitis Aguda Tipo B (B16)",
    "Hepatitis Aguda Tipo C (B17.1, B18.2)",
    "Hepatitis Otras Agudas (B17)",
    "Hepatitis No Específicas (B19)",
    "Parálisis Flácida (G82.0)",
    "Neumonías (J12-J18)",
    "Intoxicación por Plaguicidas (T60)",
    "Mordedura Sospechosa de Rabia (A82)",
    "Fiebre (R50)",
    "Efectos Adversos de Medicamentos (Y40-Y57)",
    "Efectos Adversos de Vacunas (Y58-Y59)"
]

# Plantilla de prompt actualizada
EPI12_PROMPT_TEMPLATE = """Eres un asistente médico especializado en estadísticas epidemiológicas del Ministerio de Salud de Venezuela.

Analiza este documento médico y extrae información para clasificar casos por enfermedad, edad y sexo.

RANGOS DE EDAD EXACTOS:
- menores_1_ano: menores de 1 año
- entre_1_4_anos: 1 a 4 años
- entre_5_6_anos: 5 a 6 años
- entre_7_9_anos: 7 a 9 años
- entre_10_11_anos: 10 a 11 años
- entre_12_14_anos: 12 a 14 años
- entre_15_19_anos: 15 a 19 años
- entre_20_24_anos: 20 a 24 años
- entre_25_44_anos: 25 a 44 años
- entre_45_59_anos: 45 a 59 años
- entre_60_64_anos: 60 a 64 años
- mayores_65_anos: 65 años y más
- edad_ignorada: edad no especificada

IMPORTANTE: Solo incluye enfermedades que encuentres en el documento. No inventes datos.

Devuelve SOLO un JSON válido:
{{
  "enfermedades": [
    {{
      "orden": 1,
      "enfermedad": "Nombre exacto con código CIE-10 si está disponible",
      "menores_1_ano": {{ "hombres": 0, "mujeres": 0 }},
      "entre_1_4_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_5_6_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_7_9_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_10_11_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_12_14_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_15_19_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_20_24_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_25_44_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_45_59_anos": {{ "hombres": 0, "mujeres": 0 }},
      "entre_60_64_anos": {{ "hombres": 0, "mujeres": 0 }},
      "mayores_65_anos": {{ "hombres": 0, "mujeres": 0 }},
      "edad_ignorada": {{ "hombres": 0, "mujeres": 0 }},
      "total_hombres": 0,
      "total_mujeres": 0,
      "total_general": 0
    }}
  ]
}}

Documento médico:
{context}"""

# Configuración de la cadena
prompt = ChatPromptTemplate.from_template(EPI12_PROMPT_TEMPLATE)

def procesar_con_gemini(context: str) -> Optional[Dict[str, Any]]:
    """Envía un prompt a Gemini y procesa la respuesta"""
    try:
        print(f"\n📤 Enviando a Gemini el siguiente contexto:\n{context[:500]}...")  # Log parcial del contexto
        
        response = llm.invoke(prompt.format(context=context))
        
        if not response or not hasattr(response, 'content'):
            print("Error: Respuesta inválida de Gemini")
            return None
        
        # Limpieza de la respuesta
        content = response.content.strip()
        print(f"\n📥 Respuesta cruda de Gemini:\n{content}")  # Log de la respuesta completa
        
        # Extraer JSON de diferentes formatos de respuesta
        json_str = None
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content
        
        # Intentar parsear el JSON
        try:
            data = json.loads(json_str)
            print("✅ JSON parseado correctamente")
            return data
        except json.JSONDecodeError as e:
            print(f"❌ Error al decodificar JSON. Contenido:\n{json_str}")
            return None
            
    except Exception as e:
        print(f"❌ Error en procesar_con_gemini: {str(e)}")
        return None

def calcular_totales(enfermedad_data):
    """Calcula los totales de una enfermedad"""
    total_hombres = 0
    total_mujeres = 0
    
    rangos = ["menores_1_ano", "entre_1_4_anos", "entre_5_6_anos", "entre_7_9_anos",
              "entre_10_11_anos", "entre_12_14_anos", "entre_15_19_anos", "entre_20_24_anos",
              "entre_25_44_anos", "entre_45_59_anos", "entre_60_64_anos", "mayores_65_anos", "edad_ignorada"]
    
    for rango in rangos:
        if rango in enfermedad_data:
            total_hombres += enfermedad_data[rango].get("hombres", 0)
            total_mujeres += enfermedad_data[rango].get("mujeres", 0)
    
    enfermedad_data["total_hombres"] = total_hombres
    enfermedad_data["total_mujeres"] = total_mujeres
    enfermedad_data["total_general"] = total_hombres + total_mujeres
    
    return enfermedad_data

def generar_estadisticas_individual(file_path: str) -> Optional[ResultadoEPI12]:
    """Procesa UN archivo PDF individual y genera estadísticas EPI-12"""
    print(f"\nProcesando archivo: {file_path}")
    start_time = time.time()
    
    try:
        # Cargar PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if not pages:
            print("❌ No se pudo cargar el PDF")
            return None
        
        # División de texto en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        
        print(f"✅ PDF cargado - {len(chunks)} chunks generados")
        
       # Configurar directorio de persistencia
        persist_directory = os.path.join(os.getcwd(), "chroma_db_epi12")
        
        # Verificar y limpiar directorio
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
                print("♻️ Base de datos existente eliminada")
            except Exception as e:
                print(f"⚠️ Error eliminando base de datos existente: {e}")
                # Intentar continuar con directorio existente
        
        # Asegurar que el directorio existe
        os.makedirs(persist_directory, exist_ok=True)
        
        # Crear base de datos vectorial con manejo de errores
        try:
            #configurar chroma
            global vector_db
            vector_db_manager.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            
            print("✅ Base de datos vectorial creada")
        except Exception as e:
            print(f"❌ Error creando base de datos vectorial: {e}")
            # Intentar con Chroma en memoria como fallback
            try:
                print("⚠️ Intentando con Chroma en memoria...")
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                print("✅ Chroma en memoria creado como fallback")
            except Exception as e:
                print(f"❌ Error crítico: No se pudo crear Chroma: {e}")
                return None 

        # Procesar en lotes pequeños
        resultados = []
        batch_size = 3
        delay = 1
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            context = "\n---\n".join(doc.page_content for doc in batch)
            
            print(f"Procesando lote {(i//batch_size)+1}/{(len(chunks)//batch_size)+1}")
            data = procesar_con_gemini(context)
            
            if data and "enfermedades" in data:
                for enfermedad in data["enfermedades"]:
                    if enfermedad.get("enfermedad") and enfermedad.get("enfermedad").strip():
                        # Calcular totales
                        enfermedad = calcular_totales(enfermedad)
                        resultados.append(enfermedad)
                
                print(f"✅ Lote completado - {len(data['enfermedades'])} enfermedades encontradas")
            else:
                print("⚠️ No se encontraron enfermedades en este lote")
            
            time.sleep(delay)
        
        if resultados:
            # Consolidar resultados por enfermedad
            enfermedades_consolidadas = {}
            
            for resultado in resultados:
                nombre = resultado["enfermedad"]
                if nombre not in enfermedades_consolidadas:
                    enfermedades_consolidadas[nombre] = resultado
                else:
                    # Sumar los valores
                    rangos = ["menores_1_ano", "entre_1_4_anos", "entre_5_6_anos", "entre_7_9_anos",
                             "entre_10_11_anos", "entre_12_14_anos", "entre_15_19_anos", "entre_20_24_anos",
                             "entre_25_44_anos", "entre_45_59_anos", "entre_60_64_anos", "mayores_65_anos", "edad_ignorada"]
                    
                    for rango in rangos:
                        enfermedades_consolidadas[nombre][rango]["hombres"] += resultado[rango]["hombres"]
                        enfermedades_consolidadas[nombre][rango]["mujeres"] += resultado[rango]["mujeres"]
                    
                    # Recalcular totales
                    enfermedades_consolidadas[nombre] = calcular_totales(enfermedades_consolidadas[nombre])
            
            # Asignar orden basado en la lista oficial
            enfermedades_finales = []
            orden_counter = 1
            
            for enfermedad_oficial in ENFERMEDADES_OFICIALES:
                encontrada = None
                for nombre, datos in enfermedades_consolidadas.items():
                    # Buscar coincidencias más flexibles
                    nombre_limpio = nombre.lower().replace("(", "").replace(")", "")
                    oficial_limpio = enfermedad_oficial.lower().replace("(", "").replace(")", "")
                    
                    if (oficial_limpio in nombre_limpio or 
                        nombre_limpio in oficial_limpio or
                        any(word in oficial_limpio for word in nombre_limpio.split() if len(word) > 3)):
                        encontrada = datos
                        break
                
                if encontrada:
                    encontrada["orden"] = orden_counter
                    encontrada["enfermedad"] = enfermedad_oficial  # Usar nombre oficial
                    enfermedades_finales.append(encontrada)
                    orden_counter += 1
            
            # Agregar enfermedades no encontradas en la lista oficial
            for nombre, datos in enfermedades_consolidadas.items():
                if not any(ef["enfermedad"] == nombre for ef in enfermedades_finales):
                    datos["orden"] = orden_counter
                    enfermedades_finales.append(datos)
                    orden_counter += 1
            
            estadisticas = ResultadoEPI12(enfermedades=enfermedades_finales)
            
            # Guardar resultado
            with open('estadisticas_epi12_individual.json', 'w', encoding='utf-8') as f:
                json.dump(estadisticas.model_dump(), f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Proceso completado en {time.time()-start_time:.2f} segundos")
            print(f"Enfermedades procesadas: {len(enfermedades_finales)}")
            
            return estadisticas
        
        else:
            print("\n❌ No se encontraron enfermedades en el documento")
            return None
            
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")
        return None

def generar_estadisticas_multiples(file_paths: List[str]) -> Optional[ResultadoEPI12]:
    """Procesa múltiples archivos PDF y genera estadísticas EPI-12"""
    print(f"\nIniciando procesamiento de {len(file_paths)} archivos...")
    start_time = time.time()
    
    # Procesar todos los archivos
    all_chunks = []
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
                separators=["\nnombre paciente:", "\nApellido paciente:", "\nCédula paciente:", "\n\n"]
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"✅ Procesado: {file_path} - {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")
            continue
    
    if not all_chunks:
        print("❌ No se pudieron procesar archivos")
        return None
    
    # Crear base de datos vectorial
    persist_directory = "./epi12_vector_db_multi"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    global vector_db
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Procesar en lotes
    resultados = []
    batch_size = 3
    delay = 1
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        context = "\n---\n".join(doc.page_content for doc in batch)
        
        print(f"\nProcesando lote {(i//batch_size)+1}/{(len(all_chunks)//batch_size)+1}")
        data = procesar_con_gemini(context)
        
        if data and "enfermedades" in data:
            enfermedades_validas = [e for e in data["enfermedades"] if e.get("enfermedad")]
            resultados.extend(enfermedades_validas)
            print(f"✅ Lote completado - {len(enfermedades_validas)} enfermedades")
        else:
            print("❌ Error en lote, continuando...")
        
        time.sleep(delay)
    
    if resultados:
        try:
            # Consolidar resultados por enfermedad
            enfermedades_consolidadas = {}
            
            for resultado in resultados:
                nombre = resultado["enfermedad"]
                if nombre not in enfermedades_consolidadas:
                    enfermedades_consolidadas[nombre] = resultado
                else:
                    # Sumar los valores
                    for rango in ["menores_1_ano", "entre_1_4_anos", "entre_5_6_anos", "entre_7_9_anos",
                                "entre_10_11_anos", "entre_12_14_anos", "entre_15_19_anos", "entre_20_24_anos",
                                "entre_25_44_anos", "entre_45_59_anos", "entre_60_64_anos", "mayores_65_anos", "edad_ignorada"]:
                        enfermedades_consolidadas[nombre][rango]["hombres"] += resultado[rango]["hombres"]
                        enfermedades_consolidadas[nombre][rango]["mujeres"] += resultado[rango]["mujeres"]
                    
                    enfermedades_consolidadas[nombre]["total_hombres"] += resultado["total_hombres"]
                    enfermedades_consolidadas[nombre]["total_mujeres"] += resultado["total_mujeres"]
                    enfermedades_consolidadas[nombre]["total_general"] += resultado["total_general"]
            
            # Asignar orden basado en la lista oficial
            enfermedades_finales = []
            for i, enfermedad_oficial in enumerate(ENFERMEDADES_OFICIALES, 1):
                encontrada = None
                for nombre, datos in enfermedades_consolidadas.items():
                    if enfermedad_oficial.lower() in nombre.lower() or nombre.lower() in enfermedad_oficial.lower():
                        encontrada = datos
                        break
                
                if encontrada:
                    encontrada["orden"] = i
                    encontrada["enfermedad"] = enfermedad_oficial
                    enfermedades_finales.append(encontrada)
            
            estadisticas = ResultadoEPI12(enfermedades=enfermedades_finales)
            
            # Guardar resultado
            with open('estadisticas_epi12_multi.json', 'w', encoding='utf-8') as f:
                json.dump(estadisticas.model_dump(), f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Proceso completado en {time.time()-start_time:.2f} segundos")
            print(f"Resultados guardados - {len(enfermedades_finales)} enfermedades procesadas")
            return estadisticas
            
        except Exception as e:
            print(f"Error al validar los resultados: {e}")
            return None
    
    print("\n❌ No se generaron resultados válidos")
    return None

# Variable global para la base de datos vectorial
vector_db = None

def inicializar_vector_db():
    """Inicializa la base de datos vectorial si existe"""
    global vector_db
    persist_directory = "./epi12_vector_db_single"
    if os.path.exists(persist_directory):
        try:
            vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print("✅ Base de datos vectorial cargada")
        except Exception as e:
            print(f"⚠️ Error cargando base de datos vectorial: {e}")
            vector_db = None
    else:
        print("ℹ️ No hay base de datos vectorial previa")

if __name__ == "__main__":
    # Verificar conexión con Gemini
    try:
        test_response = llm.invoke("Hola")
        if not test_response or not test_response.content:
            raise ConnectionError("No se recibió respuesta de Gemini")
        print("✅ Conexión con Gemini establecida correctamente")
    except Exception as e:
        print(f"❌ Error de conexión con Gemini: {e}")
        exit(1)
    
    # Ejemplo de uso
    # stats = generar_estadisticas_individual("archivo.pdf")
    # stats = generar_estadisticas_multiples(["archivo1.pdf", "archivo2.pdf"])

