from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import uvicorn
import tempfile
import os
import json
from datetime import datetime
import uuid
import shutil

# Importar el código RAG
from dataExtractionGem import (
    generar_estadisticas_individual,
    ResultadoEPI12,
    EstadisticaEPI12,
    vector_db,
    llm,
    inicializar_vector_db,
    ENFERMEDADES_OFICIALES
)

from pdf_generator import generar_pdf_epi12

app = FastAPI(title="EpiFlow API", version="1.0.0")

# Configurar CORS correctamente
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración global
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("generated_pdfs", exist_ok=True)

# Almacenamiento temporal (en producción usar base de datos)
documentos_db = {}
chat_sessions = {}

@app.on_event("startup")
async def startup_event():
    """Inicializar la aplicación"""
    print("🚀 Iniciando EpiFlow API...")
    print("📊 Inicializando base de datos vectorial...")
    inicializar_vector_db()
    print("✅ EpiFlow API lista para usar")

@app.get("/")
async def read_root():
    return {"message": "EpiFlow API funcionando correctamente", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/process-pdfs")
async def process_single_pdf(file: UploadFile = File(...)):
    """Procesa UN solo archivo PDF a la vez"""
    try:
        # Validar que sea PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        print(f"📄 Recibido archivo: {file.filename}")
        
        # Guardar archivo temporalmente
        file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.pdf")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"💾 Archivo guardado temporalmente: {file_path}")
        print(f"🔄 Procesando con RAG...")
        
        # Procesar con RAG
        resultado = generar_estadisticas_individual(file_path)
        
        # Eliminar archivo temporal
        os.unlink(file_path)
        print(f"🗑️ Archivo temporal eliminado")
        
        if resultado:
            print(f"✅ Procesamiento exitoso: {len(resultado.enfermedades)} enfermedades encontradas")
            return JSONResponse(content={
                "status": "success",
                "message": f"Archivo {file.filename} procesado correctamente",
                "data": resultado.model_dump()
            })
        else:
            print("❌ No se pudieron extraer datos del PDF")
            raise HTTPException(status_code=400, detail="No se pudieron extraer datos del PDF")
            
    except Exception as e:
        # Limpiar archivo temporal si existe
        #if 'file_path' in locals() and os.path.exists(file_path):
        #    os.unlink(file_path)
        
        print(f"❌ Error procesando PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")

@app.post("/api/chat")
async def chat_with_documents(request: dict):
    """Chat con los documentos cargados"""
    try:
        message = request.get("message")
        session_id = request.get("session_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Mensaje requerido")
        
        print(f"💬 Chat - Usuario: {message[:50]}...")
        
        # Inicializar vector_db si no existe
        if vector_db is None:
            return {"response": "No hay documentos cargados para consultar. Por favor, sube un archivo PDF primero."}
        
        # Búsqueda semántica
        similar_docs = vector_db.similarity_search(message, k=3)
        context = "\n---\n".join([doc.page_content for doc in similar_docs])
        
        # Generar respuesta con Gemini
        prompt = f"""Contexto de documentos médicos:
{context}

Pregunta del usuario: {message}

Responde de manera clara y profesional basándote en la información médica disponible:"""
        
        response = llm.invoke(prompt)
        
        # Guardar en historial de chat
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].extend([
            {"role": "user", "content": message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response.content, "timestamp": datetime.now().isoformat()}
        ])
        
        print(f"🤖 Respuesta generada: {response.content[:50]}...")
        return {"response": response.content}
        
    except Exception as e:
        print(f"❌ Error en chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")

@app.post("/api/documents")
async def save_document(documento: dict):
    """Guardar documento EPI-12"""
    try:
        doc_id = str(uuid.uuid4())
        documento["id"] = doc_id
        documento["fecha_creacion"] = datetime.now().isoformat()
        
        documentos_db[doc_id] = documento
        
        print(f"💾 Documento guardado: {doc_id}")
        return documento
        
    except Exception as e:
        print(f"❌ Error guardando documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al guardar documento: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    """Obtener todos los documentos"""
    print(f"📋 Consultando documentos: {len(documentos_db)} encontrados")
    return list(documentos_db.values())

@app.get("/api/document/{document_id}")
async def get_document(document_id: str):
    """Obtener un documento específico"""
    if document_id not in documentos_db:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return documentos_db[document_id]

@app.get("/api/export-pdf/{document_id}")
async def export_pdf(document_id: str):
    """Exportar documento como PDF oficial"""
    if document_id not in documentos_db:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    try:
        print(f"📄 Generando PDF para documento: {document_id}")
        doc_data = documentos_db[document_id]
        
        # Metadatos del documento
        metadata = {
            'entidad': 'Estado Bolívar',
            'municipio': 'Caroní',
            'establecimiento': 'Hospital General',
            'año': datetime.now().year,
            'semana': datetime.now().isocalendar()[1]
        }
        
        # Generar PDF oficial
        pdf_path = generar_pdf_epi12(doc_data.get('datos', {}), metadata)
        
        print(f"✅ PDF generado: {pdf_path}")
        
        # Retornar el archivo
        return FileResponse(
            pdf_path,
            media_type='application/pdf',
            filename=f"EPI12_Oficial_{document_id[:8]}.pdf"
        )
        
    except Exception as e:
        print(f"❌ Error generando PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {str(e)}")

@app.get("/api/enfermedades-oficiales")
async def get_enfermedades_oficiales():
    """Retorna las 52 enfermedades oficiales del EPI-12"""
    print("📋 Consultando enfermedades oficiales")
    enfermedades = []
    for i, enfermedad in enumerate(ENFERMEDADES_OFICIALES, 1):
        enfermedades.append({
            "orden": i,
            "enfermedad": enfermedad,
            "menores_1_ano": {"hombres": 0, "mujeres": 0},
            "entre_1_4_anos": {"hombres": 0, "mujeres": 0},
            "entre_5_6_anos": {"hombres": 0, "mujeres": 0},
            "entre_7_9_anos": {"hombres": 0, "mujeres": 0},
            "entre_10_11_anos": {"hombres": 0, "mujeres": 0},
            "entre_12_14_anos": {"hombres": 0, "mujeres": 0},
            "entre_15_19_anos": {"hombres": 0, "mujeres": 0},
            "entre_20_24_anos": {"hombres": 0, "mujeres": 0},
            "entre_25_44_anos": {"hombres": 0, "mujeres": 0},
            "entre_45_59_anos": {"hombres": 0, "mujeres": 0},
            "entre_60_64_anos": {"hombres": 0, "mujeres": 0},
            "mayores_65_anos": {"hombres": 0, "mujeres": 0},
            "edad_ignorada": {"hombres": 0, "mujeres": 0},
            "total_hombres": 0,
            "total_mujeres": 0,
            "total_general": 0
        })
    
    return {"enfermedades": enfermedades}

if __name__ == "__main__":
    print("🚀 Iniciando servidor EpiFlow...")
    print("📊 Inicializando base de datos vectorial...")
    inicializar_vector_db()
    print("✅ Servidor listo en http://localhost:8000")
    print("📖 Documentación en http://localhost:8000/docs")
    
    # Quitar reload=True para evitar el warning
    uvicorn.run(app, host="0.0.0.0", port=8000)

