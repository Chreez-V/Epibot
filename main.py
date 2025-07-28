from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
from fpdf import FPDF  # pip install fpdf2
import uvicorn
import tempfile
import os
import json
from datetime import datetime
import uuid
import shutil

# Importa tu código RAG existente
from dataExtractionGem import (
    procesar_con_gemini, 
    generar_estadisticas,
    ResultadoEPI12,
    EstadisticaEPI12,
    vector_db,
    llm,
    Chroma,
    GoogleGenerativeAIEmbeddings,
    PyPDFLoader,
    RecursiveCharacterTextSplitter
)

app = FastAPI(title="EpiFlow API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración global
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Almacenamiento temporal (en producción usar base de datos)
documentos_db = {}
chat_sessions = {}

@app.post("/api/process-pdfs")
async def process_pdfs(files: List[UploadFile] = File(...)):
    try:
        # Configuración de embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Directorio para persistencia de Chroma
        persist_directory = "./epi12_vector_db_new"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        
        # Procesar cada archivo PDF
        all_chunks = []
        for file in files:
            # Guardar archivo temporalmente
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Cargar y dividir el PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=30,
                length_function=len,
                separators=["\nnombre paciente:", "\nApellido paciente:", "\nCédula paciente:", "\n\n"]
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            
            # Eliminar archivo temporal después de procesar
            os.unlink(file_path)
        
        # Crear base de datos vectorial con todos los chunks
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Generar estadísticas EPI-12
        resultados = []
        batch_size = 2
        delay = 1
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            context = "\n---\n".join(doc.page_content for doc in batch)
            
            data = procesar_con_gemini(context)
            
            if data and "enfermedades" in data:
                # Filtrar enfermedades vacías o con nombres None
                enfermedades_validas = [e for e in data["enfermedades"] if e.get("enfermedad")]
                resultados.extend(enfermedades_validas)
            
            time.sleep(delay)
        
        if resultados:
            # Crear objeto ResultadoEPI12 validado
            estadisticas = ResultadoEPI12(enfermedades=resultados)
            
            # Guardar en la base de datos temporal
            doc_id = str(uuid.uuid4())
            documentos_db[doc_id] = {
                "id": doc_id,
                "fecha_creacion": datetime.now().isoformat(),
                "data": estadisticas.model_dump(),
                "tipo": "EPI-12"
            }
            
            return JSONResponse(content={
                "status": "success",
                "document_id": doc_id,
                "data": estadisticas.model_dump()
            })
        else:
            raise HTTPException(status_code=400, detail="No se pudieron extraer datos estadísticos de los documentos")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documentos: {str(e)}")

@app.post("/api/chat")
async def chat_with_documents(request: dict):
    try:
        message = request.get("message")
        session_id = request.get("session_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Mensaje requerido")
        
        # Usar tu vector_db existente para búsqueda semántica
        similar_docs = vector_db.similarity_search(message, k=3)
        context = "\n---\n".join([doc.page_content for doc in similar_docs])
        
        # Generar respuesta con Gemini
        prompt = f"Contexto de documentos médicos:\n{context}\n\nPregunta del usuario: {message}\n\nResponde de manera clara y profesional:"
        response = llm.invoke(prompt)
        
        # Guardar en historial de chat
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].extend([
            {"role": "user", "content": message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response.content, "timestamp": datetime.now().isoformat()}
        ])
        
        return {"response": response.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def save_document(documento: dict):
    try:
        doc_id = str(uuid.uuid4())
        documento["id"] = doc_id
        documento["fecha_creacion"] = datetime.now().isoformat()
        
        documentos_db[doc_id] = documento
        
        return documento
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    return list(documentos_db.values())

@app.get("/api/document/{document_id}")
async def get_document(document_id: str):
    if document_id not in documentos_db:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return documentos_db[document_id]

@app.get("/api/export-pdf/{document_id}")
async def export_pdf(document_id: str):
    if document_id not in documentos_db:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    # En producción usarías reportlab o similar
    # Esto es solo un ejemplo básico
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    doc_data = documentos_db[document_id]
    
    # Agregar contenido al PDF
    pdf.cell(0, 10, f"Reporte EPI-12 - {document_id}", 0, 1, 'C')
    pdf.cell(0, 10, f"Fecha: {doc_data['fecha_creacion']}", 0, 1, 'L')
    pdf.ln(10)
    
    # Agregar datos de enfermedades
    pdf.cell(0, 10, "Enfermedades reportadas:", 0, 1, 'L')
    for enfermedad in doc_data['data']['enfermedades']:
        pdf.cell(0, 10, f"- {enfermedad['enfermedad']}: Hombres={enfermedad.get('total_hombres', 0)}, Mujeres={enfermedad.get('total_mujeres', 0)}", 0, 1, 'L')
    
    # Guardar PDF
    pdf_path = f"generated_pdfs/{document_id}.pdf"
    os.makedirs("generated_pdfs", exist_ok=True)
    pdf.output(pdf_path)
    
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"EPI12_{document_id}.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
