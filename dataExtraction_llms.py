# Import Langchain module
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os
import tempfile
import pandas as pd
from dotenv import load_dotenv

from docling import DocumentProcessor
from PIL import Image
import pytesseract
import re
import json

# Configuración de OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajusta según tu instalación

def procesar_imagen_tabla(ruta_imagen):
    """Procesa imágenes con tablas médicas y estructura los datos"""
    # 1. Usar Docling para preprocesamiento (opcional)
    doc_proc = DocumentProcessor()
    
    # 2. Extraer texto con Tesseract OCR
    imagen = Image.open(ruta_imagen)
    texto = pytesseract.image_to_string(imagen, lang='spa')


#llm = OllamaLLM(model="tinyllama")
#for chunk in llm.stream("What is UNEG?"):
#    print(chunk, end="", flush=True)
#print("\n") 

