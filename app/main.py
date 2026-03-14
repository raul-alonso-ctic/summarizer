from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
import sys

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force load .env file to override any conflicting shell environment variables
load_dotenv(override=True)

from app.models import (
    SummarizeRequest, SummarizeResponse, 
    ProcessFolderRequest, ProcessFolderResponse
)
from app.services.processor import DocumentProcessor
from app.services.gdrive import GoogleDriveService
from app.services.checkpoint import CheckpointService
import os
import shutil
import tempfile
from typing import List, Optional
from pathlib import Path

app = FastAPI(title="Summarizer Service", version="0.2.0")
templates = Jinja2Templates(directory="app/templates")
processor = DocumentProcessor()

# Ruta base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Inicializar servicio de Google Drive solo si está habilitado
gdrive_service = None
try:
    if os.getenv("GOOGLE_DRIVE_ENABLED", "true").lower() == "true":
        gdrive_service = GoogleDriveService()
except Exception as e:
    print(f"Warning: No se pudo inicializar Google Drive Service: {e}")
    gdrive_service = None

@app.get("/favicon.ico")
async def favicon():
    """Sirve el favicon"""
    favicon_path = BASE_DIR / "assets" / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request, 
    files: List[UploadFile] = File(...),
    max_tokens: int = Form(1024),
    initial_pages: int = Form(2),
    final_pages: int = Form(2),
    temperature_vllm: float = Form(0.1),
    temperature_llm: float = Form(0.3),
    top_p: float = Form(0.9),
    process_all: bool = Form(False),
    max_archive_files: int = Form(0)
):
    """Endpoint para subir archivos directamente desde la web UI"""
    results_html = ""
    temp_dir = tempfile.mkdtemp()
    
    # Logic for "Process All"
    if process_all:
        logger.info("Process All requested. Overriding page limits.")
        initial_pages = 1000000
        final_pages = 0

    logger.info(f"Received upload request with {len(files)} files. Max tokens: {max_tokens}, Pages: {initial_pages}/{final_pages}, Temp VLLM: {temperature_vllm}, Temp LLM: {temperature_llm}, Top P: {top_p}")
    
    try:
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            try:
                # Guardar archivo temporalmente
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                
                # Procesar archivo
                source_config = {
                    "mode": "upload",
                    "path": file_path,
                    "file_name": file.filename,  # Añadir file_name para mejor detección y mensajes de error
                    "language": "es",
                    "initial_pages": initial_pages,
                    "final_pages": final_pages,
                    "max_tokens": max_tokens,
                    "temperature_vllm": temperature_vllm,
                    "temperature_llm": temperature_llm,
                    "top_p": top_p,
                    "max_inner_files": max_archive_files
                }
                
                result = processor.process_file_from_source(source_config)
                
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    results_html += f"""
                <div class="result-item error">
                    <div class="result-title">{file.filename}</div>
                    <p><strong>⚠️ Archivo vacío:</strong> Este archivo está completamente vacío (0 bytes) y ha sido ignorado.</p>
                </div>
                """
                    continue
                
                # Formatear resultado HTML - Simplificado a texto plano sin markdown
                children_html = ""
                if result.children:
                    children_html = '<div class="children-list">'
                    for child in result.children:
                        children_html += f'<div class="child-item"><strong>{child.name}</strong>: {child.description}</div>'
                    children_html += "</div>"
                
                results_html += f"""
                <div class="result-item">
                    <script type="application/json" class="result-data">
                        {result.model_dump_json()}
                    </script>
                    <div class="result-title">{result.name}</div>
                    <div class="result-meta">
                        <span class="badge {result.type}">{result.type.upper()}</span>
                    </div>
                    <div class="result-description">
                        {result.description}
                    </div>
                    {children_html}
                </div>
                """
            except Exception as e:
                results_html += f"""
                <div class="result-item error">
                    <div class="result-title">{file.filename}</div>
                    <p><strong>Error:</strong> {str(e)}</p>
                </div>
                """
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results_html

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "summarizer", "version": "0.2.0"}

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Endpoint genérico para procesar documentos desde diferentes fuentes"""
    results = []
    
    for doc in request.documents:
        try:
            source_config = {
                "mode": doc.source.mode,
                "path": doc.source.path,
                "file_id": doc.source.file_id,
                "file_name": doc.source.file_name,
                "folder_id": doc.source.folder_id,
                "language": doc.source.language,
                "initial_pages": doc.source.initial_pages,
                "final_pages": doc.source.final_pages,
                "max_tokens": doc.source.max_tokens,
                "temperature": doc.source.temperature,
                "top_p": doc.source.top_p,
                "max_inner_files": doc.source.max_inner_files
            }
            
            result = processor.process_file_from_source(source_config)
            # Si el archivo está vacío, result será None y lo ignoramos
            if result is None:
                logger.info(f"Archivo vacío ignorado en API: {doc.source.file_name or doc.id}")
                # Añadir un resultado de error indicando que el archivo está vacío
                results.append({
                    "file_id": doc.source.file_id,
                    "name": doc.source.file_name or doc.id,
                    "title": doc.source.file_name or doc.id,
                    "description": "⚠️ Archivo vacío: Este archivo está completamente vacío (0 bytes) y ha sido ignorado.",
                    "type": doc.type,
                    "children": [],
                    "metadata": {"error": True, "empty_file": True}
                })
                continue
            # file_id ya está en el DocumentResult
            results.append(result)
        except Exception as e:
            results.append({
                "file_id": doc.source.file_id,
                "name": doc.source.file_name or doc.id,
                "description": f"Error procesando: {str(e)}",
                "type": doc.type,
                "children": [],
                "metadata": {"error": True}
            })
    
    return SummarizeResponse(results=results)

@app.get("/health/gdrive")
async def health_gdrive():
    if not processor.gdrive_service:
        return {"status": "disabled", "message": "Google Drive service is not enabled"}
    
    try:
        # Intentar listar archivos en la carpeta raíz para validar conexión
        files = processor.gdrive_service.list_files(limit=1)
        return {
            "status": "ok", 
            "message": "Google Drive connection successful",
            "files_visible": len(files)
        }
    except Exception as e:
        return {"status": "error", "message": f"Google Drive connection failed: {str(e)}"}

@app.get("/health/llm")
async def health_llm():
    """Prueba la conexión con el modelo de texto (LLMService)"""
    return processor.llm_service.test_connection()

@app.get("/health/vllm")
async def health_vllm():
    """Prueba la conexión con el modelo multimodal (VLLMService)"""
    return processor.vllm_service.test_connection()

@app.get("/checkpoint/{folder_id}")
async def get_checkpoint_status(folder_id: str):
    """Obtiene el estado del checkpoint para una carpeta"""
    unattended_mode = os.getenv("UNATTENDED_MODE", "false").lower() == "true"
    
    if not unattended_mode:
        raise HTTPException(
            status_code=400, 
            detail="Modo desatendido no está habilitado (UNATTENDED_MODE=true)"
        )
    
    try:
        checkpoint_service = CheckpointService()
        existing_checkpoint = checkpoint_service._find_existing_checkpoint(folder_id)
        
        if not existing_checkpoint:
            return {
                "status": "not_found",
                "message": f"No se encontró checkpoint para la carpeta {folder_id}",
                "folder_id": folder_id
            }
        
        checkpoint_service.current_checkpoint = str(existing_checkpoint)
        checkpoint_service._load_checkpoint()
        
        progress = checkpoint_service.get_progress()
        
        return {
            "status": "found",
            "folder_id": folder_id,
            "folder_name": checkpoint_service.checkpoint_data.get("folder_name", "Unknown"),
            "checkpoint_file": checkpoint_service.current_checkpoint,
            "progress": progress,
            "started_at": checkpoint_service.checkpoint_data.get("started_at"),
            "last_updated": checkpoint_service.checkpoint_data.get("last_updated"),
            "failed_files": checkpoint_service.get_failed_files()
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del checkpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estado del checkpoint: {str(e)}"
        )


@app.post("/process-folder", response_model=ProcessFolderResponse)
async def process_folder(request: ProcessFolderRequest):
    """Procesa todos los archivos PDF y ZIP de una carpeta de Google Drive"""
    if not gdrive_service:
        raise HTTPException(status_code=500, detail="Servicio de Google Drive no disponible")
    
    folder_id = None
    
    # Si se proporciona folder_name, buscar la carpeta
    if request.folder_name:
        parent_id = request.parent_folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not parent_id:
            raise HTTPException(
                status_code=400, 
                detail="parent_folder_id es requerido cuando se usa folder_name (o configurar GOOGLE_DRIVE_FOLDER_ID)"
            )
        folder_id = gdrive_service.find_folder_by_name(
            parent_id, 
            request.folder_name
        )
        if not folder_id:
            raise HTTPException(
                status_code=404, 
                detail=f"Carpeta '{request.folder_name}' no encontrada"
            )
    elif request.folder_id:
        folder_id = gdrive_service.extract_folder_id(request.folder_id)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Se requiere folder_id o folder_name"
        )
    
    # Obtener nombre de la carpeta
    try:
        folder_info = gdrive_service.get_file_info(folder_id)
        folder_name = folder_info.get('name', 'Unknown')
    except Exception as e:
        logger.error(f"Error accessing folder {folder_id}: {e}")
        # Check if it's a 404 or permission error
        error_msg = str(e)
        if "404" in error_msg or "notFound" in error_msg:
            raise HTTPException(
                status_code=404, 
                detail=f"Folder not found: {folder_id}. Please check the ID and permissions."
            )
        raise HTTPException(
            status_code=500, 
            detail=f"Error accessing Google Drive: {error_msg}"
        )
    
    # Verificar modo desatendido y mostrar información
    unattended_mode = os.getenv("UNATTENDED_MODE", "false").lower() == "true"
    checkpoint_info = None
    
    if unattended_mode:
        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/data/checkpoints")
        logger.info(f"MODO DESATENDIDO ACTIVADO - Checkpoints en: {checkpoint_dir}")
        checkpoint_info = {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "message": f"El procesamiento se guardará en checkpoints. Consulta el progreso en: {checkpoint_dir}"
        }
    
    # Procesar carpeta
    response = processor.process_gdrive_folder(
        folder_id,
        folder_name,
        request.language,
        request.initial_pages,
        request.final_pages,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.max_inner_files
    )
    
    # Agregar información de checkpoint a la respuesta si está activo
    # Nota: El checkpoint ya fue creado/consultado durante process_gdrive_folder
    # Aquí solo agregamos la información a la respuesta
    if unattended_mode and checkpoint_info:
        try:
            checkpoint_service = CheckpointService()
            existing_checkpoint = checkpoint_service._find_existing_checkpoint(folder_id)
            if existing_checkpoint:
                checkpoint_service.current_checkpoint = str(existing_checkpoint)
                checkpoint_service._load_checkpoint()
                progress = checkpoint_service.get_progress()
                checkpoint_info.update({
                    "checkpoint_file": checkpoint_service.current_checkpoint,
                    "progress": progress
                })
        except Exception as e:
            logger.warning(f"No se pudo obtener información del checkpoint: {e}")
    
    # Agregar checkpoint_info a metadata de la respuesta
    if checkpoint_info:
        if not hasattr(response, 'metadata') or response.metadata is None:
            response.metadata = {}
        response.metadata['checkpoint'] = checkpoint_info
    
    return response

