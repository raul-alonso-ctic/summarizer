from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SourceConfig(BaseModel):
    mode: Literal["local", "upload", "gdrive"]
    path: Optional[str] = None
    file_id: Optional[str] = None  # ID directo del archivo en Google Drive
    file_name: Optional[str] = None  # Nombre del archivo (para buscar en folder_id)
    folder_id: Optional[str] = None # Para Google Drive
    folder_name: Optional[str] = None # Para buscar carpeta por nombre
    language: str = "es"
    initial_pages: int = Field(default=2, ge=0, description="Número de páginas iniciales a procesar")
    final_pages: int = Field(default=2, ge=0, description="Número de páginas finales a procesar")
    max_tokens: int = Field(default=1024, ge=10, description="Máximo tokens para la respuesta")
    temperature_vllm: Optional[float] = Field(default=0.1, ge=0.0, le=2.0, description="Temperatura para el modelo VLLM (multimodal, PDF/DOCX)")
    temperature_llm: Optional[float] = Field(default=0.3, ge=0.0, le=2.0, description="Temperatura para el modelo LLM (texto, ZIP/XML/EML)")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperatura para ambos modelos (deprecated, usar temperature_vllm y temperature_llm)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top P para el modelo")
    max_inner_files: int = Field(default=0, ge=0, description="Max files to process inside archives (0=unlimited)")

class DocumentSource(BaseModel):
    id: str
    type: Literal["pdf", "zip", "xml", "eml"]
    source: SourceConfig

class SummarizeRequest(BaseModel):
    documents: List[DocumentSource]

class ProcessFolderRequest(BaseModel):
    folder_id: Optional[str] = None
    folder_name: Optional[str] = None
    parent_folder_id: Optional[str] = None # Para buscar carpeta por nombre dentro de otra
    language: str = "es"
    initial_pages: int = Field(default=2, ge=0, description="Número de páginas iniciales a procesar de cada PDF")
    final_pages: int = Field(default=2, ge=0, description="Número de páginas finales a procesar de cada PDF")
    max_tokens: int = Field(default=1024, ge=10, description="Máximo tokens para la respuesta")
    temperature_vllm: Optional[float] = Field(default=0.1, ge=0.0, le=2.0, description="Temperatura para el modelo VLLM (multimodal, PDF/DOCX)")
    temperature_llm: Optional[float] = Field(default=0.3, ge=0.0, le=2.0, description="Temperatura para el modelo LLM (texto, ZIP/XML/EML)")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperatura para ambos modelos (deprecated, usar temperature_vllm y temperature_llm)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top P para el modelo")
    max_inner_files: int = Field(default=0, ge=0, description="Max files to process inside archives (0=unlimited)")

class DocumentResult(BaseModel):
    name: str
    title: str
    description: str
    type: str
    path: Optional[str] = None
    file_id: Optional[str] = None  # ID del archivo en Google Drive (si aplica, para mode = "gdrive")
    children: Optional[List['DocumentResult']] = None
    metadata: Optional[Dict[str, Any]] = None
 
class ProcessFolderResponse(BaseModel):
    folder_id: str
    folder_name: str
    processed_at: datetime
    total_files: int
    results: List[DocumentResult]

class SummarizeResponse(BaseModel):
    results: List[DocumentResult]
