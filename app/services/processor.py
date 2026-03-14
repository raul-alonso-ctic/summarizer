import os
import tempfile
import zipfile
import tarfile
import shutil
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from app.services.pdf import PDFProcessor
from app.services.docx import DOCXProcessor
from app.services.vllm import VLLMService
from app.services.llm import LLMService
from app.services.gdrive import GoogleDriveService
from app.services.checkpoint import CheckpointService
from app.services.xml_eml import XMLEMLProcessor
from app.models import DocumentResult, ProcessFolderResponse
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.xml_eml_processor = XMLEMLProcessor()
        
        # Initialize VLLM service for PDF and DOCX processing (multimodal with images)
        vllm_model = os.getenv("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
        self.vllm_service = VLLMService(model=vllm_model)
        
        # Initialize LLM service for ZIP macro-summaries, XML and EML (text-only, faster)
        # Si USE_VLLM_FOR_ALL=true, no pasar el modelo para que LLMService use VLLM_MODEL
        use_vllm_for_all = os.getenv("USE_VLLM_FOR_ALL", "false").lower() == "true"
        if use_vllm_for_all:
            # Pasar None para que LLMService decida según USE_VLLM_FOR_ALL
            self.llm_service = LLMService(model=None)
            logger.info("USE_VLLM_FOR_ALL is true - LLMService will use VLLM_MODEL")
        else:
            llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen3-32B")
            self.llm_service = LLMService(model=llm_model)
            logger.info(f"Initialized LLM service with model: {llm_model}")
        
        # Lock para serializar descargas de Google Drive (evitar rate limiting y colisiones)
        self.gdrive_download_lock = threading.Lock()

        # Semáforo global para limitar inferencias concurrentes (evitar saturar el modelo)
        max_concurrent_inference = int(os.getenv("MAX_CONCURRENT_INFERENCE", "16"))
        self.inference_semaphore = threading.Semaphore(max_concurrent_inference)
        logger.info(f"Initialized inference semaphore with max {max_concurrent_inference} concurrent requests")
        
        logger.info(f"Initialized VLLM service with model: {vllm_model}")
        
        self.gdrive_service = GoogleDriveService() if os.getenv("GOOGLE_DRIVE_ENABLED", "true").lower() == "true" else None
        
    def _get_vllm_prompt_and_schema(self, language: str = "es") -> Tuple[str, dict]:
        """Genera el prompt y schema unificados para PDF y DOCX (VLLM multimodal)"""
        # Convertir código de idioma a nombre completo
        language_names = {
            "es": "español",
            "en": "inglés",
            "fr": "francés",
            "de": "alemán",
            "it": "italiano",
            "pt": "portugués"
        }
        language_name = language_names.get(language.lower(), "español")
        
        # Obtener lista de nombres para normalizar (opcional)
        normalize_names_str = os.getenv("NORMALIZE_NAMES", "").strip()
        normalize_names_instruction = ""
        if normalize_names_str:
            # Parsear la lista de nombres (separados por comas)
            names_list = [name.strip() for name in normalize_names_str.split(",") if name.strip()]
            if names_list:
                names_text = ", ".join([f'"{name}"' for name in names_list])
                normalize_names_instruction = f"""
NORMALIZACIÓN DE NOMBRES IMPORTANTES:
Si detectas en el documento nombres de personas que puedan corresponder a alguno de estos nombres normalizados, DEBES usar la versión normalizada exacta:
{names_text}

REGLAS CRÍTICAS para normalización:
- SOLO usa estos nombres cuando estés ABSOLUTAMENTE SEGURO de que el nombre detectado corresponde a uno de estos nombres normalizados
- NO inventes ni fuerces estos nombres si no aparecen claramente en el documento
- NO incluyas estos nombres si no están presentes en el documento, incluso si el contexto podría sugerirlos
- Si detectas una variación o error ortográfico de estos nombres, usa la versión normalizada correcta
- La normalización debe ser precisa: si ves ligeras variaciones ortográficas de los nombres de la lista, usa exactamente los nombres de la lista
- Si tienes dudas sobre si un nombre corresponde a uno de estos, NO lo normalices y usa el nombre tal como aparece en el documento
"""

        # Instrucción fija para normalización de términos comerciales
        normalize_terms_instruction = """
NORMALIZACIÓN DE TÉRMINOS COMERCIALES:
Cuando el documento sea una simple lista de precios de materiales, productos o una actuación/servicio concreto (sin un alcance de proyecto mayor), utiliza el término "presupuesto" para describirlo, independientemente de cómo se titule el documento original.

Ejemplos que DEBEN normalizarse a "presupuesto":
- "Factura proforma" con lista de materiales y precios
- "Cotización de materiales"
- "Oferta de precios" para productos o servicios puntuales
- Documentos que solo listan artículos/servicios con sus precios

Ejemplos que NO deben normalizarse (mantener su denominación original):
- "Propuesta económica" o "Propuesta técnico-económica" para proyectos
- "Oferta" con alcance de proyecto, fases, entregables
- Documentos que describen la realización de un proyecto completo"""

        # Prompt unificado para PDF y DOCX
        prompt = f"""Analiza este documento y genera un título y una descripción en texto plano.

El título debe ser representativo del contenido del documento, autocontenido y descriptivo del significado y propósito del documento. Máximo 15-20 palabras. El título debe resumir de forma concisa la esencia del documento.
La descripción debe ser completa, directa y capturar el propósito y los detalles clave del documento (entidades, fechas, montos).

IMPORTANTE: La descripción debe capturar en no más de 250 palabras los conceptos más importantes para luego poder ser utilizada en un sistema de búsqueda semántica.

{normalize_names_instruction}
{normalize_terms_instruction}

Tu respuesta DEBE ser un objeto JSON con las claves "title" y "description".

Responde en {language_name}."""
        
        # Schema unificado para Structured Outputs
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A representative title (maximum 15-20 words) that semantically describes the document content. The title should be self-contained and summarize the essence and meaning of the document."
                },
                "description": {
                    "type": "string",
                    "description": "A complete plain text description of the document."
                }
            },
            "required": ["title", "description"],
            "additionalProperties": False
        }
        
        return prompt, schema

    def _get_description_prompt(self, content: str, content_type: str, language: str = "es") -> str:
        """
        Genera un prompt unificado para obtener descripciones de documentos (ZIP, XML, EML)
        
        Args:
            content: Contenido a analizar (descripciones de ZIP, contenido XML, o contenido EML)
            content_type: Tipo de contenido ("zip", "xml", o "eml")
            language: Idioma para la respuesta (código: "es", "en", etc.)
            
        Returns:
            Prompt formateado para el LLM
        """
        # Convertir código de idioma a nombre completo
        language_names = {
            "es": "español",
            "en": "inglés",
            "fr": "francés",
            "de": "alemán",
            "it": "italiano",
            "pt": "portugués"
        }
        language_name = language_names.get(language.lower(), "español")
        
        # Obtener lista de nombres para normalizar (opcional)
        normalize_names_str = os.getenv("NORMALIZE_NAMES", "").strip()
        normalize_names_instruction = ""
        if normalize_names_str:
            # Parsear la lista de nombres (separados por comas)
            names_list = [name.strip() for name in normalize_names_str.split(",") if name.strip()]
            if names_list:
                names_text = ", ".join([f'"{name}"' for name in names_list])
                normalize_names_instruction = f"""
NORMALIZACIÓN DE NOMBRES IMPORTANTES:
Si detectas nombres de personas que puedan corresponder a alguno de estos nombres normalizados, DEBES usar la versión normalizada exacta:
{names_text}

REGLAS CRÍTICAS para normalización:
- SOLO usa estos nombres cuando estés ABSOLUTAMENTE SEGURO de que el nombre detectado corresponde a uno de estos nombres normalizados
- NO inventes ni fuerces estos nombres si no aparecen claramente en el contenido
- NO incluyas estos nombres si no están presentes, incluso si el contexto podría sugerirlos
- Si detectas una variación o error ortográfico de estos nombres, usa la versión normalizada correcta
- La normalización debe ser precisa: usa exactamente la versión normalizada cuando corresponda
- Si tienes dudas sobre si un nombre corresponde a uno de estos, NO lo normalices y usa el nombre tal como aparece
"""

        # Instrucción fija para normalización de términos comerciales
        normalize_terms_instruction = """
NORMALIZACIÓN DE TÉRMINOS COMERCIALES:
Cuando el documento sea una simple lista de precios de materiales, productos o una actuación/servicio concreto (sin un alcance de proyecto mayor), utiliza el término "presupuesto" para describirlo, independientemente de cómo se titule el documento original.

Ejemplos que DEBEN normalizarse a "presupuesto":
- "Factura proforma" con lista de materiales y precios
- "Cotización de materiales"
- "Oferta de precios" para productos o servicios puntuales
- Documentos que solo listan artículos/servicios con sus precios

Ejemplos que NO deben normalizarse (mantener su denominación original):
- "Propuesta económica" o "Propuesta técnico-económica" para proyectos
- "Oferta" con alcance de proyecto, fases, entregables
- Documentos que describen la realización de un proyecto completo"""

        if content_type == "zip":
            prompt = f"""Analiza las siguientes descripciones de documentos contenidos en un archivo ZIP y genera una breve descripción en TEXTO PLANO que resuma semánticamente el contenido de la colección completa.

Descripciones:
{content}

IMPORTANTE:
- Responde ÚNICAMENTE con texto plano, sin formato JSON ni otro formato que no sea texto plano
- NO uses comillas, llaves, corchetes, saltos de línea, ni ningún formato estructurado
- NO incluyas etiquetas como "description:", "resumen:" o similares
- Responde directamente con el texto de la descripción
- El resumen debe ser completo, directo y capturar el propósito y los detalles clave del conjunto (entidades, fechas, montos)
- La descripción debe capturar en no más de 250 palabras los conceptos más importantes para luego poder ser utilizada en un sistema de búsqueda semántica

{normalize_names_instruction}
{normalize_terms_instruction}

Responde en {language_name}."""
        elif content_type == "xml":
            prompt = f"""Analiza el siguiente contenido XML y genera una descripción en texto plano.

Contenido XML:
{content}

El resumen debe ser completo, directo y capturar el propósito y los detalles clave del documento (entidades, fechas, montos, estructura).

IMPORTANTE:
- Responde ÚNICAMENTE con texto plano, sin formato JSON ni otro formato que no sea texto plano
- NO uses comillas, llaves, corchetes, saltos de línea, ni ningún formato estructurado
- NO incluyas etiquetas como "description:", "resumen:" o similares
- Responde directamente con el texto de la descripción
- La descripción debe capturar en no más de 250 palabras los conceptos más importantes para luego poder ser utilizada en un sistema de búsqueda semántica

{normalize_names_instruction}
{normalize_terms_instruction}

Responde en {language_name}."""
        elif content_type == "eml":
            prompt = f"""Analiza el siguiente email y genera una descripción en texto plano.

Email:
{content}

El resumen debe ser completo, directo y capturar el propósito del email, asunto, remitente, destinatario y contenido principal.

IMPORTANTE:
- Responde ÚNICAMENTE con texto plano, sin formato JSON ni otro formato que no sea texto plano
- NO uses comillas, llaves, corchetes, saltos de línea, ni ningún formato estructurado
- NO incluyas etiquetas como "description:", "resumen:" o similares
- Responde directamente con el texto de la descripción
- La descripción debe capturar en no más de 250 palabras los conceptos más importantes para luego poder ser utilizada en un sistema de búsqueda semántica

{normalize_names_instruction}
{normalize_terms_instruction}

Responde en {language_name}."""
        else:
            raise ValueError(f"Tipo de contenido no soportado: {content_type}")
        
        return prompt
    
    def _get_title_prompt(self, description: str, content_type: str, language: str = "es") -> str:
        """
        Genera un prompt unificado para obtener títulos de documentos (ZIP, XML, EML)
        
        Args:
            description: Descripción del documento sobre la cual generar el título
            content_type: Tipo de contenido ("zip", "xml", o "eml")
            language: Idioma para la respuesta (código: "es", "en", etc.)
            
        Returns:
            Prompt formateado para el LLM
        """
        # Convertir código de idioma a nombre completo
        language_names = {
            "es": "español",
            "en": "inglés",
            "fr": "francés",
            "de": "alemán",
            "it": "italiano",
            "pt": "portugués"
        }
        language_name = language_names.get(language.lower(), "español")
        # Partes comunes a todos los tipos
        common_rules = """REGLAS ESTRICTAS:
- El título DEBE ser representativo del contenido, autocontenido y descriptivo del significado y propósito. Máximo 15-20 palabras
- El título debe resumir de forma concisa la esencia del documento, centrándose en el significado y contenido
- NO incluyas: montos, fechas específicas, ubicaciones detalladas, programas de financiación, ni información secundaria
- Responde ÚNICAMENTE con el título en texto plano, sin formato JSON, sin comillas, sin etiquetas, sin puntos finales
- Solo el título, nada más
"""
        
        if content_type == "zip":
            prompt = f"""Basándote en la siguiente descripción de una COLECCIÓN/CONJUNTO de documentos contenidos en un archivo ZIP, genera un título representativo que identifique la colección completa.

Descripción de la colección:
{description}

{common_rules}
- El título DEBE indicar que es una COLECCIÓN/CONJUNTO de documentos (ej: "Colección", "Documentos", "Archivo", o similar)
- DEBE incluir: el tipo/tema común de los documentos (si hay uno)
- Si los documentos son de diferentes tipos/temas, el título debe ser más genérico y representativo del conjunto
- El título debe resumir la esencia y propósito común de la colección

Ejemplos de buenos títulos para colecciones:
- "Colección Documentos Administrativos 2025"
- "Archivo Facturas y Contratos Comerciales"
- "Documentos Proyecto Transformación Digital"
- "Colección Asesoramiento y Consultoría"

Ejemplos de títulos MALOS (específicos de un solo documento, no de la colección):
- "Asesoramiento Transformación Digital" (solo describe uno de los documentos)
- "Factura Proforma A1263-25" (solo describe un documento)
- "Correo Electrónico" (solo describe un documento)

INSTRUCCIONES CRÍTICAS:
- NO escribas tu razonamiento, NO expliques cómo llegaste al título
- NO uses frases como "Voy a analizar..." o "Necesito pensar..." o "Déjame pensar..."
- NO muestres tu proceso de pensamiento o chain of thought
- Responde DIRECTAMENTE con el título, sin explicaciones ni razonamiento
- Escribe ÚNICAMENTE el título dentro de las etiquetas <answer></answer>
- NO escribas nada fuera de las etiquetas <answer></answer>

Responde en {language_name}.

<answer>"""
        elif content_type == "xml" or content_type == "eml":
            # XML y EML comparten el mismo formato de título
            prompt = f"""Basándote en la siguiente descripción, genera un título representativo que identifique el documento.

Descripción:
{description}

{common_rules}
- El título debe resumir la esencia y propósito del documento
- Debe ser autocontenido y descriptivo del significado y contenido
- Enfócate en el tipo de documento, servicio o concepto principal

Ejemplo de buen título: "Asesoramiento Transformación Digital"
Ejemplo de título MALO (demasiado largo o específico): "Transacción entre entidades en Gijón, Asturias: Asesoramiento 360 en Transformación digital, 7260 EUR..."

INSTRUCCIONES CRÍTICAS:
- NO escribas tu razonamiento, NO expliques cómo llegaste al título
- NO uses frases como "Voy a analizar..." o "Necesito pensar..." o "Déjame pensar..."
- NO muestres tu proceso de pensamiento o chain of thought
- Responde DIRECTAMENTE con el título, sin explicaciones ni razonamiento
- Escribe ÚNICAMENTE el título dentro de las etiquetas <answer></answer>
- NO escribas nada fuera de las etiquetas <answer></answer>

Responde en {language_name}.

<answer>"""
        else:
            raise ValueError(f"Tipo de contenido no soportado para título: {content_type}")
        
        return prompt
    
    def _is_error_description(self, description: str) -> bool:
        """Verifica si una descripción indica un error"""
        if not description:
            return True
        
        error_indicators = [
            "Error:",
            "error:",
            "Error al procesar",
            "Error procesando",
            "Error descargando",
            "Error generando",
            "El modelo no devolvió contenido",
            "no devolvió contenido",
            "failed",
            "Failed",
            "FAILED"
        ]
        
        description_lower = description.lower()
        return any(indicator.lower() in description_lower for indicator in error_indicators)
    
    def _clean_description(self, text: str) -> str:
        """
        Limpia agresivamente una descripción eliminando TODAS las comillas y backslashes
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio sin comillas ni backslashes
        """
        if not text:
            return ""
        
        # Primero limpiar escapes
        text = text.replace('\\"', '"')  # Escapes de comillas dobles
        text = text.replace("\\'", "'")  # Escapes de comillas simples
        text = text.replace('\\\\', '')   # Backslashes dobles
        text = text.replace('\\n', ' ')   # Saltos de línea escapados -> espacio
        text = text.replace('\\t', ' ')   # Tabs escapados -> espacio
        text = text.replace('\\r', ' ')   # Retornos de carro escapados -> espacio
        
        # Eliminar TODAS las comillas (simples y dobles)
        text = text.replace('"', '')
        text = text.replace("'", '')
        
        # Eliminar cualquier backslash restante
        text = text.replace('\\', '')
        
        return text.strip()
        
    def _extract_description(self, response_content: str, fallback_msg: str = "Resumen no disponible") -> str:
        """Extrae la descripción de un JSON de forma robusta, manejando markdown y texto extra"""
        if not response_content:
            return fallback_msg

        clean_content = response_content.strip()
        
        # 1. Intentar limpiar bloques de código Markdown (```json ... ```)
        import re
        if "```" in clean_content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean_content)
            if match:
                clean_content = match.group(1).strip()
        
        # 2. Intentar buscar el primer '{' y el último '}' por si hay texto alrededor
        start_idx = clean_content.find('{')
        end_idx = clean_content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = clean_content[start_idx:end_idx+1]
            try:
                data = json.loads(json_str)
                # Lista de claves posibles (insensible a mayúsculas/minúsculas)
                keys_to_try = ["description", "descripcion", "macro-description", "macro-descripcion", "summary", "resumen"]
                
                # Normalizar claves del diccionario para búsqueda insensible
                data_lower = {k.lower(): v for k, v in data.items()}
                
                for key in keys_to_try:
                    if key in data_lower:
                        result = str(data_lower[key]).strip()
                        # Limpiar escapes de comillas y backslashes
                        result = result.replace('\\"', '"').replace("\\'", "'")
                        result = result.replace('\\\\', '')
                        return result
                
                # Si no encontramos las claves, coger el valor string más largo del objeto
                str_values = [str(v) for v in data.values() if isinstance(v, (str, dict, list))]
                if str_values:
                    result = max(str_values, key=len).strip()
                    # Limpiar escapes
                    result = result.replace('\\"', '"').replace("\\'", "'")
                    result = result.replace('\\\\', '')
                    return result
            except json.JSONDecodeError:
                # Si no es JSON válido tras el recorte, seguimos al paso 3
                pass

        # 3. Si no se pudo parsear como JSON, devolver el contenido limpio
        # Pero si parece JSON serializado (contiene {"description":), intentamos limpiar solo eso
        if '"description":' in clean_content or '"descripcion":' in clean_content:
            # Fallback simple: extraer solo lo que esté entre las segundas comillas tras la clave
            simple_match = re.search(r'"descrip(?:tion|cion)"\s*:\s*"([^"]*)"', clean_content, re.IGNORECASE)
            if simple_match:
                extracted = simple_match.group(1).strip()
                # Limpiar escapes
                extracted = extracted.replace('\\"', '"').replace("\\'", "'")
                extracted = extracted.replace('\\\\', '')
                return extracted

        # Limpiar escapes en el contenido final
        clean_content = clean_content.replace('\\"', '"').replace("\\'", "'")
        clean_content = clean_content.replace('\\\\', '')

        return clean_content.strip()

    def process_pdf(self, pdf_path: str, language: str = "es", initial_pages: int = 2, final_pages: int = 2, max_tokens: int = 1024, temperature_vllm: float = 0.1, top_p: float = 0.9) -> Dict[str, Any]:
        """Procesa un PDF y genera su resumen"""
        logger.info(f"Starting PDF processing: {os.path.basename(pdf_path)} (Language: {language})")
        
        # Verificar si el archivo está vacío
        try:
            if os.path.getsize(pdf_path) == 0:
                logger.warning(f"Archivo PDF vacío ignorado: {os.path.basename(pdf_path)}")
                return None  # Retornar None para indicar que debe ser ignorado
        except OSError as e:
            logger.warning(f"No se pudo verificar tamaño del archivo {pdf_path}: {e}")
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Convertir PDF a imágenes
            logger.info("Converting PDF to images...")
            try:
                images = self.pdf_processor.convert_to_images(pdf_path, temp_dir, initial_pages, final_pages)
            except Exception as e:
                error_msg = str(e).lower()
                # Detectar PDFs corruptos o truncados
                if 'truncated' in error_msg or 'corrupt' in error_msg or 'image file is truncated' in error_msg:
                    logger.warning(f"PDF corrupto/truncado detectado: {os.path.basename(pdf_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                else:
                    logger.error(f"Error al convertir PDF a imágenes: {e}")
                    return {
                        "title": os.path.basename(pdf_path),
                        "description": f"Error: No se pudieron extraer imágenes del PDF: {str(e)}",
                        "metadata": {"error": True}
                    }
            
            if not images:
                logger.error("Failed to extract images from PDF")
                return {
                    "title": os.path.basename(pdf_path),
                    "description": "Error: No se pudieron extraer imágenes del PDF",
                    "metadata": {"error": True}
                }
            
            logger.info(f"Extracted {len(images)} images. Preparing model prompt.")
            
            # Obtener prompt y schema unificados
            prompt, schema = self._get_vllm_prompt_and_schema(language)
            
            # Analizar con LLM multimodal usando Structured Outputs
            logger.info("Calling Multimodal Service...")
            with self.inference_semaphore:
                response_content = self.vllm_service.analyze_vllm(images, prompt, max_tokens, schema, temperature_vllm, top_p)

            # Extraer title y description del JSON
            try:
                import json
                response_json = json.loads(response_content)
                title = response_json.get("title", "").strip()
                description = response_json.get("description", "").strip()

                # Fallback si no se obtienen correctamente
                if not title:
                    title = os.path.basename(pdf_path)
                if not description:
                    description = self._extract_description(response_content) or "Sin descripción disponible"
            except Exception as e:
                logger.warning(f"Error parsing structured output: {e}. Falling back to description extraction.")
                description = self._extract_description(response_content) or "Sin descripción disponible"
                title = os.path.basename(pdf_path)
            
            # Asegurar que siempre haya título y descripción
            if not title:
                title = os.path.basename(pdf_path)
            if not description:
                description = "Sin descripción disponible"
            
            # Limpiar descripción: eliminar comillas y backslashes (SIEMPRE, sin importar de dónde venga)
            description = self._clean_description(description)
            
            logger.info("Response parsed successfully")

            return {
                "title": title,
                "description": description,
                "metadata": {
                    "pages_processed": len(images),
                    "language": language
                }
            }
        finally:
            # Limpiar archivos temporales
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_docx(self, docx_path: str, language: str = "es", initial_pages: int = 2, final_pages: int = 2, max_tokens: int = 1024, temperature_vllm: float = 0.1, top_p: float = 0.9) -> Dict[str, Any]:
        """Procesa un DOCX/DOC/ODT y genera su resumen (igual que PDFs)"""
        file_ext = os.path.splitext(docx_path)[1].lower()
        file_type_name = {"docx": "DOCX", "doc": "DOC", "odt": "ODT"}.get(file_ext[1:], "DOCUMENTO")
        logger.info(f"Starting {file_type_name} processing: {os.path.basename(docx_path)} (Language: {language})")
        
        # Verificar si el archivo está vacío
        try:
            if os.path.getsize(docx_path) == 0:
                logger.warning(f"Archivo {file_type_name} vacío ignorado: {os.path.basename(docx_path)}")
                return None  # Retornar None para indicar que debe ser ignorado
        except OSError as e:
            logger.warning(f"No se pudo verificar tamaño del archivo {docx_path}: {e}")
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Convertir DOCX/DOC/ODT a imágenes (primero convierte a PDF, luego a imágenes)
            logger.info(f"Converting {file_type_name} to images...")
            images = self.docx_processor.convert_to_images(docx_path, temp_dir, initial_pages, final_pages)
            
            if not images:
                logger.error(f"Failed to extract images from {file_type_name}")
                return {
                    "title": os.path.basename(docx_path),
                    "description": f"Error: No se pudieron extraer imágenes del {file_type_name}",
                    "metadata": {}
                }
            
            logger.info(f"Extracted {len(images)} images. Preparing model prompt.")
            
            # Obtener prompt y schema unificados
            prompt, schema = self._get_vllm_prompt_and_schema(language)
            
            # Analizar con LLM multimodal usando Structured Outputs
            logger.info("Calling Multimodal Service...")
            with self.inference_semaphore:
                response_content = self.vllm_service.analyze_vllm(images, prompt, max_tokens, schema, temperature_vllm, top_p)

            # Extraer title y description del JSON
            try:
                import json
                response_json = json.loads(response_content)
                title = response_json.get("title", "").strip()
                description = response_json.get("description", "").strip()

                # Fallback si no se obtienen correctamente
                if not title:
                    title = os.path.basename(docx_path)
                if not description:
                    description = self._extract_description(response_content) or "Sin descripción disponible"
            except Exception as e:
                logger.warning(f"Error parsing structured output: {e}. Falling back to description extraction.")
                description = self._extract_description(response_content) or "Sin descripción disponible"
                title = os.path.basename(docx_path)
            
            # Asegurar que siempre haya título y descripción
            if not title:
                title = os.path.basename(docx_path)
            if not description:
                description = "Sin descripción disponible"
            
            # Limpiar descripción: eliminar comillas y backslashes (SIEMPRE, sin importar de dónde venga)
            description = self._clean_description(description)
            
            logger.info("Response parsed successfully")

            return {
                "title": title,
                "description": description,
                "metadata": {
                    "pages_processed": len(images),
                    "language": language
                }
            }
        finally:
            # Limpiar archivos temporales
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_archive(self, archive_path: str, extracted_dir: str) -> None:
        """
        Extrae un archivo comprimido (ZIP, RAR, 7Z, TAR) al directorio especificado.
        
        Args:
            archive_path: Ruta al archivo comprimido
            extracted_dir: Directorio donde extraer los archivos
            
        Raises:
            ValueError: Si el formato del archivo no es soportado
            Exception: Si hay un error al extraer el archivo
        """
        archive_name = os.path.basename(archive_path).lower()
        
        if archive_name.endswith('.zip'):
            logger.info("Extracting ZIP file...")
            zip_basename = os.path.basename(archive_path)
            extraction_successful = False

            # Lista de codificaciones a intentar en orden de preferencia:
            # 1. UTF-8: estándar moderno
            # 2. CP1252: Windows-1252, común en ZIPs creados en Windows con caracteres españoles/europeos
            # 3. CP437: DOS, común en ZIPs antiguos
            encodings_to_try = [
                ('utf-8', 'UTF-8'),
                ('cp1252', 'CP1252 (Windows)'),
                ('cp437', 'CP437 (DOS)'),
            ]

            last_error = None
            failed_encodings = []
            for encoding, encoding_name in encodings_to_try:
                try:
                    with zipfile.ZipFile(archive_path, 'r', metadata_encoding=encoding) as zip_ref:
                        zip_ref.extractall(extracted_dir)
                    if encoding != 'utf-8':
                        logger.info(f"ZIP {zip_basename}: extracción exitosa usando codificación {encoding_name}")
                    extraction_successful = True
                    break
                except (UnicodeDecodeError, ValueError) as e:
                    error_msg = str(e).lower()
                    is_encoding_error = (
                        isinstance(e, UnicodeDecodeError) or
                        'file name in directory' in error_msg or
                        'header' in error_msg or
                        'differ' in error_msg or
                        "can't decode" in error_msg or
                        'codec' in error_msg
                    )
                    if is_encoding_error:
                        last_error = e
                        failed_encodings.append(encoding_name)
                        logger.debug(f"ZIP {zip_basename}: codificación {encoding_name} falló: {e}")
                        continue  # Intentar siguiente codificación
                    else:
                        raise  # Error no relacionado con codificación

            if failed_encodings and extraction_successful:
                logger.info(f"ZIP {zip_basename}: codificaciones fallidas [{', '.join(failed_encodings)}] antes de éxito")

            if not extraction_successful:
                # Fallback final: extraer archivo por archivo sin especificar codificación
                logger.warning(f"ZIP {zip_basename}: ninguna codificación funcionó ({last_error}). Intentando extracción archivo por archivo...")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    extracted_count = 0
                    for member in zip_ref.infolist():
                        try:
                            zip_ref.extract(member, extracted_dir)
                            extracted_count += 1
                        except Exception as member_error:
                            logger.warning(f"Error extrayendo {member.filename}: {member_error}. Continuando...")
                            continue
                    if extracted_count == 0:
                        raise Exception("No se pudo extraer ningún archivo del ZIP")
                    logger.info(f"Extraídos {extracted_count} archivo(s) del ZIP con extracción manual")
        elif archive_name.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
            logger.info("Extracting TAR file...")
            # Determinar el modo de apertura según la extensión
            if archive_name.endswith('.tar.gz') or archive_name.endswith('.tgz'):
                mode = 'r:gz'
            elif archive_name.endswith('.tar.bz2') or archive_name.endswith('.tbz2'):
                mode = 'r:bz2'
            elif archive_name.endswith('.tar.xz'):
                mode = 'r:xz'
            else:
                mode = 'r'
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extracted_dir)
        elif archive_name.endswith(('.rar', '.cbr')):
            logger.info("Extracting RAR file...")
            try:
                import rarfile
            except ImportError:
                raise ImportError("rarfile no está instalado. Instálalo con: pip install rarfile")
            try:
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(extracted_dir)
            except rarfile.RarCannotExec as e:
                raise ImportError(
                    "No se encontró el binario 'unrar' necesario para extraer archivos RAR. "
                    "En sistemas Debian/Ubuntu, instálalo con: apt-get install unrar"
                ) from e
        elif archive_name.endswith('.7z'):
            logger.info("Extracting 7Z file...")
            try:
                import py7zr
            except ImportError:
                raise ImportError("py7zr no está instalado. Instálalo con: pip install py7zr")
            with py7zr.SevenZipFile(archive_path, mode='r') as zip7_ref:
                zip7_ref.extractall(extracted_dir)
        else:
            raise ValueError(f"Formato de archivo no soportado: {archive_name}")

    def process_archive(self, archive_path: str, language: str = "es", initial_pages: int = 2, final_pages: int = 2, max_tokens: int = 1024, temperature_vllm: float = 0.1, temperature_llm: float = 0.3, top_p: float = 0.9, max_inner_files: int = 0) -> Dict[str, Any]:
        """
        Procesa un archivo comprimido (ZIP, RAR, 7Z, TAR), extrae PDFs/DOCX/XML/EML y genera resúmenes.
        Esta función es genérica y funciona para ZIP, RAR, 7Z y TAR.
        
        Para archivos RAR problemáticos, si falla el procesamiento normal, intenta un fallback:
        extrae el RAR, lo comprime como ZIP y procesa el ZIP.
        """
        archive_name = os.path.basename(archive_path)
        archive_type = "ZIP" if archive_path.lower().endswith('.zip') else "RAR" if archive_path.lower().endswith(('.rar', '.cbr')) else "7Z" if archive_path.lower().endswith('.7z') else "TAR"
        is_rar = archive_path.lower().endswith(('.rar', '.cbr'))
        
        logger.info(f"Starting {archive_type} processing: {archive_name}")
        temp_dir = tempfile.mkdtemp()
        extracted_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        
        children_results = []
        
        try:
            # Extraer archivo comprimido
            # Si es RAR y falla la extracción, intentar fallback inmediatamente
            try:
                self._extract_archive(archive_path, extracted_dir)
            except Exception as extract_error:
                if is_rar:
                    logger.warning(f"Error extrayendo RAR {archive_name}: {extract_error}. Intentando fallback: extraer, comprimir como ZIP y procesar...")
                    # Intentar fallback: extraer RAR manualmente y comprimir como ZIP
                    try:
                        # Limpiar el directorio extraído anterior (si existe)
                        if os.path.exists(extracted_dir):
                            shutil.rmtree(extracted_dir, ignore_errors=True)
                        os.makedirs(extracted_dir, exist_ok=True)
                        
                        # Intentar extraer el RAR usando rarfile directamente con manejo de errores
                        import rarfile
                        try:
                            # Intentar abrir el RAR con manejo de errores
                            try:
                                rar_ref = rarfile.RarFile(archive_path, 'r')
                            except Exception as open_error:
                                logger.error(f"No se pudo abrir el RAR: {open_error}")
                                raise extract_error from open_error
                            
                            try:
                                # Intentar extraer con diferentes métodos si falla
                                try:
                                    rar_ref.extractall(extracted_dir)
                                    logger.info("RAR extraído exitosamente con extractall")
                                except Exception as extract_error2:
                                    logger.warning(f"Extractall falló ({extract_error2}), intentando extraer archivo por archivo...")
                                    # Intentar extraer archivo por archivo
                                    extracted_count = 0
                                    for member in rar_ref.infolist():
                                        try:
                                            # Crear directorios necesarios
                                            member_path = os.path.join(extracted_dir, member.filename)
                                            member_dir = os.path.dirname(member_path)
                                            if member_dir and not os.path.exists(member_dir):
                                                os.makedirs(member_dir, exist_ok=True)
                                            
                                            rar_ref.extract(member, extracted_dir)
                                            extracted_count += 1
                                        except Exception as member_error:
                                            error_msg = str(member_error).lower()
                                            # Si es un error de "read enough data", continuar con el siguiente archivo
                                            if 'read enough data' in error_msg or 'failed' in error_msg:
                                                logger.warning(f"Error leyendo {member.filename} (posible archivo corrupto): {member_error}. Continuando...")
                                            else:
                                                logger.warning(f"Error extrayendo {member.filename}: {member_error}. Continuando...")
                                            continue
                                    
                                    if extracted_count == 0:
                                        logger.error("No se pudo extraer ningún archivo del RAR")
                                        raise extract_error
                                    else:
                                        logger.info(f"Extraídos {extracted_count} archivos del RAR (algunos pueden haber fallado)")
                            finally:
                                rar_ref.close()
                                
                        except Exception as rar_error:
                            logger.error(f"No se pudo extraer el RAR: {rar_error}")
                            raise extract_error from rar_error
                        
                        # Crear un ZIP temporal con el contenido extraído
                        zip_fallback_path = os.path.join(temp_dir, f"{os.path.splitext(archive_name)[0]}_fallback.zip")
                        logger.info(f"Comprimiendo contenido extraído del RAR como ZIP: {zip_fallback_path}")
                        
                        with zipfile.ZipFile(zip_fallback_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                            for root, dirs, files in os.walk(extracted_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    # Calcular el path relativo para mantener la estructura
                                    arcname = os.path.relpath(file_path, extracted_dir)
                                    try:
                                        zip_ref.write(file_path, arcname)
                                    except Exception as zip_write_error:
                                        logger.warning(f"Error añadiendo {file_path} al ZIP: {zip_write_error}. Continuando...")
                                        continue
                        
                        logger.info(f"ZIP de fallback creado. Procesando como ZIP...")
                        # Procesar el ZIP de fallback recursivamente
                        return self.process_archive(zip_fallback_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, temperature_llm, top_p, max_inner_files)
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback RAR->ZIP también falló: {fallback_error}")
                        # Si el fallback también falla, lanzar el error original de extracción
                        raise extract_error from fallback_error
                else:
                    # Si no es RAR, lanzar el error normalmente
                    raise
            
            # Buscar todos los archivos soportados recursivamente (PDF, DOCX, DOC, ODT, XML, EML, imágenes)
            # También detectar archivos comprimidos anidados para procesarlos recursivamente
            pdf_files = []
            docx_files = []
            xml_files = []
            eml_files = []
            image_files = []
            nested_archives = []  # Archivos comprimidos anidados

            for root, dirs, files in os.walk(extracted_dir):
                # Excluir directorios de metadatos de macOS (__MACOSX)
                dirs[:] = [d for d in dirs if d != '__MACOSX']

                for file in files:
                    # Excluir archivos de metadatos de macOS (resource forks con prefijo ._)
                    if file.startswith('._'):
                        logger.debug(f"Archivo de metadatos macOS ignorado: {file}")
                        continue

                    file_path = os.path.join(root, file)
                    # Excluir explícitamente archivos .xsig
                    if file.lower().endswith('.xsig'):
                        logger.info(f"Archivo .xsig ignorado dentro de {archive_type} (no soportado): {file}")
                        continue  # Saltar archivos .xsig
                    elif file.lower().endswith('.pdf'):
                        pdf_files.append(file_path)
                    elif file.lower().endswith(('.docx', '.doc', '.odt')):
                        docx_files.append(file_path)
                    elif file.lower().endswith('.xml'):
                        xml_files.append(file_path)
                    elif file.lower().endswith('.eml'):
                        eml_files.append(file_path)
                    elif file.lower().endswith(self.IMAGE_EXTENSIONS):
                        image_files.append(file_path)
                    # Detectar archivos comprimidos anidados (ZIP, RAR, 7Z, TAR dentro del archivo comprimido principal)
                    elif file.lower().endswith(('.zip', '.rar', '.cbr', '.7z', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
                        nested_archives.append(file_path)
                        logger.info(f"Archivo comprimido anidado detectado: {file_path}")

            total_files = len(pdf_files) + len(docx_files) + len(xml_files) + len(eml_files) + len(image_files)
            logger.info(f"Found {len(pdf_files)} PDF, {len(docx_files)} DOCX/DOC/ODT, {len(xml_files)} XML, {len(eml_files)} EML, and {len(image_files)} image files in {archive_type} (total: {total_files})")

            # Preparar lista unificada de archivos a procesar con su tipo
            all_inner_files = []
            for f in pdf_files:
                all_inner_files.append(('pdf', f))
            for f in docx_files:
                all_inner_files.append(('docx', f))
            for f in xml_files:
                all_inner_files.append(('xml', f))
            for f in eml_files:
                all_inner_files.append(('eml', f))
            for f in image_files:
                all_inner_files.append(('image', f))

            # Apply ARCHIVE_MAX_FILES limit: read from env if not explicitly provided
            if max_inner_files <= 0:
                max_inner_files = int(os.getenv("ARCHIVE_MAX_FILES", "0"))

            total_found = len(all_inner_files) + len(nested_archives)
            skipped_files = 0

            if max_inner_files > 0 and len(all_inner_files) > max_inner_files:
                # Reorder: PDFs first, then DOCX, images, XML, EML
                type_priority = {'pdf': 0, 'docx': 1, 'image': 2, 'xml': 3, 'eml': 4}
                all_inner_files.sort(key=lambda x: type_priority.get(x[0], 99))
                skipped_files = len(all_inner_files) - max_inner_files
                all_inner_files = all_inner_files[:max_inner_files]
                logger.info(f"ARCHIVE_MAX_FILES={max_inner_files}: processing {len(all_inner_files)} files, skipping {skipped_files}")

            # Configuración de workers para procesamiento paralelo dentro de archivos
            archive_workers = int(os.getenv("ARCHIVE_WORKERS", "4"))
            content_limit = int(os.getenv("XML_EML_CONTENT_LIMIT", "5000"))

            def process_inner_file(file_info: tuple) -> Optional[DocumentResult]:
                """Procesa un archivo individual dentro del archivo comprimido"""
                file_type, file_path = file_info
                relative_path = os.path.relpath(file_path, extracted_dir)
                file_name = os.path.basename(file_path)

                try:
                    result = None

                    if file_type == 'pdf':
                        logger.info(f"Processing inner PDF: {relative_path}")
                        result = self.process_pdf(file_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, top_p)
                        doc_type = "pdf"
                    elif file_type == 'docx':
                        file_ext = os.path.splitext(file_path)[1].lower()
                        file_type_name = {".docx": "DOCX", ".doc": "DOC", ".odt": "ODT"}.get(file_ext, "DOCUMENTO")
                        logger.info(f"Processing inner {file_type_name}: {relative_path}")
                        result = self.process_docx(file_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, top_p)
                        doc_type = "docx"  # Usar "docx" como tipo genérico para Word/ODT
                    elif file_type == 'xml':
                        logger.info(f"Processing inner XML: {relative_path}")
                        result = self.process_xml(file_path, language, max_tokens, temperature_llm, top_p, content_limit)
                        doc_type = "xml"
                    elif file_type == 'eml':
                        logger.info(f"Processing inner EML: {relative_path}")
                        result = self.process_eml(file_path, language, max_tokens, temperature_llm, top_p, content_limit)
                        doc_type = "eml"
                    elif file_type == 'image':
                        logger.info(f"Processing inner image: {relative_path}")
                        result = self.process_image(file_path, language, max_tokens, temperature_vllm, top_p)
                        doc_type = "image"
                    else:
                        logger.warning(f"Unknown file type {file_type} for {relative_path}")
                        return None

                    # Si el archivo está vacío, result será None y lo ignoramos
                    if result is None:
                        logger.info(f"Archivo {file_type} vacío ignorado: {relative_path}")
                        return None

                    # Asegurar que title y description siempre estén presentes
                    title = result.get("title") or file_name
                    description = result.get("description") or "Sin descripción disponible"

                    # Para EML: si el título contiene un error, usar el nombre del archivo
                    if file_type == 'eml' and title and ("Error" in title or "error" in title.lower() or "no devolvió contenido" in title.lower()):
                        title = file_name

                    description = self._clean_description(description)

                    return DocumentResult(
                        name=file_name,
                        title=title,
                        description=description,
                        type=doc_type,
                        path=relative_path,
                        metadata=result.get("metadata", {})
                    )

                except Exception as e:
                    logger.error(f"Error processing inner {file_type} {file_path}: {e}")
                    return DocumentResult(
                        name=file_name,
                        title=file_name,
                        description=f"Error procesando: {str(e)}",
                        type=file_type,
                        path=relative_path,
                        metadata={"error": True}
                    )

            # Procesar archivos en paralelo si hay más de uno
            if len(all_inner_files) > 1 and archive_workers > 1:
                logger.info(f"Processing {len(all_inner_files)} inner files in parallel with {archive_workers} workers")
                with ThreadPoolExecutor(max_workers=archive_workers) as executor:
                    futures = {executor.submit(process_inner_file, f): f for f in all_inner_files}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            children_results.append(result)
            else:
                # Procesamiento secuencial para un solo archivo o si ARCHIVE_WORKERS=1
                for file_info in all_inner_files:
                    result = process_inner_file(file_info)
                    if result:
                        children_results.append(result)

            # Procesar archivos comprimidos anidados recursivamente
            if nested_archives:
                # Compute remaining budget for nested archives
                remaining_budget = max_inner_files - len(all_inner_files) if max_inner_files > 0 else 0

                logger.info(f"Found {len(nested_archives)} nested archive(s). Processing recursively...")
                for nested_archive_path in nested_archives:
                    # Skip nested archives if budget exhausted
                    if max_inner_files > 0 and remaining_budget <= 0:
                        skipped_files += 1
                        logger.info(f"ARCHIVE_MAX_FILES budget exhausted, skipping nested archive: {os.path.basename(nested_archive_path)}")
                        continue

                    relative_path = os.path.relpath(nested_archive_path, extracted_dir)
                    logger.info(f"Processing nested archive: {relative_path}")
                    try:
                        # Procesar el archivo comprimido anidado recursivamente
                        nested_result = self.process_archive(
                            nested_archive_path,
                            language,
                            initial_pages,
                            final_pages,
                            max_tokens,
                            temperature_vllm,
                            temperature_llm,
                            top_p,
                            max_inner_files=remaining_budget
                        )
                        
                        if nested_result and nested_result.get("children"):
                            # Añadir los children del archivo comprimido anidado como children del archivo principal
                            nested_archive_name = os.path.basename(nested_archive_path)
                            for child in nested_result["children"]:
                                # Construir path que refleje la estructura anidada: nested_archive/child_path
                                if child.path:
                                    nested_child_path = f"{relative_path}/{child.path}"
                                else:
                                    nested_child_path = f"{relative_path}/{child.name}"
                                
                                children_results.append(DocumentResult(
                                    name=child.name,
                                    title=child.title,
                                    description=child.description,
                                    type=child.type,
                                    path=nested_child_path,
                                    metadata=child.metadata
                                ))
                            logger.info(f"Added {len(nested_result['children'])} documents from nested archive {nested_archive_name}")
                            # Decrement remaining budget
                            if max_inner_files > 0:
                                remaining_budget -= len(nested_result["children"])
                        else:
                            # Si el archivo comprimido anidado no tiene children, añadir un resultado indicando que está vacío
                            logger.warning(f"Nested archive {relative_path} produced no documents")
                            children_results.append(DocumentResult(
                                name=os.path.basename(nested_archive_path),
                                title=os.path.basename(nested_archive_path),
                                description=f"Archivo comprimido anidado procesado pero no se encontraron documentos soportados dentro.",
                                type="zip",  # Tipo genérico para archivos comprimidos
                                path=relative_path,
                                metadata={"nested_archive": True, "empty": True}
                            ))
                    except Exception as e:
                        logger.error(f"Error processing nested archive {nested_archive_path}: {e}")
                        # Añadir un resultado de error para el archivo comprimido anidado
                        children_results.append(DocumentResult(
                            name=os.path.basename(nested_archive_path),
                            title=os.path.basename(nested_archive_path),
                            description=f"Error procesando archivo comprimido anidado: {str(e)}",
                            type="zip",  # Tipo genérico para archivos comprimidos
                            path=relative_path,
                            metadata={"error": True, "nested_archive": True}
                        ))

            # Generar resumen agregado inteligente
            total_docs = len(children_results)
            logger.info(f"{archive_type} processing complete. {total_docs} documents processed ({len(pdf_files)} PDFs, {len(docx_files)} DOCX/DOC/ODT, {len(xml_files)} XMLs, {len(eml_files)} EMLs, {len(image_files)} images, {len(nested_archives)} nested archives). Generating macro-summary.")
            
            if total_docs > 0:
                # Construir contexto para macro-resumen
                descriptions_text = "\n".join([f"- {r.name}: {r.description}" for r in children_results])
                
                # Primera llamada: obtener descripción usando prompt unificado
                macro_prompt = self._get_description_prompt(descriptions_text, "zip", language)

                try:
                    logger.info(f"Calling LLM Service for {archive_type} macro-summary (description)...")
                    with self.inference_semaphore:
                        macro_description_raw = self.llm_service.analyze_llm(
                            prompt=macro_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature_llm,
                            top_p=top_p
                        )

                    # Asegurar que es texto plano (ya viene limpio de analyze_llm, pero por si acaso)
                    macro_description = self.llm_service._clean_plain_text_response(macro_description_raw)
                    macro_description = self._clean_description(macro_description)  # Limpiar comillas y backslashes

                    logger.info(f"Macro-description generado: {len(macro_description)} caracteres")

                    # Pequeño delay para evitar rate limiting entre llamadas secuenciales
                    time.sleep(0.5)

                    # Segunda llamada: obtener título basado en la descripción usando prompt unificado
                    title_prompt = self._get_title_prompt(macro_description, "zip", language)

                    logger.info(f"Calling LLM Service for {archive_type} title...")
                    with self.inference_semaphore:
                        macro_title_raw = self.llm_service.analyze_llm(
                            prompt=title_prompt,
                            max_tokens=512,  # Títulos cortos, no necesitamos muchos tokens, pero si pedimos muy pocos, a veces falla el modelo
                            temperature=temperature_llm,
                            top_p=top_p
                        )

                    macro_title = self.llm_service._clean_plain_text_response(macro_title_raw).strip()
                    
                    # Si el título está vacío o contiene un mensaje de error, usar el nombre del archivo
                    if not macro_title or "Error" in macro_title or "error" in macro_title.lower() or "no devolvió contenido" in macro_title.lower():
                        macro_title = archive_name
                        logger.warning(f"LLM returned empty or error content for {archive_type} title. Using filename: {macro_title}")
                    else:
                        logger.info(f"Macro-title generado: {macro_title}")
                    
                except Exception as e:
                    logger.error(f"Error generating macro-summary: {e}")
                    macro_description = f"Colección de {total_docs} documento(s). (Error generando resumen automático)"
                    macro_title = archive_name  # Usar nombre del archivo como título
            else:
                macro_description = f"{archive_type} procesado pero no se encontraron documentos soportados (PDF, XML, EML) dentro."
                macro_title = archive_name  # Usar nombre del archivo como título
            
            metadata = {
                    "total_documents": total_docs,
                    "total_pdfs": len(pdf_files),
                    "total_docx": len(docx_files),
                    "total_xmls": len(xml_files),
                    "total_emls": len(eml_files),
                    "total_images": len(image_files),
                    "language": language
                }
            if skipped_files > 0:
                metadata["files_found"] = total_found
                metadata["files_skipped"] = skipped_files

            return {
                "title": macro_title,
                "description": macro_description,
                "children": children_results,
                "metadata": metadata
            }
        except Exception as e:
            # Si es un RAR y falla, intentar fallback: extraer, comprimir como ZIP y procesar
            if is_rar:
                logger.warning(f"Error procesando RAR {archive_name}: {e}. Intentando fallback: extraer, comprimir como ZIP y procesar...")
                try:
                    # Limpiar el directorio extraído anterior (si existe)
                    if os.path.exists(extracted_dir):
                        shutil.rmtree(extracted_dir, ignore_errors=True)
                    os.makedirs(extracted_dir, exist_ok=True)
                    
                    # Intentar extraer el RAR de nuevo
                    self._extract_archive(archive_path, extracted_dir)
                    
                    # Crear un ZIP temporal con el contenido extraído
                    zip_fallback_path = os.path.join(temp_dir, f"{os.path.splitext(archive_name)[0]}_fallback.zip")
                    logger.info(f"Comprimiendo contenido extraído del RAR como ZIP: {zip_fallback_path}")
                    
                    with zipfile.ZipFile(zip_fallback_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        for root, dirs, files in os.walk(extracted_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # Calcular el path relativo para mantener la estructura
                                arcname = os.path.relpath(file_path, extracted_dir)
                                zip_ref.write(file_path, arcname)
                    
                    logger.info(f"ZIP de fallback creado. Procesando como ZIP...")
                    # Procesar el ZIP de fallback recursivamente
                    return self.process_archive(zip_fallback_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, temperature_llm, top_p, max_inner_files)
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback RAR->ZIP también falló: {fallback_error}")
                    # Si el fallback también falla, lanzar el error original
                    raise e from fallback_error
            else:
                # Si no es RAR, lanzar el error normalmente
                raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_zip(self, zip_path: str, language: str = "es", initial_pages: int = 2, final_pages: int = 2, max_tokens: int = 1024, temperature_vllm: float = 0.1, temperature_llm: float = 0.3, top_p: float = 0.9, max_inner_files: int = 0) -> Dict[str, Any]:
        """
        Procesa un ZIP. Esta función es un alias de process_archive para mantener compatibilidad.
        """
        return self.process_archive(zip_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, temperature_llm, top_p, max_inner_files)
    
    def process_xml(self, xml_path: str, language: str = "es", max_tokens: int = 1024, temperature_llm: float = 0.3, top_p: float = 0.9, content_limit: int = None) -> Dict[str, Any]:
        """Procesa un archivo XML y genera su resumen"""
        logger.info(f"Starting XML processing: {os.path.basename(xml_path)} (Language: {language})")
        
        # Verificar si el archivo está vacío
        try:
            if os.path.getsize(xml_path) == 0:
                logger.warning(f"Archivo XML vacío ignorado: {os.path.basename(xml_path)}")
                return None  # Retornar None para indicar que debe ser ignorado
        except OSError as e:
            logger.warning(f"No se pudo verificar tamaño del archivo {xml_path}: {e}")
        
        # Obtener límite de contenido desde variable de entorno o parámetro
        if content_limit is None:
            content_limit = int(os.getenv("XML_EML_CONTENT_LIMIT", "5000"))
        
        try:
            # Extraer contenido del XML
            xml_content = self.xml_eml_processor.process_xml(xml_path)
            
            if not xml_content:
                logger.error("Failed to extract content from XML")
                return {
                    "title": os.path.basename(xml_path),
                    "description": "Error: No se pudo extraer contenido del XML",
                    "metadata": {}
                }
            
            logger.info(f"Extracted XML content ({len(xml_content)} characters). Preparing model prompt.")
            
            # Primera llamada: obtener descripción usando prompt unificado
            xml_preview = xml_content[:content_limit]  # Limitar tamaño del prompt según configuración
            prompt = self._get_description_prompt(xml_preview, "xml", language)
            
            # Analizar con LLM (text-only)
            logger.info("Calling LLM Service for XML description...")
            with self.inference_semaphore:
                description = self.llm_service.analyze_llm(prompt, max_tokens, temperature=temperature_llm, top_p=top_p)

            # Limpiar la respuesta
            description = self.llm_service._clean_plain_text_response(description)
            description = self._clean_description(description)  # Limpiar comillas y backslashes

            logger.info("Response parsed successfully")

            # Pequeño delay para evitar rate limiting entre llamadas secuenciales
            time.sleep(0.5)

            # Segunda llamada: obtener título basado en la descripción usando prompt unificado
            title_prompt = self._get_title_prompt(description, "xml", language)

            try:
                logger.info("Calling LLM Service for XML title...")
                with self.inference_semaphore:
                    title_raw = self.llm_service.analyze_llm(
                        prompt=title_prompt,
                        max_tokens=512,  # Títulos cortos, no necesitamos muchos tokens, pero si pedimos muy pocos, a veces falla el modelo
                        temperature=temperature_llm,
                        top_p=top_p
                    )

                title = self.llm_service._clean_plain_text_response(title_raw).strip()
                
                # Fallback si el título está vacío o contiene error
                if not title or "Error" in title or "error" in title.lower() or "no devolvió contenido" in title.lower():
                    title = os.path.basename(xml_path)
                    logger.warning(f"LLM returned empty or error content for XML title. Using filename: {title}")
                else:
                    logger.info(f"XML title generated: {title}")
            except Exception as e:
                logger.warning(f"Error generating XML title: {e}. Using fallback.")
                title = os.path.basename(xml_path)
            
            return {
                "title": title,
                "description": description,
                "metadata": {
                    "content_length": len(xml_content),
                    "language": language
                }
            }
        except Exception as e:
            logger.error(f"Error processing XML: {e}")
            return {
                "title": os.path.basename(xml_path),
                "description": f"Error procesando XML: {str(e)}",
                "metadata": {"error": True}
            }
    
    def process_eml(self, eml_path: str, language: str = "es", max_tokens: int = 1024, temperature_llm: float = 0.3, top_p: float = 0.9, content_limit: int = None) -> Dict[str, Any]:
        """Procesa un archivo EML (email) y genera su resumen"""
        logger.info(f"Starting EML processing: {os.path.basename(eml_path)} (Language: {language})")
        
        # Verificar si el archivo está vacío
        try:
            if os.path.getsize(eml_path) == 0:
                logger.warning(f"Archivo EML vacío ignorado: {os.path.basename(eml_path)}")
                return None  # Retornar None para indicar que debe ser ignorado
        except OSError as e:
            logger.warning(f"No se pudo verificar tamaño del archivo {eml_path}: {e}")
        
        # Obtener límite de contenido desde variable de entorno o parámetro
        if content_limit is None:
            content_limit = int(os.getenv("XML_EML_CONTENT_LIMIT", "5000"))
        
        try:
            # Extraer contenido del EML
            eml_content = self.xml_eml_processor.process_eml(eml_path)
            
            if not eml_content:
                logger.error("Failed to extract content from EML")
                return {
                    "title": os.path.basename(eml_path),
                    "description": "Error: No se pudo extraer contenido del email",
                    "metadata": {}
                }
            
            logger.info(f"Extracted EML content ({len(eml_content)} characters). Preparing model prompt.")
            
            # Primera llamada: obtener descripción usando prompt unificado
            eml_preview = eml_content[:content_limit]  # Limitar tamaño del prompt según configuración
            prompt = self._get_description_prompt(eml_preview, "eml", language)
            
            # Primera llamada: obtener descripción
            logger.info("Calling LLM Service for EML description...")
            with self.inference_semaphore:
                description = self.llm_service.analyze_llm(prompt, max_tokens, temperature=temperature_llm, top_p=top_p)

            # Limpiar la respuesta
            description = self.llm_service._clean_plain_text_response(description)
            description = self._clean_description(description)  # Limpiar comillas y backslashes

            logger.info("Response parsed successfully")

            # Pequeño delay para evitar rate limiting entre llamadas secuenciales
            time.sleep(0.5)

            # Segunda llamada: obtener título basado en la descripción usando prompt unificado
            title_prompt = self._get_title_prompt(description, "eml", language)

            try:
                logger.info("Calling LLM Service for EML title...")
                with self.inference_semaphore:
                    title_raw = self.llm_service.analyze_llm(
                        prompt=title_prompt,
                        max_tokens=512,  # Títulos cortos, no necesitamos muchos tokens, pero si pedimos muy pocos, a veces falla el modelo
                        temperature=temperature_llm,
                        top_p=top_p
                    )

                title = self.llm_service._clean_plain_text_response(title_raw).strip()
                
                # Fallback si el título está vacío o contiene error
                if not title or "Error" in title or "error" in title.lower() or "no devolvió contenido" in title.lower():
                    title = os.path.basename(eml_path)
                    logger.warning(f"LLM returned empty or error content for EML title. Using filename: {title}")
                else:
                    logger.info(f"EML title generated: {title}")
            except Exception as e:
                logger.warning(f"Error generating EML title: {e}. Using fallback.")
                title = os.path.basename(eml_path)
            
            return {
                "title": title,
                "description": description,
                "metadata": {
                    "content_length": len(eml_content),
                    "language": language
                }
            }
        except Exception as e:
            logger.error(f"Error processing EML: {e}")
            return {
                "title": os.path.basename(eml_path),
                "description": f"Error procesando email: {str(e)}",
                "metadata": {"error": True}
            }

    # Supported image extensions for direct image processing
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif')

    def process_image(self, image_path: str, language: str = "es", max_tokens: int = 1024, temperature_vllm: float = 0.1, top_p: float = 0.9) -> Dict[str, Any]:
        """Procesa un archivo de imagen y genera su resumen usando el modelo multimodal"""
        logger.info(f"Starting image processing: {os.path.basename(image_path)} (Language: {language})")

        # Verificar si el archivo está vacío
        try:
            if os.path.getsize(image_path) == 0:
                logger.warning(f"Archivo de imagen vacío ignorado: {os.path.basename(image_path)}")
                return None
        except OSError as e:
            logger.warning(f"No se pudo verificar tamaño del archivo {image_path}: {e}")

        try:
            # Obtener prompt y schema unificados (reutilizar los de PDF/DOCX)
            prompt, schema = self._get_vllm_prompt_and_schema(language)

            # Enviar imagen directamente al modelo multimodal
            logger.info("Calling Multimodal Service for image...")
            with self.inference_semaphore:
                response_content = self.vllm_service.analyze_vllm(
                    [image_path], prompt, max_tokens, schema, temperature_vllm, top_p
                )

            # Extraer title y description del JSON
            try:
                response_json = json.loads(response_content)
                title = response_json.get("title", "").strip()
                description = response_json.get("description", "").strip()

                # Fallback si no se obtienen correctamente
                if not title:
                    title = os.path.basename(image_path)
                if not description:
                    description = self._extract_description(response_content) or "Sin descripción disponible"
            except Exception as e:
                logger.warning(f"Error parsing structured output: {e}. Falling back to description extraction.")
                description = self._extract_description(response_content) or "Sin descripción disponible"
                title = os.path.basename(image_path)

            # Asegurar que siempre haya título y descripción
            if not title:
                title = os.path.basename(image_path)
            if not description:
                description = "Sin descripción disponible"

            # Limpiar descripción
            description = self._clean_description(description)

            logger.info("Image response parsed successfully")

            return {
                "title": title,
                "description": description,
                "metadata": {
                    "image_format": os.path.splitext(image_path)[1].lower(),
                    "language": language
                }
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "title": os.path.basename(image_path),
                "description": f"Error procesando imagen: {str(e)}",
                "metadata": {"error": True}
            }

    def process_file_from_source(self, source_config: Dict[str, Any], file_id: Optional[str] = None, file_name: Optional[str] = None) -> Optional[DocumentResult]:
        """Procesa un archivo desde diferentes fuentes"""
        mode = source_config["mode"]
        logger.info(f"Processing file from source: mode={mode}, file_name={file_name}")
        
        language = source_config.get("language", "es")
        initial_pages = source_config.get("initial_pages", 2)
        final_pages = source_config.get("final_pages", 2)
        max_tokens = source_config.get("max_tokens", 1024)
        temperature_vllm = source_config.get("temperature_vllm", source_config.get("temperature", 0.1))
        temperature_llm = source_config.get("temperature_llm", source_config.get("temperature", 0.3))
        top_p = source_config.get("top_p", 0.9)
        content_limit = source_config.get("content_limit", None)  # Para XML/EML
        max_inner_files = source_config.get("max_inner_files", 0)  # Para archivos comprimidos
        temp_dir = tempfile.mkdtemp()
        
        try:
            file_path = None
            file_type = None
            
            if mode == "gdrive":
                if not self.gdrive_service:
                    raise Exception("Servicio de Google Drive no está habilitado")
                
                # Obtener file_id si no se proporcionó directamente
                if not file_id:
                    file_id = source_config.get("file_id")
                
                # Si no tenemos file_id, buscar por nombre en la carpeta
                if not file_id:
                    folder_id = source_config.get("folder_id")
                    search_file_name = source_config.get("file_name") or file_name
                    
                    if not folder_id or not search_file_name:
                        raise Exception("Se requiere file_id O (folder_id + file_name) para modo gdrive")
                    
                    # Buscar archivo en la carpeta
                    logger.info(f"Searching for file '{search_file_name}' in folder {folder_id}")
                    folder_contents = self.gdrive_service.list_folder_contents(folder_id)
                    
                    for item in folder_contents:
                        # Buscar por nombre exacto o con extensión
                        if (item['name'] == search_file_name or 
                            item['name'] == f"{search_file_name}.pdf" or 
                            item['name'] == f"{search_file_name}.docx" or 
                            item['name'] == f"{search_file_name}.doc" or 
                            item['name'] == f"{search_file_name}.odt" or 
                            item['name'] == f"{search_file_name}.zip" or
                            item['name'] == f"{search_file_name}.rar" or
                            item['name'] == f"{search_file_name}.7z" or
                            item['name'] == f"{search_file_name}.tar" or
                            item['name'] == f"{search_file_name}.tar.gz" or
                            item['name'] == f"{search_file_name}.tgz"):
                            file_id = item['id']
                            file_name = item['name']
                            logger.info(f"Found file: {file_name} (ID: {file_id})")
                            break
                    
                    if not file_id:
                        raise Exception(f"Archivo '{search_file_name}' no encontrado en la carpeta {folder_id}")

                
                # Obtener información del archivo si no tenemos el nombre
                if not file_name:
                    # Usar lock para serializar llamadas a Google Drive API
                    with self.gdrive_download_lock:
                        file_info = self.gdrive_service.get_file_info(file_id)
                    file_name = file_info.get('name', 'unknown_file')
                
                logger.info(f"Downloading from GDrive: {file_name}")
                file_path = os.path.join(temp_dir, file_name)
                # Usar lock para serializar descargas de Google Drive (evitar rate limiting y colisiones)
                with self.gdrive_download_lock:
                    self.gdrive_service.download_file(file_id, file_path)
                
                # Determinar tipo por extensión o mimeType
                # Excluir explícitamente archivos .xsig
                if file_name.lower().endswith('.xsig'):
                    logger.info(f"Archivo .xsig ignorado (no soportado): {file_name}")
                    return None  # Retornar None para indicar que debe ser ignorado
                elif file_name.lower().endswith('.pdf'):
                    file_type = "pdf"
                elif file_name.lower().endswith(('.docx', '.doc', '.odt')):
                    file_type = "docx"  # Usar "docx" como tipo genérico para todos los documentos de Word/ODT
                elif file_name.lower().endswith(('.zip', '.rar', '.cbr', '.7z', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
                    file_type = "zip"  # Usar "zip" como tipo genérico para todos los archivos comprimidos
                elif file_name.lower().endswith('.xml'):
                    file_type = "xml"
                elif file_name.lower().endswith('.eml'):
                    file_type = "eml"
                elif file_name.lower().endswith(self.IMAGE_EXTENSIONS):
                    file_type = "image"
                else:
                    # Intentar determinar por mimeType
                    # Excluir explícitamente archivos .xsig
                    if file_name.lower().endswith('.xsig'):
                        logger.info(f"Archivo .xsig ignorado (no soportado): {file_name}")
                        return None  # Retornar None para indicar que debe ser ignorado
                    # Usar lock para serializar llamadas a Google Drive API
                    with self.gdrive_download_lock:
                        file_info = self.gdrive_service.get_file_info(file_id)
                    mime_type = file_info.get('mimeType', '')
                    if 'pdf' in mime_type:
                        file_type = "pdf"
                    elif 'word' in mime_type or 'docx' in mime_type or 'document' in mime_type or 'msword' in mime_type or 'opendocument.text' in mime_type:
                        file_type = "docx"  # Usar "docx" como tipo genérico para todos los documentos de Word/ODT
                    elif 'zip' in mime_type or 'rar' in mime_type or '7z' in mime_type or 'x-7z' in mime_type or 'tar' in mime_type or 'compressed' in mime_type or 'x-tar' in mime_type or 'x-rar' in mime_type:
                        file_type = "zip"  # Usar "zip" como tipo genérico para todos los archivos comprimidos
                    elif 'xml' in mime_type:
                        # Verificar que no sea .xsig antes de clasificar como XML
                        if not file_name.lower().endswith('.xsig'):
                            file_type = "xml"
                        else:
                            logger.info(f"Archivo .xsig ignorado (no soportado): {file_name}")
                            return None
                    elif 'message' in mime_type or 'rfc822' in mime_type or 'eml' in mime_type:
                        file_type = "eml"
                    elif 'image/' in mime_type:
                        file_type = "image"
            
            elif mode == "local":
                file_path = source_config.get("path")
                if not file_path or not os.path.exists(file_path):
                    raise Exception(f"Archivo no encontrado: {file_path}")
                # Excluir explícitamente archivos .xsig
                if file_path.lower().endswith('.xsig'):
                    logger.info(f"Archivo .xsig ignorado (no soportado): {os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                elif file_path.lower().endswith('.pdf'):
                    file_type = "pdf"
                elif file_path.lower().endswith(('.docx', '.doc', '.odt')):
                    file_type = "docx"  # Usar "docx" como tipo genérico para todos los documentos de Word/ODT
                elif file_path.lower().endswith(('.zip', '.rar', '.cbr', '.7z', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
                    file_type = "zip"  # Usar "zip" como tipo genérico para todos los archivos comprimidos
                elif file_path.lower().endswith('.xml'):
                    file_type = "xml"
                elif file_path.lower().endswith('.eml'):
                    file_type = "eml"
                elif file_path.lower().endswith(self.IMAGE_EXTENSIONS):
                    file_type = "image"
            
            elif mode == "upload":
                # En modo upload, el archivo ya está en file_path
                file_path = source_config.get("path")
                if not file_path:
                    raise Exception("path es requerido para modo upload")
                # Excluir explícitamente archivos .xsig
                if file_path.lower().endswith('.xsig'):
                    logger.info(f"Archivo .xsig ignorado (no soportado): {os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                elif file_path.lower().endswith('.pdf'):
                    file_type = "pdf"
                elif file_path.lower().endswith(('.docx', '.doc', '.odt')):
                    file_type = "docx"  # Usar "docx" como tipo genérico para todos los documentos de Word/ODT
                elif file_path.lower().endswith(('.zip', '.rar', '.cbr', '.7z', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
                    file_type = "zip"  # Usar "zip" como tipo genérico para todos los archivos comprimidos
                elif file_path.lower().endswith('.xml'):
                    file_type = "xml"
                elif file_path.lower().endswith('.eml'):
                    file_type = "eml"
                elif file_path.lower().endswith(self.IMAGE_EXTENSIONS):
                    file_type = "image"
            
            if not file_path or not file_type:
                display_name = file_name or os.path.basename(file_path) if file_path else "unknown"
                logger.error(f"Could not determine file type for {display_name}")
                raise Exception(f"No se pudo determinar el tipo de archivo para {display_name}")
            
            logger.info(f"Detected file type: {file_type}")
            
            # Procesar según el tipo
            if file_type == "pdf":
                result = self.process_pdf(file_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, top_p)
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    logger.info(f"Archivo PDF vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type="pdf",
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    metadata=result.get("metadata", {})
                )
            elif file_type == "docx":
                result = self.process_docx(file_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, top_p)
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    file_ext = os.path.splitext(file_path)[1].lower()
                    file_type_name = {"docx": "DOCX", "doc": "DOC", "odt": "ODT"}.get(file_ext[1:], "DOCUMENTO")
                    logger.info(f"Archivo {file_type_name} vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                # Determinar el tipo real del archivo por su extensión
                file_ext = os.path.splitext(file_path)[1].lower()
                actual_type = file_ext[1:] if file_ext else "docx"  # "docx", "doc", o "odt"
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type=actual_type,
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    metadata=result.get("metadata", {})
                )
            elif file_type == "zip":
                result = self.process_archive(file_path, language, initial_pages, final_pages, max_tokens, temperature_vllm, temperature_llm, top_p, max_inner_files)
                # Si el archivo comprimido está vacío, result será None y lo ignoramos
                if result is None:
                    archive_type = "ZIP" if file_path.lower().endswith('.zip') else "RAR" if file_path.lower().endswith(('.rar', '.cbr')) else "7Z" if file_path.lower().endswith('.7z') else "TAR"
                    logger.info(f"Archivo {archive_type} vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                # Agregar file_id a los children si vienen de Google Drive
                children = result.get("children", [])
                if mode == "gdrive" and file_id and children:
                    for child in children:
                        # Los children de un ZIP no tienen file_id individual... pero el ZIP padre sí
                        pass
                
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type="zip",
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    children=children,
                    metadata=result.get("metadata", {})
                )
            elif file_type == "xml":
                result = self.process_xml(file_path, language, max_tokens, temperature_llm, top_p, content_limit)
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    logger.info(f"Archivo XML vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type="xml",
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    metadata=result.get("metadata", {})
                )
            elif file_type == "eml":
                result = self.process_eml(file_path, language, max_tokens, temperature_llm, top_p, content_limit)
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    logger.info(f"Archivo EML vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type="eml",
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    metadata=result.get("metadata", {})
                )
            elif file_type == "image":
                result = self.process_image(file_path, language, max_tokens, temperature_vllm, top_p)
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    logger.info(f"Archivo de imagen vacío ignorado: {file_name or os.path.basename(file_path)}")
                    return None  # Retornar None para indicar que debe ser ignorado
                description = self._clean_description(result["description"])  # Limpiar comillas y backslashes
                return DocumentResult(
                    name=file_name or os.path.basename(file_path),
                    title=result.get("title") or file_name or os.path.basename(file_path),
                    description=description,
                    type="image",
                    path=source_config.get("path"),
                    file_id=file_id if mode == "gdrive" else None,
                    metadata=result.get("metadata", {})
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_gdrive_folder(self, folder_id: str, folder_name: str, language: str = "es", initial_pages: int = 2, final_pages: int = 2, max_tokens: int = 1024, temperature_vllm: float = 0.1, temperature_llm: float = 0.3, top_p: float = 0.9, max_inner_files: int = 0) -> ProcessFolderResponse:
        """Procesa todos los archivos PDF, DOCX/DOC/ODT, ZIP/RAR/TAR, XML y EML de una carpeta de Google Drive
        
        Args:
            folder_id: ID de la carpeta de Google Drive
            folder_name: Nombre de la carpeta
            language: Idioma para el procesamiento (default: es)
            initial_pages: Número de páginas iniciales a procesar (default: 2)
            final_pages: Número de páginas finales a procesar (default: 2)
            max_tokens: Máximo tokens para la respuesta
        """
        # Iniciar cronómetro
        start_time = time.time()
        
        # Verificar si está en modo desatendido
        unattended_mode = os.getenv("UNATTENDED_MODE", "false").lower() == "true"
        checkpoint_service = None
        
        if unattended_mode:
            logger.info("=" * 80)
            logger.info("MODO DESATENDIDO ACTIVADO")
            logger.info("=" * 80)
            checkpoint_service = CheckpointService()
            config = {
                "language": language,
                "initial_pages": initial_pages,
                "final_pages": final_pages,
                "max_tokens": max_tokens,
                "temperature_vllm": temperature_vllm,
                "temperature_llm": temperature_llm,
                "top_p": top_p
            }
        
        # Obtener TODOS los archivos recursivamente (incluyendo no procesables)
        all_files_all = self.gdrive_service.get_all_files_recursive_all(folder_id)
        # Obtener solo los archivos procesables
        all_files = self.gdrive_service.get_all_files_recursive(folder_id)
        
        # Identificar archivos ignorados (no procesables)
        processable_file_ids = {f['id'] for f in all_files}
        ignored_files = [f for f in all_files_all if f['id'] not in processable_file_ids]
        
        total_files = len(all_files_all)  # Total incluyendo ignorados
        
        logger.info(f"Total de archivos encontrados: {total_files} ({len(all_files)} procesables, {len(ignored_files)} ignorados)")
        
        # Inicializar checkpoint si está en modo desatendido
        if checkpoint_service:
            checkpoint_path = checkpoint_service.start_checkpoint(
                folder_id, folder_name, total_files, config
            )
            
            # Filtrar archivos ya procesados (pero incluir fallidos para reintentar)
            processed_files = checkpoint_service.get_processed_files()
            
            # Archivos pendientes (incluye fallidos para reintentar) = todos los archivos - procesados_exitosos
            pending_files = [
                f for f in all_files 
                if f['id'] not in processed_files
            ]
            
            # Informar sobre archivos fallidos que se van a reintentar
            failed_files = checkpoint_service.get_failed_files()
            if failed_files:
                logger.info(f"⚠️  Se reintentarán {len(failed_files)} archivo(s) que fallaron anteriormente")
            
            # Actualizar la lista de pending_files en el checkpoint
            checkpoint_service.checkpoint_data["pending_files"] = [f['id'] for f in pending_files]
            checkpoint_service._save_checkpoint()
            
            # Cargar resultados previos
            previous_results = checkpoint_service.get_results()
            results = []
            
            # Convertir resultados previos a DocumentResult
            for prev_result in previous_results:
                result_data = prev_result.get("result", {})
                if isinstance(result_data, dict):
                    try:
                        doc_result = DocumentResult(**result_data)
                        results.append(doc_result)
                    except Exception as e:
                        logger.warning(f"Error cargando resultado previo: {e}")
            
            all_files = pending_files
            if len(all_files) > 0:
                logger.info(f"Iniciando procesamiento de {len(all_files)} archivos pendientes...")
            else:
                logger.info("✓ Todos los archivos ya han sido procesados. No hay archivos pendientes.")
        else:
            results = []
        
        # Configuración de procesamiento por batches
        batch_size = int(os.getenv("BATCH_SIZE", "1"))  # Por defecto sin batches
        max_workers = int(os.getenv("MAX_WORKERS", "1"))  # Por defecto sin threading
        
        source_config = {
            "mode": "gdrive",
            "language": language,
            "initial_pages": initial_pages,
            "final_pages": final_pages,
            "max_tokens": max_tokens,
            "temperature_vllm": temperature_vllm,
            "temperature_llm": temperature_llm,
            "top_p": top_p,
            "max_inner_files": max_inner_files
        }
        
        # Procesar archivos
        if batch_size > 1 and max_workers > 1:
            # Procesamiento por batches con threading
            logger.info(f"Procesando en batches de {batch_size} archivos con {max_workers} workers")
            results.extend(self._process_files_batch_parallel(
                all_files, source_config, checkpoint_service, batch_size, max_workers
            ))
        else:
            # Procesamiento secuencial
            for file_info in all_files:
                try:
                    result = self.process_file_from_source(
                        source_config,
                        file_id=file_info['id'],
                        file_name=file_info['name']
                    )
                    # Si el archivo está vacío, result será None y lo ignoramos
                    if result is None:
                        logger.info(f"Archivo vacío ignorado: {file_info['name']}")
                        continue
                    result.path = file_info['path']
                    # Asegurar que el file_id esté presente
                    if not result.file_id:
                        result.file_id = file_info['id']
                    
                    # Verificar si la descripción indica error
                    description = result.description or ""
                    if self._is_error_description(description):
                        # Marcar como fallido si la descripción indica error
                        error_msg = f"Error en descripción: {description}"
                        logger.error(f"Error en descripción para {file_info['name']}: {description}")
                        if checkpoint_service:
                            checkpoint_service.mark_file_failed(
                                file_info['id'],
                                file_info['name'],
                                error_msg
                            )
                        # Cambiar el resultado a error
                        result.description = error_msg
                        result.metadata = result.metadata or {}
                        result.metadata["error"] = True
                    else:
                        # Procesamiento exitoso
                        if checkpoint_service:
                            checkpoint_service.mark_file_processed(
                                file_info['id'],
                                file_info['name'],
                                result.model_dump()
                            )
                    
                    results.append(result)
                    
                    # Mostrar progreso periódicamente
                    if checkpoint_service:
                        progress = checkpoint_service.get_progress()
                        if progress['processed'] % 10 == 0:  # Cada 10 archivos
                            logger.info(f"Progreso: {progress['processed']}/{progress['total']} "
                                      f"({progress['progress_percent']:.1f}%)")
                except Exception as e:
                    error_msg = f"Error al procesar: {str(e)}"
                    logger.error(f"Error procesando {file_info['name']}: {e}")
                    error_result = DocumentResult(
                        name=file_info['name'],
                        title=file_info['name'],  # Usar nombre como título en caso de error
                        description=error_msg,
                        type=file_info.get('mimeType', 'unknown'),
                        path=file_info.get('path', ''),
                        file_id=file_info['id'],
                            metadata={"error": True}
                    )
                    results.append(error_result)
                    
                    # Actualizar checkpoint con error
                    if checkpoint_service:
                        checkpoint_service.mark_file_failed(
                            file_info['id'],
                            file_info['name'],
                            str(e)
                        )
        
        # Finalizar checkpoint
        if checkpoint_service:
            checkpoint_service.finalize("completed")
            progress = checkpoint_service.get_progress()
            logger.info("=" * 80)
            logger.info("PROCESAMIENTO COMPLETADO")
            logger.info(f"Total procesados: {progress['processed']}")
            logger.info(f"Total fallidos: {progress['failed']}")
            logger.info(f"Archivo de checkpoint: {checkpoint_service.get_checkpoint_path()}")
            logger.info("=" * 80)
        
        # Calcular tiempo total transcurrido
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        # Formatear tiempo de forma legible
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        logger.info("=" * 80)
        logger.info(f"⏱️  TIEMPO TOTAL DE PROCESAMIENTO: {time_str} ({elapsed_time:.2f} segundos)")
        logger.info(f"📊 Archivos procesados: {len(results)}")
        logger.info("=" * 80)
        
        # Incluir archivos ignorados (no procesables) con metadata.ignored = True
        for ignored_file in ignored_files:
            # Determinar tipo de archivo simplemente por extensión
            file_name = ignored_file.get('name', '')
            file_type = 'unknown'
            
            if '.' in file_name:
                # Obtener extensión (manejar extensiones compuestas como .tar.gz)
                name_lower = file_name.lower()
                # Verificar extensiones compuestas primero
                compound_extensions = ['.tar.gz', '.tar.bz2', '.tar.xz']
                for ext in compound_extensions:
                    if name_lower.endswith(ext):
                        file_type = ext[1:].replace('.', '_')  # tar.gz -> tar_gz
                        break
                else:
                    # Extensión simple
                    file_type = name_lower.rsplit('.', 1)[-1] if '.' in name_lower else 'unknown'
            
            ignored_result = DocumentResult(
                name=ignored_file['name'],
                title="",  # Title vacío
                description="",  # Description vacío
                type=file_type,
                path=ignored_file.get('path', ''),
                file_id=ignored_file['id'],
                metadata={"ignored": True}
            )
            results.append(ignored_result)
            
            # Guardar en checkpoint como procesado (pero ignorado)
            if checkpoint_service:
                checkpoint_service.mark_file_processed(
                    ignored_file['id'],
                    ignored_file['name'],
                    ignored_result.model_dump()
                )
        
        # Incluir archivos fallidos del checkpoint con description y title vacíos
        if checkpoint_service:
            failed_files = checkpoint_service.get_failed_files()
            # Obtener todos los archivos de nuevo para buscar paths (incluyendo ignorados)
            all_files_dict = {f['id']: f for f in all_files_all}
            
            for failed_file in failed_files:
                file_id = failed_file.get("file_id")
                file_name = failed_file.get("file_name", "unknown")
                
                # Buscar información del archivo en la lista original
                file_info = all_files_dict.get(file_id)
                
                # Si no está en la lista, intentar obtenerlo de Google Drive
                if not file_info and hasattr(self, 'gdrive_service') and self.gdrive_service:
                    try:
                        gdrive_info = self.gdrive_service.get_file_info(file_id)
                        # Crear estructura similar a la de get_all_files_recursive
                        file_info = {
                            'id': file_id,
                            'name': gdrive_info.get('name', file_name),
                            'mimeType': gdrive_info.get('mimeType', 'unknown'),
                            'path': ''  # No tenemos path sin recorrer la estructura
                        }
                    except Exception:
                        file_info = None
                
                # Determinar tipo de archivo
                file_type = 'unknown'
                if file_info:
                    mime_type = file_info.get('mimeType', '')
                    if 'pdf' in mime_type.lower():
                        file_type = 'pdf'
                    elif 'word' in mime_type.lower() or 'document' in mime_type.lower():
                        file_type = 'docx'
                    elif 'zip' in mime_type.lower():
                        file_type = 'zip'
                    elif 'rar' in mime_type.lower():
                        file_type = 'rar'
                    elif 'xml' in mime_type.lower():
                        file_type = 'xml'
                    elif 'message' in mime_type.lower() or 'email' in mime_type.lower():
                        file_type = 'eml'
                    else:
                        # Intentar deducir del nombre
                        name_lower = file_info.get('name', '').lower()
                        if name_lower.endswith('.pdf'):
                            file_type = 'pdf'
                        elif name_lower.endswith(('.docx', '.doc', '.odt')):
                            file_type = 'docx'
                        elif name_lower.endswith(('.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz')):
                            file_type = 'zip'
                        elif name_lower.endswith('.xml'):
                            file_type = 'xml'
                        elif name_lower.endswith('.eml'):
                            file_type = 'eml'
                
                # Crear DocumentResult para archivo fallido
                failed_result = DocumentResult(
                    name=file_name,
                    title="",  # Title vacío
                    description="",  # Description vacío
                    type=file_type,
                    path=file_info.get('path', '') if file_info else '',
                    file_id=file_id,
                    metadata={"error": True, "error_message": failed_file.get("error", "")}
                )
                results.append(failed_result)
        
        # Ordenar resultados por ruta
        results.sort(key=lambda x: x.path or "")
        
        return ProcessFolderResponse(
            folder_id=folder_id,
            folder_name=folder_name,
            processed_at=datetime.now(),
            total_files=len(results),
            results=results
        )
    
    def _process_files_batch_parallel(self, files: List[Dict], source_config: Dict, 
                                     checkpoint_service: Optional[CheckpointService],
                                     batch_size: int, max_workers: int) -> List[DocumentResult]:
        """Procesa archivos en batches paralelos"""
        results = []
        results_lock = threading.Lock()
        
        def process_single_file(file_info: Dict) -> Optional[DocumentResult]:
            """Procesa un solo archivo"""
            try:
                result = self.process_file_from_source(
                    source_config,
                    file_id=file_info['id'],
                    file_name=file_info['name']
                )
                # Si el archivo está vacío, result será None y lo ignoramos
                if result is None:
                    logger.info(f"Archivo vacío ignorado: {file_info['name']}")
                    return None  # Retornar None para indicar que debe ser ignorado
                result.path = file_info['path']
                # Asegurar que el file_id esté presente
                if not result.file_id:
                    result.file_id = file_info['id']
                
                # Verificar si la descripción indica error
                description = result.description or ""
                if self._is_error_description(description):
                    # Marcar como fallido si la descripción indica error
                    error_msg = f"Error en descripción: {description}"
                    logger.error(f"Error en descripción para {file_info['name']}: {description}")
                    if checkpoint_service:
                        checkpoint_service.mark_file_failed(
                            file_info['id'],
                            file_info['name'],
                            error_msg
                        )
                    # Cambiar el resultado a error
                    result.description = error_msg
                    result.metadata = result.metadata or {}
                    result.metadata["error"] = True
                else:
                    # Procesamiento exitoso
                    if checkpoint_service:
                        checkpoint_service.mark_file_processed(
                            file_info['id'],
                            file_info['name'],
                            result.model_dump()
                        )
                
                return result
            except Exception as e:
                error_msg = f"Error al procesar: {str(e)}"
                logger.error(f"Error procesando {file_info['name']}: {e}")
                
                error_result = DocumentResult(
                    name=file_info['name'],
                    title=file_info['name'], # Usar nombre como título en caso de error
                    description=error_msg,
                    type=file_info.get('mimeType', 'unknown'),
                    path=file_info.get('path', ''),
                    file_id=file_info.get('id'), # file_id del Google Drive
                    metadata={"error": True}
                )
                
                # Actualizar checkpoint con error
                if checkpoint_service:
                    checkpoint_service.mark_file_failed(
                        file_info['id'],
                        file_info['name'],
                        str(e)
                    )
                
                return error_result
        
        # Procesar en batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(files) + batch_size - 1) // batch_size
            
            logger.info(f"Procesando batch {batch_num}/{total_batches} ({len(batch)} archivos)")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_file, file_info): file_info 
                          for file_info in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        with results_lock:
                            results.append(result)
            
            # Mostrar progreso después de cada batch
            if checkpoint_service:
                progress = checkpoint_service.get_progress()
                logger.info(f"Progreso total: {progress['processed']}/{progress['total']} "
                          f"({progress['progress_percent']:.1f}%)")
        
        return results

