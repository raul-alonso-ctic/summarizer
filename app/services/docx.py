from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import os
import shutil
import signal
import subprocess
import tempfile
from typing import List


class DOCXProcessor:
    def convert_to_images(self, docx_path: str, output_folder: str, initial_pages: int = 2, final_pages: int = 2) -> List[str]:
        """Convierte las primeras N y últimas M páginas del DOCX/DOC/ODT a imágenes

        Primero convierte DOCX/DOC/ODT a PDF usando LibreOffice, luego PDF a imágenes (igual que PDFs)

        Args:
            docx_path: Ruta al archivo DOCX, DOC o ODT
            output_folder: Carpeta donde guardar las imágenes
            initial_pages: Número de páginas iniciales a procesar (default: 2)
            final_pages: Número de páginas finales a procesar (default: 2)
        """
        images = []
        temp_pdf = None
        lo_profile_dir = None
        try:
            # Convertir DOCX a PDF usando LibreOffice
            # LibreOffice genera el PDF con el mismo nombre base pero extensión .pdf
            docx_basename = os.path.splitext(os.path.basename(docx_path))[0]
            temp_pdf = os.path.join(output_folder, f"{docx_basename}.pdf")

            # Isolated user profile per conversion to prevent lock file conflicts.
            # Without this, a timed-out soffice.bin leaves a lock on the shared
            # profile, causing all subsequent conversions to hang.
            lo_profile_dir = tempfile.mkdtemp(prefix="lo_profile_")

            proc = subprocess.Popen(
                [
                    'libreoffice',
                    '--headless',
                    '--nodefault',
                    f'-env:UserInstallation=file://{lo_profile_dir}',
                    '--convert-to', 'pdf',
                    '--outdir', output_folder,
                    docx_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # start_new_session puts LibreOffice in its own process group
                # so we can kill the entire group (including child soffice.bin)
                # on timeout — preventing orphan processes.
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                # Kill entire process group to catch orphan soffice.bin children
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.kill()
                proc.wait()
                print(f"LibreOffice timed out converting {docx_path}")
                return []

            # Verificar si la conversión fue exitosa
            if proc.returncode != 0:
                print(f"Error converting document to PDF: {stderr}")
                return []

            # Verificar que el PDF se haya generado
            if not os.path.exists(temp_pdf):
                # Intentar buscar cualquier PDF recién generado en el directorio
                pdf_files = [f for f in os.listdir(output_folder) if f.endswith('.pdf')]
                if pdf_files:
                    # Usar el PDF más reciente (probablemente el que acabamos de generar)
                    pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
                    temp_pdf = os.path.join(output_folder, pdf_files[0])
                else:
                    print(f"Error: No se pudo convertir documento a PDF: {docx_path}")
                    return []

            # Obtener número total de páginas del PDF generado
            reader = PdfReader(temp_pdf)
            total_pages = len(reader.pages)

            # Procesar páginas iniciales
            if total_pages >= 1 and initial_pages > 0:
                first_batch = convert_from_path(temp_pdf, first_page=1, last_page=min(initial_pages, total_pages))
                for i, img in enumerate(first_batch):
                    path = os.path.join(output_folder, f"page_{i+1}.jpg")
                    img.save(path, "JPEG")
                    images.append(path)

            # Procesar páginas finales (si hay suficientes páginas y no se solapan con las iniciales)
            if total_pages > initial_pages and final_pages > 0:
                last_start = max(initial_pages + 1, total_pages - final_pages + 1)
                last_batch = convert_from_path(temp_pdf, first_page=last_start, last_page=total_pages)
                for i, img in enumerate(last_batch):
                    page_num = last_start + i
                    path = os.path.join(output_folder, f"page_{page_num}.jpg")
                    img.save(path, "JPEG")
                    images.append(path)

            return images
        except Exception as e:
            print(f"Error processing document {docx_path}: {e}")
            return []
        finally:
            # Limpiar PDF temporal si existe
            if temp_pdf and os.path.exists(temp_pdf):
                try:
                    os.remove(temp_pdf)
                except:
                    pass
            # Limpiar perfil temporal de LibreOffice
            if lo_profile_dir and os.path.exists(lo_profile_dir):
                try:
                    shutil.rmtree(lo_profile_dir)
                except:
                    pass
