"""
Document loaders supporting PDF (with multimodal adaptive vision extraction),
standalone images (with CLIP embeddings and Gemini Vision transcription),
Markdown, and plain text.
"""
import io
import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from langchain_core.documents import Document

try:
    import pymupdf  # PyMuPDF / fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
ALLOWED_TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_ALL_EXTENSIONS = ALLOWED_PDF_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS | ALLOWED_TEXT_EXTENSIONS


class DocumentLoaderError(Exception):
    """Exception raised when document loading fails."""
    pass


def describe_image_with_vision(
    image_input: Union[bytes, Image.Image],
    api_key: Optional[str] = None,
    context_hint: str = "",
) -> str:
    """
    Transcribe and analyze an image or document page using Google Gemini Vision.
    Extracts verbatim text, architecture flowcharts, diagrams, tables, and relationships.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.warning("No GOOGLE_API_KEY available for visual analysis.")
        return ""

    try:
        from google import genai
        client = genai.Client(api_key=key)

        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            return ""

        prompt = (
            "You are an expert document and visual content extraction engine.\n"
            "Transcribe and analyze this image or document slide in complete detail:\n"
            "1. Extract ALL visible text verbatim, preserving headings, bullet points, and notes.\n"
            "2. If this image contains architecture diagrams, flowcharts, or system components, describe every block, arrow, direction, and relationship explicitly.\n"
            "3. If this image contains tables, charts, or graphs, transcribe all tabular data as markdown tables with exact labels and values.\n"
            "4. Output clean, structured Markdown. Do not omit technical terms or data points."
        )
        if context_hint:
            prompt += f"\nContext hint: {context_hint}"

        # Try gemini-2.5-flash first, then gemini-1.5-flash, then gemini-flash-lite-latest
        models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-lite-latest"]
        for model in models:
            try:
                response = client.models.generate_content(model=model, contents=[img, prompt])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning("Gemini Vision model '%s' failed: %s", model, str(e))

        return ""
    except Exception as e:
        logger.warning("Vision transcription failed: %s", str(e))
        return ""


class DocumentLoader:
    """
    Unified document loader supporting PDFs, standalone images, Markdown, and plain text.
    Implements adaptive multimodal processing for visual pages, flowcharts, and scans.
    """

    @classmethod
    def load_from_bytes(
        cls,
        file_bytes: bytes,
        filename: str,
        api_key: Optional[str] = None,
        vector_store: Optional[Any] = None,
    ) -> List[Document]:
        """
        Load documents from raw in-memory bytes based on file extension.
        """
        ext = Path(filename).suffix.lower()

        if ext in ALLOWED_PDF_EXTENSIONS:
            return cls._load_pdf_from_bytes(file_bytes, filename, api_key=api_key, vector_store=vector_store)
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            return cls._load_image_from_bytes(file_bytes, filename, api_key=api_key, vector_store=vector_store)
        elif ext in [".md", ".markdown"]:
            return cls._load_text_from_bytes(file_bytes, filename, doc_type="markdown")
        elif ext in [".txt", ".text"]:
            return cls._load_text_from_bytes(file_bytes, filename, doc_type="text")
        else:
            raise DocumentLoaderError(
                f"Unsupported file format: '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_ALL_EXTENSIONS))}"
            )

    @classmethod
    def load_from_path(
        cls,
        file_path: str,
        api_key: Optional[str] = None,
        vector_store: Optional[Any] = None,
    ) -> List[Document]:
        """
        Load documents from a filesystem path.
        """
        path = Path(file_path)
        if not path.is_file():
            raise DocumentLoaderError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        return cls.load_from_bytes(file_bytes, path.name, api_key=api_key, vector_store=vector_store)

    @classmethod
    def _load_image_from_bytes(
        cls,
        file_bytes: bytes,
        filename: str,
        api_key: Optional[str] = None,
        vector_store: Optional[Any] = None,
    ) -> List[Document]:
        """
        Load a standalone image (.png, .jpg, etc.):
        1. Saves image locally in data/extracted_images/ and data/raw/
        2. Computes 512-dim CLIP embedding
        3. Transcribes full visual details, flowcharts, and labels via Gemini Vision
        4. Upserts CLIP embedding into ChromaDB image collection
        5. Returns LangChain Document with rich text and visual metadata
        """
        extracted_dir = Path("data/extracted_images")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        saved_img_path = extracted_dir / filename
        with open(saved_img_path, "wb") as f:
            f.write(file_bytes)

        # Also ensure saved in data/raw for inline preview
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_img_path = raw_dir / filename
        if not raw_img_path.exists():
            with open(raw_img_path, "wb") as f:
                f.write(file_bytes)

        # 1. CLIP embedding
        clip_vector = None
        try:
            from src.embeddings.image_embedder import ImageEmbedder
            clip_vector = ImageEmbedder.get_instance().embed_image(file_bytes)
        except Exception as e:
            logger.warning("Could not compute CLIP embedding for '%s': %s", filename, str(e))

        # 2. Gemini Vision transcription
        vision_text = describe_image_with_vision(
            file_bytes, api_key=api_key, context_hint=f"Standalone image document: {filename}"
        )
        if not vision_text.strip():
            vision_text = f"Visual Image Document: {filename}\nContains diagram, photo, or visual content."

        # 3. Store CLIP embedding in ChromaDB image collection
        if clip_vector is not None:
            try:
                from src.vectorstore.client import ChromaVectorStore
                vs = vector_store or ChromaVectorStore()
                vs.add_image_embedding(
                    image_id=f"{filename}_img_1",
                    embedding=clip_vector,
                    metadata={
                        "source": filename,
                        "filename": filename,
                        "page_number": 1,
                        "total_pages": 1,
                        "doc_type": "image",
                        "image_path": str(saved_img_path),
                    },
                    document_text=vision_text[:500],
                )
            except Exception as e:
                logger.warning("Could not save image embedding to ChromaDB: %s", str(e))

        doc = Document(
            page_content=vision_text,
            metadata={
                "source": filename,
                "filename": filename,
                "page_number": 1,
                "total_pages": 1,
                "doc_type": "image",
                "image_path": str(saved_img_path),
            },
        )
        logger.info("Successfully loaded image document '%s' (content length: %d chars)", filename, len(vision_text))
        return [doc]

    @classmethod
    def _load_pdf_from_bytes(
        cls,
        file_bytes: bytes,
        filename: str,
        api_key: Optional[str] = None,
        vector_store: Optional[Any] = None,
    ) -> List[Document]:
        """
        Extract text and visual diagrams page-by-page from PDF bytes.
        Employs adaptive classification:
        - Category A (Text-only): Fast digital extraction.
        - Category B (Visual/Diagram/Scanned/Low text): Pixmap rendering + Gemini Vision + CLIP embedding.
        - Category C (Mixed text + figures): Native text + Gemini Vision diagram transcription + CLIP embedding.
        Uses ThreadPoolExecutor for concurrent vision extraction.
        """
        extracted_dir = Path("data/extracted_images")
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if HAS_PYMUPDF:
            try:
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                total_pages = len(doc)
                documents: List[Document] = []

                # Plan pages requiring vision
                vision_tasks: List[Tuple[int, bytes, str, str]] = []  # (page_num, pixmap_bytes, native_text, category)
                page_info: List[Dict[str, Any]] = []

                from src.embeddings.image_embedder import ImageEmbedder
                embedder = ImageEmbedder.get_instance()

                vector_store_inst = None
                try:
                    from src.vectorstore.client import ChromaVectorStore
                    vector_store_inst = vector_store or ChromaVectorStore()
                except Exception:
                    pass

                for page_idx in range(total_pages):
                    page_num = page_idx + 1
                    page = doc.load_page(page_idx)
                    native_text = (page.get_text() or "").strip()
                    images = page.get_images(full=True)

                    has_images = len(images) > 0
                    is_low_text = len(native_text) < 150

                    category = "text"
                    if is_low_text and (has_images or len(native_text) == 0):
                        category = "vision_full"  # Category B: Architecture diagrams, slides, scans
                    elif len(native_text) >= 150 and has_images:
                        category = "vision_mixed"  # Category C: Text with accompanying diagram/chart
                    else:
                        category = "text"  # Category A: Pure text

                    if category in ("vision_full", "vision_mixed"):
                        pix = page.get_pixmap(dpi=150)
                        pix_bytes = pix.tobytes("png")

                        # Save snapshot for visual reference
                        page_img_path = extracted_dir / f"{filename}_p{page_num}.png"
                        try:
                            with open(page_img_path, "wb") as f:
                                f.write(pix_bytes)
                        except Exception:
                            pass

                        # Compute and store CLIP embedding
                        try:
                            clip_vec = embedder.embed_image(pix_bytes)
                            if clip_vec and vector_store_inst:
                                vector_store_inst.add_image_embedding(
                                    image_id=f"{filename}_p{page_num}",
                                    embedding=clip_vec,
                                    metadata={
                                        "source": filename,
                                        "filename": filename,
                                        "page_number": page_num,
                                        "total_pages": total_pages,
                                        "doc_type": "pdf_page_image",
                                        "image_path": str(page_img_path),
                                    },
                                    document_text=native_text[:300],
                                )
                        except Exception as e:
                            logger.warning("Error generating CLIP embedding for page %d: %s", page_num, e)

                        vision_tasks.append((page_num, pix_bytes, native_text, category))
                        page_info.append({
                            "page_num": page_num,
                            "category": category,
                            "native_text": native_text,
                        })
                    else:
                        page_info.append({
                            "page_num": page_num,
                            "category": "text",
                            "native_text": native_text,
                        })

                # Concurrently execute Gemini Vision for visual pages
                vision_results: Dict[int, str] = {}
                if vision_tasks:
                    logger.info("Dispatching %d visual pages in '%s' to Gemini Vision concurrently...", len(vision_tasks), filename)
                    with ThreadPoolExecutor(max_workers=min(4, len(vision_tasks))) as executor:
                        future_to_page = {
                            executor.submit(
                                describe_image_with_vision,
                                t[1],
                                api_key,
                                f"{filename} Page {t[0]}",
                            ): t[0]
                            for t in vision_tasks
                        }
                        for future in as_completed(future_to_page):
                            p_num = future_to_page[future]
                            try:
                                transcribed = future.result()
                                vision_results[p_num] = transcribed
                            except Exception as e:
                                logger.warning("Vision transcription failed for page %d: %s", p_num, e)
                                vision_results[p_num] = ""

                # Assemble final Document list in page order
                for p_meta in page_info:
                    p_num = p_meta["page_num"]
                    cat = p_meta["category"]
                    n_text = p_meta["native_text"]

                    if cat == "vision_full":
                        v_text = vision_results.get(p_num, "")
                        if v_text.strip():
                            final_text = f"{n_text}\n\n{v_text}".strip() if n_text else v_text
                        else:
                            final_text = n_text or f"[Visual diagram / scanned content on Page {p_num} of {filename}]"

                    elif cat == "vision_mixed":
                        v_text = vision_results.get(p_num, "")
                        if v_text.strip():
                            final_text = f"{n_text}\n\n### Visual Elements & Diagrams (Page {p_num}):\n{v_text}"
                        else:
                            final_text = n_text

                    else:
                        final_text = n_text

                    if final_text.strip():
                        documents.append(
                            Document(
                                page_content=final_text,
                                metadata={
                                    "source": filename,
                                    "filename": filename,
                                    "page_number": p_num,
                                    "total_pages": total_pages,
                                    "doc_type": "pdf",
                                },
                            )
                        )

                if documents:
                    logger.info(
                        "Parsed PDF '%s' via PyMuPDF: %d total pages, %d visual pages processed",
                        filename,
                        total_pages,
                        len(vision_tasks),
                    )
                    return documents

            except Exception as e:
                logger.warning("PyMuPDF parsing encountered an issue: %s. Falling back to pypdf.", str(e))

        # Fallback to pypdf
        if HAS_PYPDF:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                total_pages = len(reader.pages)
                documents: List[Document] = []
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "filename": filename,
                                    "page_number": page_idx + 1,
                                    "total_pages": total_pages,
                                    "doc_type": "pdf",
                                },
                            )
                        )
                if documents:
                    return documents
            except Exception as e:
                raise DocumentLoaderError(f"Failed to parse PDF with pypdf: {str(e)}") from e

        raise DocumentLoaderError(f"No extractable text or visual content found in '{filename}'.")

    @classmethod
    def _load_text_from_bytes(
        cls, file_bytes: bytes, filename: str, doc_type: str
    ) -> List[Document]:
        """
        Load plain text or markdown from bytes.
        """
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "filename": filename,
                        "page_number": 1,
                        "total_pages": 1,
                        "doc_type": doc_type,
                    },
                )
            ]
        except Exception as e:
            raise DocumentLoaderError(f"Failed to read text file '{filename}': {str(e)}") from e
