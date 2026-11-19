"""
Multimodal Image and Cross-Modal Text Embedder using CLIP.
Projects images and queries into a shared 512-dimensional vector space.
"""
import io
import logging
from typing import List, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_EMBEDDER_INSTANCE = None


class ImageEmbedder:
    """
    Singleton wrapper for sentence-transformers CLIP model (clip-ViT-B-32).
    Generates normalized 512-dimensional embeddings for:
    1. Images (PIL Image or raw bytes)
    2. Text queries (natural language search against visual figures)
    """

    def __init__(self, model_name: str = "clip-ViT-B-32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None
        self._is_available = True

    def _load_model(self):
        if self._model is None and self._is_available:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading CLIP model '%s' on %s...", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("CLIP model loaded successfully.")
            except Exception as e:
                logger.warning("Failed to load CLIP model '%s': %s. Visual embedding disabled.", self.model_name, e)
                self._is_available = False

    @classmethod
    def get_instance(cls) -> "ImageEmbedder":
        global _IMAGE_EMBEDDER_INSTANCE
        if _IMAGE_EMBEDDER_INSTANCE is None:
            _IMAGE_EMBEDDER_INSTANCE = cls()
        return _IMAGE_EMBEDDER_INSTANCE

    def embed_image(self, image_input: Union[bytes, Image.Image]) -> Optional[List[float]]:
        """
        Embed an image (PIL Image or raw bytes) into a 512-dim vector.
        """
        try:
            self._load_model()
            if not self._model:
                return None

            if isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                return None

            vec = self._model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            logger.warning("Error generating CLIP image embedding: %s", e)
            return None

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed a text query into the CLIP cross-modal vector space.
        """
        try:
            self._load_model()
            if not self._model or not text.strip():
                return None

            vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            logger.warning("Error generating CLIP text embedding: %s", e)
            return None
