import os
import io
from typing import List, Union
from PIL import Image
import pytesseract

class OcrExtractor:
    """Extracts text from image files using Tesseract OCR."""

    IMAGE_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".pnm", ".pbm", ".pgm"
    }

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in cls.IMAGE_EXTENSIONS

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        return sorted(list(cls.IMAGE_EXTENSIONS))

    def extract_text(self, file_path_or_bytes: Union[str, bytes, io.BytesIO], filename: str) -> str:
        """
        Perform OCR extraction on an image file.
        `file_path_or_bytes` can be a path string, bytes, or file stream.
        """
        if isinstance(file_path_or_bytes, str) and os.path.isfile(file_path_or_bytes):
            image = Image.open(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, bytes):
            image = Image.open(io.BytesIO(file_path_or_bytes))
        elif hasattr(file_path_or_bytes, "read"):
            file_bytes = file_path_or_bytes.read()
            if hasattr(file_path_or_bytes, "seek"):
                file_path_or_bytes.seek(0)
            image = Image.open(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported input type for OCR image extraction.")

        # Convert image to RGB if needed (e.g. RGBA, P, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Configure tessdata path if local traineddata exists
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_tessdata = os.path.join(project_dir, "tessdata")
        config = ""
        if os.path.exists(local_tessdata) and os.path.isfile(os.path.join(local_tessdata, "eng.traineddata")):
            config = f'--tessdata-dir "{local_tessdata}"'

        text = pytesseract.image_to_string(image, config=config) if config else pytesseract.image_to_string(image)
        return text.strip()
