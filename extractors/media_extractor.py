import os
import tempfile
import whisper
from typing import List, Callable, Optional

class MediaExtractor:
    """Extracts text/transcripts from audio and video files using OpenAI Whisper."""

    MEDIA_EXTENSIONS = {
        # Audio
        ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma",
        # Video
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"
    }

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            model_name_str = str(self.model_name).strip()
            expanded_path = os.path.expanduser(model_name_str)
            cache_dir = os.path.expanduser("~/.cache/whisper")

            try:
                # 1. If model_name is a direct path to a local .pt model file
                if os.path.isfile(expanded_path):
                    self._model = whisper.load_model(expanded_path)
                # 2. If model file exists in local whisper cache ~/.cache/whisper/<name>.pt
                elif os.path.isfile(os.path.join(cache_dir, f"{model_name_str}.pt")):
                    self._model = whisper.load_model(os.path.join(cache_dir, f"{model_name_str}.pt"))
                # 3. Download / Load from default Whisper registry
                else:
                    self._model = whisper.load_model(model_name_str)
            except Exception as err:
                err_msg = str(err)
                if "name resolution" in err_msg.lower() or "connection" in err_msg.lower() or "urlopen" in err_msg.lower() or "gai_error" in err_msg.lower():
                    raise RuntimeError(
                        f"Failed to download Whisper model '{model_name_str}' (Network error: {err_msg}).\n\n"
                        f"💡 How to Fix:\n"
                        f"1. Connect your computer to the internet so Whisper can download '{model_name_str}.pt'.\n"
                        f"2. Or download '{model_name_str}.pt' manually on another device and place it in '{cache_dir}/', or pass the absolute path to the .pt file."
                    ) from err
                raise RuntimeError(f"Error loading Whisper model '{model_name_str}': {err_msg}") from err
        return self._model

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in cls.MEDIA_EXTENSIONS

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        return sorted(list(cls.MEDIA_EXTENSIONS))

    @classmethod
    def list_cached_models(cls) -> List[str]:
        """Lists Whisper models currently cached in ~/.cache/whisper."""
        cache_dir = os.path.expanduser("~/.cache/whisper")
        if not os.path.exists(cache_dir):
            return []
        cached = []
        for f in os.listdir(cache_dir):
            if f.endswith(".pt"):
                cached.append(f.rsplit(".", 1)[0])
        return sorted(cached)

    def transcribe(self, file_input, filename: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Transcribe audio or video input using Whisper.
        `file_input` can be a file path string or bytes/UploadedFile stream.
        """
        temp_file_path = None
        try:
            if isinstance(file_input, str) and os.path.isfile(file_input):
                media_path = file_input
            else:
                ext = os.path.splitext(filename)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    if hasattr(file_input, "read"):
                        tmp.write(file_input.read())
                    elif isinstance(file_input, bytes):
                        tmp.write(file_input)
                    else:
                        raise ValueError("Invalid file_input provided.")
                    temp_file_path = tmp.name
                media_path = temp_file_path

            if progress_callback:
                progress_callback(f"Loading Whisper model ('{self.model_name}')...")

            model = self._get_model()

            if progress_callback:
                progress_callback("Transcribing media audio track...")

            # Transcribe audio track from video/audio file
            result = model.transcribe(media_path, verbose=False)
            return result.get("text", "").strip()

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
