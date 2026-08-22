import os
import tempfile
import whisper
from typing import List, Callable, Optional, Dict, Any, Tuple

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
                if os.path.isfile(expanded_path):
                    self._model = whisper.load_model(expanded_path)
                elif os.path.isfile(os.path.join(cache_dir, f"{model_name_str}.pt")):
                    self._model = whisper.load_model(os.path.join(cache_dir, f"{model_name_str}.pt"))
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

    @staticmethod
    def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
        """Format seconds into HH:MM:SS,mmm timestamp string for SRT/VTT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"

    @classmethod
    def generate_srt(cls, segments: List[Dict[str, Any]]) -> str:
        """Convert Whisper segments into SRT subtitle text format."""
        srt_lines = []
        for idx, seg in enumerate(segments, start=1):
            start = cls.format_timestamp(seg.get("start", 0.0), decimal_marker=",")
            end = cls.format_timestamp(seg.get("end", 0.0), decimal_marker=",")
            text = seg.get("text", "").strip()
            srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        return "\n".join(srt_lines)

    @classmethod
    def generate_vtt(cls, segments: List[Dict[str, Any]]) -> str:
        """Convert Whisper segments into WebVTT subtitle text format."""
        vtt_lines = ["WEBVTT\n"]
        for idx, seg in enumerate(segments, start=1):
            start = cls.format_timestamp(seg.get("start", 0.0), decimal_marker=".")
            end = cls.format_timestamp(seg.get("end", 0.0), decimal_marker=".")
            text = seg.get("text", "").strip()
            vtt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        return "\n".join(vtt_lines)

    def transcribe_full(
        self,
        file_input,
        filename: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Transcribes media file and returns complete result dictionary:
        {
          "text": str,
          "language": str,
          "segments": list,
          "srt": str,
          "vtt": str
        }
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
                        raise ValueError("Invalid file_input provided for media extraction.")
                    temp_file_path = tmp.name
                media_path = temp_file_path

            if progress_callback:
                progress_callback(f"Loading Whisper model ('{self.model_name}')...")

            model = self._get_model()

            if progress_callback:
                progress_callback("Transcribing media audio & auto-detecting language...")

            result = model.transcribe(media_path, verbose=False)
            text = result.get("text", "").strip()
            detected_lang = result.get("language", "unknown")
            segments = result.get("segments", [])

            srt_content = self.generate_srt(segments)
            vtt_content = self.generate_vtt(segments)

            return {
                "text": text,
                "language": detected_lang,
                "segments": segments,
                "srt": srt_content,
                "vtt": vtt_content
            }

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    def transcribe(self, file_input, filename: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Legacy helper returning plain transcribed text."""
        result = self.transcribe_full(file_input, filename, progress_callback=progress_callback)
        return result["text"]

