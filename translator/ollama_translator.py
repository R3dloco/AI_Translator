import ollama
import re
from typing import List, Dict, Any, Optional, Callable, Generator

class OllamaTranslator:
    """Handles text translation using local Ollama LLM provider with streaming support."""

    def __init__(self, host: str = "http://localhost:11434", model_name: str = "llama3.2:latest"):
        self.host = host
        self.model_name = model_name
        self.client = ollama.Client(host=self.host)

    @classmethod
    def list_models(cls, host: str = "http://localhost:11434") -> List[str]:
        """Fetch list of installed models from local Ollama service."""
        try:
            client = ollama.Client(host=host)
            response = client.list()
            models = []
            if hasattr(response, 'models'):
                models = [m.model for m in response.models]
            elif isinstance(response, dict) and 'models' in response:
                models = [m.get('name') or m.get('model') for m in response['models']]
            return models if models else ["llama3.2:latest"]
        except Exception as e:
            return ["llama3.2:latest", "qwen3-coder:latest", "llama3.1:latest"]

    def chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> List[str]:
        """Split text into manageable chunks at paragraph/newline boundaries."""
        text = text.strip()
        if not text:
            return []
        
        if len(text) <= max_chars_per_chunk:
            return [text]

        paragraphs = text.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for p in paragraphs:
            if current_length + len(p) + 1 > max_chars_per_chunk:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                if len(p) > max_chars_per_chunk:
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_chunk = []
                    sub_len = 0
                    for s in sentences:
                        if sub_len + len(s) + 1 > max_chars_per_chunk:
                            if sub_chunk:
                                chunks.append(" ".join(sub_chunk))
                                sub_chunk = []
                                sub_len = 0
                        sub_chunk.append(s)
                        sub_len += len(s) + 1
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                else:
                    current_chunk.append(p)
                    current_length += len(p) + 1
            else:
                current_chunk.append(p)
                current_length += len(p) + 1

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def stream_translate_text(
        self,
        text: str,
        target_language: str = "English",
        custom_instructions: str = "",
        max_chunk_chars: int = 3500
    ) -> Generator[str, None, None]:
        """
        Yields translated tokens in real-time as a stream.
        """
        if not text or not text.strip():
            return

        chunks = self.chunk_text(text, max_chars_per_chunk=max_chunk_chars)
        system_prompt = (
            f"You are an expert AI translator. Your task is to accurately translate the input text "
            f"into fluent, high-quality natural {target_language}.\n"
            "Guidelines:\n"
            "1. Preserve the original meaning, structure, tone, and formatting (e.g. headings, lists, tables, markdown).\n"
            "2. Do NOT add commentary, intro headers, or explanations (like 'Here is the translation:').\n"
            f"3. Provide ONLY the translated {target_language} text directly.\n"
        )
        if custom_instructions:
            system_prompt += f"Additional user instruction: {custom_instructions}\n"

        for idx, chunk in enumerate(chunks):
            if idx > 0:
                yield "\n\n"

            user_prompt = f"Translate the following text into {target_language}:\n\n{chunk}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                stream = self.client.chat(model=self.model_name, messages=messages, stream=True)
                for chunk_resp in stream:
                    content = ""
                    if hasattr(chunk_resp, 'message') and hasattr(chunk_resp.message, 'content'):
                        content = chunk_resp.message.content or ""
                    elif isinstance(chunk_resp, dict):
                        content = chunk_resp.get("message", {}).get("content", "")
                    if content:
                        yield content
            except Exception as e:
                err_msg = str(e)
                if "connection" in err_msg.lower() or "refused" in err_msg.lower():
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self.host}. Please verify Ollama is running (`ollama serve`)."
                    ) from e
                elif "not found" in err_msg.lower():
                    raise RuntimeError(
                        f"Ollama model '{self.model_name}' not found. Run `ollama pull {self.model_name}` to download it."
                    ) from e
                else:
                    raise RuntimeError(f"Ollama translation error: {err_msg}") from e

    def translate_text(
        self,
        text: str,
        target_language: str = "English",
        custom_instructions: str = "",
        max_chunk_chars: int = 3500,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        Translates input text into specified target_language using Ollama LLM.
        """
        if not text or not text.strip():
            return ""

        chunks = self.chunk_text(text, max_chars_per_chunk=max_chunk_chars)
        translated_chunks = []

        system_prompt = (
            f"You are an expert AI translator. Your task is to accurately translate the input text "
            f"into fluent, high-quality natural {target_language}.\n"
            "Guidelines:\n"
            "1. Preserve the original meaning, structure, tone, and formatting (e.g. headings, lists, tables, markdown).\n"
            "2. Do NOT add commentary, intro headers, or explanations (like 'Here is the translation:').\n"
            f"3. Provide ONLY the translated {target_language} text directly.\n"
        )
        if custom_instructions:
            system_prompt += f"Additional user instruction: {custom_instructions}\n"

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(idx + 1, len(chunks), f"Translating chunk {idx + 1} of {len(chunks)} to {target_language}...")

            user_prompt = f"Translate the following text into {target_language}:\n\n{chunk}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                response = self.client.chat(model=self.model_name, messages=messages)
                if hasattr(response, 'message'):
                    msg_obj = response.message
                    content = getattr(msg_obj, 'content', '') if hasattr(msg_obj, 'content') else (msg_obj.get('content', '') if isinstance(msg_obj, dict) else '')
                elif isinstance(response, dict):
                    content = response.get("message", {}).get("content", "")
                else:
                    content = str(response)
                translated_chunks.append(content.strip())
            except Exception as e:
                err_msg = str(e)
                if "connection" in err_msg.lower() or "refused" in err_msg.lower():
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self.host}. Please verify Ollama is running (`ollama serve`)."
                    ) from e
                elif "not found" in err_msg.lower():
                    raise RuntimeError(
                        f"Ollama model '{self.model_name}' not found. Run `ollama pull {self.model_name}` to download it."
                    ) from e
                else:
                    raise RuntimeError(f"Ollama translation error: {err_msg}") from e

        return "\n\n".join(translated_chunks)


