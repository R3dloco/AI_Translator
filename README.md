# 🌐 OmniTranslate AI

**OmniTranslate AI** is a multi-format AI translation web application built with **Python**, **Streamlit**, **Ollama**, **Tesseract OCR**, and **OpenAI Whisper**. It enables offline, privacy-first translation of documents, scanned PDFs, images, spreadsheets, ebooks, code files, audio, and video content without sending data to third-party cloud APIs.

---

## 🐍 Python Version Requirement

* **Python Target**: **Python 3.10 or higher** (Tested on Python 3.10 through Python 3.14).

---

## ✨ Key Features

- **Privacy-First Local Translation**: Powered by local LLMs via [Ollama](https://ollama.com/) (e.g., `llama3.2`, `qwen3-coder`, `aya-expanse`).
- **Multi-Format Extraction (40+ file extensions)**:
  - **Documents & Code**: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.epub`, `.eml`, `.msg`, `.html`, `.json`, `.py`, `.js`, `.md`, `.txt`, etc.
  - **Images & Scanned PDFs**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` + scanned PDFs via Tesseract OCR.
  - **Audio & Video**: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.avi`, `.webm` via OpenAI Whisper.
- **Context-Aware Chunking**: Intelligently splits large files along paragraph/sentence boundaries for token-safe translation.
- **Dual Processing Modes**:
  - **Single File Mode**: Side-by-side interactive comparison of original text and translation.
  - **Directory Scan Mode**: Recursive folder batch processing.
- **Flexible Export**: Export outputs to `.txt`, `.md`, `.docx`, or `.json`.

---

## 🛠️ System Prerequisites

Before running the application, ensure the following tools are installed:

1. **Python 3.10+**: Runtime environment.
2. **Ollama**: Local LLM server ([ollama.com](https://ollama.com/)).
3. **Tesseract OCR**: OCR binary for scanning images and PDFs.
4. **FFmpeg**: Multimedia framework for audio/video decoding (used by Whisper).

---

## 🚀 Quick Start Guide

### 1. Navigate to Project Directory
```bash
cd /home/voldemort/Dev/Python/AI_Project
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Ollama & Download Model
```bash
ollama serve
ollama pull llama3.2
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📁 Project Structure

```
AI_Project/
├── app.py                          # Streamlit application dashboard
├── extractors/                     # Multi-format content extractors
│   ├── document_extractor.py       # Plain text, PDF, Office, Email, Ebook
│   ├── ocr_extractor.py            # Image OCR via Tesseract
│   └── media_extractor.py          # Audio/Video transcription via Whisper
├── translator/                     # Ollama translation engine & chunker
│   └── ollama_translator.py
├── utils/                          # File I/O and batch directory processor
│   └── file_handler.py
├── tessdata/                       # Local Tesseract OCR trained data
├── requirements.txt                # Python dependencies
├── information.md                  # Comprehensive technical documentation
└── README.md                       # High-level overview & quickstart guide
```

---

## 📘 Documentation

For complete technical specifications, architectural details, and developer notes, refer to [`information.md`](file:///home/voldemort/Dev/Python/AI_Project/information.md).
