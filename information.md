# 🌐 OmniTranslate AI — Comprehensive System & Program Documentation

**OmniTranslate AI** is an enterprise-grade, multi-format AI translation application built with **Python**, **Streamlit**, **Ollama**, **Tesseract OCR**, and **OpenAI Whisper**.

It processes documents, images, scanned PDFs, audio recordings, and video files, extracts plain text/transcripts, and translates the content into any target language using local LLMs (Large Language Models) served by **Ollama**.

---

## 📋 Table of Contents
1. [Overview & Key Features](#-overview--key-features)
2. [Supported File Formats](#-supported-file-formats)
3. [System Architecture & Directory Structure](#-system-architecture--directory-structure)
4. [Component & Module Breakdown](#-component--module-breakdown)
   - [1. App Interface (`app.py`)](#1-app-interface-apppy)
   - [2. Document Extractor (`extractors/document_extractor.py`)](#2-document-extractor-extractorsdocument_extractorpy)
   - [3. OCR Extractor (`extractors/ocr_extractor.py`)](#3-ocr-extractor-extractorsocr_extractorpy)
   - [4. Media Extractor (`extractors/media_extractor.py`)](#4-media-extractor-extractorsmedia_extractorpy)
   - [5. Ollama Translator (`translator/ollama_translator.py`)](#5-ollama-translator-translatorollama_translatorpy)
   - [6. File Handler (`utils/file_handler.py`)](#6-file-handler-utilsfile_handlerpy)
5. [Requirements & Dependencies](#-requirements--dependencies)
6. [Installation & Setup Guide](#-installation--setup-guide)
7. [Running the Application](#-running-the-application)
8. [Configuring the Server Port](#-configuring-the-server-port)
9. [Stopping the Server](#-stopping-the-server)
10. [Standalone Compilation (PyInstaller)](#-standalone-compilation-pyinstaller)

---

## 🌐 Overview & Key Features

- **Multi-Format Ingestion**: Ingests files across 40+ extensions, including standard documents, eBooks, e-mails, spreadsheets, images, audio, and video formats.
- **Scanned PDF & OCR Support**: Automatically detects image-only/scanned PDF pages and routes them to Tesseract OCR via `pdf2image` and `pytesseract`.
- **Speech-to-Text Transcription**: Extracts audio tracks from video (`.mp4`, `.mkv`, `.avi`, etc.) and audio (`.mp3`, `.wav`, `.m4a`, etc.) files using OpenAI Whisper.
- **Local & Private LLM Translations**: All translation queries are routed to local LLM models (e.g. `llama3.2`, `qwen3-coder`, `aya-expanse`) via Ollama. No data is sent to cloud APIs.
- **Smart Text Chunking**: Breaks large documents into context-aware chunks (paragraph/sentence boundaries) to prevent token limit truncations during LLM generation.
- **Single File & Batch Directory Modes**:
  - **Single File**: Side-by-side original text vs. translated output preview, downloadable in multiple formats.
  - **Directory Scan**: Recursively scans folders, processes all matching files, saves output directly to disk, and packages results into ZIP archives.
- **Multi-Format Export**: Save translations to `.txt`, `.md`, `.docx`, or `.json` formats.

---

## 📂 Supported File Formats

| Category | Supported File Extensions | Underlying Engine / Library |
| :--- | :--- | :--- |
| **Documents** | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.xls`, `.csv`, `.epub`, `.eml`, `.msg`, `.html`, `.htm` | `pypdf`, `python-docx`, `python-pptx`, `pandas`, `openpyxl`, `ebooklib`, `extract_msg`, `BeautifulSoup4` |
| **Code & Text** | `.txt`, `.md`, `.rtf`, `.json`, `.yaml`, `.yml`, `.xml`, `.log`, `.rst`, `.tex`, `.ini`, `.conf`, `.cfg`, `.env`, `.properties`, `.toml`, `.sql`, `.srt`, `.vtt`, `.py`, `.js`, `.ts`, `.css`, `.cpp`, `.c`, `.h`, `.hpp`, `.hxx`, `.java`, `.cs`, `.go`, `.php`, `.rb`, `.swift`, `.kt`, `.scala`, `.r`, `.sh`, `.bash`, `.jsx`, `.tsx`, `.vue` | Built-in Python I/O (UTF-8 encoding fallback) |
| **Images & Scanned PDFs** | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.pnm`, `.pbm`, `.pgm`, plus scanned PDF pages | Tesseract OCR (`pytesseract`), `PIL` / `Pillow`, `pdf2image` |
| **Audio** | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma` | OpenAI Whisper (`whisper`) |
| **Video** | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv` | FFmpeg + OpenAI Whisper |

---

## 🏗️ System Architecture & Directory Structure

```
AI_Project/
├── app.py                          # Streamlit UI & application workflow controller
├── omnitranslate_ai.spec           # PyInstaller build specification file
├── requirements.txt                # Python package dependency definitions
├── README.md                       # High-level overview & quickstart guide
├── information.md                  # Comprehensive technical documentation
├── .streamlit/
│   └── config.toml                 # Streamlit server port & execution configuration
├── extractors/
│   ├── __init__.py
│   ├── document_extractor.py       # Extract text from docs, code, PDF, Office, e-mails
│   ├── ocr_extractor.py            # OCR text extraction from images via Tesseract
│   └── media_extractor.py          # Audio & video transcription via OpenAI Whisper
├── translator/
│   ├── __init__.py
│   └── ollama_translator.py        # Ollama API interface with chunking & translation prompts
├── utils/
│   ├── __init__.py
│   └── file_handler.py             # Directory scanner, disk writer, and ZIP packager
├── venv/                           # Python 3.14 Virtual Environment
└── tessdata/                       # Local Tesseract OCR language trained data
```

---

## 🧩 Component & Module Breakdown

### 1. App Interface (`app.py`)
- **Role**: Provides a dark-themed, wide-layout web dashboard using Streamlit.
- **Sidebar Options**:
  - **Single File Mode** vs. **Directory Scan Mode** selection.
  - **Ollama Host URL** input (default: `http://localhost:11434`).
  - **Ollama Model Picker** (fetches installed models dynamically).
  - **Target Language Picker** (English, Spanish, French, German, Italian, Portuguese, Japanese, Chinese, Korean, Arabic, or custom).
  - **Whisper Model Picker** (`tiny`, `base`, `small`, `medium`, `large`).
  - **Custom Prompt Instructions** field for fine-tuning translation style.
  - **Export Format Selector** (`txt`, `md`, `docx`, `json`).

### 2. Document Extractor (`extractors/document_extractor.py`)
- **Class**: `DocumentExtractor`
- **Responsibilities**:
  - Checks extension support using `is_supported(filename)`.
  - Reads binary bytes or file paths and routes to specific format handlers:
    - `_extract_pdf`: Extracts native text using `pypdf.PdfReader`. If text content is under 10 characters (indicating a scan), falls back to `pdf2image` + `OcrExtractor`.
    - `_extract_docx`: Extracts paragraph text and table cell contents.
    - `_extract_pptx`: Extracts text from slide shapes and tables.
    - `_extract_excel` / `_extract_csv`: Reads sheets via `pandas` and formats tables into clean string representations.
    - `_extract_epub`: Extracts HTML chapters from eBook files via `ebooklib` and strips tags.
    - `_extract_eml` / `_extract_msg`: Extracts body text and subject headers from email files using `extract_msg` and `email`.
    - `_extract_html`: Strips scripts/styles and extracts text via `BeautifulSoup`.
    - `_extract_plain_text`: Text fallback with UTF-8 / Latin-1 decoding.

### 3. OCR Extractor (`extractors/ocr_extractor.py`)
- **Class**: `OcrExtractor`
- **Responsibilities**:
  - Ingests image inputs (`.png`, `.jpg`, `.bmp`, `.webp`, etc.).
  - Converts image color modes to `RGB` or `L`.
  - Checks for local `tessdata/eng.traineddata` path and configures `pytesseract` accordingly.
  - Executes `pytesseract.image_to_string` and returns cleaned plain text.

### 4. Media Extractor (`extractors/media_extractor.py`)
- **Class**: `MediaExtractor`
- **Responsibilities**:
  - Instantiated with model size (e.g. `"base"`).
  - Checks cached models in `~/.cache/whisper`.
  - Downloads model weights automatically or loads custom local `.pt` files.
  - Calls `model.transcribe(media_path)` to transcribe audio tracks from video and sound files.

### 5. Ollama Translator (`translator/ollama_translator.py`)
- **Class**: `OllamaTranslator`
- **Responsibilities**:
  - `list_models(host)`: Queries the local Ollama daemon for installed LLM models.
  - `chunk_text(text, max_chars_per_chunk)`: Splitting logic:
    1. Splits on paragraph breaks (`\n`).
    2. If a paragraph exceeds length limit, splits further by sentence boundaries (`.!?`).
  - `translate_text(...)`: Sends chunked system/user prompts to `ollama.Client.chat()` and concatenates translated outputs.

### 6. File Handler (`utils/file_handler.py`)
- **Class**: `FileHandler`
- **Responsibilities**:
  - `scan_directory(dir_path, recursive)`: Ignores `venv`, `.git`, `node_modules`, `__pycache__`, etc., and lists supported files.
  - `save_translation_to_file(...)`: Formats and writes output files (`.txt`, `.md`, `.docx`, `.json`) to disk.
  - `create_zip_archive(files_dict)`: Compiles translated outputs into a downloadable ZIP buffer.

---

## 📦 Requirements & Dependencies

Dependencies defined in [`requirements.txt`](file:///home/voldemort/Dev/Python/AI_Project/requirements.txt):

```text
streamlit>=1.30.0
ollama>=0.1.6
openai-whisper>=20231117
pypdf>=4.0.0
python-docx>=1.1.0
python-pptx>=0.6.23
pandas>=2.0.0
openpyxl>=3.1.0
extract-msg>=0.47.0
ebooklib>=0.18
beautifulsoup4>=4.12.0
pytesseract>=0.3.10
pdf2image>=1.17.0
```

### System Tools Required:
1. **Python**: Version 3.10 or higher.
2. **FFmpeg**: Required for audio/video decoding (used by OpenAI Whisper).
3. **Tesseract OCR**: Binary installed on system (or local `tessdata` folder).
4. **Ollama**: Local service for LLM execution ([ollama.com](https://ollama.com/)).

---

## 🛠️ Installation & Setup Guide

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/voldemort/Dev/Python/AI_Project
   ```

2. **Activate the Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```powershell
     venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies** (if creating a new environment):
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running & model is pulled**:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

---

## 🚀 Running the Application

Start the Streamlit application using python/streamlit within the active virtual environment:

```bash
streamlit run app.py
```

Upon launching, the app displays:
- **Local Access URL**: `http://localhost:8503`
- **Network Access URL**: `http://<YOUR-IP>:8503`

---

## ⚙️ Configuring the Server Port

You can change the server port using any of these methods:

### Method 1: Command Line Flag (Quickest)
```bash
streamlit run app.py --server.port 8501
```

### Method 2: Permanent Change in `.streamlit/config.toml`
Edit [.streamlit/config.toml](file:///home/voldemort/Dev/Python/AI_Project/.streamlit/config.toml):
```toml
[server]
port = 8501
headless = false
```

### Method 3: Environment Variable
```bash
STREAMLIT_SERVER_PORT=8501 streamlit run app.py
```

---

## 🛑 Stopping the Server

- **Active Terminal**: Press `Ctrl + C` in the running terminal.
- **Terminal Command**: Run `pkill -f "streamlit run app.py"`.
- **AI Session**: Reply to the assistant asking to stop the server.

---

## 🛠️ Standalone Compilation (PyInstaller)

To compile the standalone application executable using **PyInstaller**, use the included specification file [`omnitranslate_ai.spec`](file:///home/voldemort/Dev/Python/AI_Project/omnitranslate_ai.spec):

```bash
pyinstaller omnitranslate_ai.spec
```

This packages the application along with all project dependencies, internal modules (`extractors`, `translator`, `utils`), configuration (`.streamlit`), and OCR data (`tessdata`) into the `dist/` folder.
