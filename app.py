import os
import time
import streamlit as st
import pandas as pd
from typing import List, Dict

# Import custom modules
from extractors.document_extractor import DocumentExtractor
from extractors.media_extractor import MediaExtractor
from extractors.ocr_extractor import OcrExtractor
from translator.ollama_translator import OllamaTranslator
from utils.file_handler import FileHandler

# Page Configuration
st.set_page_config(
    page_title="OmniTranslate AI - Multi-Format AI Translation Suite",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium UI design
st.markdown("""
<style>
    /* Dark / Vibrant modern theme styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #9333ea 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
    }
    .card-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success { background-color: #065f46; color: #34d399; }
    .badge-info { background-color: #1e40af; color: #60a5fa; }
    .badge-warning { background-color: #92400e; color: #fbbf24; }
    .badge-ocr { background-color: #701a75; color: #f0abfc; }
    
    /* Code/Text box polish */
    .stTextArea textarea {
        background-color: #0f172a;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>🌐 OmniTranslate AI</h1>
    <p>Translate Documents, Images (OCR), Video & Audio into Any Language using local Ollama LLMs & Whisper AI</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "ollama_host" not in st.session_state:
    st.session_state["ollama_host"] = "http://localhost:11434"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = ""
if "selected_lang" not in st.session_state:
    st.session_state["selected_lang"] = "English"
if "custom_target_lang" not in st.session_state:
    st.session_state["custom_target_lang"] = "Spanish"
if "whisper_model_name" not in st.session_state:
    st.session_state["whisper_model_name"] = "base"
if "output_dir" not in st.session_state:
    st.session_state["output_dir"] = os.path.abspath("./translated_output")
if "save_format" not in st.session_state:
    st.session_state["save_format"] = "txt"
if "custom_instructions" not in st.session_state:
    st.session_state["custom_instructions"] = "Maintain original technical terms and domain jargon."

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Ollama LLM Settings")
    ollama_host = st.text_input("Ollama Host URL", key="ollama_host")
    
    # Fetch Ollama models
    available_models = OllamaTranslator.list_models(ollama_host)
    if available_models:
        if st.session_state["selected_model"] not in available_models:
            st.session_state["selected_model"] = available_models[0]
        model_index = available_models.index(st.session_state["selected_model"])
    else:
        available_models = ["llama3.2:latest"]
        model_index = 0

    selected_model = st.selectbox(
        "Select Ollama Model",
        options=available_models,
        index=model_index,
        key="selected_model_widget"
    )
    st.session_state["selected_model"] = selected_model

    st.markdown("---")
    st.subheader("2. Target Language Settings")
    language_options = [
        "English", "Spanish", "French", "German", "Italian",
        "Portuguese", "Dutch", "Russian", "Japanese", "Chinese (Simplified)",
        "Chinese (Traditional)", "Korean", "Arabic", "Hindi", "Turkish",
        "Polish", "Swedish", "Danish", "Finnish", "Norwegian", "Custom..."
    ]
    lang_idx = language_options.index(st.session_state["selected_lang"]) if st.session_state["selected_lang"] in language_options else 0
    selected_lang = st.selectbox("Select Target Language", options=language_options, index=lang_idx, key="selected_lang_widget")
    st.session_state["selected_lang"] = selected_lang

    if selected_lang == "Custom...":
        target_language = st.text_input("Enter Custom Language", key="custom_target_lang")
    else:
        target_language = selected_lang
    
    st.markdown("---")
    st.subheader("3. Audio / Video Whisper Model")
    whisper_options = ["tiny", "base", "small", "medium", "large"]
    whisper_idx = whisper_options.index(st.session_state["whisper_model_name"]) if st.session_state["whisper_model_name"] in whisper_options else 1
    whisper_model_name = st.selectbox(
        "Whisper Model Size",
        options=whisper_options,
        index=whisper_idx,
        key="whisper_model_widget",
        help="Larger models increase accuracy for audio/video transcription but require more RAM/VRAM."
    )
    st.session_state["whisper_model_name"] = whisper_model_name

    st.markdown("---")
    st.subheader("4. Output Settings")
    output_dir = st.text_input("Output Directory on Disk", key="output_dir")
    
    format_options = ["txt", "md", "docx", "json"]
    format_idx = format_options.index(st.session_state["save_format"]) if st.session_state["save_format"] in format_options else 0
    save_format = st.selectbox("Save Format", options=format_options, index=format_idx, key="save_format_widget")
    st.session_state["save_format"] = save_format

    st.markdown("---")
    st.subheader("5. Custom Translation Instructions")
    custom_instructions = st.text_area(
        "Optional Instructions for LLM",
        key="custom_instructions",
        height=70,
        help="Custom prompt additions passed to Ollama during translation."
    )

# Instantiate Extractor objects & Translator
doc_extractor = DocumentExtractor()
media_extractor = MediaExtractor(model_name=whisper_model_name)
ocr_extractor = OcrExtractor()
translator = OllamaTranslator(host=ollama_host, model_name=selected_model)

# Navigation Tabs
tab_single, tab_directory, tab_info = st.tabs([
    "📄 Single File (Doc / Image / Video / Audio)",
    "📁 Directory Batch Processing",
    "ℹ️ System Info & Supported Formats"
])

# ==========================================
# TAB 1: SINGLE FILE TRANSLATION
# ==========================================
with tab_single:
    st.subheader(f"Translate a Single Document, Image (OCR), Audio, or Video File into {target_language}")
    
    input_mode = st.radio("Input Source", ["Upload File", "Local File Path"], horizontal=True)
    
    file_bytes = None
    file_name = None
    local_path = None

    all_supported_extensions = (
        DocumentExtractor.get_supported_extensions() +
        MediaExtractor.get_supported_extensions() +
        OcrExtractor.get_supported_extensions()
    )

    if input_mode == "Upload File":
        uploaded_file = st.file_uploader(
            f"Choose a file to translate into {target_language}",
            type=[ext.lstrip(".") for ext in sorted(list(set(all_supported_extensions)))]
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file
            file_name = uploaded_file.name
    else:
        local_path = st.text_input("Enter absolute file path on disk (e.g. /home/user/document.pdf, photo.png, or video.mp4)")
        if local_path and local_path.strip():
            expanded_path = os.path.expanduser(local_path.strip())
            if os.path.isfile(expanded_path):
                file_name = os.path.basename(expanded_path)
                file_bytes = expanded_path
            else:
                st.error(f"❌ File not found at path: `{expanded_path}`")

    if file_name:
        is_ocr = OcrExtractor.is_supported(file_name)
        is_media = MediaExtractor.is_supported(file_name)
        is_doc = DocumentExtractor.is_supported(file_name)
        
        col_type, col_model, col_lang = st.columns(3)
        with col_type:
            if is_ocr:
                st.markdown(f'<span class="status-badge badge-ocr">Image OCR ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            elif is_doc:
                st.markdown(f'<span class="status-badge badge-info">Document Type ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            elif is_media:
                st.markdown(f'<span class="status-badge badge-warning">Media Type ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Extension not officially recognized. Will attempt plain text extraction.")
        with col_model:
            st.markdown(f'<span class="status-badge badge-success">Ollama Model: {selected_model}</span>', unsafe_allow_html=True)
        with col_lang:
            st.markdown(f'<span class="status-badge badge-info">Target: {target_language}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(f"🚀 Process & Translate File into {target_language}", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Extraction
                extracted_text = ""
                if is_ocr:
                    status_placeholder.info(f"🔍 Performing Tesseract OCR on image '{file_name}'...")
                    progress_bar.progress(20)
                    extracted_text = ocr_extractor.extract_text(file_bytes, file_name)
                elif is_media:
                    status_placeholder.info(f"🎙️ Transcribing media '{file_name}' with Whisper ({whisper_model_name})...")
                    progress_bar.progress(20)
                    extracted_text = media_extractor.transcribe(
                        file_bytes, file_name,
                        progress_callback=lambda msg: status_placeholder.info(f"🎙️ {msg}")
                    )
                else:
                    status_placeholder.info(f"📑 Extracting text from document '{file_name}'...")
                    progress_bar.progress(20)
                    extracted_text = doc_extractor.extract_text(file_bytes, file_name)

                progress_bar.progress(50)
                
                if not extracted_text.strip():
                    status_placeholder.error("❌ Failed to extract text from the file (empty or unreadable).")
                else:
                    # Step 2: Translation
                    status_placeholder.info(f"🤖 Translating text into {target_language} using Ollama ({selected_model})...")
                    
                    def on_trans_progress(current, total, msg):
                        percent = 50 + int((current / total) * 45)
                        progress_bar.progress(percent)
                        status_placeholder.info(f"🤖 {msg}")

                    translated_text = translator.translate_text(
                        extracted_text,
                        target_language=target_language,
                        custom_instructions=custom_instructions,
                        progress_callback=on_trans_progress
                    )
                    
                    progress_bar.progress(100)
                    status_placeholder.success(f"✅ Translation to {target_language} completed successfully!")
                    
                    # Step 3: Save to Disk
                    saved_path = FileHandler.save_translation_to_file(
                        file_name, extracted_text, translated_text, output_dir, save_format, target_language=target_language
                    )
                    st.success(f"💾 **Result saved to disk**: `{saved_path}`")
                    
                    # Step 4: Display Side-by-Side Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Extracted Text")
                        st.text_area("Extracted", extracted_text, height=350, key="orig_text")
                    with col2:
                        st.subheader(f"{target_language} Translation")
                        st.text_area("Translated", translated_text, height=350, key="trans_text")

                    # Download button
                    with open(saved_path, "rb") as sf:
                        file_download_bytes = sf.read()
                    
                    mime_map = {
                        "txt": "text/plain",
                        "md": "text/markdown",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "json": "application/json"
                    }
                    st.download_button(
                        label=f"📥 Download {target_language} Translation (.{save_format})",
                        data=file_download_bytes,
                        file_name=os.path.basename(saved_path),
                        mime=mime_map.get(save_format, "application/octet-stream")
                    )

            except Exception as e:
                status_placeholder.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)

# ==========================================
# TAB 2: DIRECTORY BATCH TRANSLATION
# ==========================================
with tab_directory:
    st.subheader(f"Translate an Entire Directory of Documents, Images & Media into {target_language}")
    
    dir_path_input = st.text_input(
        "Directory Path to Scan",
        value=os.path.abspath("."),
        help="Enter the folder path containing documents, images, audio, or video files."
    )
    
    col_rec, col_scan = st.columns([2, 1])
    with col_rec:
        recursive_scan = st.checkbox("Include subdirectories (Recursive Scan)", value=True)
    with col_scan:
        scan_btn = st.button("🔍 Scan Directory", use_container_width=True)

    if scan_btn or "scanned_files" in st.session_state:
        if scan_btn:
            expanded_dir = os.path.expanduser(dir_path_input.strip()) if dir_path_input else ""
            if not expanded_dir or not os.path.exists(expanded_dir):
                st.error(f"❌ Directory path does not exist: `{dir_path_input}`")
                st.session_state["scanned_files"] = []
                st.session_state["scanned_dir"] = dir_path_input
            elif not os.path.isdir(expanded_dir):
                st.error(f"❌ Path is not a directory: `{dir_path_input}`")
                st.session_state["scanned_files"] = []
                st.session_state["scanned_dir"] = dir_path_input
            else:
                found_files = FileHandler.scan_directory(dir_path_input, recursive=recursive_scan)
                st.session_state["scanned_files"] = found_files
                st.session_state["scanned_dir"] = expanded_dir
        
        found_files = st.session_state.get("scanned_files", [])
        scanned_dir = st.session_state.get("scanned_dir", dir_path_input)
        
        if not found_files:
            st.warning(f"No supported document, image, or media files found in `{scanned_dir}`.")
        else:
            st.info(f"📂 Found **{len(found_files)}** supported file(s) in `{scanned_dir}`.")
            
            # Helper for type preview
            def get_file_type_label(file_path: str) -> str:
                if OcrExtractor.is_supported(file_path):
                    return "Image (OCR)"
                elif MediaExtractor.is_supported(file_path):
                    return "Audio/Video"
                else:
                    return "Document"

            # Preview files table
            df_files = pd.DataFrame([
                {
                    "File Name": os.path.basename(f),
                    "Type": get_file_type_label(f),
                    "Path": f
                } for f in found_files
            ])
            st.dataframe(df_files, use_container_width=True, height=200)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🚀 Batch Translate All Files to {target_language}", type="primary", use_container_width=True):
                batch_progress = st.progress(0)
                batch_status = st.empty()
                
                results_dict = {}
                saved_paths = []

                for i, file_path in enumerate(found_files):
                    file_name = os.path.basename(file_path)
                    batch_status.info(f"Processing ({i+1}/{len(found_files)}): **{file_name}**...")
                    
                    try:
                        # Extract
                        if OcrExtractor.is_supported(file_name):
                            extracted_text = ocr_extractor.extract_text(file_path, file_name)
                        elif MediaExtractor.is_supported(file_name):
                            extracted_text = media_extractor.transcribe(file_path, file_name)
                        else:
                            extracted_text = doc_extractor.extract_text(file_path, file_name)

                        # Translate
                        if extracted_text.strip():
                            translated_text = translator.translate_text(
                                extracted_text,
                                target_language=target_language,
                                custom_instructions=custom_instructions
                            )
                        else:
                            translated_text = "[Extraction produced no text]"

                        # Save to disk
                        saved_file = FileHandler.save_translation_to_file(
                            file_name, extracted_text, translated_text, output_dir, save_format, target_language=target_language
                        )
                        saved_paths.append(saved_file)
                        with open(saved_file, "rb") as sf:
                            results_dict[os.path.basename(saved_file)] = sf.read()

                    except Exception as err:
                        st.error(f"Error processing {file_name}: {err}")

                    batch_progress.progress(int(((i + 1) / len(found_files)) * 100))

                batch_status.success(f"🎉 Batch translation finished! **{len(saved_paths)}** files translated to {target_language} and saved to `{output_dir}`.")
                
                # Provide ZIP download for all translated files
                if results_dict:
                    zip_data = FileHandler.create_zip_archive(results_dict)
                    lang_suffix = target_language.lower().replace(" ", "_")
                    st.download_button(
                        label=f"📦 Download All Batch Translations (.zip)",
                        data=zip_data,
                        file_name=f"batch_{lang_suffix}_translations.zip",
                        mime="application/zip"
                    )

# ==========================================
# TAB 3: SYSTEM INFO & SUPPORTED FORMATS
# ==========================================
with tab_info:
    st.subheader("System Diagnostics & Supported File Formats")
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.markdown("### 🖥️ Ollama Service Status")
        st.write(f"**Host URL:** `{ollama_host}`")
        models = OllamaTranslator.list_models(ollama_host)
        st.write(f"**Installed Models ({len(models)}):**")
        for m in models:
            st.markdown(f"- `{m}`")
            
        st.markdown("---")
        st.markdown("### 🎙️ OpenAI Whisper Models")
        cached_whisper = MediaExtractor.list_cached_models()
        if cached_whisper:
            st.write(f"**Downloaded Local Models ({len(cached_whisper)}):**")
            for wm in cached_whisper:
                st.markdown(f"- `{wm}`")
        else:
            st.warning("⚠️ No Whisper models downloaded locally yet (`~/.cache/whisper`). The selected model will be downloaded automatically when an audio/video file is processed (internet required on first run).")
            
    with col_diag2:
        st.markdown("### 📑 Supported Document Formats")
        doc_exts = DocumentExtractor.get_supported_extensions()
        st.write(", ".join([f"`{e}`" for e in doc_exts]))
        
        st.markdown("### 🔍 Supported Image Formats (Tesseract OCR)")
        ocr_exts = OcrExtractor.get_supported_extensions()
        st.write(", ".join([f"`{e}`" for e in ocr_exts]))

        st.markdown("### 🎙️ Supported Video & Audio Formats")
        media_exts = MediaExtractor.get_supported_extensions()
        st.write(", ".join([f"`{e}`" for e in media_exts]))

    st.markdown("---")
    st.markdown(f"""
    ### 💡 Quick Tips
    1. **Target Language**: Choose any target language from the sidebar preset list or type a custom target language.
    2. **Image & Scanned PDF OCR**: Upload images (`.png`, `.jpg`, `.bmp`, `.tiff`, `.webp`) or scanned PDFs. Tesseract OCR will automatically extract the text before translating.
    3. **Ollama LLM**: Ensure your Ollama server is running (`ollama serve`). You can pull any translation model like `ollama pull llama3.2` or `ollama pull qwen3-coder`.
    4. **Whisper Transcription**: For faster video/audio transcription, use `tiny` or `base`. For higher multilingual audio precision, select `small` or `medium`.
    5. **Directory Processing**: Specify any local directory path (e.g. `/home/user/Documents`). The app will scan all documents, images, audio/video files, translate each into {target_language}, and store the output on disk.
    """)

