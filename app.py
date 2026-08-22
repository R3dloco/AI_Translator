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

# Session state initialization
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "🔮 Vibrant Dark Mode"
if "ollama_host" not in st.session_state:
    st.session_state["ollama_host"] = "http://localhost:11434"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = ""
if "enable_streaming" not in st.session_state:
    st.session_state["enable_streaming"] = True
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

# Read theme_mode_widget at top of execution cycle so theme updates instantly
current_theme = st.session_state.get("theme_mode_widget", st.session_state.get("theme_mode", "🔮 Vibrant Dark Mode"))
st.session_state["theme_mode"] = current_theme

# Theme Color Palettes Dictionary
THEMES = {
    "🔮 Vibrant Dark Mode": {
        "bg": "#0e1117", "sidebar": "#161b22", "card": "#1e293b", "text": "#e0e6ed",
        "header": "linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #9333ea 100%)",
        "border": "#334155", "input_bg": "#0f172a", "input_text": "#f1f5f9",
        "btn": "#3b82f6", "btn_hover": "#2563eb", "badge_success_bg": "#065f46", "badge_success_text": "#34d399"
    },
    "⚪ Clean Light Mode": {
        "bg": "#f8fafc", "sidebar": "#f1f5f9", "card": "#ffffff", "text": "#0f172a",
        "header": "linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%)",
        "border": "#cbd5e1", "input_bg": "#ffffff", "input_text": "#0f172a",
        "btn": "#2563eb", "btn_hover": "#1d4ed8", "badge_success_bg": "#d1fae5", "badge_success_text": "#047857"
    },
    "🌌 Cyberpunk Neon": {
        "bg": "#090a0f", "sidebar": "#0d0e17", "card": "#121424", "text": "#e2e8f0",
        "header": "linear-gradient(135deg, #7928ca 0%, #ff0080 50%, #00f3ff 100%)",
        "border": "#00f3ff", "input_bg": "#16192e", "input_text": "#00f3ff",
        "btn": "#ff007f", "btn_hover": "#d6006b", "badge_success_bg": "#00f3ff33", "badge_success_text": "#00f3ff"
    },
    "🌲 Nordic Emerald": {
        "bg": "#111827", "sidebar": "#1f2937", "card": "#1f2937", "text": "#f3f4f6",
        "header": "linear-gradient(135deg, #064e3b 0%, #059669 50%, #10b981 100%)",
        "border": "#374151", "input_bg": "#111827", "input_text": "#f3f4f6",
        "btn": "#10b981", "btn_hover": "#059669", "badge_success_bg": "#064e3b", "badge_success_text": "#34d399"
    },
    "☀️ Solarized Warm Light": {
        "bg": "#fdf6e3", "sidebar": "#eee8d5", "card": "#fffbf0", "text": "#073642",
        "header": "linear-gradient(135deg, #b58900 0%, #cb4b16 50%, #d33682 100%)",
        "border": "#d33682", "input_bg": "#ffffff", "input_text": "#073642",
        "btn": "#d33682", "btn_hover": "#b58900", "badge_success_bg": "#eee8d5", "badge_success_text": "#b58900"
    }
}

t = THEMES.get(current_theme, THEMES["🔮 Vibrant Dark Mode"])

css = f"""
<style>
    .stApp, .stApp > header {{
        background-color: {t["bg"]} !important;
        color: {t["text"]} !important;
    }}

    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stAppHeader {{
        background-color: {t["bg"]} !important;
        color: {t["text"]} !important;
    }}
    header[data-testid="stHeader"] *,
    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] *,
    .stAppHeader * {{
        color: {t["text"]} !important;
        fill: {t["text"]} !important;
    }}
    header[data-testid="stHeader"] button,
    [data-testid="stToolbar"] button {{
        background-color: {t["card"]} !important;
        color: {t["text"]} !important;
        border: 1px solid {t["border"]} !important;
    }}

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {t["sidebar"]} !important;
        border-right: 1px solid {t["border"]} !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        color: {t["text"]} !important;
    }}
    .stApp p, .stApp span, .stApp label, .stApp [data-testid="stWidgetLabel"],
    .stMarkdown, h1, h2, h3, h4, h5, h6 {{
        color: {t["text"]} !important;
    }}
    .main-header {{
        background: {t["header"]} !important;
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .main-header h1, .main-header p {{
        color: #ffffff !important;
    }}
    .card-box {{
        background-color: {t["card"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }}
    .status-badge {{
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .badge-success {{ background-color: {t["badge_success_bg"]} !important; color: {t["badge_success_text"]} !important; }}
    .badge-info {{ background-color: {t["card"]} !important; color: {t["btn"]} !important; border: 1px solid {t["border"]} !important; }}
    .badge-warning {{ background-color: #92400e !important; color: #fbbf24 !important; }}
    .badge-ocr {{ background-color: #701a75 !important; color: #f0abfc !important; }}
    .badge-url {{ background-color: #0369a1 !important; color: #7dd3fc !important; }}
    
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div {{
        background-color: {t["input_bg"]} !important;
        color: {t["input_text"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] svg {{
        color: {t["input_text"]} !important;
        fill: {t["input_text"]} !important;
    }}
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {{
        background-color: {t["card"]} !important;
        color: {t["text"]} !important;
    }}
    li[role="option"] * {{
        color: {t["text"]} !important;
    }}
    li[role="option"]:hover, li[aria-selected="true"] {{
        background-color: {t["sidebar"]} !important;
    }}

    /* Code Blocks & Text Display Styling */
    .stCodeBlock, pre, code, [data-testid="stCodeBlock"], [data-testid="stMarkdownContainer"] {{
        background-color: {t["input_bg"]} !important;
        color: {t["input_text"]} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stCodeBlock"] code, .stCodeBlock span {{
        color: {t["input_text"]} !important;
    }}

    /* File Uploader Component Styling */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {t["input_bg"]} !important;
        border: 2px dashed {t["border"]} !important;
        border-radius: 10px !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] svg {{
        color: {t["text"]} !important;
        fill: {t["text"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button {{
        background-color: {t["btn"]} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] button:hover {{
        background-color: {t["btn_hover"]} !important;
        color: #ffffff !important;
    }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)



# Function: Render Desktop-Grade File Explorer UI
def render_file_explorer(
    state_key: str,
    title: str = "Desktop File Explorer",
    on_select_folder: Optional[Callable[[str], None]] = None,
    show_files: bool = True
):
    """Renders a desktop-grade File Explorer UI with detailed file metadata."""
    if state_key not in st.session_state:
        st.session_state[state_key] = os.path.abspath(".")

    curr_path = st.session_state[state_key]
    if not os.path.exists(curr_path) or not os.path.isdir(curr_path):
        curr_path = os.path.abspath(".")
        st.session_state[state_key] = curr_path

    st.markdown(f"#### 🖥️ {title}")
    st.caption(f"📍 Location: `{curr_path}`")

    # Breadcrumb Navigation Bar
    path_parts = [p for p in curr_path.split(os.sep) if p]
    bc_cols = st.columns(min(len(path_parts) + 2, 8))
    with bc_cols[0]:
        if st.button("🏠 /", key=f"{state_key}_bc_root"):
            st.session_state[state_key] = "/"
            st.rerun()

    running_path = ""
    for idx, part in enumerate(path_parts):
        running_path += "/" + part
        col_target = (idx + 1) % len(bc_cols)
        with bc_cols[col_target]:
            if st.button(part, key=f"{state_key}_bc_{idx}_{part}"):
                st.session_state[state_key] = running_path
                st.rerun()

    # Action buttons: Up One Level & Select Folder
    parent_dir = os.path.dirname(curr_path)
    c_up, c_sel = st.columns([1, 2])
    with c_up:
        if parent_dir and parent_dir != curr_path:
            if st.button("⬆️ Up One Level", key=f"{state_key}_up", use_container_width=True):
                st.session_state[state_key] = parent_dir
                st.rerun()
    with c_sel:
        if on_select_folder:
            if st.button(f"✅ Select `{os.path.basename(curr_path) or '/'}`", key=f"{state_key}_sel_curr", type="primary", use_container_width=True):
                on_select_folder(curr_path)
                st.rerun()

    # Read items
    try:
        entries = sorted(os.listdir(curr_path))
    except Exception as e:
        st.error(f"Cannot read directory: {e}")
        return

    subdirs = []
    files = []
    for item in entries:
        if item.startswith("."):
            continue
        full_item = os.path.join(curr_path, item)
        if os.path.isdir(full_item):
            subdirs.append(item)
        elif show_files and os.path.isfile(full_item):
            files.append(item)

    st.markdown("---")
    
    # Render File Explorer Header Row
    h1, h2, h3, h4 = st.columns([0.6, 3.5, 1.5, 1.8])
    with h1: st.caption("**Type**")
    with h2: st.caption("**Name**")
    with h3: st.caption("**Size**")
    with h4: st.caption("**Modified / Action**")

    # Render Subfolders
    for sd in subdirs:
        full_sd = os.path.join(curr_path, sd)
        r1, r2, r3, r4 = st.columns([0.6, 3.5, 1.5, 1.8])
        with r1: st.write("📁")
        with r2:
            if st.button(f"{sd}/", key=f"{state_key}_open_{sd}", use_container_width=True):
                st.session_state[state_key] = full_sd
                st.rerun()
        with r3: st.caption("Folder")
        with r4:
            if on_select_folder:
                if st.button("Select", key=f"{state_key}_choose_{sd}"):
                    on_select_folder(full_sd)
                    st.rerun()


    # Render Files
    if show_files:
        for f in files:
            full_f = os.path.join(curr_path, f)
            ext = os.path.splitext(f)[1].lower()

            if OcrExtractor.is_supported(f):
                icon = "🖼️"
            elif MediaExtractor.is_supported(f):
                icon = "🎞️" if ext in (".mp4", ".mkv", ".avi", ".mov", ".webm") else "🎵"
            elif DocumentExtractor.is_supported(f):
                icon = "📄"
            else:
                icon = "📄"

            try:
                stat = os.stat(full_f)
                sz = stat.st_size
                if sz < 1024:
                    sz_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    sz_str = f"{sz / 1024:.1f} KB"
                else:
                    sz_str = f"{sz / (1024 * 1024):.1f} MB"
                mod_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            except Exception:
                sz_str = "-"
                mod_str = "-"

            r1, r2, r3, r4 = st.columns([0.6, 3.5, 1.5, 1.8])
            with r1: st.write(icon)
            with r2: st.write(f)
            with r3: st.caption(sz_str)
            with r4: st.caption(mod_str)

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>🌐 OmniTranslate AI</h1>
    <p>Multi-Format AI Translation Suite • Documents, Images (OCR), Video & Audio, Web URLs into Any Language via Ollama LLMs & Whisper AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration & Controls")
    
    st.subheader("🎨 Appearance & Theme Palette")
    theme_options = [
        "🔮 Vibrant Dark Mode",
        "⚪ Clean Light Mode",
        "🌌 Cyberpunk Neon",
        "🌲 Nordic Emerald",
        "☀️ Solarized Warm Light"
    ]
    theme_idx = theme_options.index(current_theme) if current_theme in theme_options else 0
    theme_mode = st.selectbox(
        "Theme Preset",
        options=theme_options,
        index=theme_idx,
        key="theme_mode_widget"
    )
    st.session_state["theme_mode"] = theme_mode



    st.markdown("---")
    st.subheader("1. Ollama LLM Settings")
    ollama_host = st.text_input("Ollama Host URL", key="ollama_host")
    
    # Fetch Ollama models safely
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

    enable_streaming = st.checkbox("⚡ Real-time Token Streaming", key="enable_streaming")

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
        help="Larger models increase transcription accuracy but require more VRAM/RAM."
    )
    st.session_state["whisper_model_name"] = whisper_model_name

    st.markdown("---")
    st.subheader("4. Output Settings")
    
    # Sidebar Output Directory Browser Toggle
    if "show_out_browser" not in st.session_state:
        st.session_state["show_out_browser"] = False
    if "out_tree_browse_dir" not in st.session_state:
        st.session_state["out_tree_browse_dir"] = st.session_state.get("output_dir", os.path.abspath("./translated_output"))

    browse_label = "❌ Close File Explorer" if st.session_state["show_out_browser"] else "🖥️ Open File Explorer"
    if st.button(browse_label, key="toggle_out_browser_btn", use_container_width=True):
        st.session_state["show_out_browser"] = not st.session_state["show_out_browser"]
        st.rerun()

    if st.session_state["show_out_browser"]:
        with st.container():
            def set_out_dir(folder_path):
                st.session_state["output_dir"] = folder_path
                st.session_state["show_out_browser"] = False
                st.rerun()

            render_file_explorer("out_tree_browse_dir", title="Output Directory Picker", on_select_folder=set_out_dir, show_files=False)

    output_dir = st.text_input("Output Directory on Disk", key="output_dir")
    
    format_options = ["txt", "md", "docx", "json", "srt", "vtt"]
    format_idx = format_options.index(st.session_state["save_format"]) if st.session_state["save_format"] in format_options else 0
    save_format = st.selectbox("Save Format", options=format_options, index=format_idx, key="save_format_widget")
    st.session_state["save_format"] = save_format

    st.markdown("---")
    st.subheader("5. Custom Translation Instructions")
    custom_instructions = st.text_area(
        "Optional Instructions for LLM",
        key="custom_instructions",
        height=70,
        help="Custom prompt guidance passed to Ollama during translation."
    )

# Instantiate Extractor objects & Translator
doc_extractor = DocumentExtractor()
media_extractor = MediaExtractor(model_name=whisper_model_name)
ocr_extractor = OcrExtractor()
translator = OllamaTranslator(host=ollama_host, model_name=selected_model)

# Navigation Tabs
tab_single, tab_directory, tab_info = st.tabs([
    "📄 Single Source (Doc / Image / Media / URL)",
    "📁 Directory Batch Processing",
    "ℹ️ System Diagnostics & Formats"
])

# ==========================================
# TAB 1: SINGLE SOURCE TRANSLATION
# ==========================================
with tab_single:
    st.subheader(f"Translate Document, Image, Video/Audio, or Web URL into {target_language}")
    
    input_mode = st.radio("Select Input Source Type", ["Upload File", "Local File Path", "🌐 Web Page URL"], horizontal=True)
    
    file_bytes = None
    file_name = None
    local_path = None
    url_input = None

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
    elif input_mode == "Local File Path":
        local_path = st.text_input("Enter absolute file path on disk (e.g. /home/user/document.pdf or video.mp4)")
        if local_path and local_path.strip():
            expanded_path = os.path.expanduser(local_path.strip())
            if os.path.isfile(expanded_path):
                file_name = os.path.basename(expanded_path)
                file_bytes = expanded_path
            else:
                st.error(f"❌ File not found at path: `{expanded_path}`")
    else: # Web Page URL
        url_input = st.text_input("Enter Web Article / Page URL (e.g. https://example.com/article)")
        if url_input and url_input.strip():
            file_name = "web_article.html"

    if file_name:
        is_url = (input_mode == "🌐 Web Page URL")
        is_ocr = OcrExtractor.is_supported(file_name) and not is_url
        is_media = MediaExtractor.is_supported(file_name) and not is_url
        is_doc = DocumentExtractor.is_supported(file_name) and not is_url
        
        col_type, col_model, col_lang = st.columns(3)
        with col_type:
            if is_url:
                st.markdown('<span class="status-badge badge-url">Web Page URL</span>', unsafe_allow_html=True)
            elif is_ocr:
                st.markdown(f'<span class="status-badge badge-ocr">Image OCR ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            elif is_doc:
                st.markdown(f'<span class="status-badge badge-info">Document ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            elif is_media:
                st.markdown(f'<span class="status-badge badge-warning">Media ({os.path.splitext(file_name)[1]})</span>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Plain text extraction mode")
        with col_model:
            st.markdown(f'<span class="status-badge badge-success">Ollama: {selected_model}</span>', unsafe_allow_html=True)
        with col_lang:
            st.markdown(f'<span class="status-badge badge-info">Target: {target_language}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(f"🚀 Process & Translate into {target_language}", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Text / Media Extraction
                extracted_text = ""
                media_info = None

                if is_url:
                    status_placeholder.info(f"🌐 Scraping web article from `{url_input}`...")
                    progress_bar.progress(20)
                    extracted_text = doc_extractor.extract_url_text(url_input)
                elif is_ocr:
                    status_placeholder.info(f"🔍 Extracting text via Tesseract OCR from '{file_name}'...")
                    progress_bar.progress(20)
                    extracted_text = ocr_extractor.extract_text(file_bytes, file_name)
                elif is_media:
                    status_placeholder.info(f"🎙️ Transcribing media '{file_name}' with Whisper ({whisper_model_name})...")
                    progress_bar.progress(20)
                    media_info = media_extractor.transcribe_full(
                        file_bytes, file_name,
                        progress_callback=lambda msg: status_placeholder.info(f"🎙️ {msg}")
                    )
                    extracted_text = media_info["text"]
                    detected_lang = media_info.get("language", "unknown")
                    status_placeholder.success(f"🎙️ Whisper Transcription Complete! Detected Language: **{detected_lang.upper()}**")
                else:
                    status_placeholder.info(f"📑 Extracting document text from '{file_name}'...")
                    progress_bar.progress(20)
                    extracted_text = doc_extractor.extract_text(file_bytes, file_name)

                progress_bar.progress(40)
                
                if not extracted_text.strip():
                    status_placeholder.error("❌ Extraction failed or returned empty text.")
                else:
                    # Provide original SRT/VTT download if media
                    if is_media and media_info:
                        st.markdown("### 🎙️ Subtitles & Transcription")
                        col_srt, col_vtt = st.columns(2)
                        with col_srt:
                            st.download_button(
                                "📥 Download Original SRT Subtitles",
                                data=media_info["srt"],
                                file_name=f"{os.path.splitext(file_name)[0]}_original.srt",
                                mime="text/plain"
                            )
                        with col_vtt:
                            st.download_button(
                                "📥 Download Original VTT Subtitles",
                                data=media_info["vtt"],
                                file_name=f"{os.path.splitext(file_name)[0]}_original.vtt",
                                mime="text/plain"
                            )

                    # Step 2: Translation Phase (Streaming or Batch)
                    translated_text = ""
                    col_orig, col_trans = st.columns(2)
                    
                    with col_orig:
                        st.subheader("Original Extracted Text")
                        st.text_area("Original", extracted_text, height=380, key="orig_text_display")

                    with col_trans:
                        st.subheader(f"{target_language} Translation")
                        trans_container = st.empty()

                        if enable_streaming:
                            status_placeholder.info(f"⚡ Streaming real-time translation into {target_language} via Ollama ({selected_model})...")
                            stream_gen = translator.stream_translate_text(
                                extracted_text,
                                target_language=target_language,
                                custom_instructions=custom_instructions
                            )
                            translated_parts = []
                            for token in stream_gen:
                                translated_parts.append(token)
                                full_so_far = "".join(translated_parts)
                                trans_container.markdown(full_so_far)
                            translated_text = "".join(translated_parts)
                            # Render full result in scrollable text_area after streaming completes
                            trans_container.text_area("Translated", translated_text, height=380, key="trans_text_display")
                        else:
                            status_placeholder.info(f"🤖 Translating text into {target_language} via Ollama ({selected_model})...")
                            translated_text = translator.translate_text(
                                extracted_text,
                                target_language=target_language,
                                custom_instructions=custom_instructions,
                                progress_callback=lambda curr, tot, msg: status_placeholder.info(f"🤖 {msg}")
                            )
                            trans_container.text_area("Translated", translated_text, height=380, key="trans_text_display")


                    progress_bar.progress(100)
                    status_placeholder.success(f"✅ Translation to {target_language} finished successfully!")

                    # Floating Quick Action Toolbar Component
                    st.markdown("---")
                    st.markdown("### 📋 Quick Action Toolbar")
                    tb_c1, tb_c2, tb_c3, tb_c4 = st.columns([1.5, 1.5, 1.5, 2])
                    
                    orig_words = len(extracted_text.split()) if extracted_text else 0
                    trans_words = len(translated_text.split()) if translated_text else 0

                    with tb_c1:
                        st.metric("📝 Source Words", f"{orig_words:,}")
                    with tb_c2:
                        st.metric("🌐 Target Words", f"{trans_words:,}")
                    with tb_c3:
                        st.metric("⚡ Output Format", f".{save_format.upper()}")
                    with tb_c4:
                        if st.button("🧹 Clear Results", key="clear_res_btn", use_container_width=True):
                            st.rerun()

                    # 1-Click Copy Box
                    with st.expander("📋 1-Click Copy Translated Text (Click top-right icon)", expanded=True):
                        st.code(translated_text, language=None)
                    
                    # Step 3: Save to Disk & Download
                    saved_path = FileHandler.save_translation_to_file(
                        file_name, extracted_text, translated_text, output_dir, save_format, target_language=target_language
                    )
                    st.success(f"💾 **Translation saved on disk**: `{saved_path}`")
                    
                    with open(saved_path, "rb") as sf:
                        file_download_bytes = sf.read()
                    
                    mime_map = {
                        "txt": "text/plain",
                        "md": "text/markdown",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "json": "application/json",
                        "srt": "text/plain",
                        "vtt": "text/vtt"
                    }
                    st.download_button(
                        label=f"📥 Download {target_language} Result (.{save_format})",
                        data=file_download_bytes,
                        file_name=os.path.basename(saved_path),
                        mime=mime_map.get(save_format, "application/octet-stream"),
                        type="primary"
                    )


            except Exception as e:
                status_placeholder.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)

# ==========================================
# TAB 2: DIRECTORY BATCH TRANSLATION
# ==========================================
with tab_directory:
    st.subheader(f"Translate an Entire Directory of Documents, Images & Media into {target_language}")
    
    def set_batch_dir(folder_path):
        st.session_state["dir_path_input_val"] = folder_path
        st.session_state["scanned_files"] = FileHandler.scan_directory(folder_path, recursive=st.session_state.get("rec_scan_val", True))
        st.session_state["scanned_dir"] = folder_path

    # Desktop File Explorer UI Component
    with st.expander("🖥️ Desktop File Explorer (Navigate folders & view detailed file metadata)", expanded=True):
        render_file_explorer("tree_browse_dir", title="Directory Batch Explorer", on_select_folder=set_batch_dir, show_files=True)

    st.markdown("---")
    
    default_dir_val = st.session_state.get("dir_path_input_val", st.session_state.get("tree_browse_dir", os.path.abspath(".")))
    dir_path_input = st.text_input(
        "Directory Path to Scan",
        value=default_dir_val,
        help="Folder path containing documents, images, audio, or video files."
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
            
            def get_file_type_label(file_path: str) -> str:
                if OcrExtractor.is_supported(file_path):
                    return "Image (OCR)"
                elif MediaExtractor.is_supported(file_path):
                    return "Audio/Video"
                else:
                    return "Document"

            df_files = pd.DataFrame([
                {
                    "File Name": os.path.basename(f),
                    "Type": get_file_type_label(f),
                    "Path": f
                } for f in found_files
            ])
            st.dataframe(df_files, use_container_width=True, height=220)

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

                batch_status.success(f"🎉 Batch translation complete! **{len(saved_paths)}** files translated to {target_language} and saved to `{output_dir}`.")
                
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
# TAB 3: SYSTEM DIAGNOSTICS & FORMATS
# ==========================================
with tab_info:
    st.subheader("System Diagnostics & Supported File Formats")
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.markdown("### 🖥️ Ollama Service Status")
        st.write(f"**Host URL:** `{ollama_host}`")
        models = OllamaTranslator.list_models(ollama_host)
        st.write(f"**Installed LLM Models ({len(models)}):**")
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
            st.warning("⚠️ No Whisper models cached locally yet (`~/.cache/whisper`). Selected model will be downloaded automatically when an audio/video file is processed.")
            
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
    ### 💡 Quick Tips & Features
    1. **Real-time Token Streaming**: Toggle `⚡ Real-time Token Streaming` in sidebar to see LLM output rendered token-by-token.
    2. **Web Article Translation**: Select `🌐 Web Page URL` to extract and translate online articles directly.
    3. **Audio / Video Subtitles**: When translating audio or video files, original SRT and VTT subtitles are auto-generated with timestamps.
    4. **Theme Switcher**: Switch between Vibrant Dark Mode and Clean Light Mode in the sidebar appearance section.
    """)
