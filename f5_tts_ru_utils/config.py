from pathlib import Path


# === Paths ===
# Dynamically resolve project root (where config.py is located)
BASE_DIR = Path(__file__).parent.resolve()

PROJECT_DIR = "f5_tts_ru_utils"
PROJECT_PATH = BASE_DIR.parent# / PROJECT_DIR
WORKING_DIR = "Working_files"
DEBUG_TXT_SUBDIR = "debug_text"
TMP_CONVERTED_SUBDIR = "temp_converted"
TMP_AUDIO_SUBDIR = "temp_audio"
FINAL_OUTPUT_SUBDIR = "Book"

# === Input / Output Formats ===
#SUPPORTED_INPUT_FORMATS = ['.epub', '.pdf', '.fb2', '.txt', '.docx']
#DEFAULT_OUTPUT_FORMAT = 'm4b'

# === EPUB & Text Handling ===
#EPUB_EXTRACT_MAX_LENGTH = 100_000
#MAX_TEXT_CHUNK_LENGTH = 3000
#CHUNK_OVERLAP = 100
USE_BREAKS_TAG = False

# === TTS Settings ===
SPLIT_CHAPTERS_TO_BATCHES = False
MAX_CHARS_FOR_BATCH = 4000
TTS_SPEED = 1 #Default 1
#TTS_BATCH_SIZE = 5
#TTS_LANGUAGE = 'ru'
#DEFAULT_VOICE_NAME = "ru-RU-DmitryNeural"
#USE_VOICE_REFERENCE = True
MODEL_DIR = PROJECT_PATH / PROJECT_DIR / "ru_model"
VOCAB_FILE_PATH = MODEL_DIR / "F5TTS_v1_Base/vocab.txt"
MODEL_CHECKPOINT_PATH = MODEL_DIR / "F5TTS_v1_Base/model_240000_inference.safetensors"
DEFAULT_REF_AUDIO_PATH = PROJECT_PATH / PROJECT_DIR / "ref" / "Edge_Дмитрий_voice_sample/ru-RU-DmitryNeural.mp3"
DEFAULT_REF_TEXT = "Здравствуйте. Это пример предложения на русском языке."

# === Audio Processing ===
AAC_AUDIO_BITRATE = "192k" #128k,256k, AAC is more efficient, so 192k AAC ≈ 320k MP3 quality
USE_AAC_FILTERS = False
EXPORT_MP3 = False  # Set to True if you want MP3 exports
MP3_AUDIO_BITRATE = "320k" #"128k"
#SAMPLE_RATE = 44100
#NORMALIZE_AUDIO = True
#MP3_TAG_ENCODING = "utf-8"

# === Metadata ===
DEFAULT_AUTHOR = "-"
#DEFAULT_ALBUM = "Converted Audiobook"
#EMBED_COVER_IN_MP3 = True

# === 1. Russian Text Normalization ===
ALL_CAPS_TO_LOWER = False #When ALL CAPS words are found convert to lower case except for abbreviation exceptions
VERIFICATION_DICT_DEFAULT_TO_ZERO = False #When ambiguous word found auto accept option 0 (to run faster for debugging)+
#NORMALIZE_ALL_CAPS = False
#NORMALIZE_CAPS_INTERACTIVE = False
#ENABLE_CURRENCY_NORMALIZATION = True
#ENABLE_PHONE_NORMALIZATION = True
#ENABLE_DATE_NORMALIZATION = True
#ENABLE_TIME_NORMALIZATION = True
#ENABLE_NUMBER_NORMALIZATION = True
#ENABLE_LATIN_TO_CYRILLIC = True
#MAX_LRU_CACHE_SIZE = 1000
#CONTEXT_CHARS = 30

# === 2. Inference ===
AUTO_CONTINUE_TO_INFERENCE = False #Auto press Inference button, for single GPU permits manual edits of _accentuated.txt files, for multiple GPU is ignored
CURRENT_GPU = 0
SPLIT_INFERENCE_BETWEEN_TWO_GPU = True # Set to True to use multiple GPUs for batch processing

# === 3. M4B & Chapter Settings ===
CREATE_MP3 = False
#CREATE_M4B = True #Create single M4B audiobook file (unchecked leaves separate MP3s)
NORMALIZE_AUDIO_FILES = False
#CHAPTERS_ENABLED = True
#M4B_CHAPTER_LENGTH = 300  # seconds

# === Gradio UI ===
GRADIO_TITLE = "📚🔊 Ebook to Audiobook with F5-TTS! (Russian)"
GRADIO_DESCRIPTION = (
    "Upload your ebook (EPUB, FB2, TXT, HTML), and it will be converted into an audiobook "
    "with full Russian text normalization (dates, numbers, currencies, etc.) and speech synthesis."
)

# === Logging ===
#LOG_LEVEL = "INFO"#debug,info,warning,error,critical

# === External Tools ===
#FFMPEG_BINARY = "ffmpeg"
import logging 
LOGGINGLEVEL = logging.INFO  # Minimum level of messages to capture, DEBUG, INFO, WARNING, ERROR
LOGGINGFORMAT = '%(asctime)s - %(levelname)s - %(message)s'  # Log format