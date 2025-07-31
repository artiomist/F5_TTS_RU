# =========================
# 📦 Standard Library Imports
# =========================
import os
import re
import sys
import gc
import tempfile

import time
from datetime import timedelta


from pathlib import Path

# =========================
# 📦 Third-Party Libraries
# =========================
#Second tab
import subprocess
import socket
import argparse

# Audio
import torchaudio
import soundfile as sf
from pydub import AudioSegment

#Audio to text
import whisper

# Machine Learning / Transformers
import torch
import numpy as np

# Web / EPUB / HTML Processing


def get_f5_tts_path():
    if os.path.exists("/content/F5-TTS"):
        return "/content/F5-TTS"  # Colab path
    else:
        try:
            # Script context
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Notebook context
            current_dir = os.getcwd()

        f5_tts_path = os.path.abspath(os.path.join(current_dir, "..", "F5-TTS"))
        return f5_tts_path

sys.path.insert(0, os.path.abspath(get_f5_tts_path()))

def get_f5_tts_ru_path():
    if os.path.exists("/content/F5_TTS_RU"):
        return "/content/F5_TTS_RU"
    else:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        return os.path.abspath(os.path.join(current_dir, "..", "F5_TTS_RU"))

sys.path.insert(0, get_f5_tts_ru_path())

# F5-TTS Model
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    #load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    #chunk_text,
)
from huggingface_hub import hf_hub_download

from vocos import Vocos


# GUI / App Framework
import gradio as gr

#Cahes
_f5tts_model_cache = {}

# =========================
# 📦 Custom Project Modules / Local Imports
# =========================
from f5_tts_ru_utils import config
#from f5_tts_ru_utils.m4b_utils import combine_m4a_to_m4b
from f5_tts_ru_utils.utils import (
    #init_directories,
    sanitize_filename,
    embed_metadata_into_mp3,
    embed_metadata_into_m4a
)


# logging
import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=config.LOGGINGLEVEL,
    format=config.LOGGINGFORMAT, 
)
    

USING_SPACES = False
# GPU Decorator
def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func



#parser = argparse.ArgumentParser()
#parser.add_argument("--port", type=int, default=7860)
#parser.add_argument("--gpu", type=int, default=config.DEFAULT_GPU)
#args = parser.parse_args()
#config.CURRENT_GPU = args.gpu
       
#logging.info(f"🔧 Using GPU {config.CURRENT_GPU} | Split Inference between 2 GPU: {config.SPLIT_INFERENCE_BETWEEN_TWO_GPU}") 



 


# Audio ref to text
def load_asr_model(gpu_id):
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device_str) # or "small", "medium", etc.
    logging.info(f"📢 Whisper ASR model loaded on {device_str}")
    return model
    
asr_model = load_asr_model(config.CURRENT_GPU)  

def transcribe_ref_audio(audio_path):
    if not audio_path:
        raise ValueError("No reference audio provided for transcription.")
    
    try:
        result = asr_model.transcribe(audio_path, language="ru")
        if not result or not result.get("text", "").strip():
            raise RuntimeError("Transcription result is empty or invalid.")
        return result["text"]
    except Exception as e:
        logging.error(f"ASR transcription failed: {e}")
        raise RuntimeError(f"❌ Failed to transcribe reference audio: {e}")


#----INFERENCE----
@gpu_decorator
def infer(
    f5tts_model,
    vocoder,
    ref_audio_orig, 
    ref_text, 
    gen_text, 
    #vocoder_model,      # pass Vocos model
    cross_fade_duration=0.0, 
    speed=1, 
    show_info=gr.Info, 
    progress=gr.Progress(),
    device=None,
    #progress_start_fraction=0.0, 
    #progress_end_fraction=1.0, 
    #ebook_idx=0, num_ebooks=1
          ): # Added ebook_idx, num_ebooks
    """Perform inference to generate audio from text without truncation."""
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing reference audio and text: {e}")

    if not gen_text.strip():
        raise ValueError("Generated text is empty. Please provide valid text content.")

    logging.debug(f"DEBUG - gen_text: {gen_text} ({type(gen_text)})")
    logging.debug(f"DEBUG - ref_text: {ref_text} ({type(ref_text)})")



    try:
        #with torch.no_grad():
        with torch.inference_mode():
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                f5tts_model,
                vocoder,
                cross_fade_duration=cross_fade_duration,
                speed=speed,
                show_info=show_info,
                progress=progress,
                device=device,
            )
    except Exception as e:
        raise RuntimeError(f"Error during inference process: {e}")

    logging.info(f"Generated audio length: {len(final_wave)} samples at {final_sample_rate} Hz.")
    return (final_sample_rate, final_wave), ref_text
    
def get_f5tts_model(device):
    """Loads and caches the F5-TTS model for a specific device within a process."""
    global _f5tts_model_cache
    
    if device not in _f5tts_model_cache:
        logging.info(f"Loading F5-TTS model on {device}...")
        
        #_f5tts_model_cache = {}
        model_cfg = {
            "dim": 1024, "depth": 22, "heads": 16,
            "ff_mult": 2, "text_dim": 512, "conv_layers": 4
        }

        model_dir = config.MODEL_DIR
        #ckpt_path = config.MODEL_CHECKPOINT_PATH
        #vocab_path = config.VOCAB_FILE_PATH

        model_dir.mkdir(parents=True, exist_ok=True)

        # Load paths from config or fallback to HuggingFace download
        try:
            ckpt_path = config.MODEL_CHECKPOINT_PATH
        except AttributeError:
            ckpt_path = None
        
        try:
            vocab_file = config.VOCAB_FILE_PATH
        except AttributeError:
            vocab_file = None


        ckpt_path = os.path.join(ckpt_path, "model_240000_inference.safetensors")
        vocab_file = os.path.join(vocab_file, "vocab.txt")

        # Download checkpoint if not found
        if not Path(ckpt_path).exists():
            logging.warning("Checkpoint not found locally. Downloading from Hugging Face...")
            ckpt_path = hf_hub_download(
                repo_id="Misha24-10/F5-TTS_RUSSIAN",
                filename="F5TTS_v1_Base/model_240000_inference.safetensors",#
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            ckpt_path = Path(ckpt_path)

        # Download vocab if not found
        if not Path(vocab_file).exists():
            logging.warning("Vocab file not found locally. Downloading from Hugging Face...")
            vocab_file = hf_hub_download(
                repo_id="Misha24-10/F5-TTS_RUSSIAN",
                filename="F5TTS_v1_Base/vocab.txt",#
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            vocab_file = Path(vocab_file)

        
        f5tts_model = load_model(DiT, model_cfg, str(ckpt_path), vocab_file=str(vocab_file))
        f5tts_model.eval()
        #_f5tts_model_cache[device] = f5tts_model.to(device)
        f5tts_model.to(device)

        _f5tts_model_cache[device] = f5tts_model
        logging.info(f"✅ Model cached on {device}")
    return _f5tts_model_cache[device]


def list_pending_books(working_dir, ebook_title):
    """
    Lists all subdirectories in the working directory that do not end with 'DONE'
    and match the ebook_title (if provided).
    """
    working_path = Path(working_dir)
    if not working_path.exists():
        raise FileNotFoundError(f"WORKING_DIR not found: {working_path}")

    pending_dirs = []
    for subdir in working_path.iterdir():
        if subdir.is_dir() and not subdir.name.endswith("DONE"):
            if ebook_title in subdir.name:
                pending_dirs.append(subdir)
    return pending_dirs

def split_text_into_batches(text, max_chars=config.MAX_CHARS_FOR_BATCH):
    """Split text into batches no longer than max_chars, ideally at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        if len(current_batch) + len(sentence) <= max_chars:
            current_batch += sentence + " "
        else:
            batches.append(current_batch.strip())
            current_batch = sentence + " "
    if current_batch.strip():
        batches.append(current_batch.strip())
    return batches

def chunk_text_for_total_inference_steps_estimate(text, max_chars=135):
    """
    Splits input text into chunks, preserving <break time='Xs'/> as special chunks,
    and avoids skipping any lines or content.

    Args:
        text (str): The annotated text with optional <break time='Xs'/>.
        max_chars (int): Max characters (UTF-8 bytes) per chunk.

    Returns:
        List[str]: Chunked list including <break time='Xs'/> as separate entries.
    """
    chunks = []
    current_chunk = ""

    # Normalize all <break> without time to <break time='1s'/>
    text = re.sub(r"<break\s*/?>", "<break time='1s'/>", text, flags=re.IGNORECASE)

    # Split based on <break ...> tags
    parts = re.split(r"(<break\s+time=['\"]?[\d.]+s['\"]?\s*\/?>)", text, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if this part is a <break time='Xs'/>
        if re.match(r"<break\s+time=['\"]?[\d.]+s['\"]?\s*\/?>", part, flags=re.IGNORECASE):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(part)
        else:
            # Split by basic punctuation or newlines to better preserve content
            sub_sentences = re.split(r"(?<=[;:,.!?。！？])\s+|\n+", part)
            for sentence in sub_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Chunk based on max UTF-8 encoded length
                if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def infer_text_only(gen_text, gen_text_file):
    import tempfile
    import shutil

    # If a file is uploaded, read its content and overwrite gen_text
    if gen_text_file:
        # gen_text_file is a dict with 'name' and 'data' keys if using gr.File(type='file')
        # But since type="file", gen_text_file is a Path string
        with open(gen_text_file, "r", encoding="utf-8") as f:
            gen_text = f.read()

    if not gen_text.strip():
        return None, None, "❌ No text provided for inference."

    # Setup dummy/default parameters for inference (adapt as needed)
    ref_audio = config.DEFAULT_REF_AUDIO_PATH
    ref_text = ""  # Could be blank or something else
    cross_fade_duration = 0.0
    speed = 1.0
    create_mp3 = False
    default_to_zero = False
    skip_edits = False

    # Run basic_tts or directly infer here:
    # We simplify and just generate one m4a file in temp dir for playback and download.

    try:
        # Run basic_tts but limited to just this one text as one "chapter"
        # For this simplified case, we do inference directly:

        # Prepare a temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="gradio_tts_")

        device = torch.device(f"cuda:{config.CURRENT_GPU}" if config.CURRENT_GPU is not None and torch.cuda.is_available() else "cpu")
        f5tts_model = get_f5tts_model(device)
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

        audio_out, _ = infer(
            f5tts_model,
            vocoder,
            ref_audio,
            ref_text,
            gen_text,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            device=device
        )

        sample_rate, waveform = audio_out

        # Save to wav file temporarily
        wav_path = Path(temp_dir) / "output.wav"
        sf.write(wav_path, waveform, sample_rate)

        # Convert wav to m4a with pydub
        audio_seg = AudioSegment.from_wav(wav_path)
        m4a_path = Path(temp_dir) / "output.m4a"
        audio_seg.export(m4a_path, format="ipod", bitrate=config.AAC_AUDIO_BITRATE)

        # Return paths for player and downloads
        return str(m4a_path), [str(m4a_path)], "✅ Inference done on input text."

    except Exception as e:
        return None, None, f"❌ Error during inference: {str(e)}"
        
# --- Main function ---
@gpu_decorator
def basic_tts(
    file, ref_audio, ref_text,
    cross_fade_duration, speed,
    create_mp3, default_to_zero,
    skip_edits#, gpu_id
):
    try:
        logging.info("🗣️ Starting TTS inference...")
        
        # === Config values ===
        #MODEL_CHECKPOINT_PATH = config.MODEL_CHECKPOINT_PATH
        #VOCAB_FILE_PATH = config.VOCAB_FILE_PATH
        #REF_TEXT = config.DEFAULT_REF_TEXT

        # === Load model ===
        device = torch.device(f"cuda:{config.CURRENT_GPU}" if config.CURRENT_GPU is not None and torch.cuda.is_available() else "cpu")
        f5tts_model = get_f5tts_model(device)
        logging.info(f"📦 Model device: {next(f5tts_model.parameters()).device}")
        
        # === Load vocoder ===
        logging.info("Loading vocoder")
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

        # === Load reference audio ===
        logging.info(f"Loading reference audio: {config.DEFAULT_REF_AUDIO_PATH}")
        #ref_audio, sr = torchaudio.load(config.DEFAULT_REF_AUDIO_PATH)
        #ref_audio = ref_audio.to(device)
        ref_audio = config.DEFAULT_REF_AUDIO_PATH
        #ref_text = REF_TEXT

        all_outputs = []

        #checks if WORKING_DIR exists if not 
        PROJECT_DIR = Path(__file__).parent# / config.PROJECT_DIR
        WORKING_DIR = PROJECT_DIR / config.WORKING_DIR
        logging.debug(f"[DEBUG] Checking WORKING_DIR: {WORKING_DIR}")

        if not WORKING_DIR.exists() or not WORKING_DIR.is_dir():
            raise ValueError("[Error] Process ebook first")
            
        # === Process all book folders that don't end in _INFERED ===
        for book_dir in WORKING_DIR.iterdir():
            logging.debug(f"Found: {book_dir.name}")
            if not book_dir.is_dir():
                logging.debug(f"Skipping {book_dir.name}: Not a directory")
                continue
            #if book_dir.name.endswith("INFERED"):
            #    logging.debug(f"Skipping {book_dir.name}: Already marked as INFERED")
            #    continue

            logging.info(f"📘 Processing book folder: {book_dir.name}")
            txt_debug_dir = book_dir / config.DEBUG_TXT_SUBDIR
            if not txt_debug_dir.exists():
                #logging.warning(f"⚠️ No debug text directory in: {book_dir}")
                raise ValueError(f"⚠️ No debug text directory in: {book_dir}")

            # Extract base filename from folder name for chapter filenames
            base_name = book_dir.name

            # === Load chapter files ===
            chapters = []
            chapter_idx = 1
            logging.debug(f"Loading chapter files")
            """while True:
                title_path = txt_debug_dir / f"{base_name}_chapter{chapter_idx:02d}_title.txt"
                text_path = txt_debug_dir / f"{base_name}_chapter{chapter_idx:02d}_accented.txt"
                logging.debug(f"Looking for chapter files: {title_path}, {text_path}")
                if not title_path.exists() or not text_path.exists():
                    logging.warning(f"❌ There is no: {chapter_idx}. Total {chapter_idx-1} chapters.")
                    break

                with open(title_path, encoding="utf-8") as f:
                    ch_title = f.read().strip()
                with open(text_path, encoding="utf-8") as f:
                    ch_text = f.read().strip()

                chapters.append((ch_title, ch_text))
                logging.debug(f"✅ Loaded chapter {chapter_idx}: {ch_title[:50]}")
                chapter_idx += 1
            """
            # Step 1: Get all accented files
            accented_files = list(txt_debug_dir.glob(f"{base_name}_chapter*_accented.txt"))

            # Step 2: Extract chapter number using regex, and sort
            chapter_data = []
            for file in accented_files:
                match = re.search(rf"{base_name}_chapter(\d+)_accented\.txt", file.name)
                if match:
                    chapter_number = int(match.group(1))
                    chapter_data.append((chapter_number, file))

            # Step 3: Sort by chapter number
            chapter_data.sort(key=lambda x: x[0])

            # Step 4: Process each file
            for chapter_idx, text_path in chapter_data:
                title_path = txt_debug_dir / f"{base_name}_chapter{chapter_idx:02d}_title.txt"

                #if not title_path.exists():
                #    logging.info(f"❌ Missing title file for chapter {chapter_idx}. SKIPPING")
                #    continue

                try:
                    with open(title_path, encoding="utf-8") as f:
                        ch_title = f.read().strip()
                    with open(text_path, encoding="utf-8") as f:
                        ch_text = f.read().strip()
                    chapters.append((ch_title, ch_text))
                    logging.debug(f"✅ Loaded chapter {chapter_idx}: {ch_title[:50]}")
                except Exception as e:
                    logging.warning(f"❌ Error reading chapter {chapter_idx}: {e}")
                    continue
        
        
        
        
            if not chapters:
                #logging.warning(f"❌ No chapters found in: {txt_debug_dir}")
                raise ValueError(f"❌ No chapters found in: {txt_debug_dir}")


            # === Count total batches across all chapters ===
            total_book_steps = 0
            #gpu0_steps = 0
            #gpu1_steps = 0
            gpu_name = "GPU 0"
            # Step 1: Load reference audio to get duration
            ref_text = ref_text or ""
            if ref_text == "":
                logging.warning("⚠️ Reference text is missing. Transcribing from ref_audio")
                ref_text = transcribe_ref_audio(ref_audio)
            #audio, sr = torchaudio.load(ref_audio)   
            #audio, sr = ref_audio, sr  
            #audio = ref_audio        

            # Step 2: Combine all chapter text
            #full_text = "\n\n".join(chapter_text for _, chapter_text in chapters)

            # Step 3: Calculate max_chars per batch based on reference audio and speed
            #audio_duration_sec = audio.shape[-1] / sr
            audio_tensor, sr = torchaudio.load(ref_audio)
            audio_duration_sec = audio_tensor.shape[-1] / sr

            max_chars = int(len(ref_text.encode("utf-8")) / audio_duration_sec * (22 - audio_duration_sec) * speed)
            logging.debug(f"📝 Reference text length: {len(ref_text) if ref_text else 'None'}")


            # Step 4: Split full_text using chunking logic
            #gen_text_batches = chunk_text(full_text, max_chars=max_chars)
            #total_book_steps = len(gen_text_batches)

            #print(f"Estimated total inference chapters: {total_book_steps}")
            logging.info("Estimated inference steps per chapter:")
            
            start_time = time.time()
            chapter_step_counts = {}
            gpu_step_counts = {0: 0, 1: 0}
            completed_steps = 0

            for i, (ch_title, ch_text) in enumerate(chapters):
                chapter_batches_estimate = chunk_text_for_total_inference_steps_estimate(ch_text, max_chars=max_chars)
                #batches_per_chapter.append(chapter_batches_estimate)
                step_count = len(chapter_batches_estimate)
                total_book_steps += step_count
                chapter_step_counts[i] = step_count
                if config.SPLIT_INFERENCE_BETWEEN_TWO_GPU:
                    if i%2:
                        #gpu0_steps += step_count
                        gpu_step_counts[0] += step_count
                        gpu_name = "GPU 0"
                    else:
                        #gpu1_steps += step_count
                        gpu_step_counts[1] += step_count
                        gpu_name = "GPU 1"
                print(f"  Chapter {i+1}: '{ch_title[:40]}...' → {step_count} step(s) on {gpu_name}")

            logging.info(f"Total inference steps: {total_book_steps}\n")
            #if config.SPLIT_INFERENCE_BETWEEN_TWO_GPU:
            #    logging.info(f"GPU 0 inference steps: {gpu0_steps}\n")
            #    logging.info(f"GPU 1 inference steps: {gpu1_steps}\n")

            
            # === Synthesize speech per chapter ===
            #config.CURRENT_GPU = args.gpu  # 0 or 1 depending on which instance is running
            logging.info(f"🚀 Starting inference on GPU {config.CURRENT_GPU} (SPLIT_INFERENCE_BETWEEN_TWO_GPU = {config.SPLIT_INFERENCE_BETWEEN_TWO_GPU})")

            tmp_m4a_dir = book_dir / config.FINAL_OUTPUT_SUBDIR
            
            successfully_processed_all = True  # Assume success unless a failure is detected

            for chapter_idx, (ch_title, ch_text) in enumerate(chapters):
                # 🔁 Skip based on GPU role when using multiple GPU setup
                if config.SPLIT_INFERENCE_BETWEEN_TWO_GPU:
                    if chapter_idx % 2 != config.CURRENT_GPU:
                        logging.info(f"⏭️ Skipping Chapter {chapter_idx+1} on GPU {config.CURRENT_GPU} (not assigned).")
                        continue
            
                logging.info(f"🎙️ Synthesizing Chapter {chapter_idx+1}: {ch_title[:50]}")

                #define m4a path
                filename = f"{base_name}_Chapter{chapter_idx+1:02d}"
                filenameext = ".m4a"

                m4a_path = tmp_m4a_dir / (filename + filenameext)
            
                if m4a_path.exists():
                    logging.info(f"✅ M4A already exists: {m4a_path}, skipping inference.")
                else:
                    if config.SPLIT_CHAPTERS_TO_BATCHES:
                        batches = split_text_into_batches(ch_text, config.MAX_CHARS_FOR_BATCH)
                    else:
                        batches = [ch_text]

                    chapter_audio = []
                    chapter_failed = False  # Track if this chapter has any failed batch
                    
                    for batch_idx, batch_text in enumerate(batches):
                        try:
                            logging.debug(f"Running inference: chapter {chapter_idx+1}, batch {batch_idx+1}")
                            #audio_out, _ = infer(
                            #    f5tts_model, vocoder, ref_audio, ref_text,
                            #    batch_text, speed, crossfade=0.0, device=device
                            #)
                            audio_out, _ = infer(
                                f5tts_model, vocoder,
                                ref_audio, ref_text, batch_text,
                                cross_fade_duration=0.0, speed=speed,
                                device=device
                            )
                            #chapter_audio.append(audio_out)
                            if isinstance(audio_out[1], torch.Tensor):
                                audio_out_cpu = (audio_out[0], audio_out[1].detach().cpu().numpy())
                            else:
                                audio_out_cpu = (audio_out[0], audio_out[1])
                            chapter_audio.append(audio_out_cpu)
                        except Exception as e:
                            logging.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                            chapter_failed = True
                            break  # Optionally skip the rest of the batches for this chapter

                    if chapter_failed or not chapter_audio:
                        logging.warning(f"⚠️ Skipping Chapter {chapter_idx} due to previous errors")
                        successfully_processed_all = False
                        continue  # Skip saving audio and move to next chapter

                    audio_concat = np.concatenate([a for sr, a in chapter_audio])
                    sample_rate = chapter_audio[0][0]
                    
                    #define cover file placment
                    cover_image = book_dir / config.TMP_CONVERTED_SUBDIR / f"cover_{base_name}.jpg"
                    
                    #load title and author
                    title_author_path = os.path.join(book_dir / config.TMP_CONVERTED_SUBDIR, f"title_author_{base_name}.txt")

                    with open(title_author_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    ebook_title = lines[0].strip() if len(lines) > 0 else ""
                    ebook_author = lines[1].strip() if len(lines) > 1 else ""

                    logging.debug(f"📖 Loaded EBook Title: {ebook_title}")
                    logging.debug(f"✍️ Loaded Author: {ebook_author}")

                    #Define chepter file title   
                    title_to_embed=ch_title

                    # Write .wav then convert to .m4a
                    temp_wav = m4a_path.with_suffix(".wav")
                    sf.write(temp_wav, audio_concat, sample_rate)
                    try:
                        audio = AudioSegment.from_wav(temp_wav)
                        audio.export(m4a_path, format="ipod", bitrate=config.AAC_AUDIO_BITRATE)
                        logging.info(f"💾 Saved chapter: {m4a_path}")
                        
                        try:
                            logging.info(f"Embedding metadata into M4A")
                            embed_metadata_into_m4a(
                                m4a_path,
                                cover_image_path=cover_image,
                                title=title_to_embed,
                                author=ebook_author,
                                album_title=ebook_title
                            )
                        except Exception as e:
                            logging.error(f"❌ Failed to embed metadata into M4A: {e}")
                        
                    except Exception as e:
                        logging.error(f"❌ Failed to convert WAV to M4A: {e}")
                        if temp_wav.exists():
                            os.remove(temp_wav)
                        successfully_processed_all = False
                        
                    # Optional MP3 export
                    if create_mp3:
                        filenameext = ".mp3"
                        mp3_path = tmp_m4a_dir / (filename + filenameext)
                        if mp3_path.exists():
                            logging.info(f"✅ MP3 already exists: {mp3_path}, skipping inference.")
                        else:
                            try:
                                audio.export(mp3_path, format="mp3", bitrate=config.MP3_AUDIO_BITRATE, parameters=["-q:a", "0"])
                                logging.info(f"💾 Saved chapter: {mp3_path}")
                                while not os.path.exists(mp3_path):
                                    time.sleep(0.1)  # wait a moment to ensure file is ready
                                try:
                                    embed_metadata_into_mp3(
                                        mp3_path=mp3_path,
                                        cover_image_path=cover_image,
                                        title=title_to_embed,
                                        author=ebook_author,
                                        album_title=ebook_title
                                    )
                                except Exception as e:
                                    logging.error(f"❌ Failed to embed metadata into MP3: {e}")
                            except Exception as e:
                                logging.error(f"❌ Failed to convert WAV to MP3: {e}")

                    os.remove(temp_wav)
                    
                    
                    # === Track and Log ETA ===
                    completed_steps += chapter_step_counts.get(chapter_idx, 0)
                    elapsed_time = time.time() - start_time
                    avg_time_per_step = elapsed_time / completed_steps if completed_steps else 0

                    steps_remaining = gpu_step_counts[config.CURRENT_GPU] - completed_steps
                    eta_seconds = steps_remaining * avg_time_per_step
                    
                    # Round ETA to nearest minute, format as ~Hh Mm
                    eta_minutes = int(round(eta_seconds / 60))
                    eta_hours = eta_minutes // 60
                    eta_minutes = eta_minutes % 60
                    eta_formatted = f"~{eta_hours}h {eta_minutes}m"

                    percent_complete = (completed_steps / gpu_step_counts[config.CURRENT_GPU]) * 100 if gpu_step_counts[config.CURRENT_GPU] else 0

                    chapters_remaining = [
                        i for i in range(len(chapters))
                        if i % 2 == config.CURRENT_GPU and i > chapter_idx
                    ]
                    remaining_chapters_count = len(chapters_remaining)

                    #eta_message = f"⏳ {remaining_chapters_count} chapters remaining ({steps_remaining} steps), " \
                    #              f"{percent_complete:.2f}% complete, ETA: {eta_formatted}"
                    chapter_title_short = ch_title[:50].replace("\n", " ").strip()

                    eta_message = (
                        f'chapter #{chapter_idx + 1:02d} "{chapter_title_short}" done, '
                        f'{remaining_chapters_count} chapters remaining '
                        f'({steps_remaining} steps) on GPU {config.CURRENT_GPU}, '
                        f'{percent_complete:6.2f}% complete, ETA: {eta_formatted}'
                    )

                    logging.info(eta_message)

                    # === Log ETA to a separate file ===
                    try:
                        #eta_log_path = book_dir.parent / "ETA_log.txt"
                        eta_log_path = book_dir / "ETA_log.txt"
                        with open(eta_log_path, "a", encoding="utf-8") as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {eta_message}\n")
                    except Exception as e:
                        logging.error(f"❌ Failed to write ETA log: {e}")
                    #eta_log_path = book_dir.parent / "ETA_log.txt"
                    #with open(eta_log_path, "a", encoding="utf-8") as f:
                    #    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {eta_message}\n")

                    
                    
                    
                    #Explicitly clear GPU cache, After each chapter.
                    torch.cuda.empty_cache()
                    gc.collect()  # import gc
            
            # Only rename folder if all chapters were successful
            if successfully_processed_all:
                logging.info("✅ Done batch processing")
                #done_path = book_dir.with_name(book_dir.name + "_INFERED")
                #book_dir.rename(done_path)
                #logging.info(f"✅ Marked as INFERED: {done_path.name}")
            #else:
                #logging.warning(f"⚠️ Book '{book_dir.name}' not marked as INFERED due to errors.")

        return None, None, "✅ TTS process completed. Check output folders for results."

    #return all_outputs
    except Exception as e:
        logging.error(f"❌ Exception in basic_tts: {e}", exc_info=True)
        return None, None, f"❌ Error: {str(e)}"    
    
    
    

    

# --- Create and configure the Gradio application ---
def create_gradio_app():
    from f5_tts_ru_utils.preprocess_ru_text import preprocess_ru_text
    from f5_tts_ru_utils.m4b_utils import combine_m4a_to_m4b
    logging.info("🛠️ Creating Gradio interface...")

    with gr.Blocks() as app:
        # Title
        gr.Markdown(f"# {config.GRADIO_TITLE}")

        # Description
        gr.Markdown(config.GRADIO_DESCRIPTION)

        ref_audio_input = gr.Audio(
            label="Upload Voice File (<15 sec) or Record with Mic Icon (Ensure Natural Phrasing, Trim Silence). There is default voice Edge Alexander",
            type="filepath",
            value=config.DEFAULT_REF_AUDIO_PATH
        )

        gen_file_input = gr.Files(
            label="Upload eBook (epub, fb2, txt, html)",
            file_types=[".epub", ".fb2", ".txt", ".html"],
            #file_count="multiple",
        )
        with gr.Group():
            gr.Markdown("Russian text normalisation for inference")
            preprocess_btn = gr.Button("1. Preprocess Text", variant="primary")
            verification_dict_all_default_to_zero = gr.Checkbox(
                label="When ambiguous word found auto accept option 0 (to run faster for debugging)",
                value=config.VERIFICATION_DICT_DEFAULT_TO_ZERO,  # Checked by default
            )
            all_caps_to_lower = gr.Checkbox(
                label="When ALL CAPS words are found convert to lower case except for abbreviation exceptions",
                value=config.ALL_CAPS_TO_LOWER, 
            )
            
        
        with gr.Group():
            gr.Markdown("Inference")

            inference_btn = gr.Button("2. Run Inference", variant="primary")
            
            gr.Markdown(f"🧠 **GPU Mode:** {'Two' if config.SPLIT_INFERENCE_BETWEEN_TWO_GPU else 'Single'}")
            
            create_mp3 = gr.Checkbox(
                label="Create MP3 alongside with MP3. If not only M4A will be created.",
                value=config.CREATE_MP3, 
            )  
            
        with gr.Group():
            gr.Markdown("M4B")    
        
            combine_m4b_btn = gr.Button("3. Combine m4a to m4b", variant="primary")
            
            #create_m4b = gr.Checkbox(
            #    label="Create single M4B audiobook file. If checked auto executes on single GPU after inference has completed. If unchecked leaves separate M4As.",
            #    value=config.CREATE_M4B,  
            #)    
            
        status = gr.Textbox(label="Status", interactive=False)

        #show_audiobooks_btn = gr.Button("Show All Completed Audiobooks", variant="secondary")

        audiobooks_output = gr.Files(label="Converted Audiobooks (Download Links)")
        player = gr.Audio(label="Play Latest Converted Audiobook", interactive=False)

        #status_label = gr.Textbox(label="Progress Detail", interactive=False, lines=2, visible=True)
        #status_label_text = gr.Textbox(label="Detailed TTS Progress", lines=3, interactive=False)
            
        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text (Leave Blank for Automatic Transcription)",
                lines=2,
                value=config.DEFAULT_REF_TEXT
            )
            speed_slider = gr.Slider(
                label="Speech Speed (Adjusting Can Cause Artifacts)",
                minimum=0.3,
                maximum=2.0,
                value=config.TTS_SPEED,
                step=0.1,
            )
            cross_fade_duration_slider = gr.Slider(
                label="Cross-Fade Duration (Between Generated Audio Chunks)",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
            )
            
            """
            gr.Markdown("Which GPU to use:")
            current_gpu = gr.Radio(choices=[0, 1], value=config.DEFAULT_GPU)
            def set_current_gpu(gpu_id):
                config.CURRENT_GPU = int(gpu_id)
                return f"✅ GPU {gpu_id} selected"

            current_gpu.change(fn=set_current_gpu, inputs=current_gpu, outputs=status)
            """
            """
            gr.Markdown("If multiple GPUs are used those options are ignored as manual press of Inference and M4B buttons is expected")
            auto_skip_manual_edits_pause = gr.Checkbox(
                label="Auto press Inference button. If not checked permits manual edits of _accentuated.txt files. For multiple GPU is ignored",
                value=config.AUTO_CONTINUE_TO_INFERENCE, 
            )
            """
                          
        with gr.Row():
            gen_text_input = gr.Textbox(
                label="Text to Generate",
                lines=10,
                max_lines=40,
                scale=4,
            )
            gen_text_file = gr.File(
                label="Load Text File (.txt)",
                file_types=[".txt"],
                scale=1
            )
        text_input_btn = gr.Button("Infer text Input Block", variant="primary")
        
        preprocess_event = preprocess_btn.click(
            preprocess_ru_text,
            inputs=[gen_file_input, all_caps_to_lower, verification_dict_all_default_to_zero],
            outputs=status
        )
        # Then: Conditionally auto-trigger inference logic (Button #2) if config allows
        if config.AUTO_CONTINUE_TO_INFERENCE and not config.SPLIT_INFERENCE_BETWEEN_TWO_GPU:
            preprocess_event = preprocess_event.then(
                basic_tts,
                inputs=[
                    gen_file_input,
                    ref_audio_input,
                    ref_text_input,
                    cross_fade_duration_slider,
                    speed_slider,
                    create_mp3,
                    verification_dict_all_default_to_zero,
                    #auto_skip_manual_edits_pause,
                    #current_gpu  
                ],
                outputs=[player, audiobooks_output, status]
            )

        inference_btn.click(
            basic_tts,
            inputs=[
                gen_file_input,
                ref_audio_input,
                ref_text_input,
                cross_fade_duration_slider,
                speed_slider,
                create_mp3,
                verification_dict_all_default_to_zero,
                #auto_skip_manual_edits_pause,
                #current_gpu  
            ],
            outputs=[player, audiobooks_output, status]
        )


        
        combine_m4b_btn.click(
            combine_m4a_to_m4b,
            inputs=[],
            outputs=[player, audiobooks_output, status]
        )
        
        text_input_btn.click(
            fn=infer_text_only,
            inputs=[gen_text_input, gen_text_file],
            outputs=[player, audiobooks_output, status]
        )

        ref_audio_input.change(
            fn=transcribe_ref_audio,
            inputs=ref_audio_input,
            outputs=ref_text_input
        )


    logging.info("✅ Gradio interface created.")
    return app

#python run_f5_app.py --share --port 7860 --gpu 0



    


