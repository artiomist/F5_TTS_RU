import os
import re
import magic
import shutil
import subprocess
from bs4 import BeautifulSoup
from ebooklib import epub
from ebooklib.epub import EpubHtml
from .utils import (
    init_directories,
    detect_file_type,
    ensure_directory,
    sanitize_filename,
    convert_to_epub,
    extract_text_and_title_and_author_from_epub,
    extract_cover,
    ensure_even_dimensions,
    create_placeholder_cover
)
from . import config
import logging
logging.basicConfig(
    level=config.LOGGINGLEVEL,
    format=config.LOGGINGFORMAT, 
)
from .ru4tts_normalization import normalize_russian
from .patch_dict import verification_dict, replacement_dict
from pathlib import Path  # Used in russian_normalisation_accentuation
import string              # Used in detect_multiple_stresses
#from ru_ebook_app import basic_tts
# =========================
# ⚙️ Environment Configs
# =========================
def running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if not running_in_colab():
    magic_path = Path(__file__).parent / "libmagic" / "magic.mgc"
    os.environ["MAGIC"] = str(magic_path.resolve())
# =========================
# 🧠  Accentizer
# =========================
from ruaccent import RUAccent 
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)


def preprocess_ru_text(file, all_caps_to_lower, verification_dict_all_default_to_zero):
    logging.info("📖 Starting text preprocessing...")
    #yield gr.Textbox.update("📖 Starting text preprocessing...")

    if file is None:
        logging.warning("⚠️ No file provided for preprocessing.")
        #yield gr.Textbox.update(value="⚠️ No file provided for preprocessing.")
        return "⚠️ No file provided for preprocessing."

    # Handle if 'file' is a list (multiple files)
    if isinstance(file, list):
        if len(file) == 0:
            logging.warning("⚠️ Empty file list provided.")
            #yield gr.Textbox.update(value="⚠️ Empty file list provided.")
            return "⚠️ Empty file list provided."
        file = file[0]  # Use the first file only (or change logic if you want all)
    
    original_path = file.name
    sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(original_path))[0])
    temp_epub_created = False
    file_ext = os.path.splitext(original_path)[1].lower()

    # --- Step 1: Basic fallback defaults ---
    ebook_title = sanitized_name
    ebook_author = config.DEFAULT_AUTHOR # Default if no author found

    tmp_audio_dir, tmp_m4a_dir, txt_debug_dir, temp_converted_dir = init_directories(sanitized_name)
    
    cover_output_path = os.path.join(temp_converted_dir, f"cover_{sanitized_name}.jpg")
    
    try:
        logging.debug(f"📂 Processing file: {original_path}")

        # ✅ Handle .txt and .html separately
        if file_ext in [".txt", ".html"]:
            logging.debug("📄 Detected plain text or HTML file. Extracting as raw text...")

            with open(original_path, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()
            
            create_placeholder_cover(sanitized_name, cover_output_path)
            logging.debug(f"🖼️ Fallback cover created: {cover_output_path}")

            # Save as single chapter
            raw_txt_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter01.txt")
            with open(raw_txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            accented_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter01_accented.txt")
            
            # Set chapters for later loop
            chapters = [(ebook_title, processed_text)]

        else:#if NOT TXT
            # ✅ Convert to EPUB if not already
            logging.debug("📁 Detecting file type for EPUB conversion...")
            file_type = detect_file_type(original_path)

            if file_type == 'application/epub+zip':
                epub_path = original_path
                logging.debug(f"✅ EPUB found: {epub_path}")
            else:
                epub_path = os.path.join(temp_converted_dir, f"{sanitized_name}.epub")

                if os.path.exists(epub_path):
                    logging.debug(f"[SKIP] EPUB conversion: Temp EPUB already exists at {epub_path}")
                else:
                    logging.debug("🔁 Converting to EPUB...")
                    convert_to_epub(original_path, epub_path)
                    temp_epub_created = True

            # ✅ Extract cover or use fallback
            try:
                if os.path.exists(cover_output_path):
                    logging.debug(f"[SKIP] Cover already exists at: {cover_output_path}")
                else:
                    logging.debug("🖼️ Extracting cover image...")
                    cover_image = extract_cover(epub_path)
                    if cover_image:
                        ensure_even_dimensions(cover_image)
                        # cover_image.save(cover_output_path)
                        shutil.copy(cover_image, cover_output_path)
                        logging.debug(f"✅ Cover saved as: {cover_output_path}")
                    else:
                        raise ValueError("❌ No cover found")
            except Exception:
                create_placeholder_cover(sanitized_name, cover_output_path)
                logging.debug(f"🖼️ Fallback cover created: {cover_output_path}")
            
            # ✅ Extract chapters, title, author
            logging.debug("📖 Extracting chapters, title, and author...")
            chapters, ebook_title, ebook_author = extract_text_and_title_and_author_from_epub(epub_path, txt_debug_dir)
            logging.info("✅ Extracting chapters, title, and author DONE!")

            # ✅ Accent and normalize EPUB chapters and TXT chapter, if not already done
            for idx, (chapter_title, _) in enumerate(chapters):
                accented_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter{idx+1:02d}_accented.txt")
                raw_txt_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter{idx+1:02d}.txt")

                if os.path.exists(accented_path):
                    logging.debug(f"[SKIP] Accentuation already done for: {accented_path}")
                else:
                    if not os.path.exists(raw_txt_path):
                        logging.error(f"❌ Missing raw chapter text for accentuation: {raw_txt_path}")
                        #yield f"❌ Missing raw chapter text: {raw_txt_path}"
                    else:
                        logging.debug(f"🔤 Accentuation: {raw_txt_path} → {accented_path}")
                        #yield gr.Textbox.update(f"🔤 Accentuation – chapter {idx+1}…")
                        #yield gr.Textbox.update("🕐 Please respond in terminal if prompted")
                        russian_normalisation_accentuation(
                            raw_txt_path,
                            accentizer,
                            all_caps_to_lower=all_caps_to_lower,
                            verification_dict_all_default_to_zero=verification_dict_all_default_to_zero
                        )


        # ✅ Save title and author
        title_author_path = os.path.join(temp_converted_dir, f"title_author_{sanitized_name}.txt")
        with open(title_author_path, 'w', encoding='utf-8') as f:
            f.write(f"{ebook_title}\n{ebook_author}")
        logging.debug(f"📝 Title/Author saved to: {title_author_path}")

        result = f"✅ Preprocessing done for: {sanitized_name}"
        logging.info(f"[PREPROCESS] {result}. Plese proceed to Inference.")
        #yield gr.Textbox.update("✅ Preprocessing done!")
        
        return result

    except Exception as e:
        logging.error(f"❌ Error in preprocess_ru_text: {e}")
        return f"❌ Error: {str(e)}"
        




# ----2.2 Russian tools----
def russian_normalisation_accentuation(txt_path, accentizer, all_caps_to_lower=False, verification_dict_all_default_to_zero=False):
    try:
        txt_path = Path(txt_path)
        normalized_path = txt_path.with_name(txt_path.stem + "_normalized.txt")
        processed_path = txt_path.with_name(txt_path.stem + "_accented.txt")

        # Step 1: Check for already processed (accented) file
        if processed_path.exists():
            logging.info(f"[SKIP] Accentuated file already exists: {processed_path}")
            with open(processed_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Step 2: Check for existing normalized file
        if normalized_path.exists():
            logging.info(f"[SKIP] Normalized file already exists: {normalized_path}")
            with open(normalized_path, 'r', encoding='utf-8') as norm_file:
                normalized_lines = [line.strip() for line in norm_file.readlines()]
        else:
            # Load and normalize original text
            with open(txt_path, 'r', encoding='utf-8-sig') as f:
                text_content = f.read()

            #interactive_caps = not all_caps_to_lower
            normalized_text = normalize_russian(text_content, all_caps_to_lower)#, interactive_caps)
            normalized_lines = [line.strip() for line in normalized_text.splitlines()]

            # Save normalized text
            with open(normalized_path, 'w', encoding='utf-8') as norm_file:
                for line in normalized_lines:
                    norm_file.write(line + '\n')
            logging.info(f"[INFO] Normalized text saved: {normalized_path}")

        # Step 3: Accentuation pipeline
        accented_lines, empty_line_counts, trailing_empty, _ = process_text_with_accentuator(normalized_lines, accentizer)

        accented_lines = verify_and_patch_stress(
            accented_lines,
            verification_dict_all_default_to_zero=verification_dict_all_default_to_zero
        )

        for line in accented_lines:
            detect_multiple_stresses(line)

        if config.USE_BREAKS_TAG:
            final_lines = insert_breaks(accented_lines, empty_line_counts, trailing_empty)
        else:
            final_lines = accented_lines

        # Save final processed file
        with open(processed_path, 'w', encoding='utf-8') as outfile:
            for line in final_lines:
                outfile.write(line + '\n')

        logging.info(f"[INFO] Accentuated + breaks inserted: {processed_path}")
        return '\n'.join(final_lines)

    except Exception as e:
        logging.error(f"[ERROR] Failed to process {txt_path}: {e}")
        raise


def process_text_with_accentuator(lines, accentizer):
    """Normalizes punctuation, accents text, returns accented lines and break info."""
    non_empty_lines = []
    empty_line_counts = []
    empty_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped == '' or stripped == '.':
            empty_count += 1
        else:
            empty_line_counts.append(empty_count)
            empty_count = 0
            non_empty_lines.append(stripped)

    trailing_empty = empty_count

    # Normalize punctuation
    normalized_lines = [normalize_punctuation(line) for line in non_empty_lines]

    # Accent each line
    accented_lines = [
        accentizer.process_all(line)
            .replace('[ELLIPSISBREAK]', "<break time='1s'/>")
            .replace('[BREAK]', "<break time='2s'/>")
        for line in normalized_lines
    ]

    return accented_lines, empty_line_counts, trailing_empty, normalized_lines

def normalize_punctuation(line):
    # Normalize rare punctuation
    if config.USE_BREAKS_TAG:
        line = line.replace('…', ". [ELLIPSISBREAK]").rstrip() #ellipsis
    else:
        line = line.replace('…', ".").rstrip() #ellipsis
    line = line.replace('⁈', '?')  # Replace interrobang
    line = line.replace('—', '-')  # em dash to regular dash
    line = line.replace('–', '-')  # en dash to regular dash
    line = line.replace('„', '"').replace('“', '"').replace('”', '"')
    line = line.replace('«', '"').replace('»', '"')
    line = line.replace('*', '.')#* * *

    if re.search(r'[:;]$', line):
        line = line[:-1] + '.'

    if not re.search(r'[.!?]$', line):
        line += '.'

    return line
   
def verify_and_patch_stress(accented_lines, verification_dict_all_default_to_zero=False):
    patched_lines = []

    for line in accented_lines:
        detect_multiple_stresses(line) #detect and log Р+айн+овского
        words = line.split()
        new_words = []

        i = 0
        while i < len(words):
            word = words[i]
            #word_clean = re.sub(r'[^\wа-яА-ЯёЁ+]', '', word)
            #word_base = word_clean
            #word_stripped = word_clean.replace('+', '')
            match = re.match(r"^([^\wа-яА-ЯёЁ+]*)([\wа-яА-ЯёЁ+]+)([^\wа-яА-ЯёЁ+]*)$", word)
            if match:
                prefix, word_core, suffix = match.groups()
            else:
                prefix, word_core, suffix = '', word, ''

            word_stripped = word_core.replace('+', '')
            word_base = word_core

            # Replacement dict — always replace
            
            replacement = get_dict_regex_replacement(word_stripped, replacement_dict)
            #if replacement is not None:
            #    logging.info(f"[REPLACED] {word} → {prefix}{replacement}{suffix}")
            #    #new_words.append(replacement)
            #    new_words.append(prefix + replacement + suffix)
            if replacement is not None:
                new_word = prefix + replacement + suffix
                if new_word != word:
                    logging.info(f"[REPLACED] {word} → {new_word}")
                new_words.append(new_word)

            # Verification dict — ask user
            elif word_stripped in verification_dict:
                options = verification_dict[word_stripped]

                # Show context
                context_start = max(0, i - 4)
                context_end = min(len(words), i + 5)
                context_words = words[context_start:context_end]

                # Underline the ambiguous word
                underline_start = "\033[4m"
                underline_end = "\033[0m"
                underlined_context = [
                    f"{underline_start}{w}{underline_end}" if j + context_start == i else w
                    for j, w in enumerate(context_words)
                ]

                print(f"\n[VERIFY] Ambiguous word found: '{word_stripped}' (original: '{word}')")
                print(f"Context: ... {' '.join(underlined_context)} ...")
                print("Options:")
                for idx, opt in enumerate(options):
                    print(f" [{idx}] {opt}")

                default_choice = None
                 
                if word_base == options[0]:
                    default_choice = 0
                elif word_base == options[1]:
                    default_choice = 1


                if verification_dict_all_default_to_zero:
                            print(f"[AUTO-ACCEPT] Using option 0: {options[0]}")
                            #new_words.append(options[0])
                            new_words.append(prefix + options[0] + suffix)
                else:
                    while True:
                        prompt = f"Choose correct stress [0/1]{' (press Enter to keep current)' if default_choice is not None else ''}: "
                        choice = input(prompt).strip()

                        if choice == '' and default_choice is not None:
                            print(f"[ACCEPTED] Using current form: {options[default_choice]}")
                            new_words.append(prefix +options[default_choice] + suffix)
                            break
                        elif choice in ['0', '1']:
                            new_words.append(prefix +options[int(choice)] + suffix)
                            break
                        else:
                            print("Invalid input. Please enter 0 or 1, or press Enter to keep the current form.")
            else:
                new_words.append(word)

            i += 1

        patched_lines.append(" ".join(new_words))

    return patched_lines
    
def get_dict_regex_replacement(word_stripped, replacement_dict):
    # Separate leading and trailing punctuation from the word
    match = re.match(r"^([^\wа-яА-ЯёЁ+]*)([\wа-яА-ЯёЁ+]+)([^\wа-яА-ЯёЁ+]*)$", word_stripped, re.UNICODE)
    if match:
        leading_punct, word_core, trailing_punct = match.groups()
    else:
        # fallback: no match (shouldn't happen unless input is malformed)
        leading_punct = ''
        word_core = word_stripped
        trailing_punct = ''
        
    # 1. Exact match
    if word_core in replacement_dict:
        replacement = replacement_dict[word_core][0]
        return replacement# + trailing_punct
    
    # 2. Wildcard match: keys with '*'     Райновск* = "Райновским" "Райновский" "Райновскому"
    for key_pattern, replacements in replacement_dict.items():
        if '*' in key_pattern:
            prefix = key_pattern.split('*')[0]
            pattern = '^' + re.escape(prefix) + '(.*)$'

            match = re.match(pattern, word_core)
            if match:
                suffix = match.group(1)
                replacement_template = replacements[0]
                replacement = replacement_template.format(suffix)
                return replacement + trailing_punct
                
    # no replacement found
    return None
    
def detect_multiple_stresses(line):
    for word in line.split():
        # Remove punctuation from both ends (like '(', ')' etc.)
        clean_word = word.strip(string.punctuation)
        if '-' in clean_word:
            continue  # Skip hyphenated words
        if clean_word.count('+') >= 2:
            logging.info(f"[MULTI-STRESS WARNING] Word with multiple '+' found: {word}")
            
            
def insert_breaks(accented_lines, empty_line_counts, trailing_empty):
    """Insert <break> tags into accented_lines based on empty_line_counts and trailing_empty,
       also treat accented lines that are just '.' or empty as 1s breaks."""
    output_lines = []

    for i, accented_line in enumerate(accented_lines):
        stripped_line = accented_line.strip()
        
        # Check for break insertion based on empty line logic
        insert_break = False
        break_time = None

        if i > 0 and i < len(empty_line_counts):
            if empty_line_counts[i] == 1:
                insert_break = True
                break_time = '1s'
            elif empty_line_counts[i] >= 2:
                insert_break = True
                break_time = '2s'

        # Dot-only line also triggers 1s break
        is_dot_line = re.fullmatch(r'\.*', stripped_line) and stripped_line != ''
        if not insert_break and is_dot_line:
            insert_break = True
            break_time = '1s'

        if insert_break:
            output_lines.append(f"<break time='{break_time}'/>")

        output_lines.append(accented_line)
        """
        # Add a paragraph-level break (0.5s) unless line starts with dash or is a break line
        if (
            not stripped_line.startswith(('—', '-')) and
            not stripped_line.startswith('<break') and
            not is_dot_line
        ):
            output_lines.append("<break time='0.5s'/>")
        """
    # Handle trailing empty lines
    if trailing_empty == 1:
        output_lines.append("<break time='1s'/>")
    elif trailing_empty >= 2:
        output_lines.append("<break time='2s'/>")
    else:
        if accented_lines:
            last_line = accented_lines[-1].strip()
            if re.fullmatch(r'\.*', last_line) and last_line != '':
                output_lines.append("<break time='1s'/>")

    return output_lines
