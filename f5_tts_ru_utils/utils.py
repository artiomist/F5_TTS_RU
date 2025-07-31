import os
import re
#import time
#start = time.time()
import subprocess
import magic
from bs4 import BeautifulSoup
from ebooklib import epub
from ebooklib.epub import EpubHtml
from pathlib import Path
# Metadata handling
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, error
from mutagen.mp4 import MP4, MP4Cover

from PIL import Image, ImageDraw, ImageFont  # For image resizing in ensure_even_dimensions
#logging.debug(f"PIL.Image imported in {time.time() - start:.2f}s")
# =========================
# 📦 Custom Project Modules / Local Imports
# =========================
from . import config
# logging
import logging
logging.basicConfig(
    level=config.LOGGINGLEVEL,
    format=config.LOGGINGFORMAT, 
)


def init_directories(sanitized_name):
    # Base "Working_files" directory
    #base_dir = Path(__file__).parent / config.WORKING_DIR
    
    #project_dir = Path(__file__).parent / config.PROJECT_DIR
    #base_dir = project_dir / config.WORKING_DIR
    base_dir = config.PROJECT_PATH / config.WORKING_DIR
    
    abs_working_dir = base_dir.resolve()
    abs_working_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Absolute WORKING_DIR: {abs_working_dir}")

    # Subdirectory for the specific book
    book_dir_name = f"{sanitized_name}"
    book_working_dir = abs_working_dir / book_dir_name
    book_working_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created book working dir: {book_working_dir}")    
    
    # Create subdirectories inside the book working dir
    tmp_audio_dir = book_working_dir / config.TMP_AUDIO_SUBDIR
    tmp_m4a_dir = book_working_dir / config.FINAL_OUTPUT_SUBDIR
    txt_debug_dir = book_working_dir / config.DEBUG_TXT_SUBDIR
    temp_converted_dir = book_working_dir / config.TMP_CONVERTED_SUBDIR

    for d in [tmp_audio_dir, tmp_m4a_dir, txt_debug_dir, temp_converted_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Created subdir: {d}")

    return str(tmp_audio_dir), str(tmp_m4a_dir), str(txt_debug_dir), str(temp_converted_dir)



#----1.1 LOAD FILE----
def detect_file_type(file_path):
    """Detect the MIME type of a file."""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        raise RuntimeError(f"Error detecting file type: {e}")

def ensure_directory(directory_path):
    """Ensure that a directory exists."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error creating directory {directory_path}: {e}")

def sanitize_filename(filename):
    """Sanitize a filename by removing invalid characters."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    return sanitized.replace(" ", "_")


#----1.2 Convert all to EPUB----
def convert_to_epub(input_path, output_path):
    """Convert an ebook to EPUB format using Calibre."""
    try:
        ensure_directory(os.path.dirname(output_path))
        subprocess.run(['ebook-convert', input_path, output_path], check=True, text=True, encoding='utf-8', errors='replace')
        logging.info(f"Converted {input_path} to EPUB.")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting eBook: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during conversion: {e}")


#----2.1 Extract data for TXT->WAV----
def extract_text_and_title_and_author_from_epub(epub_path, txt_debug_dir):
    """Extract chapters, title, and author from an EPUB file using NCX or heading tags."""
    try:
        book = epub.read_epub(epub_path)
        logging.info(f"EPUB '{epub_path}' successfully read.")
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB file: {e}")

    sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(epub_path))[0])
    #text_content = []
    title = None
    author = None 
    chapters = []#chapters: List[Tuple[str, str]]

    # --- Extract metadata ---
    try:
        # Extract title
        title_metadata = book.get_metadata('DC', 'title')
        if title_metadata:
            title = title_metadata[0][0]
            logging.info(f"Extracted title: {title}")
        else:
            title = os.path.splitext(os.path.basename(epub_path))[0]
            logging.info(f"No title in metadata. Using filename: {title}")

        # Extract author
        author_metadata = book.get_metadata('DC', 'creator')
        if author_metadata:
            # Assuming the first creator is the primary author
            author = author_metadata[0][0]
            logging.info(f"Extracted author: {author}")
        else:
            author = config.DEFAULT_AUTHOR # Default if no author found
            logging.info(f"No author in metadata. Using '{author}'.")

    except Exception as e: # Catch potential errors during metadata extraction
        logging.error(f"Error extracting metadata: {e}")
        if title is None: # Ensure title has a fallback
            title = os.path.splitext(os.path.basename(epub_path))[0]
            logging.info(f"Using filename as title due to error: {title}")
        if author is None: # Ensure author has a fallback
            author = config.DEFAULT_AUTHOR # Default if no author found
            logging.info(f"Using '{author}' due to error in metadata extraction.")

    
    
    # --- Extract chapters using NCX (TOC) ---
    try:
        toc_entries = extract_chapters_ordered_by_toc(book)

        for chapter_title, href in toc_entries:
            item = book.get_item_with_href(href)
            if item:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                #chapter_text_with_marker = f"{chapter_title}\n[BREAK]\n{text}"
                chapter_text_with_marker = text
                chapters.append((chapter_title, chapter_text_with_marker))
    except Exception as e:
        logging.error(f"TOC (NCX) parsing failed: {e}")

    # --- Fallback if TOC failed ---
    if not chapters:
        # Fallback heading-based parsing
        try:
            for item in book.items:
                #if item.get_type() == epub.ITEM_DOCUMENT:
                if isinstance(item, EpubHtml):
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    h1 = soup.find('h1')
                    h2 = soup.find('h2')
                    if h1:
                        chapter_title = h1.get_text(strip=True)
                        h1.decompose()  # Remove the h1 tag from soup
                    elif h2:
                        chapter_title = h2.get_text(strip=True)
                        h2.decompose()  # Remove the h2 tag from soup
                    else:
                        chapter_title = "."
                        
                    chapter_text = soup.get_text(separator='\n', strip=True) # Extract full chapter text
                    # Insert [BREAK] after the <h1> content
                    chapter_text_with_marker = f"{chapter_title}\n[BREAK]\n{chapter_text}"
                    chapters.append((chapter_title, chapter_text_with_marker))
        except Exception as e:
            logging.error(f"Error parsing item {item.get_id()}: {e}")

    # --- Save extracted chapters ---
    # Save and process chapters if any were found
    logging.info("Processing chapters")
    if not chapters or not any(ch_text.strip() for _, ch_text in chapters):
        raise ValueError("No valid chapters could be extracted.")
        
    #processed_chapters = []
    #sanitized_title = sanitize_filename(title)

    for idx, (ch_title, ch_text) in enumerate(chapters):
        title_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter{idx+1:02d}_title.txt")
        with open(title_path, 'w', encoding='utf-8') as f:
            f.write(ch_title)
        logging.debug(f"✅ Saved title {idx+1} to {title_path} )")
        
        txt_path = os.path.join(txt_debug_dir, f"{sanitized_name}_chapter{idx+1:02d}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(ch_text)
            
        logging.debug(f"✅ Saved chapter {idx+1} to {txt_path} ({len(ch_text)} chars)")


        # Process saved chapter with normalization and accentuation

        # Store updated chapter
        #processed_chapters.append((ch_title, ch_text))

        #chapters = processed_chapters

    return chapters, title, author

def extract_chapters_ordered_by_toc(book):
    def flatten_toc(items):
        for item in items:
            if isinstance(item, epub.Link):
                yield (item.title, item.href.split('#')[0])
            elif isinstance(item, (list, tuple)):
                yield from flatten_toc(item)
    return list(flatten_toc(book.toc))
    
    
#----2.3 Extract data for TXT->WAV----
'''def extract_cover(ebook_path):
    """Extract cover image from the eBook."""
    try:
        cover_path = os.path.splitext(ebook_path)[0] + '.jpg'
        result = subprocess.run(
            ['ebook-meta', ebook_path, '--get-cover', cover_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False  # Do not raise an exception automatically
        )
        if os.path.exists(cover_path) and os.path.getsize(cover_path) > 0:
            return cover_path
        else:
            print("No embedded cover found in EPUB. Skipping cover extraction.")
    except Exception as e:
        print(f"Error extracting eBook cover: {e}")
    return None    
'''


def extract_cover(ebook_path):
    """
    Extract cover image from the eBook (EPUB).
    Tries ebook-meta (Calibre) first; falls back to Python EPUB parser.
    Returns path to cover image or None.
    """
    cover_path = Path(ebook_path).with_suffix('.jpg')

    # Try Calibre's ebook-meta if available
    try:
        result = subprocess.run(
            ['ebook-meta', ebook_path, '--get-cover', str(cover_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False  # Don't raise, we'll check manually
        )

        if cover_path.exists() and cover_path.stat().st_size > 0:
            print("Cover extracted using ebook-meta.")
            return str(cover_path)
        else:
            print("ebook-meta failed or no cover found. Trying Python fallback.")
    except FileNotFoundError:
        print("ebook-meta not found. Falling back to Python method.")

    # Fallback: use pure Python EPUB cover extractor
    return extract_cover_python(ebook_path)

# ---- Python-only fallback using ebooklib ----
def extract_cover_python(ebook_path):
    try:
        from ebooklib import epub
        import io

        book = epub.read_epub(ebook_path)
        for item in book.get_items():
            if item.get_type() == 9 and 'cover' in item.get_name().lower():  # 9 = IMAGE
                cover_path = Path(ebook_path).with_stem(Path(ebook_path).stem + '_cover').with_suffix('.jpg')
                with open(cover_path, 'wb') as f:
                    f.write(item.get_content())
                print("Cover extracted using Python EPUB parser.")
                return str(cover_path)

        print("No cover found in EPUB using Python.")
    except Exception as e:
        print(f"Python-based cover extraction failed: {e}")
    return None

def ensure_even_dimensions(image_path):
    img = Image.open(image_path)
    w, h = img.size
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    if new_w != w or new_h != h:
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        img = img.resize((new_w, new_h))
        img.save(image_path)
        
def create_placeholder_cover(title, output_path, width=600, height=800, background_color=(70, 130, 180), text_color=(255, 255, 255)):
    """
    Create a placeholder cover image with the given title.
    
    Parameters:
        title (str): Title text to put on the image.
        output_path (str): Path to save the generated image.
        width (int): Width of the image.
        height (int): Height of the image.
        background_color (tuple): RGB background color.
        text_color (tuple): RGB text color.
    """
    # Create a blank image
    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Load a font (fallback to default if custom font not found)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Word wrap the title if it's too long
    max_width = width - 40
    lines = []
    words = title.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    # Calculate position to center the text vertically
    total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines]) + (len(lines) - 1) * 10
    y_text = (height - total_text_height) // 2

    for line in lines:
        text_width = draw.textlength(line, font=font)
        x = (width - text_width) // 2
        draw.text((x, y_text), line, font=font, fill=text_color)
        y_text += draw.textbbox((0, 0), line, font=font)[3] + 10

    # Save the image
    img.save(output_path)
    print(f"Placeholder cover saved to: {output_path}")
    
#----3.0 OUTPUT----Embed into M4A
def embed_metadata_into_m4a(m4a_path, cover_image_path, title, author, album_title=None):
    try:
        tags = MP4(m4a_path)

        tags["\xa9nam"] = title
        tags["\xa9ART"] = author
        tags["\xa9alb"] = album_title if album_title else title

        if cover_image_path and os.path.exists(cover_image_path):
            with open(cover_image_path, 'rb') as img:
                cover_data = img.read()
                tags["covr"] = [MP4Cover(cover_data, imageformat=MP4Cover.FORMAT_JPEG)]
        else:
            print(f"[INFO] No cover image provided or found at {cover_image_path}")

        tags.save()
        print(f"[SUCCESS] Metadata embedded into {m4a_path}")
    except Exception as e:
        print(f"[ERROR] Failed to embed M4A metadata: {e}")


#----3.0 OUTPUT----Embed into MP3
def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author, album_title=None):
    #def embed_metadata_into_mp3(mp3_path, cover_image_path, title, author, album_title=None):
    """Embed cover image, title, author, and album into the MP3 file's metadata."""
    try:
        audio = ID3(mp3_path)
    except error: # If no ID3 tag exists, create one
        audio = ID3()
    except Exception as e:
        print(f"Error loading ID3 tags for {mp3_path}: {e}. Creating new tags.")
        audio = ID3() # Fallback to creating new tags

    # Add Title
    audio.delall("TIT2") # Remove existing title tags
    audio.add(TIT2(encoding=3, text=title))
    print(f"Set TIT2 (Title) to: {title}")

    # Add Author/Artist
    audio.delall("TPE1") # Remove existing artist tags
    audio.add(TPE1(encoding=3, text=author))
    print(f"Set TPE1 (Author/Artist) to: {author}")

    # Add Album (often used for Book Title in audiobook players)
    # If a specific album_title is provided, use it. Otherwise, use the main title.
    album_to_set = album_title if album_title else title
    audio.delall("TALB") # Remove existing album tags
    audio.add(TALB(encoding=3, text=album_to_set))
    print(f"Set TALB (Album) to: {album_to_set}")

    # Embed Cover Image (existing logic)
    if cover_image_path and os.path.exists(cover_image_path):
        audio.delall("APIC") # Remove existing cover art
        try:
            with open(cover_image_path, 'rb') as img:
                audio.add(APIC(
                    encoding=3,          # 3 is for UTF-8
                    mime='image/jpeg',   # or 'image/png'
                    type=3,              # 3 means front cover
                    desc='Front cover',
                    data=img.read()
                ))
            print(f"Embedded cover image into {mp3_path}")
        except Exception as e:
            print(f"Failed to embed cover image into MP3: {e}")
    else:
        print(f"No cover image provided or found at '{cover_image_path}'. Skipping cover embedding.")
        audio.delall("APIC") # Ensure no old APIC tag remains if new cover is not set

    try:
        # Save with ID3v2.3 for broad compatibility
        audio.save(mp3_path, v2_version=3)
        print(f"Successfully saved metadata to {mp3_path}")
    except Exception as e:
        print(f"Failed to save MP3 metadata: {e}")
