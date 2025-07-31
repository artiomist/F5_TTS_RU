import os
import re
import subprocess
from pathlib import Path
from mutagen.mp4 import MP4

from . import config

import logging
logging.basicConfig(
    level=config.LOGGINGLEVEL,
    format=config.LOGGINGFORMAT, 
)
from .utils import (
    sanitize_filename,
)

def combine_m4a_to_m4b():
    logging.info("🔗 Starting process: Combine .m4a files into .m4b")

    try:
        PROJECT_DIR = Path(__file__).parent.parent#.parent / config.PROJECT_DIR
        WORKING_DIR = PROJECT_DIR / config.WORKING_DIR
        logging.debug(f"[DEBUG] Checking WORKING_DIR: {WORKING_DIR}")

        if not WORKING_DIR.exists() or not WORKING_DIR.is_dir():
            raise ValueError("[Error] Process ebook first")

        player_audio = None
        all_outputs = []
        status = "✅ Completed"

        # Iterate over book folders
        for book_dir in WORKING_DIR.iterdir():
            logging.debug(f"Found: {book_dir.name}")
            #if not book_dir.is_dir() or not book_dir.name.endswith("INFERED"):
            #    logging.debug(f"Skipping {book_dir.name}")
            #    continue

            logging.info(f"📘 Found inferred book folder: {book_dir.name}")
            tmp_m4a_dir = book_dir / config.FINAL_OUTPUT_SUBDIR
            if not tmp_m4a_dir.exists():
                raise FileNotFoundError(f"❌ FINAL_OUTPUT_SUBDIR not found in {book_dir}")

            audio_files = get_sorted_audio_files(tmp_m4a_dir, extensions=(".m4a",))
            if not audio_files:
                raise ValueError(f"[ERROR] No .m4a files found in {tmp_m4a_dir}")

            logging.info(f"🎵 Found {len(audio_files)} audio files")

            # Normalize audio if enabled
            if config.NORMALIZE_AUDIO_FILES:
                logging.info("🎚️ Normalizing audio files")
                normalized_dir = tmp_m4a_dir / "normalized"
                normalized_dir.mkdir(exist_ok=True)

                normalized_audio_files = []
                for file in audio_files:
                    input_path = tmp_m4a_dir / file
                    output_file = normalized_dir / file
                    normalize_audio_file(str(input_path), str(output_file))
                    normalized_audio_files.append(file)

                tmp_m4a_dir = normalized_dir
                #audio_files = get_sorted_audio_files(tmp_m4a_dir, extensions=(".m4a",))
                audio_files = normalized_audio_files  # Already ordered in loop

            # Generate chapter file
            logging.info("📑 Generating chapter metadata")
            generate_chapters(tmp_m4a_dir, audio_files)

            # Extract metadata
            base_name = book_dir.name
            #ebook_title = book_dir.name.replace("_INFERED", "")
            #ebook_author = "Unknown"  # Replace if available
            #cover_image = os.path.join(tmp_m4a_dir, "cover.jpg")
            cover_image = book_dir / config.TMP_CONVERTED_SUBDIR / f"cover_{base_name}.jpg"
            if not os.path.exists(cover_image):
                logging.warning("⚠️ Cover image not found, proceeding without it")            
            #load title and author
            title_author_path = os.path.join(book_dir / config.TMP_CONVERTED_SUBDIR, f"title_author_{base_name}.txt")

            with open(title_author_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            ebook_title = lines[0].strip() if len(lines) > 0 else ""
            ebook_author = lines[1].strip() if len(lines) > 1 else ""

            logging.debug(f"📖 Loaded EBook Title: {ebook_title}")
            logging.debug(f"✍️ Loaded Author: {ebook_author}")


            # Convert to M4B
            logging.info("🎧 Converting to final .m4b")
            final_m4b = convert_to_m4b(
                tmp_m4a_dir,
                cover_image_path=cover_image if os.path.exists(cover_image) else None,
                author=ebook_author,
                album_title=ebook_title,
            )

            player_audio = final_m4b
            all_outputs.append(final_m4b)
            logging.info(f"✅ M4B created: {final_m4b}")

        return player_audio, all_outputs, status

    except Exception as e:
        logging.error(f"❌ Error in combine_m4a_to_m4b: {e}")
        return None, [], f"❌ Failed: {e}"
        
        


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

#----3.1 OUTPUT----Normalize
def normalize_audio_file(input_file, output_file):
    ffmpeg_cmd = [
        'ffmpeg', '-i', input_file,
        '-vn',  # ⬅️ STRIP EMBEDDED COVER ART (video stream)
        '-af', 'agate=threshold=-25dB:ratio=1.4:attack=10:release=250,'
               'afftdn=nf=-70,'
               'acompressor=threshold=-20dB:ratio=2:attack=80:release=200:makeup=1dB,'
               'loudnorm=I=-16:TP=-3:LRA=7:linear=true,'
               'equalizer=f=150:t=q:w=2:g=1,'
               'equalizer=f=250:t=q:w=2:g=-3,'
               'equalizer=f=3000:t=q:w=2:g=2,'
               'equalizer=f=5500:t=q:w=2:g=-4,'
               'equalizer=f=9000:t=q:w=2:g=-2,'
               'highpass=f=63',
        '-ar', '44100',
        '-ac', '2',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y', output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True, text=True, encoding='utf-8', errors='replace')
"""
def normalize_audio_folder(folder_path):
    print("[INFO] Normalizing audio files to .m4a...")
    for file in os.listdir(folder_path):
        if file.lower().endswith('.mp3'):
            input_path = os.path.join(folder_path, file)
            output_path = os.path.join(folder_path, os.path.splitext(file)[0] + '.m4a')
            print(f"Normalizing {input_path} → {output_path}")
            try:
                normalize_audio_file(input_path, output_path)
            except Exception as e:
                print(f"[ERROR] Failed to normalize {input_path}: {e}")
"""
#----3.2 OUTPUT----Generate Chapters
def generate_chapters(folder_path, audio_files):
    chapter_file = os.path.join(folder_path, "chapters.txt")
    current_time = 0.0
    
    with open(chapter_file, 'w', encoding='utf-8') as f:
        for idx, file in enumerate(audio_files):
            input_path = os.path.join(folder_path, file)
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', input_path],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            try:
                duration = round(float(result.stdout.strip()), 3)
            except Exception:
                print(f"[WARN] Failed to get duration for {file}. Skipping.")
                continue
            start_ms = int(current_time * 1000)
            end_ms = int((current_time + duration) * 1000)

            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\n")
            f.write(f"END={end_ms}\n")
            #
            try:
                tags = MP4(input_path)
                title = tags.get('\xa9nam')
                if title and len(title) > 0:
                    f.write(f"title={title[0]}\n\n")
                else:
                    f.write(f"title={idx + 1}\n\n")
            except Exception as e:
                f.write(f"title={idx + 1}\n\n")

            current_time += duration

    return chapter_file

"""def generate_chapters(folder_path, base_name):
    folder = Path(folder_path)
    chapter_file = folder / "chapters.txt"
    current_time = 0.0

    # Step 1: Find matching files, case-insensitive
    audio_files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and re.match(rf"{re.escape(base_name)}_chapter\d+\.m4a", f.name, re.IGNORECASE)
    ]

    if not audio_files:
        logging.warning(f"⚠️ No matching chapter files found in {folder}")
        return str(chapter_file)

    with open(chapter_file, 'w', encoding='utf-8') as f:
        for idx, file_path in enumerate(audio_files, start=1):
            # Get duration using ffprobe
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            try:
                duration = round(float(result.stdout.strip()), 3)
            except Exception:
                logging.warning(f"[WARN] Failed to get duration for {file_path.name}. Skipping.")
                continue

            start_ms = int(current_time * 1000)
            end_ms = int((current_time + duration) * 1000)

            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\n")
            f.write(f"END={end_ms}\n")

            # Try to read the chapter title from tags
            try:
                tags = MP4(str(file_path))
                title_list = tags.get('\xa9nam', [])
                title_str = title_list[0] if title_list else f"Chapter {idx}"
                f.write(f"title={title_str}\n\n")
            except Exception as e:
                logging.warning(f"[WARN] Failed to extract title from tags: {e}")
                f.write(f"title=Chapter {idx}\n\n")

            current_time += duration

    logging.info(f"✅ Wrote {len(audio_files)} chapters to {chapter_file}")
    return str(chapter_file)
"""   
#----3.3 OUTPUT----Output .m4b
def get_sorted_audio_files(folder_path, extensions=(".mp3", ".m4a")):
    """
    Returns a sorted list of audio files with specified extensions in the given folder.
    Defaults to .mp3 and .m4a if no filter is provided.
    """
    return sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions) and os.path.isfile(os.path.join(folder_path, f))
    )

def get_metadata(metadata_file):
    title, artist = "Unknown", "Unknown"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.lower().startswith("title="): title = line.strip().split("=", 1)[1]
                if line.lower().startswith("artist="): artist = line.strip().split("=", 1)[1]
    return title, artist

def convert_to_m4b(folder_path, cover_image_path, author, album_title=None):
    #audio_files = get_sorted_audio_files(folder_path)
    #only include .m4a files
    #audio_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith('.m4a')]

    audio_files = get_sorted_audio_files(folder_path, extensions=(".m4a",))



    if not audio_files:
        print("[ERROR] No .m4a files to combine.")
        return

    filelist_path = os.path.abspath(os.path.join(folder_path, "filelist.txt"))
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for file in audio_files:

            full_path = os.path.abspath(os.path.join(folder_path, file)).replace('\\', '/')
            f.write(f"file '{full_path}'\n")

    combined_m4b = os.path.join(folder_path, "combined_audio.m4a")

    ffmpeg_cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', filelist_path,
            '-vn'                      # <== DO NOT include the video stream
        ]
    
    if config.USE_AAC_FILTERS:#Convertion to combined_audio.m4a with filters
        ffmpeg_cmd += [
            '-af', 'loudnorm=I=-16:TP=-3:LRA=7',  # or your full filter chain
            #'agate=threshold=-25dB:ratio=1.4:attack=10:release=250,afftdn=nf=-70,acompressor=threshold=-20dB:ratio=2:attack=80:release=200:makeup=1dB,loudnorm=I=-16:TP=-3:LRA=7:linear=true,equalizer=f=150:t=q:w=2:g=1,equalizer=f=250:t=q:w=2:g=-3,equalizer=f=3000:t=q:w=2:g=2,equalizer=f=5500:t=q:w=2:g=-4,equalizer=f=9000:t=q:w=2:g=-2,highpass=f=63',
            '-ar', '44100', '-ac', '2',
            '-c:a', 'aac',              # Audio codec
            '-b:a', config.AAC_AUDIO_BITRATE             # Bitrate
        ]
    else:#Convertion to combined_audio.m4a without filters
        ffmpeg_cmd += ['-c:a', 'copy']
    
    ffmpeg_cmd += ['-y', combined_m4b]

    subprocess.run(ffmpeg_cmd, check=True, text=True, encoding='utf-8',errors='replace')
    
    #title, artist = get_metadata(os.path.join(folder_path, "metadata.txt"))
    #cover_path = os.path.join(folder_path, "cover.jpg")
    chapter_file = generate_chapters(folder_path, audio_files)
    album_title = album_title or Path(folder_path).name

    #output_file = str(Path(folder_path).with_suffix('.m4b'))
    sanitized_title = sanitize_filename(album_title)
    sanitized_author = sanitize_filename(author)
    output_file = os.path.join(folder_path, f"{sanitized_title}-{sanitized_author}.m4b")
    ffmpeg_cmd = [
        'ffmpeg', '-i', combined_m4b, '-i', cover_image_path,
        '-f', 'ffmetadata', '-i', chapter_file,
        '-map', '0:a', '-map', '1:v', '-map_metadata', '2', '-map_chapters', '2',
        '-c:a', 'copy', '-c:v', 'copy',
        '-disposition:v:0', 'attached_pic',
        '-metadata', f'title={album_title}', '-metadata', f'artist={author}',
        '-f', 'mp4', '-y', output_file
    ]

    subprocess.run(ffmpeg_cmd, check=True, text=True, encoding='utf-8', errors='replace')
    os.remove(combined_m4b)
    print(f"[SUCCESS] Created audiobook: {output_file}")
    return output_file
