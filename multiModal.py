# Run this cell first to install necessary packages
!pip install openai-whisper yt-dlp google-generativeai opencv-python Pillow pytesseract torch torchvision torchaudio gradio --quiet youtube_transcript_api
!sudo apt-get -qq install tesseract-ocr # For Pytesseract OCR fallback
!sudo apt-get -qq install ffmpeg # Ensure ffmpeg is available

#set the environment
import os
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'

# --- Combined Script (Colab + Gradio + Full Video Overview + Segment Processing + Local Files) ---

# General & System
import os
import subprocess
import datetime
import time
import sys
import json
import tempfile
import io
import traceback
import threading
import shutil

# AI & Media Processing
import whisper
import cv2
import google.generativeai as genai
from PIL import Image
import pytesseract
import yt_dlp
import torch
import gradio as gr
# youtube_transcript_api is NO LONGER THE PRIMARY METHOD for YouTube full overview per your request
# from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- Configuration ---
MODEL_NAME_GEMINI = "gemini-1.5-flash-latest"
MODEL_NAME_WHISPER = "medium" # Used for segment transcription
MODEL_NAME_WHISPER_FULL_VIDEO = "base" # Used for full video overview, "base" or "small" recommended for speed
RETRY_ATTEMPTS = 3
RATE_LIMIT_WAIT_TIME = 10  # Seconds
DEFAULT_FRAME_INTERVAL_MS = 2500
USE_OCR_FALLBACK = True
EXPORT_JSON_FRAME_TEXTS = False

# --- Global Stop Flag ---
GlobalStopFlag = threading.Event()

# --- Helper Functions (Keep all your existing helper functions as they are) ---
# seconds_to_hhmmss, ms_to_time_str, time_str_to_ms,
# extract_full_audio_ffmpeg, trim_audio_ffmpeg, transcribe_audio_whisper,
# get_text_from_image_gemini, fallback_ocr_pytesseract, extract_text_from_video_frames
# (Ensure these are exactly as in your last provided code)
def seconds_to_hhmmss(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def ms_to_time_str(ms):
    if ms < 0: ms = 0
    s, ms_rem = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(ms_rem):03d}"

def time_str_to_ms(time_str):
    parts = str(time_str).split(':')
    h, m, s_full = 0, 0, "0"
    if len(parts) == 3: h, m, s_full = parts
    elif len(parts) == 2: m, s_full = parts
    elif len(parts) == 1: s_full = parts[0]
    else: raise ValueError(f"Invalid time string format: {time_str}")
    s_parts = s_full.split('.')
    try:
        seconds_val = int(s_parts[0])
        milliseconds = int(s_parts[1].ljust(3, '0')[:3]) if len(s_parts) > 1 else 0
        return (int(h) * 3600 + int(m) * 60 + seconds_val) * 1000 + milliseconds
    except ValueError: raise ValueError(f"Invalid time component in: {time_str}")

def extract_full_audio_ffmpeg(video_file, output_audio_file):
    try:
        command = ["ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-loglevel", "error", output_audio_file]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Full audio extracted: {output_audio_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting full audio: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found.")
        return False

def trim_audio_ffmpeg(input_file, output_file, start_sec, end_sec):
    duration_sec = end_sec - start_sec
    if duration_sec <=0:
        print(f"Error: Duration for trimming is not positive (start: {start_sec}s, end: {end_sec}s)")
        return False
    try:
        command = ["ffmpeg", "-y", "-i", input_file, "-ss", seconds_to_hhmmss(start_sec), "-t", str(duration_sec), "-c", "copy", "-loglevel", "error", output_file]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio trimmed (copy): {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming audio with copy: {e.stderr}. Retrying with re-encoding...")
        try:
            command_reencode = ["ffmpeg", "-y", "-ss", seconds_to_hhmmss(start_sec), "-i", input_file, "-t", str(duration_sec), "-loglevel", "error", output_file] # Corrected output_audio_file to output_file
            result = subprocess.run(command_reencode, check=True, capture_output=True, text=True)
            print(f"Audio trimmed (re-encode): {output_file}")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"Error trimming audio (re-encoding failed): {e2.stderr}")
            return False
    except FileNotFoundError: print("Error: ffmpeg not found."); return False

def transcribe_audio_whisper(audio_path, model_name=MODEL_NAME_WHISPER): # Default for segments
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available(), language='en')
        transcript = " ".join(segment['text'].strip() for segment in result['segments'])
        print(f"Transcription complete for {audio_path}.")
        return transcript
    except Exception as e:
        print(f"Error during Whisper transcription ({audio_path}): {e}")
        if torch and not torch.cuda.is_available(): print("CUDA not available, Whisper ran on CPU.")
        return "Audio transcription failed."

def get_text_from_image_gemini(api_key_param, image_bytes):
    try:
        genai.configure(api_key=api_key_param)
        model = genai.GenerativeModel(MODEL_NAME_GEMINI)
        image_part = {"mime_type": "image/png", "data": image_bytes}
        prompt_img = "Extract all visible text from this image. If no text is clearly visible, respond with 'No text found.' Do not add any conversational filler, just the extracted text or 'No text found.'"
        safety_settings_img = [{"category": cat, "threshold": "BLOCK_NONE"} for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        request_options_img = genai.types.RequestOptions(timeout=120)
        response = model.generate_content([prompt_img, image_part], safety_settings=safety_settings_img, request_options=request_options_img)
        text_content = "".join(part.text for part in response.parts if hasattr(part, "text")).strip() if response.parts else (response.text.strip() if hasattr(response, 'text') else "")
        return text_content if text_content else "No text found."
    except Exception as e:
        print(f"Gemini API error (image text): {e}")
        if any(err_msg in str(e).lower() for err_msg in ["429", "resource has been exhausted", "rate limit", "quota"]): raise
        return None

def fallback_ocr_pytesseract(image: Image.Image):
    try: return pytesseract.image_to_string(image).strip()
    except Exception as e: print(f"Pytesseract OCR error: {e}"); return ""

def extract_text_from_video_frames(video_path, start_time_ms, end_time_ms, api_key_param,
                                   frame_interval_ms_cfg=DEFAULT_FRAME_INTERVAL_MS,
                                   use_ocr_fallback_cfg=USE_OCR_FALLBACK,
                                   export_json_cfg=EXPORT_JSON_FRAME_TEXTS,
                                   progress_tracker=None):
    if start_time_ms >= end_time_ms: return "Error: Start time must be before end time."
    if not os.path.exists(video_path): return f"Error: Video file not found at {video_path}"
    temp_segment_path = None
    all_extracted_texts_list = []
    try:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(video_path)
        sanitized_base_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in base_name)
        ts = int(time.time())
        temp_segment_path = os.path.join(temp_dir, f"segment_{sanitized_base_name}_{ts}.mp4")

        segment_processing_duration_ms = end_time_ms - start_time_ms
        ffmpeg_ss_time = ms_to_time_str(start_time_ms).split('.')[0]
        ffmpeg_to_duration = ms_to_time_str(segment_processing_duration_ms).split('.')[0]
        ffmpeg_command = ['ffmpeg', '-y', '-ss', ffmpeg_ss_time, '-i', video_path, '-to', ffmpeg_to_duration, '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-loglevel', 'error', temp_segment_path]

        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"FFmpeg (fast seek) error: {result.stderr}. Retrying accurate seek...")
            ffmpeg_ss_time_accurate = ms_to_time_str(start_time_ms)
            ffmpeg_duration_accurate = ms_to_time_str(segment_processing_duration_ms)
            ffmpeg_command_accurate = ['ffmpeg', '-y', '-i', video_path, '-ss', ffmpeg_ss_time_accurate, '-t', ffmpeg_duration_accurate, '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-loglevel', 'error', temp_segment_path]
            result = subprocess.run(ffmpeg_command_accurate, capture_output=True, text=True, check=False)
            if result.returncode != 0: return f"FFmpeg segment creation failed: {result.stderr}"

        cap = cv2.VideoCapture(temp_segment_path)
        if not cap.isOpened(): return f"Error: Could not open video segment: {temp_segment_path}"

        total_frames_to_process_estimate = segment_processing_duration_ms / frame_interval_ms_cfg if frame_interval_ms_cfg > 0 else 1
        processed_frames_count = 0

        last_processed_segment_ms = -frame_interval_ms_cfg - 1
        while True:
            if GlobalStopFlag.is_set():
                print("Frame extraction cancelled by user.")
                return "Frame extraction cancelled."

            segment_current_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if segment_current_pos_ms > segment_processing_duration_ms + 200 : break
            ret, frame = cap.read()
            if not ret: break
            if segment_current_pos_ms >= last_processed_segment_ms + frame_interval_ms_cfg:
                if progress_tracker:
                    progress_tracker(processed_frames_count / total_frames_to_process_estimate if total_frames_to_process_estimate > 0 else 0, desc="Extracting frame text...")

                pil_image = None
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    img_byte_arr = io.BytesIO(); pil_image.save(img_byte_arr, format='PNG'); image_bytes = img_byte_arr.getvalue()
                    text_candidate = None
                    for attempt in range(RETRY_ATTEMPTS):
                        if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled during Gemini call")
                        try: text_candidate = get_text_from_image_gemini(api_key_param, image_bytes); break
                        except InterruptedError: raise
                        except Exception as e_gemini_img:
                            if any(err_msg in str(e_gemini_img).lower() for err_msg in ["429", "resource", "limit", "quota"]):
                                print(f"Gemini rate/quota (image text attempt {attempt+1}): {e_gemini_img}")
                                if attempt < RETRY_ATTEMPTS - 1: time.sleep(RATE_LIMIT_WAIT_TIME)
                                else: text_candidate = None;
                            else: text_candidate = None; break

                    if text_candidate is None and use_ocr_fallback_cfg :
                        print("Gemini failed/no text, trying Pytesseract OCR...")
                        if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled during OCR")
                        if pil_image is None: pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        ocr_text = fallback_ocr_pytesseract(pil_image)
                        if ocr_text: text_candidate = ocr_text; print(f"OCR Fallback found: '{ocr_text[:30]}...'")

                    if text_candidate and text_candidate.lower().strip() not in ["no text found", ""]:
                        all_extracted_texts_list.append(text_candidate)
                except InterruptedError:
                     return "Frame extraction cancelled during processing."
                except Exception as e_frame_proc: print(f"Error processing frame: {e_frame_proc}")
                last_processed_segment_ms = segment_current_pos_ms
                processed_frames_count +=1
        unique_texts = sorted(list(set(all_extracted_texts_list)))
        return "\n---\n".join(unique_texts) if unique_texts else "No text found in video frames."
    except InterruptedError:
        return "Frame extraction cancelled."
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if temp_segment_path and os.path.exists(temp_segment_path):
            try: os.remove(temp_segment_path)
            except Exception as e_del_seg: print(f"Warn: Could not delete temp segment {temp_segment_path}: {e_del_seg}")

# Removed get_youtube_transcript_text as it's no longer the primary for overview

def summarize_full_text_with_gemini(api_key_param, full_text, video_title="this video"):
    if not full_text or "Audio transcription failed" in full_text: # Check for Whisper failure
        return f"Cannot summarize: {full_text}"
    prompt = (
        f"Provide a comprehensive yet concise summary of the following video transcript. "
        f"Video title (if known): '{video_title}'. Focus on main topics, key arguments, and important conclusions.\n\n"
        "TRANSCRIPT (first ~150k chars):\n"
        f"{full_text[:150000]}\n\n"
        "COMPREHENSIVE SUMMARY OF THE ENTIRE VIDEO:"
    )
    try:
        genai.configure(api_key=api_key_param)
        model = genai.GenerativeModel(MODEL_NAME_GEMINI)
        safety_settings = [{"category": cat, "threshold": "BLOCK_NONE"} for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        request_options = genai.types.RequestOptions(timeout=300)
        response = model.generate_content(prompt, safety_settings=safety_settings, request_options=request_options)
        summary_text = "".join(part.text for part in response.parts if hasattr(part, "text")).strip() if response.parts else (response.text.strip() if hasattr(response, 'text') else "")
        return summary_text if summary_text else "Summary generation returned no text."
    except Exception as e:
        print(f"Error summarizing full text with Gemini: {e}")
        if any(err_msg in str(e).lower() for err_msg in ["429", "resource", "limit", "quota"]):
            return f"Failed to generate summary due to API limits/quota: {e}"
        return f"Error during summary generation: {e}"

# --- Gradio Specific Helper Functions ---
DEFAULT_SUMMARY_REQUEST_PROMPT = "Please provide a detailed and coherent summary of the current video segment."

def process_video_for_gradio(video_source_path, start_time_str, end_time_str, is_youtube_url, progress=gr.Progress(track_tqdm=True)):
    global GRADIO_API_KEY
    if not GRADIO_API_KEY: return None, None, "Error: GOOGLE_API_KEY not set.", [], None

    files_to_clean = []
    # This will be the path to the video file (either downloaded or local) that segment processing uses
    actual_video_path_for_segment_processing = None

    try:
        start_sec = float(start_time_str) if ':' not in start_time_str else time_str_to_ms(start_time_str) / 1000.0
        end_sec = float(end_time_str) if ':' not in end_time_str else time_str_to_ms(end_time_str) / 1000.0
    except ValueError as e: return None, None, f"Error parsing time inputs: {e}", files_to_clean, None
    if start_sec < 0 or end_sec <= start_sec: return None, None, "Invalid start or end time.", files_to_clean, None

    temp_dir = tempfile.gettempdir()

    if is_youtube_url:
        progress(0.05, desc="Downloading YouTube video for segment...")
        ts = int(time.time())
        # Sanitize URL for filename
        sanitized_url_part = "".join(c if c.isalnum() else '_' for c in video_source_path[-30:])
        download_path_template_name = f"yt_dlp_seg_{sanitized_url_part}_{ts}"

        ydl_opts = {
            'outtmpl': os.path.join(temp_dir, f"{download_path_template_name}.%(ext)s"),
            'format': 'best[ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[height<=720]/best',
            'quiet': True, 'verbose': False, 'noplaylist': True, 'merge_output_format': 'mp4',
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (download setup)")
                info = ydl.extract_info(video_source_path, download=False)
                actual_video_path_for_segment_processing = ydl.prepare_filename(info)
                files_to_clean.append(actual_video_path_for_segment_processing)
                if not os.path.exists(actual_video_path_for_segment_processing) or ydl.params.get('overwrites', True):
                    ydl.download([video_source_path])
            if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (after download)")
        except InterruptedError: return None, None, "Segment processing cancelled.", files_to_clean, None
        except Exception as e_dl: return None, None, f"YouTube download error for segment: {e_dl}", files_to_clean, None
        if not actual_video_path_for_segment_processing or not os.path.exists(actual_video_path_for_segment_processing):
            return None, None, "YouTube video download for segment failed.", files_to_clean, None
    else:
        progress(0.05, desc="Using local video for segment...")
        actual_video_path_for_segment_processing = video_source_path
        # Local files are not added to files_to_clean here by this function; their lifecycle is managed by Gradio's temp upload or they are persistent.

    progress(0.25, desc="Segment video ready. Processing audio...")
    # Ensure unique names for temporary audio files for the segment
    base_vid_name_seg = os.path.splitext(os.path.basename(actual_video_path_for_segment_processing))[0]
    full_audio_path_seg = os.path.join(temp_dir, f"{base_vid_name_seg}_sfull_{int(time.time())}.mp3")
    trimmed_audio_path_seg = os.path.join(temp_dir, f"{base_vid_name_seg}_strim_{int(time.time())}.mp3")
    files_to_clean.extend([full_audio_path_seg, trimmed_audio_path_seg])

    audio_transcript = "Audio processing for segment failed."
    try:
        if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (audio extraction)")
        if extract_full_audio_ffmpeg(actual_video_path_for_segment_processing, full_audio_path_seg):
            progress(0.35, desc="Full audio extracted. Trimming...")
            if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (audio trim)")
            if trim_audio_ffmpeg(full_audio_path_seg, trimmed_audio_path_seg, start_sec, end_sec):
                progress(0.5, desc="Audio trimmed. Transcribing segment...")
                if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (transcription)")
                audio_transcript = transcribe_audio_whisper(trimmed_audio_path_seg, model_name=MODEL_NAME_WHISPER) # Use standard model for segment
            else: audio_transcript = "Audio trimming for segment failed."
        if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (after audio processing)")
    except InterruptedError: return audio_transcript, "Segment processing cancelled.", "Segment processing cancelled.", files_to_clean, actual_video_path_for_segment_processing

    progress(0.65, desc="Segment audio transcribed. Extracting frame text...")
    frame_text_result = "Frame text extraction failed or was cancelled."
    try:
        if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled (frame extraction setup)")
        frame_text_result = extract_text_from_video_frames(
            actual_video_path_for_segment_processing, int(start_sec * 1000), int(end_sec * 1000), GRADIO_API_KEY,
            DEFAULT_FRAME_INTERVAL_MS, USE_OCR_FALLBACK, EXPORT_JSON_FRAME_TEXTS,
            progress_tracker=lambda p, desc: progress(0.65 + p * 0.3, desc=desc)
        )
        if GlobalStopFlag.is_set() or "cancelled" in frame_text_result.lower():
             raise InterruptedError("Frame extraction was cancelled.")
    except InterruptedError:
        return audio_transcript, "Frame extraction cancelled.", "Segment processing cancelled.", files_to_clean, actual_video_path_for_segment_processing
    except Exception as e_frame:
        frame_text_result = f"Frame text extraction error: {e_frame}"
        traceback.print_exc()

    progress(0.95, desc="Segment frame text extracted.")
    final_error_message = None
    if "error" in frame_text_result.lower() or "failed" in frame_text_result.lower():
        final_error_message = f"Frame extraction issue: {frame_text_result}"
    return audio_transcript, frame_text_result, final_error_message, files_to_clean, actual_video_path_for_segment_processing


def get_gemini_chat_response(api_key_param, user_query,
                             segment_audio_context, segment_frames_context,
                             full_video_summary_context):
    if not user_query: return "Please provide a query."
    is_segment_summary_request = (user_query == DEFAULT_SUMMARY_REQUEST_PROMPT)
    context_provided_str = ""
    # Check if full_video_summary_context is valid and not an error message
    has_valid_full_summary = full_video_summary_context and not any(err_key in full_video_summary_context for err_key in ["Cannot summarize", "Could not fetch", "No transcript found", "Audio transcription failed"])

    if has_valid_full_summary and not is_segment_summary_request:
        context_provided_str += (f"---FULL VIDEO OVERVIEW---\n{full_video_summary_context}\n---END OF FULL VIDEO OVERVIEW---\n\n")

    has_valid_segment_audio = segment_audio_context and "failed" not in segment_audio_context.lower()
    has_valid_segment_frames = segment_frames_context and "no text found" not in segment_frames_context.lower() and "failed" not in segment_frames_context.lower()

    if has_valid_segment_audio or has_valid_segment_frames:
        context_provided_str += (
            "---CURRENT VIDEO SEGMENT DETAILS---\n"
            f"AUDIO (segment): {segment_audio_context if has_valid_segment_audio else 'No valid audio for this segment.'}\n"
            f"FRAMES (segment): {segment_frames_context if has_valid_segment_frames else 'No valid frame text for this segment.'}\n"
            "---END OF CURRENT VIDEO SEGMENT DETAILS---\n\n"
        )
    if not context_provided_str:
        context_provided_str = "No specific video context is currently loaded.\n\n"

    # Your specific Gemini prompt from the previous code (ensure it's the one you want)
    prompt_for_gemini = (
        "You are an advanced AI assistant with a vast knowledge base, currently tasked with analyzing a specific video segment. "
        "You have been provided with an audio transcript and text extracted from the video frames of this segment.\n\n"
        "Your goal is to answer the user's query comprehensively. To do this:\n"
        "1. Prioritize information directly found in the 'VIDEO CONTEXT' (audio transcript and frame text) provided below. This context is the most specific to the user's current focus.\n"
        "2. If the user's query can be fully and accurately answered using *only* the VIDEO CONTEXT, do so concisely and clearly, referencing the source if helpful (e.g., 'As mentioned in the audio...' or 'The screen shows...').\n"
        "3. If the VIDEO CONTEXT provides partial information or relevant clues, use that as a starting point and then **enrich your answer with your broader general knowledge** to provide a more complete and insightful response. Clearly distinguish when you are drawing from general knowledge versus the specific video context if it's important for clarity.\n"
        "4. If the VIDEO CONTEXT is irrelevant to the query or doesn't contain the answer, use your general knowledge to answer the query as helpfully as possible.\n"
        "5. If the query is a request for a summary of the video segment (e.g., user typed 'summarize' or used the `DEFAULT_SUMMARY_REQUEST_PROMPT`), then focus *primarily* on summarizing the provided 'CURRENT VIDEO SEGMENT DETAILS' in a detailed and coherent manner. General knowledge should only be used minimally for clarification if needed in a summary.\n"
        "6. If you cannot answer the question using either the video context or your general knowledge, clearly state that.\n\n"
        f"USER QUERY: \"{user_query}\"\n\n{context_provided_str}ASSISTANT RESPONSE:"
    )
    try:
        genai.configure(api_key=api_key_param)
        model = genai.GenerativeModel(MODEL_NAME_GEMINI)
        safety_settings = [{"category": cat, "threshold": "BLOCK_NONE"} for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        request_options = genai.types.RequestOptions(timeout=240)
        ai_response_text = "Could not generate a response."
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = model.generate_content(prompt_for_gemini, safety_settings=safety_settings, request_options=request_options)
                text_content = "".join(part.text for part in response.parts if hasattr(part, "text")).strip() if response.parts else (response.text.strip() if hasattr(response, 'text') else "")
                ai_response_text = text_content if text_content else "AI returned no text content."
                break
            except Exception as e_gemini_call:
                if any(err_msg in str(e_gemini_call).lower() for err_msg in ["429", "resource", "limit", "quota"]):
                    print(f"Gemini rate/quota (chat attempt {attempt+1}): {e_gemini_call}")
                    if attempt < RETRY_ATTEMPTS - 1: time.sleep(RATE_LIMIT_WAIT_TIME)
                    else: ai_response_text = f"AI failed due to API limits/quota: {e_gemini_call}"; break
                else: ai_response_text = f"Error during AI call: {e_gemini_call}"; break
        return ai_response_text
    except Exception as e_setup: return f"Critical error setting up AI model: {e_setup}"


# --- Gradio UI Event Handlers ---
def handle_full_video_overview_click(video_url, local_video_file_obj, current_chat_state_dict, progress=gr.Progress(track_tqdm=True)):
    global GRADIO_API_KEY
    GlobalStopFlag.clear()
    new_state_dict = current_chat_state_dict.copy()
    # Clear previous full summary, its source info, and related errors
    new_state_dict.update({
        'full_video_summary': None, 'error': None,
        'active_overview_source_path': None, 'is_overview_from_local': False,
        'files_to_clean_overview': new_state_dict.get('files_to_clean_overview', []) # Preserve if any from previous cancelled op
    })

    video_source_for_overview = None
    is_youtube_for_overview = False
    overview_video_filename_for_title = "this video"
    temp_audio_for_overview_path = None # Path for audio extracted for overview

    if local_video_file_obj:
        video_source_for_overview = local_video_file_obj.name
        overview_video_filename_for_title = os.path.basename(local_video_file_obj.name)
        new_state_dict['is_overview_from_local'] = True
    elif video_url:
        video_source_for_overview = video_url
        is_youtube_for_overview = True
        overview_video_filename_for_title = video_url
    else:
        new_state_dict['error'] = "Please provide a video URL or upload a local file for overview."
        return "Please provide a video URL or upload a local file.", new_state_dict, "Input missing."

    new_state_dict['active_overview_source_path'] = video_source_for_overview
    progress(0, desc="Preparing full video transcript for overview...")
    transcript_text = "Transcript generation for overview failed."

    try:
        if is_youtube_for_overview:
            progress(0.1, desc="Downloading full audio from YouTube for overview...")
            temp_dir_overview = tempfile.gettempdir()
            ts_overview = int(time.time())
            sanitized_url_part_overview = "".join(c if c.isalnum() else '_' for c in video_source_for_overview[-30:])
            # Unique name for full YouTube audio download
            download_path_template_overview = f"yt_overview_{sanitized_url_part_overview}_{ts_overview}"

            # Path for the downloaded full audio file from YouTube
            temp_audio_for_overview_path = os.path.join(temp_dir_overview, f"{download_path_template_overview}_audio.mp3")

            ydl_opts_overview = {
                'outtmpl': os.path.join(temp_dir_overview, f"{download_path_template_overview}.%(ext)s"), # Temp video
                'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]/best', # Smaller video for faster audio extract
                'quiet': True, 'verbose': False, 'noplaylist': True, 'merge_output_format': 'mp4',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192', # Standard quality
                    'nopostoverwrites': False, # Allow overwrite if temp_audio_for_overview_path exists
                }],
                # Set the audio output path directly for yt-dlp if possible or handle renaming
                # For simplicity, we'll extract audio from the downloaded temp video using our ffmpeg helper
            }
            # yt-dlp will create a video file first based on outtmpl, then extract audio from it.
            # We need the path of the *audio* file.
            # A simpler way is to download the video, then use our `extract_full_audio_ffmpeg`.

            downloaded_temp_video_for_audio = None
            with yt_dlp.YoutubeDL(ydl_opts_overview) as ydl:
                if GlobalStopFlag.is_set(): raise InterruptedError("Cancelled")
                info = ydl.extract_info(video_source_for_overview, download=False)
                # We only need audio, but yt-dlp might download video first then extract.
                # Let's try to get audio directly if possible or simplify.
                # Simpler: download best audio.
                ydl_audio_opts = {
                     'outtmpl': temp_audio_for_overview_path.replace(".mp3", ".%(ext)s"), # yt-dlp adds ext
                     'format': 'bestaudio/best',
                     'quiet': True, 'verbose': False, 'noplaylist': True,
                     'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                }
                with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl_audio:
                    ydl_audio.download([video_source_for_overview])
                # Ensure the file is named exactly temp_audio_for_overview_path
                # yt-dlp might add original extension then convert. Find the .mp3
                base_temp_audio_path = temp_audio_for_overview_path.replace(".mp3", "")
                found_mp3 = False
                for f in os.listdir(temp_dir_overview):
                    if f.startswith(os.path.basename(base_temp_audio_path)) and f.endswith(".mp3"):
                        shutil.move(os.path.join(temp_dir_overview, f), temp_audio_for_overview_path)
                        found_mp3 = True
                        break
                if not found_mp3 or not os.path.exists(temp_audio_for_overview_path):
                    raise Exception("Failed to produce final MP3 audio file from YouTube download.")

            new_state_dict.get('files_to_clean_overview', []).append(temp_audio_for_overview_path)
            progress(0.4, desc="Transcribing full YouTube audio (Whisper)...")
            transcript_text = transcribe_audio_whisper(temp_audio_for_overview_path, model_name=MODEL_NAME_WHISPER_FULL_VIDEO)
        else: # Local file
            progress(0.1, desc="Extracting full audio from local video for overview...")
            temp_audio_for_overview_path = os.path.join(tempfile.gettempdir(), f"local_overview_audio_{int(time.time())}.mp3")
            new_state_dict.get('files_to_clean_overview', []).append(temp_audio_for_overview_path)
            if extract_full_audio_ffmpeg(video_source_for_overview, temp_audio_for_overview_path):
                progress(0.4, desc="Transcribing full local audio (Whisper)...")
                transcript_text = transcribe_audio_whisper(temp_audio_for_overview_path, model_name=MODEL_NAME_WHISPER_FULL_VIDEO)
            else:
                transcript_text = "Full audio extraction failed for local video overview."
    except InterruptedError:
        new_state_dict['error'] = "Full overview cancelled (transcript generation)."
        return "Full overview cancelled.", new_state_dict, "Cancelled."
    except Exception as e_tr:
        transcript_text = f"Error during transcript generation for overview: {e_tr}"
        traceback.print_exc()
        new_state_dict['error'] = transcript_text

    if GlobalStopFlag.is_set(): # Check after potentially long transcription
        new_state_dict['error'] = "Full overview cancelled (after transcript)."
        # Cleanup temp audio if it was created
        if temp_audio_for_overview_path and os.path.exists(temp_audio_for_overview_path) and temp_audio_for_overview_path in new_state_dict.get('files_to_clean_overview',[]):
            try: os.remove(temp_audio_for_overview_path)
            except Exception as e_del: print(f"Could not clean temp audio {temp_audio_for_overview_path}: {e_del}")
        return "Full overview cancelled.", new_state_dict, "Cancelled."

    status_update = "Transcript ready for overview. Summarizing..."
    if "failed" in transcript_text.lower(): # Check for failure string from Whisper
        status_update = transcript_text
        new_state_dict.update({'full_video_summary': transcript_text, 'error': transcript_text})
        progress(1, desc=status_update)
        # Cleanup temp audio
        if temp_audio_for_overview_path and os.path.exists(temp_audio_for_overview_path) and temp_audio_for_overview_path in new_state_dict.get('files_to_clean_overview',[]):
            try: os.remove(temp_audio_for_overview_path)
            except Exception as e_del: print(f"Could not clean temp audio {temp_audio_for_overview_path}: {e_del}")
        return transcript_text, new_state_dict, status_update

    progress(0.7, desc=status_update)
    summary = summarize_full_text_with_gemini(GRADIO_API_KEY, transcript_text, overview_video_filename_for_title)
    if GlobalStopFlag.is_set():
        new_state_dict['error'] = "Full overview cancelled (summary generation)."
        summary = "Full overview cancelled." # Ensure summary reflects cancellation

    progress(1, desc="Full video summary generated.")
    new_state_dict['full_video_summary'] = summary
    if "Error" in summary or "Failed" in summary or "cancelled" in summary.lower():
        new_state_dict['error'] = summary

    # Cleanup the full audio file used for overview summary
    if temp_audio_for_overview_path and os.path.exists(temp_audio_for_overview_path) and temp_audio_for_overview_path in new_state_dict.get('files_to_clean_overview',[]):
        try:
            os.remove(temp_audio_for_overview_path)
            print(f"Cleaned temp audio for overview: {temp_audio_for_overview_path}")
            new_state_dict['files_to_clean_overview'].remove(temp_audio_for_overview_path)
        except Exception as e_del:
            print(f"Could not clean temp audio {temp_audio_for_overview_path}: {e_del}")

    return summary, new_state_dict, "Full video overview generated."


def handle_process_segment_button_click(video_url, local_video_file_obj, start_time, end_time, current_chat_state_dict):
    GlobalStopFlag.clear()
    initial_chat_history = [(None, "Processing video segment... This may take a few minutes.")]

    new_state_for_segment_processing = current_chat_state_dict.copy()
    # Clear previous segment data, keep full_video_summary and its source info
    new_state_for_segment_processing.update({
        'audio': None, 'frames': None, 'error': None,
        'files_to_clean_segment': [], # Specific to this segment op
        'active_segment_source_path': None, 'is_segment_from_local': False
    })

    video_source_for_segment = None
    is_youtube_for_segment = False

    if local_video_file_obj:
        video_source_for_segment = local_video_file_obj.name # This is a temporary path from Gradio
        new_state_for_segment_processing['is_segment_from_local'] = True
    elif video_url:
        video_source_for_segment = video_url
        is_youtube_for_segment = True
    else:
        yield initial_chat_history, {**new_state_for_segment_processing, 'error': "No video source (URL or local file) for segment processing."}, "Input missing for segment."
        return

    new_state_for_segment_processing['active_segment_source_path'] = video_source_for_segment
    yield initial_chat_history, new_state_for_segment_processing, "Processing segment..."

    # process_video_for_gradio returns: audio, frames, error, files_cleaned_by_it, actual_video_path_used
    audio_data, frames_data, error_from_processing, files_created_by_segment_proc, actual_vid_path_seg_used = process_video_for_gradio(
        video_source_for_segment, start_time, end_time, is_youtube_for_segment
    )

    # Start with the state we had during processing, then update with results
    final_state_dict = new_state_for_segment_processing.copy()
    final_state_dict.update({
        'audio': audio_data, 'frames': frames_data, 'error': error_from_processing,
        # files_created_by_segment_proc includes the downloaded YouTube video (if applicable) and its derived audio files
        'files_to_clean_segment': files_created_by_segment_proc
    })
    # Record the actual path used by segment processing, could be a downloaded temp file or original local path
    final_state_dict['active_segment_source_path'] = actual_vid_path_seg_used

    current_chat_history_display = initial_chat_history[:]
    status_msg = "Segment processing finished."

    if GlobalStopFlag.is_set():
        final_state_dict['error'] = "Segment processing cancelled by user."
        current_chat_history_display.append((None, "Segment processing was cancelled."))
        status_msg = "Segment processing cancelled."
    elif error_from_processing:
        current_chat_history_display.append((None, f"Error (segment): {error_from_processing}"))
        status_msg = f"Error (segment): {error_from_processing[:100]}..."
    elif audio_data and frames_data and "failed" not in str(audio_data).lower() and "failed" not in str(frames_data).lower():
        current_chat_history_display.append((None, "Video segment processed! Ask about this segment or request its summary."))
        status_msg = "Segment processed. Ready for segment queries."
    else:
        final_state_dict['error'] = "Segment processing issues or incomplete data."
        issues = []
        if "failed" in str(audio_data).lower(): issues.append(f"Audio: {audio_data}")
        if "failed" in str(frames_data).lower() or "cancelled" in str(frames_data).lower(): issues.append(f"Frames: {frames_data}")
        current_chat_history_display.append((None, f"Segment processing issues: {'; '.join(issues) if issues else 'Unknown.'}"))
        status_msg = "Segment processing issues."

    if current_chat_history_display and current_chat_history_display[0] == initial_chat_history[0]:
        current_chat_history_display.pop(0)

    # Cleanup files created *during this specific segment processing run*
    if final_state_dict.get('files_to_clean_segment'):
        print("\n--- Cleaning up temporary files (segment processing run) ---")
        for f_path in final_state_dict['files_to_clean_segment']:
            if f_path and os.path.exists(f_path):
                try: os.remove(f_path); print(f"Cleaned: {f_path}")
                except Exception as e_del: print(f"Warn: Failed to clean {f_path}: {e_del}")
        final_state_dict['files_to_clean_segment'] = [] # Clear after attempt

    yield current_chat_history_display, final_state_dict, status_msg


def handle_user_message_submit(user_input_prompt, chat_history, current_chat_state_dict):
    global GRADIO_API_KEY
    if not GRADIO_API_KEY:
        chat_history.append((user_input_prompt, "Error: API Key is not configured."))
        return chat_history, current_chat_state_dict, ""

    segment_audio = current_chat_state_dict.get('audio')
    segment_frames = current_chat_state_dict.get('frames')
    full_summary = current_chat_state_dict.get('full_video_summary')
    last_error = current_chat_state_dict.get('error')

    # Check for valid data in contexts
    has_valid_segment_audio = segment_audio and "failed" not in str(segment_audio).lower()
    has_valid_segment_frames = segment_frames and "failed" not in str(segment_frames).lower() and "no text found" not in str(segment_frames).lower()
    has_segment_data = has_valid_segment_audio and has_valid_segment_frames

    has_valid_full_summary = full_summary and not any(err_key in full_summary for err_key in ["Cannot summarize", "Could not fetch", "No transcript found", "Audio transcription failed"])

    if not has_segment_data and not has_valid_full_summary:
        err_msg = f"No video context available. Last status: {last_error}" if last_error else "No video content processed."
        chat_history.append((user_input_prompt, err_msg))
        return chat_history, current_chat_state_dict, ""

    actual_query = user_input_prompt.strip()
    display_query_in_chat = actual_query

    is_first_query_after_segment_success = False
    if chat_history and has_segment_data: # Only relevant if segment data is present and valid
        last_bot_msg = chat_history[-1][1] if len(chat_history) > 0 and chat_history[-1][0] is None else None
        if last_bot_msg and "Video segment processed!" in last_bot_msg:
            is_first_query_after_segment_success = True

    if not actual_query and has_segment_data and is_first_query_after_segment_success:
        actual_query = DEFAULT_SUMMARY_REQUEST_PROMPT
        display_query_in_chat = "(Segment summary by empty prompt)"
    elif not actual_query:
         chat_history.append(("", "Please type a question or prompt."))
         return chat_history, current_chat_state_dict, ""

    chat_history.append((display_query_in_chat, "...Thinking..."))
    yield chat_history, current_chat_state_dict, ""

    assistant_response = get_gemini_chat_response(GRADIO_API_KEY, actual_query, segment_audio, segment_frames, full_summary)
    chat_history[-1] = (display_query_in_chat, assistant_response)
    yield chat_history, current_chat_state_dict, ""


def handle_stop_button_click():
    GlobalStopFlag.set()
    print("Stop button clicked, GlobalStopFlag set.")
    return "Stop signal sent. Processes will attempt to terminate..."


# --- Launch Gradio App ---
def launch_gradio_interface():
    global GRADIO_API_KEY
    if not GRADIO_API_KEY: print("CRITICAL: GOOGLE_API_KEY not set.")

    with gr.Blocks(theme=gr.themes.Glass(), title="Video Analysis Assistant") as demo:
        gr.Markdown("## Video Analysis Assistant")
        gr.Markdown(
            "Upload a local video OR enter a YouTube URL.\n"
            "1. Click 'Get Full Video Overview' for an entire video summary (uses full audio transcription).\n"
            "2. Optionally, specify segment times & click 'Process Video Segment' for detailed audio/frame analysis.\n"
            "3. Ask questions in chat. The AI uses available context & general knowledge."
        )
        # Expanded chat_state
        chat_state = gr.State({
            'audio': None, 'frames': None, # For segment
            'active_segment_source_path': None, 'is_segment_from_local': False,
            'full_video_summary': None, # For overview
            'active_overview_source_path': None, 'is_overview_from_local': False,
            'error': None,
            'files_to_clean_overview': [], # Files from overview op (e.g. full audio from YT)
            'files_to_clean_segment': []  # Files from segment op (e.g. downloaded YT segment, derived audios)
        })

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Video Source & Processing")
                video_url_in = gr.Textbox(label="YouTube Video URL (Option 1)", placeholder="https://www.youtube.com/watch?v=...")
                local_video_in = gr.File(label="Upload Local Video (Option 2)", file_types=['video', '.mp4', '.mkv', '.avi', '.mov', '.webm'])

                with gr.Accordion("Full Video Overview (transcribes full audio)", open=False):
                    full_overview_btn = gr.Button("Get Full Video Overview", variant="secondary")
                    full_overview_out = gr.Textbox(label="Full Video Overview Output", interactive=False, lines=7, placeholder="Summary of entire video...")

                with gr.Accordion("Specific Segment Processing (detailed analysis)", open=True):
                    start_time_in = gr.Textbox(label="Segment Start Time", placeholder="e.g., 30 or 0:30")
                    end_time_in = gr.Textbox(label="Segment End Time", placeholder="e.g., 90 or 1:30")
                    process_segment_btn = gr.Button("Process Video Segment", variant="primary")

                stop_btn = gr.Button("Stop Current Processing", variant="stop")
                status_out = gr.Textbox(label="System Status", interactive=False, placeholder="Status messages...")

            with gr.Column(scale=2):
                gr.Markdown("### Chat with Assistant")
                chatbot_out = gr.Chatbot(label="Conversation", height=650, bubble_full_width=False, show_copy_button=True)
                user_prompt_in = gr.Textbox(label="Your Question / Prompt", placeholder="Ask about the video...", show_label=True)
                send_btn = gr.Button("Send")

        # Event Handlers
        full_overview_event = full_overview_btn.click(
            fn=handle_full_video_overview_click,
            inputs=[video_url_in, local_video_in, chat_state],
            outputs=[full_overview_out, chat_state, status_out],
            show_progress="full"
        )
        segment_process_event = process_segment_btn.click(
            fn=handle_process_segment_button_click,
            inputs=[video_url_in, local_video_in, start_time_in, end_time_in, chat_state],
            outputs=[chatbot_out, chat_state, status_out],
            show_progress="full"
        )
        stop_btn.click(fn=handle_stop_button_click, inputs=None, outputs=[status_out], cancels=[full_overview_event, segment_process_event])

        user_prompt_in.submit(fn=handle_user_message_submit, inputs=[user_prompt_in, chatbot_out, chat_state], outputs=[chatbot_out, chat_state, user_prompt_in])
        send_btn.click(fn=handle_user_message_submit, inputs=[user_prompt_in, chatbot_out, chat_state], outputs=[chatbot_out, chat_state, user_prompt_in])

    demo.queue().launch(debug=True, share=True)

if __name__ == "__main__" and 'google.colab' in sys.modules:
    GRADIO_API_KEY = os.environ.get('GOOGLE_API_KEY', None)
    if not GRADIO_API_KEY: print("WARNING: GOOGLE_API_KEY is not set.")
    print("Launching Gradio Interface...")
    launch_gradio_interface()
