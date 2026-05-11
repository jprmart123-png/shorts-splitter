import os
import math
import tempfile
import shutil
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
from datetime import datetime
import json
import cv2
import numpy as np

# --- Configuration ---
SUPPORTED_VIDEO_FORMATS = [".mkv", ".mp4", ".mov", ".avi", ".flv", ".wmv", ".webm"]

# YouTube Shorts specifications
SHORTS_ASPECT_RATIO = "9:16"
SHORTS_RESOLUTION = (1080, 1920)  # width x height

# --- Helpers ---

def detect_faces_in_frame(frame):
    """Detect faces in a frame using OpenCV Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    except Exception as e:
        return []

def get_smart_crop_position(video_path: str, sample_seconds: int = 10) -> str:
    """
    Analyze video to detect faces and determine best crop position
    Returns: 'center', 'left', 'right' based on face detection
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "center"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        sample_frames = min(sample_seconds * int(fps), total_frames)
        frame_interval = max(total_frames // max(sample_frames, 1), 1)

        face_positions = []

        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces_in_frame(frame)

            for (x, y, w, h) in faces:
                face_center_x = x + w // 2
                face_positions.append(face_center_x)

        cap.release()

        if not face_positions:
            return "center"

        avg_face_x = sum(face_positions) / len(face_positions)

        if avg_face_x < frame_width * 0.33:
            return "left"
        elif avg_face_x > frame_width * 0.67:
            return "right"
        else:
            return "center"

    except Exception as e:
        return "center"

def ffprobe_duration(input_path: str) -> float:
    """Get video duration using ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception as e:
        st.error(f"Error reading video duration: {e}")
        return 0

def ffprobe_video_info(input_path: str) -> dict:
    """Get detailed video information"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path
    ]
    try:
        out = subprocess.check_output(cmd).decode()
        data = json.loads(out)

        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), {})
        audio_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'audio'), {})

        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))

        return {
            'duration': float(data.get('format', {}).get('duration', 0)),
            'size': int(data.get('format', {}).get('size', 0)),
            'bitrate': int(data.get('format', {}).get('bit_rate', 0)),
            'video_codec': video_stream.get('codec_name', 'N/A'),
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'fps': eval(video_stream.get('r_frame_rate', '0/1')) if '/' in str(video_stream.get('r_frame_rate', '0/1')) else 0,
            'audio_codec': audio_stream.get('codec_name', 'N/A'),
            'audio_sample_rate': audio_stream.get('sample_rate', 'N/A')
        }
    except Exception as e:
        return {}

def calculate_crop_for_shorts(source_width: int, source_height: int, crop_position: str = "center") -> dict:
    """Calculate crop parameters to convert video to 9:16 (YouTube Shorts format)"""
    target_ratio = 9 / 16
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = int(crop_height * target_ratio)

        if crop_position == "left":
            origin_x = 0
        elif crop_position == "right":
            origin_x = source_width - crop_width
        else:
            origin_x = (source_width - crop_width) // 2

        origin_y = 0

    else:
        crop_width = source_width
        crop_height = int(crop_width / target_ratio)

        if crop_position == "top":
            origin_y = 0
        elif crop_position == "bottom":
            origin_y = source_height - crop_height
        else:
            origin_y = (source_height - crop_height) // 2

        origin_x = 0

    return {
        'crop_width': crop_width,
        'crop_height': crop_height,
        'origin_x': origin_x,
        'origin_y': origin_y
    }

def ensure_dir(p: str):
    """Create directory if it doesn't exist"""
    Path(p).mkdir(parents=True, exist_ok=True)

def get_ext(path: str) -> str:
    """Get file extension"""
    return Path(path).suffix.lower()

def build_output_name(base_dir: str, base_stem: str, idx: int, out_ext: str, custom_prefix: str = "") -> str:
    """Build output filename with optional custom prefix"""
    prefix = f"{custom_prefix}_" if custom_prefix else ""
    return str(Path(base_dir) / f"{prefix}{base_stem}_short_{idx:03d}{out_ext}")

def ffmpeg_segment_crop_shorts(input_path: str, start: float, dur: float, output_path: str,
                                crop_params: dict, scale_to_1080p: bool = True,
                                quality_preset: str = "medium") -> None:
    """Extract video segment and convert to YouTube Shorts format (9:16)"""
    crop_w = crop_params['crop_width']
    crop_h = crop_params['crop_height']
    origin_x = crop_params['origin_x']
    origin_y = crop_params['origin_y']

    filters = f"crop={crop_w}:{crop_h}:{origin_x}:{origin_y}"

    if scale_to_1080p:
        filters += f",scale={SHORTS_RESOLUTION[0]}:{SHORTS_RESOLUTION[1]}"

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}",
        "-i", input_path,
        "-t", f"{dur}",
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", quality_preset,
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "1",
        output_path,
        "-y"
    ]
    subprocess.check_call(cmd)

def ffmpeg_segment_copy(input_path: str, start: float, dur: float, output_path: str) -> None:
    """Extract video segment using stream copy (fast, no re-encoding)"""
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}",
        "-i", input_path,
        "-t", f"{dur}",
        "-c", "copy",
        "-map", "0",
        "-avoid_negative_ts", "1",
        output_path,
        "-y"
    ]
    subprocess.check_call(cmd)

def generate_preview(input_path: str, seconds: int = 10, convert_to_shorts: bool = False,
                     crop_params: dict = None) -> bytes:
    """Generate preview of first N seconds"""
    with tempfile.TemporaryDirectory() as tmp:
        preview = Path(tmp) / "preview.mp4"

        if convert_to_shorts and crop_params:
            crop_w = crop_params['crop_width']
            crop_h = crop_params['crop_height']
            origin_x = crop_params['origin_x']
            origin_y = crop_params['origin_y']

            filters = f"crop={crop_w}:{crop_h}:{origin_x}:{origin_y},scale={SHORTS_RESOLUTION[0]}:{SHORTS_RESOLUTION[1]}"

            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-ss", "0",
                "-i", input_path,
                "-t", str(seconds),
                "-vf", filters,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                str(preview),
                "-y"
            ]
        else:
            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-ss", "0",
                "-i", input_path,
                "-t", str(seconds),
                "-c", "copy",
                "-map", "0",
                str(preview),
                "-y"
            ]

        subprocess.check_call(cmd)
        return preview.read_bytes()

def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def format_size(bytes_val: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def check_ffmpeg_installed() -> bool:
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def check_opencv_installed() -> bool:
    """Check if OpenCV is installed"""
    try:
        import cv2
        return True
    except:
        return False

# --- Main App ---

st.set_page_config(
    page_title="YouTube Shorts Splitter",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 YouTube Shorts Video Splitter")
st.markdown("**Split long videos into YouTube Shorts (9:16 vertical format) with Smart Auto-Focus**")

# Check FFmpeg
if not check_ffmpeg_installed():
    st.error("❌ **FFmpeg is not installed!** Please install FFmpeg to use this app.")
    st.stop()

# Check OpenCV (optional)
opencv_available = check_opencv_installed()
if not opencv_available:
    st.warning("⚠️ OpenCV not installed. Smart Auto-Focus will be disabled.")

# Initialize session state
if 'detected_crop_position' not in st.session_state:
    st.session_state.detected_crop_position = None
if 'saved_video_path' not in st.session_state:
    st.session_state.saved_video_path = None

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("🎬 Output Format")
    convert_to_shorts = st.checkbox(
        "Convert to YouTube Shorts format (9:16)",
        value=True,
        help="Converts video to vertical 9:16 format (1080x1920)"
    )

    if convert_to_shorts:
        st.info("📱 Videos will be converted to 1080x1920 (9:16 aspect ratio)")

        use_auto_focus = st.checkbox(
            "🎯 Smart Auto-Focus (Face Detection)",
            value=True,
            disabled=not opencv_available,
            help="Automatically detect faces and adjust crop position"
        )

        if use_auto_focus and opencv_available:
            st.success("✅ Smart cropping enabled - will focus on detected faces")
            crop_position = "auto"
        else:
            crop_position = st.selectbox(
                "Crop Position",
                options=["center", "left", "right", "top", "bottom"],
                index=0,
                help="Where to position the crop"
            )
    else:
        crop_position = "center"
        use_auto_focus = False
        st.warning("⚠️ Videos will keep original aspect ratio")

    st.divider()

    st.subheader("⏱️ Clip Duration")

    duration_mode = st.radio(
        "Duration Input",
        options=["Preset (Quick)", "Custom (Any Duration)"],
        index=0
    )

    if duration_mode == "Preset (Quick)":
        st.caption("📌 Popular durations for YouTube Shorts")
        preset_seconds = st.selectbox(
            "Select duration",
            options=[15, 30, 45, 60, 90, 120, 180, 300],
            index=3,
            format_func=lambda x: f"{x} seconds ({x//60}:{x%60:02d})" if x < 60 else f"{x} seconds ({x//60} min {x%60} sec)",
            help="Quick presets"
        )
        clip_len_seconds = preset_seconds
    else:
        st.caption("⏱️ Set any custom duration")

        col1, col2 = st.columns(2)
        with col1:
            custom_minutes = st.number_input("Minutes", min_value=0, max_value=999, value=1, step=1)
        with col2:
            custom_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=5)

        clip_len_seconds = custom_minutes * 60 + custom_seconds

        if clip_len_seconds == 0:
            st.error("❌ Duration cannot be 0!")
            clip_len_seconds = 60

        st.info(f"⏱️ Clip duration: **{format_duration(clip_len_seconds)}**")

    if clip_len_seconds > 60:
        st.warning(f"⚠️ {clip_len_seconds}s clips won't be eligible for YouTube Shorts feed (max 60s)")
    else:
        st.success(f"✅ {clip_len_seconds}s clips are perfect for YouTube Shorts!")

    st.divider()

    st.subheader("📁 Output Options")
    create_subfolder = st.checkbox("Create timestamped subfolder", value=True)
    custom_prefix = st.text_input("Custom filename prefix (optional)", "")

    st.divider()

    with st.expander("🔧 Advanced Settings"):
        preview_duration = st.slider("Preview duration (seconds)", 5, 60, 10)
        show_detailed_info = st.checkbox("Show detailed video info", value=True)
        quality_preset = st.selectbox(
            "Encoding Quality",
            options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
            index=5,
            help="Medium = balanced quality/speed"
        )

# ─── Main Layout ───────────────────────────────────────────────

st.header("📁 Upload Video")

# ── Cloud-friendly: st.file_uploader instead of tkinter dialogs ──
uploaded_file = st.file_uploader(
    "🎥 Upload your video file",
    type=["mp4", "mkv", "avi", "mov", "flv", "wmv", "webm"],
    help="Upload the long video you want to split into Shorts"
)

# Save uploaded file to a temp location so ffmpeg can read it
if uploaded_file is not None:
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.saved_video_path = video_path
    st.session_state.detected_crop_position = None  # reset on new upload
    st.success(f"✅ Uploaded: **{uploaded_file.name}**")
else:
    video_path = st.session_state.saved_video_path

st.divider()

# Output folder (cloud: use temp dir)
output_base = tempfile.mkdtemp()
if create_subfolder:
    date_folder = f"Shorts_{datetime.now().strftime('%Y%m%d_%H%M')}"
    final_dst_dir = os.path.join(output_base, date_folder)
else:
    final_dst_dir = output_base

st.info(f"💾 Clips will be available for download after processing")

# Video info display
if video_path and Path(video_path).exists():
    try:
        info = ffprobe_video_info(video_path)
        file_size = Path(video_path).stat().st_size

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 File Size", format_size(file_size))
        with col2:
            st.metric("📹 Resolution", info.get('resolution', 'N/A'))
        with col3:
            current_ratio = f"{info.get('width', 0)}:{info.get('height', 0)}"
            st.metric("📐 Current Ratio", current_ratio)
        with col4:
            if convert_to_shorts:
                st.metric("🎯 Target", "9:16 Shorts")
            else:
                st.metric("📝 Mode", "Original")

        # Smart crop detection
        if convert_to_shorts and use_auto_focus and opencv_available and st.session_state.detected_crop_position is None:
            with st.spinner("🎯 Detecting faces for smart cropping..."):
                detected_pos = get_smart_crop_position(video_path, sample_seconds=5)
                st.session_state.detected_crop_position = detected_pos
                st.success(f"✅ Smart crop position detected: **{detected_pos.upper()}**")
        elif st.session_state.detected_crop_position:
            st.info(f"🎯 Smart crop position: **{st.session_state.detected_crop_position.upper()}**")

    except:
        pass

st.divider()

# Action Buttons
st.header("🚀 Actions")

col1, col2, col3 = st.columns(3)

with col1:
    preview_btn = st.button(
        "🎥 Preview",
        use_container_width=True,
        disabled=not video_path,
        help="Preview how Shorts will look"
    )

with col2:
    analyze_btn = st.button(
        "📊 Analyze Video",
        use_container_width=True,
        disabled=not video_path,
        help="Show detailed video information"
    )

with col3:
    generate_btn = st.button(
        "⚡ Generate Shorts",
        type="primary",
        use_container_width=True,
        disabled=not video_path,
        help="Start creating YouTube Shorts"
    )

st.divider()

# Preview Section
if preview_btn and video_path:
    try:
        st.subheader("🎬 Video Preview")

        if convert_to_shorts:
            info = ffprobe_video_info(video_path)

            if use_auto_focus and st.session_state.detected_crop_position:
                preview_crop_pos = st.session_state.detected_crop_position
            else:
                preview_crop_pos = crop_position

            crop_params = calculate_crop_for_shorts(
                info['width'],
                info['height'],
                preview_crop_pos
            )
            st.info(f"📱 Preview in YouTube Shorts format (9:16) - Crop position: **{preview_crop_pos.upper()}**")
        else:
            crop_params = None
            st.info("📺 Preview in original format")

        with st.spinner(f"Generating {preview_duration}s preview..."):
            pv = generate_preview(video_path, seconds=preview_duration,
                                  convert_to_shorts=convert_to_shorts,
                                  crop_params=crop_params)
            st.video(pv)
            st.success(f"Preview ready!")
    except Exception as e:
        st.error(f"❌ Preview failed: {e}")

# Analysis Section
if analyze_btn and video_path:
    try:
        st.subheader("📊 Video Analysis")

        with st.spinner("Analyzing video..."):
            info = ffprobe_video_info(video_path)

        if info:
            duration_sec = info['duration']
            file_size = info['size']
            total_clips = math.ceil(duration_sec / clip_len_seconds)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("⏱️ Duration", format_duration(duration_sec))
            with metric_col2:
                st.metric("💾 File Size", format_size(file_size))
            with metric_col3:
                st.metric("🎞️ Total Clips", total_clips)
            with metric_col4:
                st.metric("📏 Clip Length", format_duration(clip_len_seconds))

            if convert_to_shorts:
                if use_auto_focus and st.session_state.detected_crop_position:
                    used_pos = st.session_state.detected_crop_position
                else:
                    used_pos = crop_position

                crop_params = calculate_crop_for_shorts(info['width'], info['height'], used_pos)

            if show_detailed_info:
                with st.expander("🔍 Detailed Information", expanded=True):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.write("**Video Details:**")
                        st.write(f"- Codec: {info.get('video_codec', 'N/A')}")
                        st.write(f"- Resolution: {info.get('resolution', 'N/A')}")
                        st.write(f"- FPS: {info.get('fps', 'N/A'):.2f}")
                        st.write(f"- Bitrate: {info.get('bitrate', 0) // 1000} kbps")

                        if convert_to_shorts:
                            st.write(f"\n**Crop Settings:**")
                            st.write(f"- Mode: {'Smart Auto-Focus' if use_auto_focus else 'Manual'}")
                            st.write(f"- Position: {used_pos.upper()}")
                            st.write(f"- Crop size: {crop_params['crop_width']}x{crop_params['crop_height']}")
                            st.write(f"- Output: {SHORTS_RESOLUTION[0]}x{SHORTS_RESOLUTION[1]}")

                    with detail_col2:
                        st.write("**Audio Details:**")
                        st.write(f"- Codec: {info.get('audio_codec', 'N/A')}")
                        st.write(f"- Sample Rate: {info.get('audio_sample_rate', 'N/A')}")

                        st.write(f"\n**YouTube Shorts Info:**")
                        st.write(f"- ✅ Aspect Ratio: 9:16")
                        st.write(f"- ✅ Resolution: 1080x1920")
                        st.write(f"- Recommended: Under 60 seconds")
                        st.write(f"- Your duration: {format_duration(clip_len_seconds)}")

                        if clip_len_seconds > 60:
                            st.write(f"- ⚠️ Over 60s limit")
                        else:
                            st.write(f"- ✅ Perfect for Shorts!")

            if clip_len_seconds <= 60:
                st.success(f"✅ Will create **{total_clips} YouTube Shorts** of **{format_duration(clip_len_seconds)}** each")
            else:
                st.info(f"📹 Will create **{total_clips} clips** of **{format_duration(clip_len_seconds)}** each")

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")

# Generate Clips Section
if generate_btn:
    try:
        if not video_path:
            st.error("❌ Please upload a video file first")
            st.stop()

        ensure_dir(final_dst_dir)

        st.subheader("🔄 Processing Video")

        with st.spinner("Reading video information..."):
            info = ffprobe_video_info(video_path)
            duration_sec = info['duration']
            total_clips = math.ceil(duration_sec / clip_len_seconds)

            if convert_to_shorts:
                if use_auto_focus and st.session_state.detected_crop_position:
                    final_crop_pos = st.session_state.detected_crop_position
                else:
                    final_crop_pos = crop_position

                crop_params = calculate_crop_for_shorts(
                    info['width'],
                    info['height'],
                    final_crop_pos
                )

        if total_clips == 0:
            st.error("❌ Video is too short for the selected clip length")
            st.stop()

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write("**📥 Source:**")
            st.write(f"- File: {Path(video_path).name}")
            st.write(f"- Duration: {format_duration(duration_sec)}")
            st.write(f"- Resolution: {info.get('resolution', 'N/A')}")

        with summary_col2:
            st.write("**📤 Output:**")
            st.write(f"- Total clips: {total_clips}")
            st.write(f"- Clip length: {format_duration(clip_len_seconds)}")
            if convert_to_shorts:
                st.write(f"- Format: 9:16 (1080x1920)")
                st.write(f"- Crop: {final_crop_pos.upper()}")
                if use_auto_focus:
                    st.write(f"- Mode: Smart Auto-Focus ✨")

        st.divider()

        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()

        start_time = datetime.now()
        outputs = []
        base_stem = Path(video_path).stem
        out_ext = ".mp4"

        for i in range(total_clips):
            start = i * clip_len_seconds
            dur = min(clip_len_seconds, duration_sec - start)

            if dur <= 0:
                break

            out_path = build_output_name(final_dst_dir, base_stem, i + 1, out_ext, custom_prefix)

            elapsed = (datetime.now() - start_time).total_seconds()
            if i > 0:
                eta = (elapsed / i) * (total_clips - i)
                time_text.text(f"⏱️ Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}")

            status_text.text(f"🔄 Creating clip {i+1}/{total_clips}...")

            if convert_to_shorts:
                ffmpeg_segment_crop_shorts(
                    video_path, start, dur, out_path,
                    crop_params, scale_to_1080p=True,
                    quality_preset=quality_preset
                )
            else:
                ffmpeg_segment_copy(video_path, start, dur, out_path)

            outputs.append(out_path)

            progress = int(((i + 1) / total_clips) * 100)
            progress_bar.progress(progress)

        progress_bar.progress(100)
        status_text.empty()
        time_text.empty()

        total_time = (datetime.now() - start_time).total_seconds()

        st.success(f"🎉 Successfully created {len(outputs)} clips in {format_duration(total_time)}!")

        total_output_size = sum(Path(p).stat().st_size for p in outputs)

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("✅ Clips Created", len(outputs))
        with result_col2:
            st.metric("💾 Total Size", format_size(total_output_size))
        with result_col3:
            st.metric("⚡ Processing Time", format_duration(total_time))

        # ── Download buttons for each clip (cloud-friendly) ──
        with st.expander("📁 Download Generated Clips", expanded=True):
            for idx, clip_path in enumerate(outputs, 1):
                clip_size = Path(clip_path).stat().st_size
                clip_name = Path(clip_path).name
                clip_dur = min(clip_len_seconds, duration_sec - ((idx - 1) * clip_len_seconds))

                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.write(f"{idx}. ✅ **{clip_name}** ({format_size(clip_size)}) - {format_duration(clip_dur)}")
                with col_btn:
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f,
                            file_name=clip_name,
                            mime="video/mp4",
                            key=f"dl_{idx}"
                        )

        st.balloons()

        st.info("""
        📱 **How to upload to YouTube Shorts:**
        1. Open YouTube app on mobile
        2. Tap the **+** button
        3. Select **Create a Short**
        4. Upload your video files
        5. Add **#Shorts** in title or description
        """)

    except subprocess.CalledProcessError as e:
        st.error(f"❌ FFmpeg error: {e}")
    except Exception as ex:
        st.error(f"❌ Unexpected error: {ex}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("""
**📱 Features:** 
✨ Smart Auto-Focus (Face Detection) | ⏱️ Unlimited Duration | 📐 Perfect 9:16 Format | 🎯 Auto Crop Position

**🔧 Requirements:** FFmpeg (required), OpenCV (optional, for auto-focus)
""")
