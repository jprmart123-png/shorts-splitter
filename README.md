# 📱 YouTube Shorts Video Splitter

Split long videos into YouTube Shorts (9:16 vertical format) with Smart Auto-Focus (Face Detection).

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — GitHub pe upload karo
1. GitHub.com par jao → **New Repository** banao
2. Ye 3 files upload karo:
   - `short.py`
   - `requirements.txt`
   - `packages.txt`
3. **Commit changes** click karo

### Step 2 — Streamlit Cloud pe deploy karo
1. [share.streamlit.io](https://share.streamlit.io) par jao
2. GitHub se **Sign in** karo
3. **"New app"** click karo
4. Apni repository select karo
5. **Main file path:** `short.py`
6. **Deploy!** ✅

App live ho jayega kuch minutes mein!

---

## ✨ Features

- 📐 9:16 YouTube Shorts format conversion
- 🎯 Smart Auto-Focus with Face Detection (OpenCV)
- ⏱️ Custom clip duration (presets + custom)
- 🎥 Preview before processing
- ⬇️ Direct download of all clips
- 🎬 Multiple quality presets

## 🔧 Requirements (auto-installed on cloud)

- FFmpeg (system)
- streamlit
- opencv-python-headless
- numpy
