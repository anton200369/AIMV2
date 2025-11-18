# Webcam MRZ Scanner

This repository includes `webcam_mrz_scanner.py`, a standalone script extracted from the notebook to scan ID cards via your computer's webcam instead of an external mobile camera.

## Requirements

1. **Python packages**
   ```bash
   pip install opencv-python imutils numpy pytesseract matplotlib
   ```

2. **Tesseract OCR engine** (required by `pytesseract`)
   - **Ubuntu/Debian**
     ```bash
     sudo apt-get update
     sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
     ```
   - **macOS (Homebrew)**
     ```bash
     brew install tesseract
     ```
   - **Windows (Chocolatey)**
     ```powershell
     choco install tesseract --version=5.3.4
     ```
     If you install Tesseract manually, set `TESSERACT_CMD` to the full path of `tesseract.exe`.

## Running the scanner

1. Confirm your webcam is connected and note its index (default is `0`).
2. (Optional) Set the camera index, Tesseract path, and face cascade override via environment variables:
   ```bash
   export CAM_INDEX=0
   export TESSERACT_CMD="/usr/bin/tesseract"  # adjust if needed
   export FACE_CASCADE_PATH="/path/to/haarcascade_frontalface_default.xml"  # only if OpenCV's bundled cascade is missing
   ```
3. Launch the scanner:
   ```bash
   python webcam_mrz_scanner.py
   ```
4. Place the ID card in view; the preview window will outline the detected card and stop once a plausible MRZ is read. Press `q` to exit early.
