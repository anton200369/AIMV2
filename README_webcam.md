# Webcam MRZ Scanner (Notebook V5 as a script)

This repository includes `id_scanner_v5.py`, a standalone version of the notebook logic that scans ID cards via your computer's webcam. It now normalizes lighting, suppresses busy backgrounds, and keeps the original FRONT/BACK flow for region capture and MRZ OCR.

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
2. (Optional) Set the camera index and Tesseract path via environment variables:
   ```bash
   export CAM_INDEX=0
   export TESSERACT_CMD="/usr/bin/tesseract"  # adjust if needed
   ```
3. Launch the scanner:
   ```bash
   python id_scanner_v5.py
   ```
4. Place the ID card in view; the preview window will outline the detected card and stop once a plausible MRZ is read. Press `q` to exit early. The detector normalizes lighting, rejects glare/blur, and suppresses non-uniform backgrounds so the card contour remains stable on complex surfaces. The script also requires the contour to stay steady (position + area) across consecutive frames before capturing the front or moving to the MRZ stage, preventing false triggers from faces or nearby objects.
4. Place the ID card in view; the preview window will outline the detected card and stop once a plausible MRZ is read. Press `q` to exit early. The detector normalizes lighting, rejects glare/blur, and suppresses non-uniform backgrounds so the card contour remains stable on complex surfaces.
