"""Standalone MRZ scanner that uses the local webcam instead of iVCam.

This script adapts the notebook logic to a runnable CLI program. It opens a
webcam, finds an ID card, warps it, extracts key regions, and tries to OCR the
MRZ. Stop with `q` while the preview window is focused.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import cv2
import imutils
import numpy as np
import pytesseract


# ----------------------------
# Tesseract configuration
# ----------------------------
def _configure_tesseract() -> None:
    """Ensure pytesseract points to a valid executable.

    Users can override the path with the ``TESSERACT_CMD`` environment variable.
    On Windows the default installer path is used as a fallback; on Linux/macOS
    the binary is expected to be on PATH.
    """

    custom_path = os.getenv("TESSERACT_CMD")
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return

    if os.name == "nt":
        default_win_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path
            return

    # Otherwise rely on PATH; validate availability early for clearer errors.
    cmd = pytesseract.pytesseract.tesseract_cmd
    if not cmd:
        cmd = "tesseract"
        pytesseract.pytesseract.tesseract_cmd = cmd

    if not _is_executable_available(cmd):
        raise FileNotFoundError(
            "Tesseract executable not found. Install Tesseract or set TESSERACT_CMD to the full path."
        )


def _is_executable_available(cmd: str) -> bool:
    """Return True if *cmd* resolves to an executable on the system."""

    if os.path.isabs(cmd):
        return os.path.exists(cmd)

    for path in os.getenv("PATH", "").split(os.pathsep):
        full = os.path.join(path, cmd)
        if os.path.exists(full) and os.access(full, os.X_OK):
            return True
    return False


# ----------------------------
# Detection helpers
# ----------------------------
def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(img: np.ndarray, thresh: float = 100.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = variance_of_laplacian(gray)
    return v < thresh


def has_high_glare(img: np.ndarray, v_thresh: int = 240, area_ratio_thresh: float = 0.06) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, mask = cv2.threshold(v, v_thresh, 255, cv2.THRESH_BINARY)
    ratio = (mask > 0).sum() / mask.size
    return ratio > area_ratio_thresh


def is_good_frame(frame: np.ndarray, blur_thresh: float = 90.0, glare_v: int = 240, glare_area: float = 0.06) -> bool:
    try:
        if is_blurry(frame, thresh=blur_thresh):
            return False
        if has_high_glare(frame, v_thresh=glare_v, area_ratio_thresh=glare_area):
            return False
        return True
    except Exception:
        # If something fails, avoid blocking capture loop
        return True


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _enhance_card_edges(frame: np.ndarray) -> np.ndarray:
    """Light-normalized edge map to handle non-uniform backgrounds."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    smoothed = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    background = cv2.medianBlur(smoothed, 31)
    shadow_free = cv2.subtract(smoothed, background)
    shadow_free = cv2.normalize(shadow_free, None, 0, 255, cv2.NORM_MINMAX)

    edged = cv2.Canny(shadow_free, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edged


def _mask_candidate_regions(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def find_card_contour(frame: np.ndarray, min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    """Return 4-point approx contour of detected card or None."""

    edge_map = _enhance_card_edges(frame)
    mask = _mask_candidate_regions(frame)
    combined = cv2.bitwise_or(edge_map, mask)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = frame.shape[:2]
    min_area = (w * h) * min_area_ratio
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            aspect = _aspect_ratio_from_quad(approx.reshape(4, 2))
            if 1.2 <= aspect <= 1.9 or aspect >= 0.5:
                return approx.reshape(4, 2)
    return None


def _aspect_ratio_from_quad(pts: np.ndarray) -> float:
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    width = max(widthA, widthB)
    height = max(heightA, heightB)
    if height == 0:
        return 0.0
    return width / height


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ----------------------------
# Segmentation & OCR helpers
# ----------------------------
def _load_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Load Haar cascade for face detection.

    Looks for an override via ``FACE_CASCADE_PATH``; otherwise uses the
    ``cv2.data.haarcascades`` directory. Returns ``None`` if the file is
    missing or cannot be loaded so downstream logic can fall back to a
    heuristic crop instead of crashing.
    """

    custom = os.getenv("FACE_CASCADE_PATH")
    if custom and os.path.exists(custom):
        cascade = cv2.CascadeClassifier(custom)
        if not cascade.empty():
            return cascade

    default_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(default_path):
        cascade = cv2.CascadeClassifier(default_path)
        if not cascade.empty():
            return cascade

    print(
        "⚠️ Haar cascade for face detection not found or failed to load. "
        "Set FACE_CASCADE_PATH to a valid XML file to enable face localization."
    )
    return None


face_cascade = _load_face_cascade()


def detect_face_refine(warped: np.ndarray) -> Optional[np.ndarray]:
    if face_cascade is None:
        return None

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    return warped[y : y + h, x : x + w].copy()


def extract_card_regions(warped: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """Extract MRZ, face, signature, name_block, and dni candidate regions."""

    h, w = warped.shape[:2]
    regions: Dict[str, Optional[np.ndarray]] = {}

    mrz_h = int(h * 0.18)
    mrz_y = h - mrz_h - int(h * 0.01)
    regions["mrz"] = warped[mrz_y : mrz_y + mrz_h, 0:w].copy()

    face_roi = detect_face_refine(warped)
    if face_roi is not None:
        regions["face"] = face_roi
    else:
        face_w = int(w * 0.30)
        face_h = int(h * 0.45)
        regions["face"] = warped[int(h * 0.06) : int(h * 0.06) + face_h, int(w * 0.04) : int(w * 0.04) + face_w].copy()

    sig_h = int(h * 0.12)
    sig_w = int(w * 0.5)
    sig_x = int(w * 0.25)
    sig_y = max(0, mrz_y - sig_h - int(h * 0.02))
    regions["signature"] = warped[sig_y : sig_y + sig_h, sig_x : sig_x + sig_w].copy()

    name_x = int(w * 0.38)
    name_y = int(h * 0.08)
    name_w = int(w * 0.55)
    name_h = int(h * 0.25)
    regions["name_block"] = warped[name_y : name_y + name_h, name_x : name_x + name_w].copy()

    try:
        regions["dni_candidate"] = warped[int(h * 0.32) : int(h * 0.52), int(w * 0.60) : int(w * 0.95)].copy()
    except Exception:
        regions["dni_candidate"] = None

    return regions


def preprocess_mrz_for_ocr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Preprocess MRZ for OCR: grayscale, upscale, CLAHE, threshold, morphology."""

    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = max(2, int(600 / max(w, h)))
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = 255 - cleaned
    return cleaned


def read_mrz_text(mrz_img: Optional[np.ndarray]) -> str:
    proc = preprocess_mrz_for_ocr(mrz_img)
    if proc is None:
        return ""
    config = r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ<0123456789 --psm 6"
    try:
        raw = pytesseract.image_to_string(proc, config=config)
    except Exception:
        raw = pytesseract.image_to_string(proc, config="--psm 6")
    cleaned = "".join([c for c in raw if c.isalnum() or c in {"<", "\n", " "}])
    return cleaned.strip()


def is_plausible_mrz_text(text: str) -> bool:
    """Heuristic plausibility check for MRZ output."""

    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<0123456789")
    for ln in lines:
        filtered = [c for c in ln if c in allowed]
        if len(filtered) < max(20, int(0.6 * len(ln))):
            return False
    # MRZ lines are usually 30+ chars; accept if at least one line is long enough
    return any(len(ln) >= 30 for ln in lines)


# ----------------------------
# Main capture loop
# ----------------------------
def run_detection_from_camera(
    cam_index: int = 0,
    show_intermediate: bool = True,
    stop_on_success: bool = True,
    max_frames: int = 1000,
) -> Optional[Dict[str, object]]:
    """Capture frames until an ID card is detected and MRZ recognized."""

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {cam_index}. Ensure the webcam is connected.")
        return None

    print(f"✅ Camera index {cam_index} opened. Starting capture...")
    found_result: Optional[Dict[str, object]] = None
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            frame_count += 1
            frame_proc = imutils.resize(frame, width=900)

            if not is_good_frame(frame_proc, blur_thresh=70.0, glare_v=240, glare_area=0.08):
                if show_intermediate and frame_count % 30 == 0:
                    print(f"Skipping frame {frame_count} due to blur/glare.")
                continue

            card_quad = find_card_contour(frame_proc, min_area_ratio=0.015)
            overlay = frame_proc.copy()
            if card_quad is not None:
                cv2.polylines(overlay, [card_quad.astype(int)], True, (0, 255, 0), 3)
                warped = four_point_transform(frame_proc, card_quad)
                if warped.shape[0] < warped.shape[1]:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                regions = extract_card_regions(warped)
                mrz_text = read_mrz_text(regions["mrz"])
                if is_plausible_mrz_text(mrz_text):
                    print(f"✅ Plausible MRZ detected at frame {frame_count}")
                    found_result = {
                        "frame": frame_count,
                        "warped": warped,
                        "regions": regions,
                        "mrz_text": mrz_text,
                    }
                    if stop_on_success:
                        break

            cv2.imshow("Live Camera Feed", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user.")
                break

            if frame_count >= max_frames:
                print("Reached max frames. Stopping.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return found_result


# ----------------------------
# CLI entry point
# ----------------------------
def main() -> int:
    _configure_tesseract()
    cam_index = int(os.getenv("CAM_INDEX", "0"))
    result = run_detection_from_camera(cam_index=cam_index, show_intermediate=True, stop_on_success=True, max_frames=1200)

    if result is None:
        print("No ID card with plausible MRZ detected.")
        return 1

    print("=== RESULT SUMMARY ===")
    print(f"Detected at frame: {result['frame']}")
    print("Recognized MRZ:")
    print(result["mrz_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
