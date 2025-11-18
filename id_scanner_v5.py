"""CLI ID scanner derived from the notebook flow (V5).

- Captures frames from a webcam.
- Robustly finds an ID card on non-uniform backgrounds, warps it, and extracts
  front ROIs plus the MRZ.
- Performs MRZ OCR via Tesseract.

Press ``q`` in the preview window to quit early.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import imutils
import numpy as np
import pytesseract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TESSERACT_WIN = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))

ROIS_FRONT: Dict[str, Tuple[float, float, float, float]] = {
    "Face": (0.18, 0.82, 0.04, 0.32),
    "Name_Surname": (0.12, 0.35, 0.35, 0.95),
    "Signature": (0.78, 0.95, 0.35, 0.85),
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def configure_tesseract() -> None:
    """Ensure pytesseract points to a valid executable."""

    custom = os.getenv("TESSERACT_CMD")
    if custom:
        pytesseract.pytesseract.tesseract_cmd = custom
        return

    if os.name == "nt" and os.path.exists(DEFAULT_TESSERACT_WIN):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_WIN
        return

    # Otherwise rely on PATH; pytesseract sets a default command already.
    cmd = pytesseract.pytesseract.tesseract_cmd or "tesseract"
    if os.path.isabs(cmd) and os.path.exists(cmd):
        return

    for path in os.getenv("PATH", "").split(os.pathsep):
        full = os.path.join(path, cmd)
        if os.path.exists(full) and os.access(full, os.X_OK):
            return

    raise FileNotFoundError(
        "Tesseract executable not found. Install Tesseract or set TESSERACT_CMD to the full path."
    )


def _normalize_lighting(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    background = cv2.medianBlur(enhanced, 31)
    foreground = cv2.subtract(enhanced, background)
    return cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)


def _build_card_mask(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask emphasizing rectangular card-like regions."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    norm = _normalize_lighting(gray)

    # Edge emphasis from gradient + Canny
    grad_x = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    edges = cv2.Canny(norm, 40, 160)
    edge_mix = cv2.bitwise_or(edges, grad)

    # Texture suppression to handle busy backgrounds
    adaptive = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 7
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)

    combined = cv2.bitwise_or(edge_mix, adaptive)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2
    )
    return combined


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _aspect_ratio_from_quad(pts: np.ndarray) -> float:
    rect = _order_points(pts.reshape(4, 2))
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


def find_card_contour(frame: np.ndarray, min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    mask = _build_card_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = frame.shape[:2]
    min_area = (h * w) * min_area_ratio
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            aspect = _aspect_ratio_from_quad(approx.reshape(4, 2))
            # Accept typical landscape ID ratios and rotated cards
            if 1.2 <= aspect <= 1.95 or 0.5 <= aspect <= 0.85:
                return approx.reshape(4, 2)
    return None


def four_point_transform_landscape(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Force landscape orientation
    if maxHeight > maxWidth:
        dst = np.array(
            [[0, 0], [maxHeight - 1, 0], [maxHeight - 1, maxWidth - 1], [0, maxWidth - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxHeight, maxWidth))
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else:
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32"
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return cv2.resize(warped, (1000, 630))


def draw_rois(image: np.ndarray, rois_dict: Dict[str, Tuple[float, float, float, float]], color=(0, 255, 0)) -> np.ndarray:
    h, w = image.shape[:2]
    vis = image.copy()
    for name, (y1, y2, x1, x2) in rois_dict.items():
        cv2.rectangle(vis, (int(w * x1), int(h * y1)), (int(w * x2), int(h * y2)), color, 2)
        cv2.putText(vis, name, (int(w * x1) + 5, int(h * y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def extract_front_regions(warped: np.ndarray) -> Dict[str, np.ndarray]:
    regions: Dict[str, np.ndarray] = {}
    h, w = warped.shape[:2]
    for name, (y1, y2, x1, x2) in ROIS_FRONT.items():
        regions[name] = warped[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]
    return regions


def detect_mrz_region(warped: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], np.ndarray]:
    target_width = 1000
    scale_ratio = target_width / warped.shape[1]
    image = cv2.resize(warped, (target_width, int(warped.shape[0] * scale_ratio)))

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    blackhat_closed = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(blackhat_closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    roi_box = None
    roi_img = None

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        if ar > 3 and crWidth > 0.50:
            pad = 10
            p_x = max(0, x - pad)
            p_y = max(0, y - pad)
            p_w = min(image.shape[1], w + (pad * 2))
            p_h = min(image.shape[0], h + (pad * 2))
            roi_img = image[p_y : p_y + p_h, p_x : p_x + p_w].copy()
            roi_box = (p_x, p_y, p_w, p_h)
            break

    return roi_box, roi_img, thresh


def ocr_mrz(mrz_img: np.ndarray) -> str:
    gray_roi = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    config = r"--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    text = pytesseract.image_to_string(binary, config=config).strip()
    return text


# ---------------------------------------------------------------------------
# Main scanning loop
# ---------------------------------------------------------------------------
def run_scanner(cam_index: int = 0, show_debug_mask: bool = False) -> Dict[str, np.ndarray | str]:
    configure_tesseract()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}")

    collected: Dict[str, np.ndarray | str] = {}
    state = "FRONT"
    stabilize_count = 0
    REQUIRED_STABLE = 10

    print("📸 Scanner Ready. Show FRONT.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = imutils.resize(frame, height=600)
            orig = frame.copy()

            card_cnt = find_card_contour(frame)
            display_frame = frame.copy()
            debug_mask = None

            if card_cnt is not None:
                cv2.polylines(display_frame, [card_cnt.astype(int)], True, (0, 255, 0), 2)
                warped = four_point_transform_landscape(orig, card_cnt)

                if state == "FRONT":
                    warped_vis = draw_rois(warped, ROIS_FRONT, color=(0, 255, 255))
                    small_warped = cv2.resize(warped_vis, (320, 200))
                    display_frame[0:200, 0:320] = small_warped
                    cv2.putText(display_frame, "Scan FRONT", (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    stabilize_count += 1
                    if stabilize_count > REQUIRED_STABLE:
                        collected.update(extract_front_regions(warped))
                        state = "WAIT_FLIP"
                        stabilize_count = 0

                elif state == "WAIT_FLIP":
                    cv2.putText(display_frame, "FLIP TO BACK...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    stabilize_count += 1
                    if stabilize_count > 40:
                        state = "BACK"
                        stabilize_count = 0

                elif state == "BACK":
                    mrz_box, mrz_roi_img, debug_mask = detect_mrz_region(warped)
                    if debug_mask is not None and debug_mask.ndim == 2:
                        debug_mask = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)

                    warped_vis = cv2.resize(warped, (1000, 630))

                    if mrz_box is not None:
                        (mx, my, mw, mh) = mrz_box
                        cv2.rectangle(warped_vis, (mx, my), (mx + mw, my + mh), (0, 255, 0), 3)
                        cv2.putText(display_frame, "MRZ FOUND!", (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        stabilize_count += 1
                        if stabilize_count > 5 and mrz_roi_img is not None:
                            text = ocr_mrz(mrz_roi_img)
                            if "<<" in text and len(text) > 15:
                                collected["MRZ_Image"] = mrz_roi_img
                                collected["MRZ_Text"] = text
                                state = "DONE"
                    else:
                        cv2.putText(
                            display_frame,
                            "Searching MRZ...",
                            (330, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        stabilize_count = 0

                    small_warped = cv2.resize(warped_vis, (320, 200))
                    display_frame[0:200, 0:320] = small_warped
            else:
                stabilize_count = 0

            if show_debug_mask and card_cnt is None:
                debug_mask = _build_card_mask(frame)

            if debug_mask is not None:
                if debug_mask.ndim == 2:
                    debug_mask = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
                debug_mask = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)

            if debug_mask is not None:
                debug_resized = cv2.resize(debug_mask, (frame.shape[1], frame.shape[0]))
                stacked = np.hstack((display_frame, debug_resized))
                cv2.imshow("Live + Mask", stacked)
            else:
                cv2.imshow("Live", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if state == "DONE":
                print("✅ Scanning Complete!")
                time.sleep(1)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return collected


def main() -> int:
    try:
        results = run_scanner(cam_index=CAM_INDEX, show_debug_mask=False)
    except Exception as exc:  # pragma: no cover - interactive path
        print(f"Error: {exc}")
        return 1

    if results:
        print("=" * 40)
        print("       🆔 FINAL RESULTS       ")
        print("=" * 40)
        for key in ["Face", "Name_Surname", "Signature"]:
            if key in results:
                print(f"- Captured {key} region: {results[key].shape[1]}x{results[key].shape[0]} px")
        if "MRZ_Text" in results:
            print("\n📝 MRZ TEXT:\n")
            print(results["MRZ_Text"])
    else:
        print("No results captured.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
