#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def default_downloads_folder() -> Path:
    return Path.home() / "Downloads"


def resolve_image_path(name_or_path: str, folder: Path) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    candidate = folder / name_or_path
    if candidate.exists():
        return candidate
    # Return the folder-joined path anyway so the caller gets a clear,
    # helpful "file not found" error pointing at where it looked.
    return candidate


def find_two_most_recent_images(folder: Path, exclude: Optional[Path] = None):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    exclude_name = exclude.name if exclude else None
    images = [f for f in folder.iterdir()
              if f.is_file()
              and f.suffix.lower() in IMAGE_EXTENSIONS
              and f.name != exclude_name]
    if len(images) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images in {folder} to auto-compare; found {len(images)}."
        )
    images.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    newest, second_newest = images[0], images[1]
    # "before" = the older of the two, "after" = the newer one
    return second_newest, newest


def load_and_resize(path1: str, path2: str, resize_width: Optional[int]):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None:
        raise FileNotFoundError(f"Could not read image: {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read image: {path2}")

    # Make img2 match img1's dimensions if they differ
    if img2.shape[:2] != img1.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Optional downscale for speed (keeps aspect ratio)
    if resize_width and img1.shape[1] > resize_width:
        scale = resize_width / img1.shape[1]
        new_size = (resize_width, int(img1.shape[0] * scale))
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2


def estimate_shift(gray1: np.ndarray, gray2: np.ndarray):
    h, w = gray1.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    f1 = np.float32(gray1)
    f2 = np.float32(gray2)
    (dx, dy), confidence = cv2.phaseCorrelate(f1, f2, hann)
    return dx, dy, confidence


def align_and_crop(img1, img2, gray1, gray2, dx: float, dy: float):
    h, w = gray1.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    img2_aligned = cv2.warpAffine(img2, M, (w, h), flags=cv2.INTER_LINEAR)
    gray2_aligned = cv2.warpAffine(gray2, M, (w, h), flags=cv2.INTER_LINEAR)

    margin_x = int(np.ceil(abs(dx))) + 2
    margin_y = int(np.ceil(abs(dy))) + 2
    margin_x = min(margin_x, w // 2 - 1)
    margin_y = min(margin_y, h // 2 - 1)

    def crop(im):
        return im[margin_y:h - margin_y, margin_x:w - margin_x]

    return crop(img1), crop(img2_aligned), crop(gray1), crop(gray2_aligned)


def ssim_diff(gray1: np.ndarray, gray2: np.ndarray):
    score, diff = ssim(gray1, gray2, full=True)
    diff = ((1 - diff) * 255).astype("uint8")
    return score, diff


def find_change_regions(diff: np.ndarray, threshold: int, min_area: int):
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append({"x": x, "y": y, "w": w, "h": h, "area": int(area)})
    return boxes


def annotate(img: np.ndarray, boxes: list, color=(0, 0, 255)):
    out = img.copy()
    for b in boxes:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare two images and highlight changes.")
    parser.add_argument("image1", nargs="?", default=None)
    parser.add_argument("image2", nargs="?", default=None)
    parser.add_argument("--dir", default=str(default_downloads_folder()))
    parser.add_argument("--resize", type=int, default=800)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=40)
    parser.add_argument("--out", default=None)
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-confidence-min", type=float, default=0.02)
    args = parser.parse_args()

    folder = Path(args.dir).expanduser()
    out_path = Path(args.out) if args.out else folder / "diff_result.png"

    if args.image1 and args.image2:
        path1 = resolve_image_path(args.image1, folder)
        path2 = resolve_image_path(args.image2, folder)
    elif not args.image1 and not args.image2:
        path1, path2 = find_two_most_recent_images(folder, exclude=out_path)
        print(f"No images specified — auto-selected from {folder}:")
        print(f"  before: {path1.name}")
        print(f"  after:  {path2.name}\n")
    else:
        parser.error("Provide both image1 and image2, or neither (to auto-select).")

    img1, img2, gray1, gray2 = load_and_resize(str(path1), str(path2), args.resize)

    if args.align:
        dx, dy, confidence = estimate_shift(gray1, gray2)
        if confidence >= args.align_confidence_min:
            print(f"Detected shift: dx={dx:.1f}px, dy={dy:.1f}px "
                  f"(confidence={confidence:.3f}) — correcting before comparison.\n")
            img1, img2, gray1, gray2 = align_and_crop(img1, img2, gray1, gray2, dx, dy)
        else:
            print(f"Shift estimate too unreliable to trust "
                  f"(confidence={confidence:.3f} < {args.align_confidence_min}) — "
                  f"comparing images as-is.\n")

    score, diff = ssim_diff(gray1, gray2)
    boxes = find_change_regions(diff, args.threshold, args.min_area)

    changed_px = sum(b["area"] for b in boxes)
    total_px = gray1.shape[0] * gray1.shape[1]
    pct_changed = 100 * changed_px / total_px

    annotated = annotate(img2, boxes)
    cv2.imwrite(str(out_path), annotated)

    result = {
        "ssim_score": round(float(score), 4),
        "percent_area_changed": round(pct_changed, 3),
        "num_change_regions": len(boxes),
        "regions": boxes,
    }

    print(f"Similarity score (SSIM): {score:.4f}  (1.0 = identical)")
    print(f"Changed regions found:   {len(boxes)}")
    print(f"Approx. area changed:    {pct_changed:.2f}%")
    print(f"Annotated image saved:   {out_path}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
