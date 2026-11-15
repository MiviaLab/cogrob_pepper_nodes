#!/usr/bin/env python3
"""
Closed-set vs open-vocabulary detection: YOLO11 vs YOLOE on the same image.

1. Runs a classic YOLO (80 fixed COCO classes).
2. Runs YOLOE with class names given as text prompts.
3. Prints both outputs and the classes found ONLY by YOLOE.
4. Saves a side-by-side image: COCO classes in blue, new classes in orange.

Setup (once):
    python3 -m pip install git+https://github.com/ultralytics/CLIP.git
    # the first YOLOE run downloads the MobileCLIP text encoder (~570 MB)

Usage:
    python3 compare_yolo_yoloe.py
    python3 compare_yolo_yoloe.py --image my_photo.jpg --classes person sneakers "red mug" keys
"""
import argparse
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

BLUE = (186, 148, 52)    # BGR: classes the closed-set model also knows
ORANGE = (23, 106, 232)  # BGR: classes only the open-vocabulary model can see
SAMPLE = "https://ultralytics.com/images/bus.jpg"


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO (closed-set) vs YOLOE (open-vocabulary)")
    ap.add_argument("--image", default=SAMPLE, help="image path or URL")
    ap.add_argument("--classes", nargs="+", default=["bus", "person", "sneakers"],
                    help="text prompts for YOLOE (short concrete nouns work best)")
    ap.add_argument("--yolo", default="yolo11n.pt", help="closed-set model")
    ap.add_argument("--yoloe", default="yoloe-11s-seg.pt", help="open-vocabulary model")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--out", default="comparison.jpg", help="output image")
    return ap.parse_args()


def to_list(result):
    """Ultralytics result -> list of (class_name, confidence, [x1, y1, x2, y2])."""
    return [(result.names[int(b.cls)], float(b.conf), [int(v) for v in b.xyxy[0].tolist()])
            for b in result.boxes]


def timed_predict(model, image, conf):
    """Warm-up run first, so the timing does not include model initialisation."""
    model.predict(image, conf=conf, verbose=False)
    t0 = time.perf_counter()
    result = model.predict(image, conf=conf, verbose=False)[0]
    return result, (time.perf_counter() - t0) * 1000


def print_table(title, dets, elapsed_ms):
    print(f"\n=== {title}  ({len(dets)} detections, {elapsed_ms:.0f} ms) ===")
    print(f"{'class':<16}{'conf':>6}   box (x1, y1, x2, y2)")
    for name, conf, box in sorted(dets, key=lambda d: -d[1]):
        print(f"{name:<16}{conf:>6.2f}   {box}")
    print("summary:", dict(Counter(d[0] for d in dets)))


def draw(img, dets, title, coco_names):
    out = img.copy()
    for name, conf, (x1, y1, x2, y2) in dets:
        color = BLUE if name in coco_names else ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 6 if y1 > th + 10 else y2 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (40, 40, 40), -1)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main():
    args = parse_args()

    # 1) Closed-set detector: the class list is fixed at training time
    yolo = YOLO(args.yolo)
    coco_names = set(yolo.names.values())
    r_yolo, ms = timed_predict(yolo, args.image, args.conf)
    dets_yolo = to_list(r_yolo)
    print_table(f"YOLO closed-set ({args.yolo}, {len(coco_names)} classes)", dets_yolo, ms)

    # 2) Open-vocabulary detector: the class list is text, chosen now
    yoloe = YOLOE(args.yoloe)
    yoloe.set_classes(args.classes, yoloe.get_text_pe(args.classes))
    r_yoloe, ms = timed_predict(yoloe, args.image, args.conf)
    dets_yoloe = to_list(r_yoloe)
    print_table(f"YOLOE open-vocabulary ({args.yoloe}, prompts={args.classes})", dets_yoloe, ms)

    # 3) What changed?
    found_yolo = {d[0] for d in dets_yolo}
    found_yoloe = {d[0] for d in dets_yoloe}
    not_coco = [c for c in args.classes if c not in coco_names]
    print("\n=== Differences ===")
    print("prompts that are NOT COCO classes:", not_coco or "none")
    print("found only by YOLOE:", sorted(found_yoloe - found_yolo) or "none")
    print("found only by YOLO :", sorted(found_yolo - found_yoloe) or "none",
          "(YOLOE only looks for the prompted classes)")
    missed = [c for c in args.classes if c not in found_yoloe]
    if missed:
        print("prompts with no detection:", missed, "-> try a shorter/concrete noun or a lower --conf")

    # 4) Side-by-side picture
    img = r_yolo.orig_img
    left = draw(img, dets_yolo, f"YOLO: {len(coco_names)} fixed classes", coco_names)
    right = draw(img, dets_yoloe, "YOLOE: " + ", ".join(args.classes), coco_names)
    gap = np.full((img.shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(args.out, np.hstack([left, gap, right]))
    print(f"\nsaved {args.out}  (blue = COCO class, orange = new class from text)")


if __name__ == "__main__":
    main()