"""
detection_utils.py - Core detection utilities for Parking Space Occupancy Detector.
Functions: load_model, run_detection, draw_results, get_summary
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------
def load_model(model_path: str = "models/best.pt") -> Any:
    """
    Load a YOLOv8 model from model_path.
    Prints actual class names found in the model.
    Raises FileNotFoundError if weights file does not exist.
    """
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run  python train.py  to train and save the model first."
        )

    model = YOLO(str(path))
    print(f"✅ Model loaded. Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (0, 215, 255),    # Yellow
    (255, 165, 0),    # Orange
    (255, 0, 255),    # Purple
    (0, 165, 255),    # Light-orange
    (128, 0, 128),    # Dark purple
]


def _build_color_map(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """
    Build a BGR color map from the actual class name list.
    Uses keyword matching first; falls back to index-based color.

    Priority keywords (case-insensitive):
      free / empty / vacant / open  → Green  (0, 255,   0)
      partial / half / part         → Yellow (0, 215, 255)
      occupied / full / taken / busy → Red   (0,   0, 255)
      anything else                 → color by index
    """
    color_map: dict[str, tuple[int, int, int]] = {}
    for i, name in enumerate(class_names):
        n = name.lower()
        if any(w in n for w in ["free", "empty", "vacant", "open"]):
            color_map[name] = (0, 255, 0)
        elif any(w in n for w in ["partial", "half", "part"]):
            color_map[name] = (0, 215, 255)
        elif any(w in n for w in ["occupied", "full", "taken", "busy"]):
            color_map[name] = (0, 0, 255)
        else:
            color_map[name] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    return color_map


# ---------------------------------------------------------------------------
# run_detection
# ---------------------------------------------------------------------------
def run_detection(
    model: Any,
    image_np: np.ndarray,
    conf_threshold: float = 0.4,
) -> list[dict]:
    """
    Run YOLOv8 inference on a single BGR image (numpy array).

    Args:
        model          : Loaded YOLO model.
        image_np       : BGR image as H×W×3 numpy array.
        conf_threshold : Detections below this confidence are discarded.

    Returns:
        List of detection dicts:
        [
            {
                'bbox'      : [x1, y1, x2, y2],
                'class_id'  : int,
                'class_name': str,
                'confidence': float,
            },
            ...
        ]
    """
    results = model(image_np, verbose=False)
    class_names: dict = model.names   # {0: 'free', 1: 'occupied', ...}
    detections: list[dict] = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, clss):
            if float(conf) < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box)
            cid  = int(cls_id)
            # Always use the model's actual name — never fall back to "unknown"
            name = class_names.get(cid, f"class_{cid}")
            detections.append(
                {
                    "bbox"      : [x1, y1, x2, y2],
                    "class_id"  : cid,
                    "class_name": name,
                    "confidence": float(conf),
                }
            )

    return detections


# ---------------------------------------------------------------------------
# draw_results
# ---------------------------------------------------------------------------
def draw_results(
    image_np,
    detections,
    class_names=None,
):
    """
    Draw colored bounding boxes with semi-transparent fill and labels.

    Color assignment (dynamic — no hardcoded class names):
      keyword 'free'/'empty'/'vacant'/'open'     → Green
      keyword 'partial'/'half'/'part'            → Yellow
      keyword 'occupied'/'full'/'taken'/'busy'   → Red
      unrecognised name                          → color by class index

    Label format: "<class_name>  <confidence%>" — white text, black background.
    Returns annotated BGR image (copy of input).
    """
    # If class_names not passed, derive them from detections themselves
    if class_names is None:
        seen = {}
        for det in detections:
            cid = det["class_id"]
            if cid not in seen:
                seen[cid] = det["class_name"]
        class_names = [seen[k] for k in sorted(seen)]

    color_map = _build_color_map(class_names)

    annotated = image_np.copy()
    overlay   = image_np.copy()

    # Draw semi-transparent fill
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name  = det["class_name"]
        cid   = det["class_id"]
        color = color_map.get(name, _FALLBACK_COLORS[cid % len(_FALLBACK_COLORS)])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)

    # Blend overlay (20 % transparency)
    cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)

    # Draw solid border + label on top of blended image
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name  = det["class_name"]
        cid   = det["class_id"]
        conf  = det["confidence"]
        color = color_map.get(name, _FALLBACK_COLORS[cid % len(_FALLBACK_COLORS)])
        label = f"{name}  {conf:.0%}"

        # Solid border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3
        lx1, ly1 = x1, max(y1 - th - 2 * pad, 0)
        lx2, ly2 = x1 + tw + 2 * pad, y1

        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (lx1 + pad, ly2 - pad),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------
def get_summary(detections, class_names=None):
    """
    Compute occupancy summary from a list of detections.

    Uses dynamic keyword matching + class_id fallback so it works with
    any class naming convention (free, empty, space-empty, Occupied, etc.)

    Returns:
        {
            'total'         : int,
            'counts'        : {'free': int, 'occupied': int, 'partially_occupied': int},
            'occupancy_rate': float   # (occupied + partial) / total * 100
        }
    """
    total  = len(detections)
    counts = {"free": 0, "occupied": 0, "partially_occupied": 0}

    for det in detections:
        name = det["class_name"].lower()
        cid  = det["class_id"]

        if any(w in name for w in ["free", "empty", "vacant", "open"]):
            counts["free"] += 1
        elif any(w in name for w in ["partial", "half", "part"]):
            counts["partially_occupied"] += 1
        elif any(w in name for w in ["occupied", "full", "taken", "busy"]):
            counts["occupied"] += 1
        else:
            # Fallback: use class_id position when name doesn't match any keyword
            if cid == 0:
                counts["free"] += 1
            elif cid == 1:
                counts["occupied"] += 1
            elif cid == 2:
                counts["partially_occupied"] += 1
            else:
                # Extra classes → treat as occupied
                counts["occupied"] += 1

    occupied_and_partial = counts["occupied"] + counts["partially_occupied"]
    occupancy_rate = (
        round(occupied_and_partial / total * 100, 1) if total > 0 else 0.0
    )

    return {
        "total"         : total,
        "counts"        : counts,
        "occupancy_rate": occupancy_rate,
    }
