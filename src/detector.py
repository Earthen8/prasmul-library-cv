from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np


PERSON_CLASS_ID = 0
BELONGING_CLASS_NAMES = {
    24: "backpack",
    26: "handbag",
    63: "laptop",
    73: "book",
}
# Exclude chair (56) and dining table (60) so empty furniture never triggers occupancy.
TARGET_CLASS_IDS = [PERSON_CLASS_ID, *BELONGING_CLASS_NAMES.keys()]
DEFAULT_YOLO_IMGSZ = 960
DEFAULT_YOLO_MODEL = "yolo11s.pt"


@dataclass(frozen=True)
class DetectedObject:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    @property
    def is_belonging(self) -> bool:
        return self.class_id in BELONGING_CLASS_NAMES

    @property
    def floor_point(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        foot_y = max(y1, y2 - max(2, int((y2 - y1) * 0.08)))
        return (cx, foot_y)


def apply_clahe_lighting(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def letterbox_resize(
    frame: np.ndarray, target_size: int = DEFAULT_YOLO_IMGSZ
) -> tuple[np.ndarray, float, int, int]:
    orig_h, orig_w = frame.shape[:2]
    scale = min(target_size / float(orig_w), target_size / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def load_yolo_model(model_weights_path: str = DEFAULT_YOLO_MODEL) -> Any:
    if not model_weights_path.strip():
        raise ValueError("Model weights path cannot be empty.")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Library 'ultralytics' is not installed. Run '.venv/Scripts/pip install ultralytics' first."
        ) from error

    try:
        return YOLO(model_weights_path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load YOLO11 model '{model_weights_path}': {error}. Verify internet access or file path."
        ) from error


def unscale_letterbox_coords(
    coords: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
    orig_w: int,
    orig_h: int,
) -> tuple[int, int, int, int]:
    x1 = int(np.clip(round((float(coords[0]) - pad_w) / scale), 0, orig_w))
    y1 = int(np.clip(round((float(coords[1]) - pad_h) / scale), 0, orig_h))
    x2 = int(np.clip(round((float(coords[2]) - pad_w) / scale), 0, orig_w))
    y2 = int(np.clip(round((float(coords[3]) - pad_h) / scale), 0, orig_h))
    return (x1, y1, x2, y2)


def parse_single_box(
    box: Any,
    scale: float = 1.0,
    pad_w: int = 0,
    pad_h: int = 0,
    orig_w: int = 1280,
    orig_h: int = 720,
) -> DetectedObject:
    coords = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = unscale_letterbox_coords(coords, scale, pad_w, pad_h, orig_w, orig_h)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    class_id = int(box.cls[0].item())
    confidence = float(box.conf[0].item())
    label = "person" if class_id == PERSON_CLASS_ID else BELONGING_CLASS_NAMES.get(class_id, "item")
    return DetectedObject(
        class_id=class_id,
        label=label,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        center=(cx, cy),
    )


def detect_library_objects(
    model: Any,
    frame: np.ndarray,
    confidence_threshold: float = 0.25,
    inference_size: int = DEFAULT_YOLO_IMGSZ,
    use_clahe: bool = True,
) -> list[DetectedObject]:
    if frame is None or frame.size == 0:
        raise ValueError("Input video frame is empty. Check the video stream source.")
    if not 0.05 <= confidence_threshold <= 0.95:
        raise ValueError("Confidence threshold must be between 0.05 and 0.95.")

    orig_h, orig_w = frame.shape[:2]
    processed = apply_clahe_lighting(frame) if use_clahe else frame
    letterboxed, scale, pad_w, pad_h = letterbox_resize(processed, inference_size)

    results = model.predict(
        source=letterboxed,
        imgsz=inference_size,
        conf=confidence_threshold,
        classes=TARGET_CLASS_IDS,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    return [
        parse_single_box(
            box, scale=scale, pad_w=pad_w, pad_h=pad_h, orig_w=orig_w, orig_h=orig_h
        )
        for box in results[0].boxes
        if int(box.cls[0].item()) in TARGET_CLASS_IDS
    ]
