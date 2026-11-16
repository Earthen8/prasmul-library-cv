from dataclasses import dataclass
from typing import Any
import numpy as np


PERSON_CLASS_ID = 0
BELONGING_CLASS_NAMES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    67: "cell phone",
    73: "book",
}
TARGET_CLASS_IDS = [PERSON_CLASS_ID, *BELONGING_CLASS_NAMES.keys()]


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


def load_yolo_model(model_weights_path: str = "yolo11n.pt") -> Any:
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
            f"Failed to load YOLO11 model '{model_weights_path}': {error}. Verify internet access for initial weight download or check local file path."
        ) from error


def parse_single_box(box: Any) -> DetectedObject:
    coords = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
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
    confidence_threshold: float = 0.30,
) -> list[DetectedObject]:
    if frame is None or frame.size == 0:
        raise ValueError("Input video frame is empty. Check the video stream source.")
    if not 0.05 <= confidence_threshold <= 0.95:
        raise ValueError("Confidence threshold must be between 0.05 and 0.95.")

    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        classes=TARGET_CLASS_IDS,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    return [parse_single_box(box) for box in results[0].boxes]
