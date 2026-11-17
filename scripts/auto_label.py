import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.detector import (
    DEFAULT_ROBOFLOW_MODEL_ID,
    DEFAULT_YOLO_MODEL,
    detect_library_objects,
    detect_library_objects_roboflow,
    load_yolo_model,
)


def convert_bbox_to_yolo_format(
    bbox: tuple[int, int, int, int], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / float(img_width)
    cy = ((y1 + y2) / 2.0) / float(img_height)
    w = (x2 - x1) / float(img_width)
    h = (y2 - y1) / float(img_height)
    return (cx, cy, w, h)


def preannotate_image(
    image_path: Path,
    output_labels_dir: Path,
    model,
    confidence: float = 0.25,
    inference_size: int = 960,
    backend: str = "local",
    model_id: str = DEFAULT_ROBOFLOW_MODEL_ID,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return 0

    height, width = frame.shape[:2]
    if backend == "roboflow":
        detections = detect_library_objects_roboflow(
            frame, model_id=model_id, confidence_threshold=confidence
        )
    else:
        detections = detect_library_objects(
            model, frame, confidence_threshold=confidence, inference_size=inference_size
        )

    label_lines = []
    for obj in detections:
        # Class 0: person, Class 1: belonging
        class_idx = 0 if obj.is_person else 1
        cx, cy, w, h = convert_bbox_to_yolo_format(obj.bbox, width, height)
        label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_file = output_labels_dir / f"{image_path.stem}.txt"
    label_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return len(label_lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-generate YOLO format .txt annotations using local YOLO11s model."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/screenshots"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--backend", choices=["local", "roboflow"], default="roboflow")
    parser.add_argument("--model-id", default=DEFAULT_ROBOFLOW_MODEL_ID)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images folder '{args.images_dir}' not found.")

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(args.images_dir.glob("*.jpg")) + list(args.images_dir.glob("*.png")))
    if not images:
        print("No images found.")
        return

    print(f"Auto-labeling {len(images)} images using backend '{args.backend}'...")
    model = None if args.backend == "roboflow" else load_yolo_model(DEFAULT_YOLO_MODEL)

    total_boxes = 0
    for img_path in images:
        count = preannotate_image(
            img_path,
            args.labels_dir,
            model,
            confidence=args.conf,
            inference_size=args.imgsz,
            backend=args.backend,
            model_id=args.model_id,
        )
        total_boxes += count

    print(f"\nDone! Generated {len(images)} label files in '{args.labels_dir}' ({total_boxes} total boxes).")
    print("Class mapping: 0 = person, 1 = belonging")


if __name__ == "__main__":
    main()
