import argparse
import csv
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
from src.seat_monitor import (
    DEFAULT_EMPTY_REF_PATH,
    PolygonPoints,
    draw_seat_monitoring_overlay,
    evaluate_all_table_zones,
    load_empty_reference_frame,
    load_zones_with_metadata,
    scale_zones_to_frame,
)


def parse_screenshot_filename(file_path: Path) -> tuple[str, str, str]:
    parts = file_path.stem.split("_")
    if len(parts) < 3:
        return ("Unknown", "Unknown", file_path.stem)
    return (parts[0], parts[1], parts[2])


def evaluate_single_screenshot(
    image_path: Path,
    model,
    zones: dict[str, PolygonPoints],
    reference_res: tuple[int, int],
    confidence: float,
    inference_size: int,
    empty_ref_path: Path,
    annotated_dir: Path | None,
    backend: str = "local",
    model_id: str = DEFAULT_ROBOFLOW_MODEL_ID,
    api_key: str | None = None,
) -> dict[str, str | int]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image '{image_path}'.")

    height, width = frame.shape[:2]
    day, category, number = parse_screenshot_filename(image_path)
    if backend == "roboflow":
        detections = detect_library_objects_roboflow(
            frame, model_id=model_id, api_key=api_key, confidence_threshold=confidence
        )
    else:
        detections = detect_library_objects(
            model, frame, confidence_threshold=confidence, inference_size=inference_size
        )
    person_count = sum(1 for obj in detections if obj.is_person)
    belonging_count = sum(1 for obj in detections if obj.is_belonging)

    occupied_tables = 0
    hsv_fallback_tables = 0
    if zones:
        scaled_zones = scale_zones_to_frame(zones, reference_res, width, height)
        empty_ref = load_empty_reference_frame(empty_ref_path, width, height)
        statuses = evaluate_all_table_zones(
            scaled_zones,
            detections,
            1,
            15.0,
            900.0,
            {},
            current_frame=frame,
            empty_reference=empty_ref,
        )
        occupied_tables = sum(1 for item in statuses if item.is_occupied)
        hsv_fallback_tables = sum(
            1 for item in statuses if item.detection_source == "HSV_FALLBACK"
        )
        if annotated_dir is not None:
            annotated = draw_seat_monitoring_overlay(frame, statuses, detections)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    return {
        "filename": image_path.name,
        "day": day,
        "category": category,
        "number": number,
        "resolution": f"{width}x{height}",
        "person_count": person_count,
        "belonging_count": belonging_count,
        "occupied_tables": occupied_tables,
        "hsv_fallback_tables": hsv_fallback_tables,
    }


def write_evaluation_csv(rows: list[dict[str, str | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "day",
        "category",
        "number",
        "resolution",
        "person_count",
        "belonging_count",
        "occupied_tables",
        "hsv_fallback_tables",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO11 batch evaluation on the 150 CCTV screenshots (Hari_Kategori_Nomor.jpg)."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder of screenshots.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--annotated-dir", type=Path, default=Path("results/annotated"))
    parser.add_argument("--zones", type=Path, default=Path("config/table_zones.json"))
    parser.add_argument("--reference", type=Path, default=DEFAULT_EMPTY_REF_PATH)
    parser.add_argument("--profile", default="Mezanin_1", help="Default ROI profile.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference resolution.")
    parser.add_argument("--backend", choices=["local", "roboflow"], default="local")
    parser.add_argument("--model-id", default=DEFAULT_ROBOFLOW_MODEL_ID)
    parser.add_argument("--api-key", default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Screenshot directory '{args.input_dir}' does not exist.")

    image_files = sorted(args.input_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No .jpg files found in '{args.input_dir}'.")

    if args.zones.exists():
        zones, ref_res = load_zones_with_metadata(args.zones, profile_name=args.profile)
    else:
        zones, ref_res = {}, (1280, 720)

    args.annotated_dir.mkdir(parents=True, exist_ok=True)
    model = None if args.backend == "roboflow" else load_yolo_model(DEFAULT_YOLO_MODEL)

    rows = [
        evaluate_single_screenshot(
            img_path,
            model,
            zones,
            ref_res,
            args.conf,
            args.imgsz,
            args.reference,
            args.annotated_dir,
            backend=args.backend,
            model_id=args.model_id,
            api_key=args.api_key,
        )
        for img_path in image_files
    ]
    write_evaluation_csv(rows, args.output_csv)
    print(f"Evaluated {len(rows)} screenshots ({args.backend}). CSV saved to '{args.output_csv}'.")


if __name__ == "__main__":
    main()
