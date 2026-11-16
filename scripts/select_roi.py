import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def load_reference_frame(source_path: Path) -> np.ndarray:
    if not source_path.exists():
        raise FileNotFoundError(
            f"Reference file not found at '{source_path}'. Provide a valid image or video path."
        )

    suffix = source_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError(f"Failed to decode image file '{source_path}'.")
        return image

    capture = cv2.VideoCapture(str(source_path))
    try:
        has_frame, frame = capture.read()
        if not has_frame or frame is None:
            raise RuntimeError(f"Could not read a reference frame from video '{source_path}'.")
        return frame
    finally:
        capture.release()


def normalize_box(x: int, y: int, width: int, height: int) -> list[int]:
    x1 = int(min(x, x + width))
    y1 = int(min(y, y + height))
    x2 = int(max(x, x + width))
    y2 = int(max(y, y + height))
    return [x1, y1, x2, y2]


def prompt_roi_selection(frame: np.ndarray, window_title: str) -> dict[str, list[int]]:
    print("Instructions: Drag a box around each table, press ENTER/SPACE to confirm each box.")
    print("Press ESC when you have finished selecting all table zones.")
    try:
        raw_boxes = cv2.selectROIs(
            window_title, frame, showCrosshair=True, fromCenter=False
        )
        cv2.destroyAllWindows()
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV GUI window failed to open. Install 'opencv-python' locally to use interactive ROI selection."
        ) from error

    zones: dict[str, list[int]] = {}
    if len(raw_boxes) == 0:
        return zones

    for index, (x, y, width, height) in enumerate(raw_boxes, start=1):
        if width <= 0 or height <= 0:
            continue
        zone_name = f"Meja_{index:02d}"
        zones[zone_name] = normalize_box(x, y, width, height)
    return zones


def draw_saved_zones_preview(
    frame: np.ndarray, zones: dict[str, list[int]], preview_path: Path
) -> None:
    annotated = frame.copy()
    for zone_name, (x1, y1, x2, y2) in zones.items():
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(
            annotated,
            f"{zone_name} ({x1},{y1},{x2},{y2})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 80),
            2,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), annotated)


def save_zones_to_json(zones: dict[str, list[int]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    formatted_content = json.dumps(zones, indent=2)
    output_json.write_text(formatted_content + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select (x1, y1, x2, y2) table ROI coordinates from a reference image or video."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to a reference image (.jpg/.png) or video (.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/table_zones.json"),
        help="Output JSON file for table coordinates.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("config/table_zones_preview.jpg"),
        help="Output image showing the annotated table bounding boxes.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    frame = load_reference_frame(args.source)
    zones = prompt_roi_selection(frame, "Select Table Zones (Enter=Confirm, Esc=Finish)")
    if not zones:
        print("No table zones were selected. Exiting without saving.")
        return

    save_zones_to_json(zones, args.output)
    draw_saved_zones_preview(frame, zones, args.preview)
    print(f"\nSaved {len(zones)} table zones to '{args.output}':")
    for name, coords in zones.items():
        print(f"  {name}: {tuple(coords)}")
    print(f"Annotated preview saved to '{args.preview}'.")


if __name__ == "__main__":
    main()
