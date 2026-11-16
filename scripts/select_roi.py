import argparse
import json
from pathlib import Path
import cv2
import numpy as np


PolygonPoints = list[list[int]]


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


def render_polygon_canvas(
    base_frame: np.ndarray,
    completed: dict[str, PolygonPoints],
    current_points: PolygonPoints,
) -> np.ndarray:
    canvas = base_frame.copy()
    for name, poly in completed.items():
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 200, 80), 2)
        cv2.putText(canvas, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)

    for index, pt in enumerate(current_points):
        cv2.circle(canvas, tuple(pt), 4, (0, 180, 255), -1)
        if index > 0:
            cv2.line(canvas, tuple(current_points[index - 1]), tuple(pt), (0, 180, 255), 2)
    return canvas


def prompt_polygon_selection(
    frame: np.ndarray, window_title: str, prefix: str = "Meja"
) -> dict[str, PolygonPoints]:
    print("Polygon Mode: Left-click to add corner points following 3D table perspective.")
    print("Right-click or press 'c'/ENTER to close a table polygon. Press 'u' to undo point, ESC to finish.")
    completed: dict[str, PolygonPoints] = {}
    current_pts: PolygonPoints = []
    state = {"close_requested": False}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and len(current_pts) >= 3:
            state["close_requested"] = True

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, on_mouse)
        while True:
            canvas = render_polygon_canvas(frame, completed, current_pts)
            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if (key in (13, ord("c")) or state["close_requested"]) and len(current_pts) >= 3:
                zone_name = f"{prefix}_{len(completed) + 1:02d}"
                completed[zone_name] = list(current_pts)
                current_pts.clear()
                state["close_requested"] = False
            elif key == ord("u") and current_pts:
                current_pts.pop()
            elif key in (27, ord("q")):
                break
        cv2.destroyAllWindows()
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV GUI window failed to open. Install 'opencv-python' locally for interactive ROI picking."
        ) from error
    return completed


def prompt_bbox_selection(
    frame: np.ndarray, window_title: str, prefix: str = "Meja"
) -> dict[str, PolygonPoints]:
    raw_boxes = cv2.selectROIs(window_title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    zones: dict[str, PolygonPoints] = {}
    if len(raw_boxes) == 0:
        return zones
    for index, (x, y, w, h) in enumerate(raw_boxes, start=1):
        if w > 0 and h > 0:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            zones[f"{prefix}_{index:02d}"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return zones


def save_zones_to_json(
    zones: dict[str, PolygonPoints],
    frame_shape: tuple[int, int, int],
    profile_name: str,
    output_json: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    existing: dict = {"reference_resolution": [width, height], "profiles": {}}
    if output_json.exists():
        try:
            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "profiles" in loaded:
                existing = loaded
        except json.JSONDecodeError:
            pass
    existing["reference_resolution"] = [width, height]
    existing.setdefault("profiles", {})[profile_name] = {
        "reference_resolution": [width, height],
        "zones": zones,
    }
    output_json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def draw_saved_zones_preview(
    frame: np.ndarray, zones: dict[str, PolygonPoints], preview_path: Path
) -> None:
    annotated = render_polygon_canvas(frame, zones, [])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), annotated)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select 3D polygon or bounding-box ROI coordinates from a reference image/video."
    )
    parser.add_argument("--source", required=True, type=Path, help="Reference image or video path.")
    parser.add_argument(
        "--mode", choices=["polygon", "bbox"], default="polygon", help="ROI shape mode."
    )
    parser.add_argument(
        "--profile", default="Mezanin_1", help="Camera profile key (e.g. Mezanin_1, Rak_2)."
    )
    parser.add_argument("--prefix", default="Meja", help="Zone name prefix.")
    parser.add_argument("--output", type=Path, default=Path("config/table_zones.json"))
    parser.add_argument("--preview", type=Path, default=Path("config/table_zones_preview.jpg"))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    frame = load_reference_frame(args.source)
    title = f"Select ROI [{args.profile}] ({args.mode})"
    if args.mode == "polygon":
        zones = prompt_polygon_selection(frame, title, prefix=args.prefix)
    else:
        zones = prompt_bbox_selection(frame, title, prefix=args.prefix)

    if not zones:
        print("No table zones were selected. Exiting without saving.")
        return

    save_zones_to_json(zones, frame.shape, args.profile, args.output)
    draw_saved_zones_preview(frame, zones, args.preview)
    print(f"Saved {len(zones)} zones under profile '{args.profile}' to '{args.output}'.")


if __name__ == "__main__":
    main()
