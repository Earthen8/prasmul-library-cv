from pathlib import Path
import cv2
import numpy as np
from src.detector import DetectedObject
from src.seat_monitor import PolygonPoints, coerce_to_polygon, polygon_bounding_box


def draw_synthetic_library_background(
    canvas: np.ndarray, zones: dict[str, PolygonPoints]
) -> None:
    canvas[:] = (232, 228, 220)
    for zone_name, poly in zones.items():
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], (195, 186, 172))
        cv2.polylines(canvas, [pts], isClosed=True, color=(110, 102, 90), thickness=2)
        x1, y1, _, _ = polygon_bounding_box(poly)
        cv2.putText(
            canvas,
            zone_name,
            (x1 + 10, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (55, 50, 42),
            2,
        )


def generate_sample_cctv_video(
    output_path: Path,
    zones: dict[str, PolygonPoints],
    duration_seconds: int = 8,
    fps: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create sample video at '{output_path}'.")

    normalized_zones = {
        name: coerce_to_polygon(list(poly), name) if isinstance(poly, tuple) else poly
        for name, poly in zones.items()
    }
    total_frames = duration_seconds * fps
    try:
        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_synthetic_library_background(frame, normalized_zones)
            timestamp_text = f"SIMULASI DEMO CCTV | FRAME {frame_idx + 1}/{total_frames}"
            cv2.putText(
                frame,
                timestamp_text,
                (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (35, 35, 35),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def build_simulated_detections(
    frame_index: int,
    camera_mode: str,
    zones: dict[str, PolygonPoints],
) -> list[DetectedObject]:
    detections: list[DetectedObject] = []
    if zones:
        detections.extend(_simulate_table_objects(zones))
    if "Rak Buku" in camera_mode:
        detections.extend(_simulate_aisle_walkers(frame_index))
    return detections


def _polygon_centroid(poly: PolygonPoints) -> tuple[int, int]:
    cx = int(sum(pt[0] for pt in poly) / len(poly))
    cy = int(sum(pt[1] for pt in poly) / len(poly))
    return (cx, cy)


def _simulate_table_objects(
    zones: dict[str, PolygonPoints],
) -> list[DetectedObject]:
    zone_list = [
        coerce_to_polygon(list(p), k) if isinstance(p, tuple) else p
        for k, p in zones.items()
    ]
    detections: list[DetectedObject] = []
    if len(zone_list) >= 1:
        cx, cy = _polygon_centroid(zone_list[0])
        detections.append(
            DetectedObject(0, "person", 0.91, (cx - 40, cy - 55, cx + 40, cy + 55), (cx, cy))
        )
        detections.append(
            DetectedObject(63, "laptop", 0.86, (cx - 22, cy - 8, cx + 22, cy + 22), (cx, cy + 6))
        )
    if len(zone_list) >= 2:
        cx, cy = _polygon_centroid(zone_list[1])
        detections.append(
            DetectedObject(24, "backpack", 0.88, (cx - 28, cy - 28, cx + 28, cy + 28), (cx, cy))
        )
    return detections


def _simulate_aisle_walkers(frame_index: int) -> list[DetectedObject]:
    step = (frame_index * 16) % 380
    walker_one_y = 260 + step
    walker_two_y = 620 - step
    return [
        DetectedObject(
            0, "person", 0.92, (210, walker_one_y - 45, 285, walker_one_y + 45), (248, walker_one_y)
        ),
        DetectedObject(
            0, "person", 0.89, (600, walker_two_y - 45, 675, walker_two_y + 45), (638, walker_two_y)
        ),
    ]
