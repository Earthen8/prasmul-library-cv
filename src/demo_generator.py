from pathlib import Path
import cv2
import numpy as np
from src.detector import DetectedObject


def draw_synthetic_library_background(
    canvas: np.ndarray, zones: dict[str, tuple[int, int, int, int]]
) -> None:
    canvas[:] = (232, 228, 220)
    for zone_name, (x1, y1, x2, y2) in zones.items():
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (195, 186, 172), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (110, 102, 90), 2)
        cv2.putText(
            canvas,
            zone_name,
            (x1 + 12, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (55, 50, 42),
            2,
        )


def generate_sample_cctv_video(
    output_path: Path,
    zones: dict[str, tuple[int, int, int, int]],
    duration_seconds: int = 8,
    fps: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create sample video at '{output_path}'.")

    total_frames = duration_seconds * fps
    try:
        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_synthetic_library_background(frame, zones)
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
    zones: dict[str, tuple[int, int, int, int]],
) -> list[DetectedObject]:
    if camera_mode == "Kamera Rak Buku (Foot Traffic & Heatmap)":
        return _simulate_aisle_walkers(frame_index)
    return _simulate_table_objects(zones)


def _simulate_table_objects(
    zones: dict[str, tuple[int, int, int, int]],
) -> list[DetectedObject]:
    zone_list = list(zones.values())
    detections: list[DetectedObject] = []
    if len(zone_list) >= 1:
        x1, y1, x2, y2 = zone_list[0]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append(
            DetectedObject(0, "person", 0.91, (cx - 45, cy - 60, cx + 45, cy + 60), (cx, cy))
        )
        detections.append(
            DetectedObject(63, "laptop", 0.86, (cx - 25, cy - 10, cx + 25, cy + 25), (cx, cy + 8))
        )
    if len(zone_list) >= 2:
        x1, y1, x2, y2 = zone_list[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append(
            DetectedObject(24, "backpack", 0.88, (cx - 30, cy - 30, cx + 30, cy + 30), (cx, cy))
        )
    return detections


def _simulate_aisle_walkers(frame_index: int) -> list[DetectedObject]:
    step = (frame_index * 14) % 420
    walker_one_y = 180 + step
    walker_two_y = 580 - step
    return [
        DetectedObject(
            0, "person", 0.92, (210, walker_one_y - 50, 290, walker_one_y + 50), (250, walker_one_y)
        ),
        DetectedObject(
            0, "person", 0.89, (600, walker_two_y - 50, 680, walker_two_y + 50), (640, walker_two_y)
        ),
    ]
