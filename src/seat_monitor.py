from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
from src.detector import DetectedObject


@dataclass
class TableZoneStatus:
    zone_name: str
    bounds: tuple[int, int, int, int]
    has_person: bool
    has_belonging: bool
    idle_seconds: float
    is_occupied: bool
    is_abandoned: bool
    detected_items: list[str]

    @property
    def status_label(self) -> str:
        if self.is_abandoned:
            return f"TERTINGGAL ({self.idle_seconds:.0f}s)"
        if self.has_belonging and not self.has_person:
            return f"IDLE ({self.idle_seconds:.0f}s)"
        if self.is_occupied:
            return "TERISI"
        return "KOSONG"


def load_table_zones(json_path: Path) -> dict[str, tuple[int, int, int, int]]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Table zone file '{json_path}' not found. Run scripts/select_roi.py or provide a valid JSON file."
        )
    try:
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON syntax in '{json_path}': {error}. Ensure coordinates follow [x1, y1, x2, y2]."
        ) from error

    return validate_zone_dictionary(raw_data)


def validate_zone_dictionary(raw_data: object) -> dict[str, tuple[int, int, int, int]]:
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError("Table zones JSON must be a non-empty object mapping names to [x1, y1, x2, y2].")

    validated: dict[str, tuple[int, int, int, int]] = {}
    for name, coords in raw_data.items():
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"Zone '{name}' must contain exactly 4 integers: [x1, y1, x2, y2].")
        x1, y1, x2, y2 = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Zone '{name}' has invalid dimensions ({x1}, {y1}, {x2}, {y2}).")
        validated[str(name)] = (x1, y1, x2, y2)
    return validated


def is_point_inside_zone(point: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    cx, cy = point
    x1, y1, x2, y2 = bounds
    return x1 <= cx <= x2 and y1 <= cy <= y2


def update_zone_timer(
    zone_name: str,
    has_person: bool,
    has_belonging: bool,
    current_video_time: float,
    idle_start_times: dict[str, float],
) -> float:
    if has_person or not has_belonging:
        idle_start_times.pop(zone_name, None)
        return 0.0
    if zone_name not in idle_start_times:
        idle_start_times[zone_name] = current_video_time
        return 0.0
    return max(0.0, current_video_time - idle_start_times[zone_name])


def evaluate_single_zone(
    zone_name: str,
    bounds: tuple[int, int, int, int],
    detections: list[DetectedObject],
    current_video_time: float,
    idle_threshold_seconds: float,
    idle_start_times: dict[str, float],
) -> TableZoneStatus:
    inside_objects = [
        obj for obj in detections if is_point_inside_zone(obj.center, bounds)
    ]
    has_person = any(obj.is_person for obj in inside_objects)
    has_belonging = any(obj.is_belonging for obj in inside_objects)
    idle_seconds = update_zone_timer(
        zone_name, has_person, has_belonging, current_video_time, idle_start_times
    )
    is_occupied = has_person or has_belonging
    is_abandoned = (not has_person) and has_belonging and (idle_seconds >= idle_threshold_seconds)
    items = [obj.label for obj in inside_objects]
    return TableZoneStatus(
        zone_name=zone_name,
        bounds=bounds,
        has_person=has_person,
        has_belonging=has_belonging,
        idle_seconds=idle_seconds,
        is_occupied=is_occupied,
        is_abandoned=is_abandoned,
        detected_items=items,
    )


def evaluate_all_table_zones(
    zones: dict[str, tuple[int, int, int, int]],
    detections: list[DetectedObject],
    current_video_time: float,
    idle_threshold_seconds: float,
    idle_start_times: dict[str, float],
) -> list[TableZoneStatus]:
    return [
        evaluate_single_zone(
            zone_name=name,
            bounds=bounds,
            detections=detections,
            current_video_time=current_video_time,
            idle_threshold_seconds=idle_threshold_seconds,
            idle_start_times=idle_start_times,
        )
        for name, bounds in zones.items()
    ]


def select_zone_bgr_color(status: TableZoneStatus) -> tuple[int, int, int]:
    if status.is_abandoned:
        return (36, 36, 220)
    if status.has_belonging and not status.has_person:
        return (0, 140, 230)
    if status.is_occupied:
        return (160, 90, 30)
    return (75, 150, 45)


def draw_seat_monitoring_overlay(
    frame: np.ndarray,
    statuses: list[TableZoneStatus],
    detections: list[DetectedObject],
) -> np.ndarray:
    annotated = frame.copy()
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox
        color = (210, 110, 30) if obj.is_person else (20, 140, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, obj.center, 4, color, -1)

    for status in statuses:
        x1, y1, x2, y2 = status.bounds
        box_color = select_zone_bgr_color(status)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        banner_text = f"{status.zone_name}: {status.status_label}"
        (text_w, text_h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(text_h + 10, y1)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 12, label_y),
            box_color,
            -1,
        )
        cv2.putText(
            annotated,
            banner_text,
            (x1 + 6, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return annotated
