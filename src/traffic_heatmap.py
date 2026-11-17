from dataclasses import dataclass
import math
import cv2
import numpy as np
from src.detector import DetectedObject


@dataclass(frozen=True)
class AisleTrafficSummary:
    active_person_count: int
    moving_person_count: int
    cumulative_detections: int
    sector_counts: dict[str, int]
    peak_sector_name: str


def has_significant_movement(
    center: tuple[int, int],
    previous_centers: list[tuple[int, int]],
    min_movement_px: float,
) -> bool:
    if not previous_centers:
        return True
    nearest_distance = min(
        math.hypot(center[0] - prev[0], center[1] - prev[1]) for prev in previous_centers
    )
    return nearest_distance >= min_movement_px


def is_point_in_any_polygon(
    point: tuple[int, int], polygons: dict[str, list[tuple[int, int]]] | None
) -> bool:
    if not polygons:
        return False
    for pts in polygons.values():
        contour = np.array(pts, dtype=np.int32)
        if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
            return True
    return False


class TrafficHeatmapAccumulator:
    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        decay_factor: float = 0.94,
        min_movement_px: float = 12.0,
    ) -> None:
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(
                f"Invalid frame dimensions ({frame_width}x{frame_height}) for heatmap."
            )
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.decay_factor = float(np.clip(decay_factor, 0.50, 0.999))
        self.min_movement_px = max(0.0, float(min_movement_px))
        self.grid = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.previous_centers: list[tuple[int, int]] = []
        self.cumulative_detections = 0
        self.sector_totals = {
            "Lorong Kiri": 0,
            "Lorong Tengah": 0,
            "Lorong Kanan": 0,
        }

    def record_persons(
        self,
        detections: list[DetectedObject],
        excluded_zones: dict[str, list[tuple[int, int]]] | None = None,
    ) -> AisleTrafficSummary:
        self.grid *= self.decay_factor
        persons = [
            obj
            for obj in detections
            if obj.is_person and not is_point_in_any_polygon(obj.floor_point, excluded_zones)
        ]
        current_centers: list[tuple[int, int]] = []
        moving_count = 0

        for person in persons:
            fx, fy = person.floor_point
            if not (0 <= fx < self.frame_width and 0 <= fy < self.frame_height):
                continue
            current_centers.append((fx, fy))
            if has_significant_movement((fx, fy), self.previous_centers, self.min_movement_px):
                cv2.circle(self.grid, (fx, fy), 28, 1.0, -1)
                self._increment_sector(fx)
                self.cumulative_detections += 1
                moving_count += 1

        self.previous_centers = current_centers
        peak_sector = max(self.sector_totals, key=lambda key: self.sector_totals[key])
        if self.cumulative_detections == 0:
            peak_sector = "Belum ada trafik"
        return AisleTrafficSummary(
            active_person_count=len(persons),
            moving_person_count=moving_count,
            cumulative_detections=self.cumulative_detections,
            sector_counts=dict(self.sector_totals),
            peak_sector_name=peak_sector,
        )

    def _increment_sector(self, cx: int) -> None:
        third = self.frame_width / 3.0
        if cx < third:
            self.sector_totals["Lorong Kiri"] += 1
        elif cx < 2 * third:
            self.sector_totals["Lorong Tengah"] += 1
        else:
            self.sector_totals["Lorong Kanan"] += 1


def render_heatmap_overlay(
    frame: np.ndarray,
    accumulator: TrafficHeatmapAccumulator,
    detections: list[DetectedObject],
    blend_alpha: float = 0.45,
) -> np.ndarray:
    annotated = frame.copy()
    max_value = float(np.max(accumulator.grid))
    if max_value > 0.05:
        blurred = cv2.GaussianBlur(accumulator.grid, (31, 31), 0)
        peak = max(1.0, float(np.max(blurred)))
        normalized = np.clip((blurred / peak) * 255.0, 0, 255).astype(np.uint8)
        colored_heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        active_mask = normalized > 15
        blended = cv2.addWeighted(
            annotated, 1.0 - blend_alpha, colored_heat, blend_alpha, 0
        )
        annotated[active_mask] = blended[active_mask]

    for obj in detections:
        if not obj.is_person:
            continue
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 60), 2)
        cv2.circle(annotated, obj.floor_point, 5, (30, 30, 240), -1)
    return annotated
