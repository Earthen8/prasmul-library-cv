from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
from src.detector import DetectedObject


PolygonPoints = list[tuple[int, int]]
DEFAULT_EMPTY_REF_PATH = Path("config/empty_reference.jpg")
DEFAULT_HSV_DIFF_RATIO = 0.35


@dataclass
class TableZoneStatus:
    zone_name: str
    polygon: PolygonPoints
    bounds: tuple[int, int, int, int]
    has_person: bool
    has_belonging: bool
    idle_frames: int
    idle_seconds: float
    is_occupied: bool
    is_abandoned: bool
    detected_items: list[str]
    detection_source: str = "NONE"

    @property
    def status_label(self) -> str:
        if self.is_abandoned:
            return f"TERTINGGAL ({self.idle_seconds:.0f}s)"
        if self.has_belonging and not self.has_person:
            return f"IDLE ({self.idle_seconds:.0f}s)"
        if self.is_occupied and self.detection_source == "HSV_FALLBACK":
            return "TERISI (HSV)"
        if self.is_occupied:
            return "TERISI"
        return "KOSONG"


def coerce_to_polygon(coords: object, zone_name: str) -> PolygonPoints:
    if not isinstance(coords, list) or len(coords) < 3:
        raise ValueError(f"Zone '{zone_name}' must be [x1,y1,x2,y2] or >=3 polygon points.")
    if len(coords) == 4 and all(isinstance(val, (int, float)) for val in coords):
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Zone '{zone_name}' has invalid box ({x1},{y1},{x2},{y2}).")
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    polygon: PolygonPoints = []
    for pt in coords:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"Invalid vertex {pt} in zone '{zone_name}'.")
        polygon.append((int(pt[0]), int(pt[1])))
    return polygon


def polygon_bounding_box(polygon: PolygonPoints) -> tuple[int, int, int, int]:
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def load_zones_with_metadata(
    json_path: Path, profile_name: str = "Mezanin_1"
) -> tuple[dict[str, PolygonPoints], tuple[int, int]]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Table zone file '{json_path}' not found. Run scripts/select_roi.py first."
        )
    try:
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON syntax in '{json_path}': {error}.") from error

    if not isinstance(raw_data, dict):
        raise ValueError("Table zones JSON root must be an object.")

    root_ref = raw_data.get("reference_resolution", [1024, 604])
    reference_size = (int(root_ref[0]), int(root_ref[1]))
    profiles = raw_data.get("profiles")
    if isinstance(profiles, dict):
        selected_profile = profiles.get(profile_name, next(iter(profiles.values()), {}))
        if isinstance(selected_profile, dict) and "zones" in selected_profile:
            prof_ref = selected_profile.get("reference_resolution", root_ref)
            reference_size = (int(prof_ref[0]), int(prof_ref[1]))
            selected = selected_profile["zones"]
        else:
            selected = selected_profile
    else:
        selected = {k: v for k, v in raw_data.items() if k != "reference_resolution"}

    zones = {str(name): coerce_to_polygon(coords, str(name)) for name, coords in selected.items()}
    return zones, reference_size


def load_table_zones(
    json_path: Path, profile_name: str = "Mezanin_1"
) -> dict[str, PolygonPoints]:
    zones, _ = load_zones_with_metadata(json_path, profile_name=profile_name)
    return zones


def load_profile_section_with_metadata(
    json_path: Path, profile_name: str, section_key: str
) -> tuple[dict[str, PolygonPoints], tuple[int, int]]:
    if not json_path.exists():
        return {}, (1280, 720)
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    root_ref = raw_data.get("reference_resolution", [1280, 720])
    profiles = raw_data.get("profiles", {})
    selected_profile = profiles.get(profile_name, {})
    if not isinstance(selected_profile, dict) or section_key not in selected_profile:
        return {}, (int(root_ref[0]), int(root_ref[1]))

    prof_ref = selected_profile.get("reference_resolution", root_ref)
    reference_size = (int(prof_ref[0]), int(prof_ref[1]))
    section_raw = selected_profile[section_key]
    section_zones = {
        str(name): coerce_to_polygon(coords, str(name)) for name, coords in section_raw.items()
    }
    return section_zones, reference_size


def load_aisle_zones_with_metadata(
    json_path: Path, profile_name: str = "Rak_1"
) -> tuple[dict[str, PolygonPoints], tuple[int, int]]:
    return load_profile_section_with_metadata(json_path, profile_name, "aisle_zones")


def load_ignore_zones_with_metadata(
    json_path: Path, profile_name: str = "Rak_1"
) -> tuple[dict[str, PolygonPoints], tuple[int, int]]:
    return load_profile_section_with_metadata(json_path, profile_name, "ignore_zones")


def scale_zones_to_frame(
    zones: dict[str, PolygonPoints],
    reference_resolution: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> dict[str, PolygonPoints]:
    ref_w, ref_h = reference_resolution
    if ref_w <= 0 or ref_h <= 0 or (ref_w == frame_width and ref_h == frame_height):
        return zones
    scale_x = frame_width / float(ref_w)
    scale_y = frame_height / float(ref_h)
    return {
        name: [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in poly]
        for name, poly in zones.items()
    }


def is_point_inside_zone(point: tuple[int, int], zone_shape: PolygonPoints | tuple[int, int, int, int]) -> bool:
    polygon = coerce_to_polygon(list(zone_shape), "zone") if isinstance(zone_shape, tuple) else zone_shape
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0.0


def is_object_in_zone(
    obj: DetectedObject,
    polygon: PolygonPoints,
    min_overlap_points: int = 2,
) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    if cv2.pointPolygonTest(contour, (float(obj.center[0]), float(obj.center[1])), False) >= 0.0:
        return True

    x1, y1, x2, y2 = obj.bbox
    xs = (x1 + (x2 - x1) // 4, (x1 + x2) // 2, x2 - (x2 - x1) // 4)
    ys = (y1 + (y2 - y1) // 4, (y1 + y2) // 2, y2 - (y2 - y1) // 4)
    hits = sum(
        1
        for px in xs
        for py in ys
        if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0.0
    )
    return hits >= min_overlap_points


def update_zone_frame_timer(
    zone_name: str,
    has_person: bool,
    has_belonging: bool,
    current_frame_index: int,
    video_fps: float,
    idle_start_frames: dict[str, int],
) -> tuple[int, float]:
    if has_person or not has_belonging:
        idle_start_frames.pop(zone_name, None)
        return (0, 0.0)
    if zone_name not in idle_start_frames:
        idle_start_frames[zone_name] = current_frame_index
        return (0, 0.0)
    elapsed_frames = max(0, current_frame_index - idle_start_frames[zone_name])
    safe_fps = max(1.0, float(video_fps))
    return (elapsed_frames, elapsed_frames / safe_fps)


def load_empty_reference_frame(
    reference_path: Path, target_width: int, target_height: int
) -> np.ndarray | None:
    if not reference_path.exists():
        return None
    image = cv2.imread(str(reference_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if w != target_width or h != target_height:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return image


def save_empty_reference_frame(frame: np.ndarray, reference_path: Path) -> Path:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot save an empty reference frame.")
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_path), frame)
    return reference_path


def compute_hsv_foreground_ratio(
    current_frame: np.ndarray,
    reference_frame: np.ndarray,
    polygon: PolygonPoints,
    hue_threshold: int = 22,
    sat_threshold: int = 45,
) -> float:
    frame_h, frame_w = current_frame.shape[:2]
    x1, y1, x2, y2 = polygon_bounding_box(polygon)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    ref_resized = (
        reference_frame
        if reference_frame.shape[:2] == (frame_h, frame_w)
        else cv2.resize(reference_frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    )
    curr_crop = cv2.cvtColor(current_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    ref_crop = cv2.cvtColor(ref_resized[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

    shifted_poly = np.array([(px - x1, py - y1) for px, py in polygon], dtype=np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_poly], 255)
    total_pixels = int(np.count_nonzero(mask))
    if total_pixels == 0:
        return 0.0

    # Compare only Hue and Saturation channels to stay resilient against window sunlight glare (Value).
    raw_h_diff = cv2.absdiff(curr_crop[:, :, 0], ref_crop[:, :, 0])
    hue_diff = np.minimum(raw_h_diff, 180 - raw_h_diff)
    sat_diff = cv2.absdiff(curr_crop[:, :, 1], ref_crop[:, :, 1])
    changed = ((hue_diff > hue_threshold) | (sat_diff > sat_threshold)) & (mask > 0)
    return float(np.count_nonzero(changed)) / float(total_pixels)


def evaluate_single_zone(
    zone_name: str,
    polygon: PolygonPoints,
    detections: list[DetectedObject],
    current_frame_index: int,
    video_fps: float,
    idle_threshold_seconds: float,
    idle_start_frames: dict[str, int],
    current_frame: np.ndarray | None = None,
    empty_reference: np.ndarray | None = None,
    hsv_diff_threshold: float = DEFAULT_HSV_DIFF_RATIO,
) -> TableZoneStatus:
    inside_objects = [
        obj for obj in detections if is_object_in_zone(obj, polygon)
    ]
    has_person = any(obj.is_person for obj in inside_objects)
    has_belonging = any(obj.is_belonging for obj in inside_objects)
    idle_frames, idle_seconds = update_zone_frame_timer(
        zone_name, has_person, has_belonging, current_frame_index, video_fps, idle_start_frames
    )

    is_occupied = has_person or has_belonging
    is_abandoned = (not has_person) and has_belonging and (idle_seconds >= idle_threshold_seconds)
    detection_source = "YOLO" if is_occupied else "NONE"
    items = [obj.label for obj in inside_objects]

    if not is_occupied and current_frame is not None and empty_reference is not None:
        fg_ratio = compute_hsv_foreground_ratio(current_frame, empty_reference, polygon)
        if fg_ratio > hsv_diff_threshold:
            is_occupied = True
            detection_source = "HSV_FALLBACK"
            items = [f"hsv_diff({fg_ratio:.0%})"]

    return TableZoneStatus(
        zone_name=zone_name,
        polygon=polygon,
        bounds=polygon_bounding_box(polygon),
        has_person=has_person,
        has_belonging=has_belonging,
        idle_frames=idle_frames,
        idle_seconds=idle_seconds,
        is_occupied=is_occupied,
        is_abandoned=is_abandoned,
        detected_items=items,
        detection_source=detection_source,
    )


def evaluate_all_table_zones(
    zones: dict[str, PolygonPoints],
    detections: list[DetectedObject],
    current_frame_index: int,
    video_fps: float,
    idle_threshold_seconds: float,
    idle_start_frames: dict[str, int],
    current_frame: np.ndarray | None = None,
    empty_reference: np.ndarray | None = None,
    hsv_diff_threshold: float = DEFAULT_HSV_DIFF_RATIO,
) -> list[TableZoneStatus]:
    return [
        evaluate_single_zone(
            zone_name=name,
            polygon=coerce_to_polygon(list(poly), name) if isinstance(poly, tuple) else poly,
            detections=detections,
            current_frame_index=current_frame_index,
            video_fps=video_fps,
            idle_threshold_seconds=idle_threshold_seconds,
            idle_start_frames=idle_start_frames,
            current_frame=current_frame,
            empty_reference=empty_reference,
            hsv_diff_threshold=hsv_diff_threshold,
        )
        for name, poly in zones.items()
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
    monitored_polys = [status.polygon for status in statuses]
    for obj in detections:
        if obj.is_belonging and monitored_polys:
            if not any(is_object_in_zone(obj, poly) for poly in monitored_polys):
                continue
        x1, y1, x2, y2 = obj.bbox
        color = (210, 110, 30) if obj.is_person else (20, 140, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, obj.center, 4, color, -1)

    for status in statuses:
        pts = np.array(status.polygon, dtype=np.int32).reshape((-1, 1, 2))
        box_color = select_zone_bgr_color(status)
        cv2.polylines(annotated, [pts], isClosed=True, color=box_color, thickness=2)
        x1, y1, _, _ = status.bounds
        banner_text = f"{status.zone_name}: {status.status_label}"
        (text_w, text_h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y = max(text_h + 10, y1)
        cv2.rectangle(annotated, (x1, label_y - text_h - 8), (x1 + text_w + 10, label_y), box_color, -1)
        cv2.putText(
            annotated, banner_text, (x1 + 5, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
        )
    return annotated


def filter_ignored_detections(
    detections: list[DetectedObject], ignore_zones: dict[str, PolygonPoints] | None
) -> list[DetectedObject]:
    if not ignore_zones:
        return detections
    polys = list(ignore_zones.values())
    return [
        obj
        for obj in detections
        if not any(is_point_inside_zone(obj.center, poly) for poly in polys)
    ]


def draw_ignore_zones_overlay(
    frame: np.ndarray, ignore_zones: dict[str, PolygonPoints] | None
) -> np.ndarray:
    if not ignore_zones:
        return frame
    annotated = frame.copy()
    for zone_name, poly in ignore_zones.items():
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], (40, 40, 140))
        annotated = cv2.addWeighted(annotated, 0.80, overlay, 0.20, 0)
        cv2.polylines(annotated, [pts], isClosed=True, color=(50, 50, 220), thickness=2)
        anchor = (pts[0][0][0] + 8, max(24, pts[0][0][1] + 22))
        cv2.putText(
            annotated,
            f"DIABAIKAN: {zone_name}",
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 180, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
