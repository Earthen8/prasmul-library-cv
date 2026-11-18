from pathlib import Path
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from scripts.extract_media import extract_frames_from_video, trim_video_clip
from src.detector import (
    DetectedObject,
    TARGET_CLASS_IDS,
    apply_clahe_lighting,
    letterbox_resize,
    parse_roboflow_prediction,
    unscale_letterbox_coords,
)
from src.seat_monitor import (
    draw_seat_monitoring_overlay,
    evaluate_all_table_zones,
    filter_ignored_detections,
    load_aisle_zones_with_metadata,
    load_ignore_zones_with_metadata,
    load_zones_with_metadata,
    scale_zones_to_frame,
)
from src.traffic_heatmap import TrafficHeatmapAccumulator


def verify_letterbox_and_clahe() -> None:
    dummy = np.zeros((604, 1024, 3), dtype=np.uint8)
    canvas, scale, pad_w, pad_h = letterbox_resize(dummy, 640)
    assert canvas.shape == (640, 640, 3)
    unscaled = unscale_letterbox_coords(
        np.array([pad_w, pad_h, 640 - pad_w, 640 - pad_h]),
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        orig_w=1024,
        orig_h=604,
    )
    assert unscaled == (0, 0, 1024, 604), f"Unexpected unscaled box: {unscaled}"
    enhanced = apply_clahe_lighting(dummy)
    assert enhanced.shape == dummy.shape
    print("[PASS] Letterbox 640x640 unscaling & CLAHE backlight normalization verified.")


def verify_per_profile_roi_scaling() -> None:
    assert set(TARGET_CLASS_IDS) == {0, 24, 26, 63, 73}
    zones_m1, ref_m1 = load_zones_with_metadata(Path("config/table_zones.json"), "Mezanin_1")
    zones_r2, ref_r2 = load_zones_with_metadata(Path("config/table_zones.json"), "Rak_2")
    zones_r1, ref_r1 = load_zones_with_metadata(Path("config/table_zones.json"), "Rak_1")
    aisles_r1, aisle_ref_r1 = load_aisle_zones_with_metadata(Path("config/table_zones.json"), "Rak_1")
    ignores_r1, _ = load_ignore_zones_with_metadata(Path("config/table_zones.json"), "Rak_1")
    assert ref_m1 == (1200, 720) and len(zones_m1) == 4
    assert ref_r2 == (1280, 720) and len(zones_r2) >= 1
    assert ref_r1 == (1280, 720) and len(zones_r1) == 3
    assert aisle_ref_r1 == (1280, 720) and len(aisles_r1) >= 1
    assert len(ignores_r1) >= 1
    ignored_person = DetectedObject(0, "person", 0.9, (1100, 30, 1180, 120), (1140, 75))
    valid_person = DetectedObject(0, "person", 0.9, (600, 400, 660, 520), (630, 460))
    filtered = filter_ignored_detections([ignored_person, valid_person], ignores_r1)
    assert filtered == [valid_person]

    scaled_hd = scale_zones_to_frame(zones_r2, ref_r2, 2560, 1440)
    first_r2_key = next(iter(zones_r2))
    orig_pt = zones_r2[first_r2_key][0]
    assert scaled_hd[first_r2_key][0] == (orig_pt[0] * 2, orig_pt[1] * 2)
    print("[PASS] Per-profile reference resolution & polygon scaling verified for Mezanin_1, Rak_2, Rak_1.")


def verify_frame_skip_and_bookshelf_filter() -> None:
    zones = {"Meja_Uji": [(100, 100), (300, 100), (300, 300), (100, 300)]}
    bag_in_zone = DetectedObject(24, "backpack", 0.9, (180, 180, 220, 220), (200, 200))
    shelf_book = DetectedObject(73, "book", 0.85, (500, 100, 540, 140), (520, 120))
    timers: dict[str, int] = {}

    status_f1 = evaluate_all_table_zones(zones, [bag_in_zone, shelf_book], 1, 15.0, 3.0, timers)[0]
    assert status_f1.detected_items == ["backpack"]
    status_f46 = evaluate_all_table_zones(zones, [bag_in_zone, shelf_book], 46, 15.0, 3.0, timers)[0]
    assert status_f46.is_abandoned and abs(status_f46.idle_seconds - 3.0) < 1e-6

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    overlay = draw_seat_monitoring_overlay(frame, [status_f46], [bag_in_zone, shelf_book])
    # Shelf book outside monitored polygon must remain unannotated (black pixels)
    assert np.all(overlay[100, 520] == 0)
    print("[PASS] Frame-skip idle timer & bookshelf book overlay filtering verified.")


def verify_hsv_fallback_detection() -> None:
    zones = {"Meja_01": [(50, 50), (250, 50), (250, 250), (50, 250)]}
    # Reference frame: bright yellow sofa (BGR: 0, 210, 230) inside Meja_01
    empty_ref = np.full((300, 300, 3), (0, 210, 230), dtype=np.uint8)

    # Case 1: Only sunlight glare (Value channel shift, same Hue/Saturation) -> must stay KOSONG
    glare_frame = np.full((300, 300, 3), (0, 255, 255), dtype=np.uint8)
    glare_status = evaluate_all_table_zones(
        zones, [], 1, 15.0, 3.0, {}, current_frame=glare_frame, empty_reference=empty_ref
    )[0]
    assert not glare_status.is_occupied and glare_status.detection_source == "NONE"

    # Case 2: YOLO misses person (detections=[]), but dark blue shirt covers >25% of sofa
    occupied_frame = glare_frame.copy()
    occupied_frame[90:210, 90:210] = (140, 40, 30)
    fallback_status = evaluate_all_table_zones(
        zones, [], 2, 15.0, 3.0, {}, current_frame=occupied_frame, empty_reference=empty_ref
    )[0]
    assert fallback_status.is_occupied is True
    assert fallback_status.detection_source == "HSV_FALLBACK"
    assert fallback_status.status_label == "TERISI (HSV)"
    print("[PASS] Hybrid HSV fallback verified: glare ignored, seated occlusion -> TERISI (HSV).")


def verify_heatmap_floor_point_and_decay() -> None:
    acc = TrafficHeatmapAccumulator(575, 1024, decay_factor=0.90, min_movement_px=15.0)
    person = DetectedObject(0, "person", 0.9, (300, 200, 360, 400), (330, 300))
    assert person.floor_point[1] > person.center[1]

    first = acc.record_persons([person])
    peak_first = float(acc.grid.max())
    second = acc.record_persons([person])
    peak_second = float(acc.grid.max())
    assert first.moving_person_count == 1 and second.moving_person_count == 0
    assert peak_second < peak_first

    seated_zones = {"Meja_01": [(50, 50), (200, 50), (200, 400), (50, 400)]}
    seated_person = DetectedObject(0, "person", 0.9, (100, 200, 150, 320), (125, 260))
    excluded = acc.record_persons([person, seated_person], excluded_zones=seated_zones)
    assert excluded.active_person_count == 1
    print("[PASS] Floor-point heatmap accumulation, motion filter & decay verified.")


def verify_streamlit_ui_interactions() -> None:
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial load exception: {at.exception}"

    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Mezanin_1 simulation exception: {at.exception}"
    statuses = at.session_state["table_statuses"]
    assert len(statuses) == 4
    print(f"[PASS] Mezanin_1 UI simulation: {[s.status_label for s in statuses]}.")

    at.selectbox[0].select("CCTV02 - Rak 2 (Seat Occupancy & Abandoned Object)").run()
    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Rak_2 simulation exception: {at.exception}"
    assert len(at.session_state["table_statuses"]) >= 1

    at.selectbox[0].select("CCTV01 - Rak 1 (Meja Kiri + Heatmap Rak)").run()
    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Rak_1 simulation exception: {at.exception}"
    assert len(at.session_state["table_statuses"]) >= 1
    assert at.session_state["aisle_summary"] is not None
    assert "Baris_Rak_01" in at.session_state["aisle_summary"].sector_counts
    print("[PASS] Rak_2 seat monitoring & Rak_1 row-by-row heatmap UI simulation verified.")


def verify_media_extraction_scripts() -> None:
    sample_video = Path("demo/sample_cctv.mp4")
    frames_dir = Path("demo/test_screenshots")
    assert extract_frames_from_video(sample_video, frames_dir, "Senin", "Meja", 3) == 3
    clip_path = Path("demo/test_clip_3s.mp4")
    assert trim_video_clip(sample_video, clip_path, "00:01", 2) > 0
    print("[PASS] Media extraction and clip trimming verified.")


def verify_roboflow_prediction_parsing() -> None:
    person_pred = {"x": 200.0, "y": 300.0, "width": 80.0, "height": 120.0, "confidence": 0.91, "class": "0", "class_id": 0}
    bag_pred = {"x": 500.0, "y": 400.0, "width": 40.0, "height": 40.0, "confidence": 0.78, "class": "1", "class_id": 1}
    person_obj = parse_roboflow_prediction(person_pred, 1920, 1080)
    bag_obj = parse_roboflow_prediction(bag_pred, 1920, 1080)
    assert person_obj.is_person and person_obj.label == "person"
    assert person_obj.bbox == (160, 240, 240, 360)
    assert bag_obj.is_belonging and bag_obj.label == "belonging"
    assert bag_obj.bbox == (480, 380, 520, 420)
    print("[PASS] Roboflow RF-DETR NAS prediction parsing verified.")


if __name__ == "__main__":
    verify_letterbox_and_clahe()
    verify_per_profile_roi_scaling()
    verify_frame_skip_and_bookshelf_filter()
    verify_hsv_fallback_detection()
    verify_heatmap_floor_point_and_decay()
    verify_streamlit_ui_interactions()
    verify_media_extraction_scripts()
    verify_roboflow_prediction_parsing()
