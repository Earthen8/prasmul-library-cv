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
    unscale_letterbox_coords,
)
from src.seat_monitor import (
    draw_seat_monitoring_overlay,
    evaluate_all_table_zones,
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
    assert ref_m1 == (1024, 604) and len(zones_m1) == 4
    assert ref_r2 == (1024, 575) and len(zones_r2) == 1
    assert ref_r1 == (1024, 574) and len(zones_r1) == 3

    scaled_hd = scale_zones_to_frame(zones_r2, ref_r2, 2048, 1150)
    orig_pt = zones_r2["Meja_Baca_Belakang"][0]
    assert scaled_hd["Meja_Baca_Belakang"][0] == (orig_pt[0] * 2, orig_pt[1] * 2)
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

    at.selectbox[0].select("Kamera Rak 2 (Heatmap Lorong + Meja Belakang)").run()
    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Rak_2 simulation exception: {at.exception}"
    assert len(at.session_state["table_statuses"]) == 1
    assert at.session_state["aisle_summary"] is not None
    print("[PASS] Rak_2 hybrid UI simulation verified.")


def verify_media_extraction_scripts() -> None:
    sample_video = Path("demo/sample_cctv.mp4")
    frames_dir = Path("demo/test_screenshots")
    assert extract_frames_from_video(sample_video, frames_dir, "Senin", "Meja", 3) == 3
    clip_path = Path("demo/test_clip_3s.mp4")
    assert trim_video_clip(sample_video, clip_path, "00:01", 2) > 0
    print("[PASS] Media extraction and clip trimming verified.")


if __name__ == "__main__":
    verify_letterbox_and_clahe()
    verify_per_profile_roi_scaling()
    verify_frame_skip_and_bookshelf_filter()
    verify_heatmap_floor_point_and_decay()
    verify_streamlit_ui_interactions()
    verify_media_extraction_scripts()
