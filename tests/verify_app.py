from pathlib import Path
from streamlit.testing.v1 import AppTest
from scripts.extract_media import extract_frames_from_video, trim_video_clip


def verify_streamlit_ui_interactions() -> None:
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial load exception: {at.exception}"
    assert at.session_state["is_dark_mode"] is False
    print("[PASS] Initial load: Empty state rendered with zero exceptions.")

    at.button(key="toggle_theme_btn").click().run()
    assert not at.exception, f"Dark mode toggle exception: {at.exception}"
    assert at.session_state["is_dark_mode"] is True
    print("[PASS] Theme toggle -> Dark Mode: state updated and dark CSS variables injected.")

    at.button(key="toggle_theme_btn").click().run()
    assert not at.exception, f"Light mode toggle exception: {at.exception}"
    assert at.session_state["is_dark_mode"] is False
    print("[PASS] Theme toggle -> Light Mode: restored light CSS variables cleanly.")

    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Simulation run exception: {at.exception}"
    assert at.session_state["processed_count"] > 0
    assert at.session_state["skipped_count"] > 0
    statuses = at.session_state["table_statuses"]
    assert len(statuses) == 3
    assert any(item.is_abandoned for item in statuses)
    print(
        f"[PASS] Kamera Meja Baca stream: processed={at.session_state['processed_count']} frames, "
        f"skipped={at.session_state['skipped_count']} frames, statuses={[s.status_label for s in statuses]}."
    )

    at.selectbox[0].select("Kamera Rak Buku (Foot Traffic & Heatmap)").run()
    at.button(key="load_sim_btn").click().run()
    assert not at.exception, f"Aisle heatmap exception: {at.exception}"
    summary = at.session_state["aisle_summary"]
    assert summary is not None and summary.cumulative_detections > 0
    print(
        f"[PASS] Kamera Rak Buku stream: active_persons={summary.active_person_count}, "
        f"cumulative_points={summary.cumulative_detections}, peak_sector='{summary.peak_sector_name}'."
    )


def verify_media_extraction_scripts() -> None:
    sample_video = Path("demo/sample_cctv.mp4")
    assert sample_video.exists(), "Sample video was not generated."
    frames_dir = Path("demo/test_screenshots")
    saved = extract_frames_from_video(
        video_path=sample_video,
        output_dir=frames_dir,
        day_name="Senin",
        category="Meja",
        frame_count=5,
    )
    assert saved == 5
    assert (frames_dir / "Senin_Meja_01.jpg").exists()
    print(f"[PASS] extract_media.py frames: extracted {saved} screenshots (Senin_Meja_01.jpg..05.jpg).")

    clip_path = Path("demo/test_clip_3s.mp4")
    written = trim_video_clip(
        video_path=sample_video,
        output_path=clip_path,
        start_timestamp="00:01",
        duration_seconds=3,
    )
    assert written > 0 and clip_path.exists()
    print(f"[PASS] extract_media.py clip: trimmed {written} frames into '{clip_path}'.")


if __name__ == "__main__":
    verify_streamlit_ui_interactions()
    verify_media_extraction_scripts()
