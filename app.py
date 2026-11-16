from pathlib import Path
import tempfile
import time
import cv2
import streamlit as st

from src.demo_generator import build_simulated_detections, generate_sample_cctv_video
from src.detector import detect_library_objects, load_yolo_model
from src.seat_monitor import (
    PolygonPoints,
    TableZoneStatus,
    draw_seat_monitoring_overlay,
    evaluate_all_table_zones,
    load_zones_with_metadata,
    scale_zones_to_frame,
)
from src.traffic_heatmap import (
    AisleTrafficSummary,
    TrafficHeatmapAccumulator,
    render_heatmap_overlay,
)
from src.ui_styles import apply_custom_styles


TABLE_MODE = "Kamera Meja Baca (Seat Occupancy & Abandoned Object)"
AISLE_MODE = "Kamera Rak Buku (Foot Traffic & Heatmap)"
HYBRID_RAK2_MODE = "Kamera Rak 2 (Heatmap Lorong + Meja Belakang)"
MODE_TO_PROFILE = {
    TABLE_MODE: "Mezanin_1",
    AISLE_MODE: "Rak_1",
    HYBRID_RAK2_MODE: "Rak_2",
}
DEFAULT_ZONES_PATH = Path("config/table_zones.json")
SAMPLE_VIDEO_PATH = Path("demo/sample_cctv.mp4")


@st.cache_resource(show_spinner=False)
def get_cached_yolo_model(weights_path: str):
    return load_yolo_model(weights_path)


def init_session_state() -> None:
    defaults = {
        "is_dark_mode": False,
        "is_streaming": False,
        "video_path": None,
        "is_simulation_clip": False,
        "table_statuses": [],
        "aisle_summary": None,
        "last_frame_rgb": None,
        "processed_count": 0,
        "skipped_count": 0,
        "error_message": None,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def render_header() -> None:
    col_title, col_theme = st.columns([3, 1])
    with col_title:
        st.markdown("### [LOGO] Pemantauan CCTV Perpustakaan Prasetiya Mulya")
        st.caption(
            "Status desain: Draft without direction (Dial: ENERGY 1 / RHYTHM 1 / MOTION 1). "
            "Semua metrik dihitung langsung dari frame video dan skala resolusi ROI."
        )
    with col_theme:
        theme_label = (
            "Ubah ke Mode Terang" if st.session_state.is_dark_mode else "Ubah ke Mode Gelap"
        )
        if st.button(theme_label, key="toggle_theme_btn", width="stretch"):
            st.session_state.is_dark_mode = not st.session_state.is_dark_mode
            st.rerun()


def handle_uploaded_video(uploaded_file) -> None:
    if uploaded_file is None:
        return
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.getbuffer())
        st.session_state.video_path = Path(temp_video.name)
        st.session_state.is_simulation_clip = False
        st.session_state.error_message = None


def activate_simulation_demo(zones: dict[str, PolygonPoints]) -> None:
    try:
        clip_path = generate_sample_cctv_video(SAMPLE_VIDEO_PATH, zones)
        st.session_state.video_path = clip_path
        st.session_state.is_simulation_clip = True
        st.session_state.is_streaming = True
        st.session_state.error_message = None
    except Exception as error:
        st.session_state.error_message = (
            f"Gagal membuat klip simulasi demo: {error}. Periksa izin tulis folder demo/."
        )


def render_sidebar_controls() -> tuple[str, int, float, float, bool]:
    with st.sidebar:
        st.markdown("## Panel Kontrol Kamera")
        camera_mode = st.selectbox(
            "Pilih Kategori Kamera CCTV",
            options=[TABLE_MODE, AISLE_MODE, HYBRID_RAK2_MODE],
            index=0,
        )
        frame_skip = st.slider(
            "Interval Frame Skipping (Inferensi tiap N frame)",
            min_value=1,
            max_value=15,
            value=5,
            help="Melewati frame di antara inferensi YOLO11 (640x640); waktu idle tetap dihitung akurat berdasarkan indeks frame video.",
        )
        idle_threshold = float(
            st.number_input(
                "Batas Peringatan Barang Tertinggal (detik video)",
                min_value=2,
                max_value=1800,
                value=3,
                step=1,
            )
        )
        confidence = float(
            st.slider(
                "Ambang Kepercayaan Deteksi (Confidence)",
                min_value=0.10,
                max_value=0.85,
                value=0.20,
                step=0.05,
            )
        )
        use_clahe = st.checkbox(
            "Normalisasi Pencahayaan Backlight (CLAHE)",
            value=True,
            help="Menyeimbangkan kontras area gelap di dekat jendela terang sebelum inferensi YOLO11.",
        )
    return camera_mode, frame_skip, idle_threshold, confidence, use_clahe


def render_video_source_buttons(zones: dict[str, PolygonPoints]) -> None:
    with st.sidebar:
        st.markdown("### Sumber Stream Video")
        if st.button("Muat & Jalankan Simulasi Demo", key="load_sim_btn", width="stretch"):
            activate_simulation_demo(zones)

        uploaded = st.file_uploader(
            "Unggah Klip Video CCTV (.mp4 / .avi)",
            type=["mp4", "avi", "mov"],
        )
        if uploaded is not None:
            handle_uploaded_video(uploaded)

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button(
                "Mulai Stream",
                key="start_stream_btn",
                disabled=st.session_state.video_path is None,
                width="stretch",
            ):
                st.session_state.is_streaming = True
        with col_stop:
            if st.button(
                "Hentikan Stream",
                key="stop_stream_btn",
                disabled=not st.session_state.is_streaming,
                width="stretch",
            ):
                st.session_state.is_streaming = False


def render_empty_or_error_state() -> bool:
    if st.session_state.error_message:
        st.markdown(
            f'<div class="status-banner-alert" role="alert">'
            f"<strong>Kesalahan Sistem:</strong> {st.session_state.error_message}"
            f"</div>",
            unsafe_allow_html=True,
        )
        return True
    if st.session_state.video_path is None and st.session_state.last_frame_rgb is None:
        st.markdown(
            '<div class="panel-box">'
            "<h3>Belum Ada Sumber Video Aktif (Empty State)</h3>"
            "<p>Pemantauan belum berjalan karena belum ada file video CCTV yang dipilih. "
            "Klik tombol <strong>Muat & Jalankan Simulasi Demo</strong> pada panel kontrol kiri "
            "untuk menguji deteksi poligon meja dan lorong secara instan, atau unggah klip MP4 CCTV.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return True
    return False


def format_table_metrics_html(statuses: list[TableZoneStatus]) -> str:
    if not statuses:
        occupied_text = "-- [Belum ada stream]"
        empty_text = "-- [Belum ada stream]"
        abandoned_text = "-- [Belum ada stream]"
    else:
        occupied_count = sum(1 for item in statuses if item.is_occupied)
        empty_count = len(statuses) - occupied_count
        abandoned_count = sum(1 for item in statuses if item.is_abandoned)
        occupied_text = f"{occupied_count} / {len(statuses)} Meja"
        empty_text = f"{empty_count} Meja"
        abandoned_text = f"{abandoned_count} Zona"

    return f"""
    <div class="metric-grid" role="region" aria-label="Ringkasan Ketersediaan Meja">
        <div class="metric-item">
            <div class="metric-label">Status Meja Terisi</div>
            <div class="metric-value">{occupied_text}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Status Meja Kosong</div>
            <div class="metric-value">{empty_text}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Peringatan Barang Tertinggal</div>
            <div class="metric-value">{abandoned_text}</div>
        </div>
    </div>
    """


def format_aisle_metrics_html(summary: AisleTrafficSummary | None) -> str:
    if summary is None:
        active_text = "-- [Belum ada stream]"
        total_text = "-- [Belum ada stream]"
        peak_text = "-- [Belum ada stream]"
    else:
        active_text = f"{summary.active_person_count} Orang ({summary.moving_person_count} bergerak)"
        total_text = f"{summary.cumulative_detections} Titik Gerak"
        peak_text = summary.peak_sector_name

    return f"""
    <div class="metric-grid" role="region" aria-label="Ringkasan Kepadatan Lorong">
        <div class="metric-item">
            <div class="metric-label">Pengunjung Aktif di Frame</div>
            <div class="metric-value">{active_text}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Akumulasi Pergerakan (Fading Aktif)</div>
            <div class="metric-value">{total_text}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Sektor Lorong Terpadat</div>
            <div class="metric-value">{peak_text}</div>
        </div>
    </div>
    """


def build_table_zones_markdown(statuses: list[TableZoneStatus]) -> str:
    if not statuses:
        return "*Tabel status poligon zona meja akan terisi otomatis saat frame video diproses.*"
    lines = [
        "| Zona Meja | Status Keputusan | Durasi Idle (Video) | Frame Idle | Objek Terdeteksi |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for item in statuses:
        detected = ", ".join(item.detected_items) if item.detected_items else "Tidak ada objek"
        lines.append(
            f"| {item.zone_name} | **{item.status_label}** | {item.idle_seconds:.1f} dtk | {item.idle_frames} frame | {detected} |"
        )
    return "\n".join(lines)


def build_aisle_sectors_markdown(summary: AisleTrafficSummary | None) -> str:
    if summary is None:
        return "*Tabel distribusi trafik lorong akan tampil begitu frame kamera rak buku diproses.*"
    lines = [
        "| Sektor Lorong Rak Buku | Akumulasi Titik Gerak | Keterangan Kepadatan |",
        "| :--- | :--- | :--- |",
    ]
    for sector_name, count in summary.sector_counts.items():
        is_peak = count > 0 and sector_name == summary.peak_sector_name
        note = "Prioritas Pemantauan (Terpadat)" if is_peak else "Normal"
        lines.append(f"| {sector_name} | {count} titik | {note} |")
    return "\n".join(lines)


def resolve_frame_detections(
    frame,
    frame_index: int,
    camera_mode: str,
    confidence: float,
    use_clahe: bool,
    scaled_zones: dict[str, PolygonPoints],
):
    if st.session_state.is_simulation_clip:
        return build_simulated_detections(frame_index, camera_mode, scaled_zones)
    model = get_cached_yolo_model("yolo11n.pt")
    return detect_library_objects(
        model, frame, confidence_threshold=confidence, use_clahe=use_clahe
    )


def update_ui_placeholders(
    camera_mode: str,
    annotated_bgr,
    metrics_slot,
    alert_slot,
    video_slot,
    table_slot,
) -> None:
    rgb_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = rgb_frame
    video_slot.image(
        rgb_frame,
        caption=f"Stream Aktif | Diproses: {st.session_state.processed_count} | Dilewati (skip): {st.session_state.skipped_count}",
        width="stretch",
    )
    statuses = st.session_state.table_statuses
    summary = st.session_state.aisle_summary
    if camera_mode == TABLE_MODE:
        metrics_slot.markdown(format_table_metrics_html(statuses), unsafe_allow_html=True)
        table_slot.markdown(build_table_zones_markdown(statuses))
    elif camera_mode == AISLE_MODE:
        metrics_slot.markdown(format_aisle_metrics_html(summary), unsafe_allow_html=True)
        table_slot.markdown(build_aisle_sectors_markdown(summary))
    else:
        metrics_slot.markdown(format_table_metrics_html(statuses), unsafe_allow_html=True)
        combined_md = build_table_zones_markdown(statuses) + "\n\n" + build_aisle_sectors_markdown(summary)
        table_slot.markdown(combined_md)

    abandoned = [item.zone_name for item in statuses if item.is_abandoned]
    if abandoned and camera_mode != AISLE_MODE:
        alert_slot.markdown(
            f'<div class="status-banner-alert" role="alert">'
            f"PERINGATAN BARANG TERTINGGAL: Barang tanpa pemilik melebihi batas waktu pada <strong>{', '.join(abandoned)}</strong>."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        alert_slot.empty()


def process_single_video_frame(
    frame,
    frame_idx: int,
    fps: float,
    camera_mode: str,
    idle_threshold: float,
    confidence: float,
    use_clahe: bool,
    scaled_zones: dict[str, PolygonPoints],
    idle_frame_timers: dict[str, int],
    heatmap_acc: TrafficHeatmapAccumulator,
):
    st.session_state.processed_count += 1
    detections = resolve_frame_detections(
        frame, frame_idx, camera_mode, confidence, use_clahe, scaled_zones
    )
    annotated = frame.copy()
    if camera_mode in (AISLE_MODE, HYBRID_RAK2_MODE):
        st.session_state.aisle_summary = heatmap_acc.record_persons(detections)
        annotated = render_heatmap_overlay(annotated, heatmap_acc, detections)
    if camera_mode in (TABLE_MODE, HYBRID_RAK2_MODE) and scaled_zones:
        st.session_state.table_statuses = evaluate_all_table_zones(
            scaled_zones, detections, frame_idx, fps, idle_threshold, idle_frame_timers
        )
        annotated = draw_seat_monitoring_overlay(
            annotated, st.session_state.table_statuses, detections
        )
    return annotated


def run_streaming_loop(
    camera_mode: str,
    frame_skip: int,
    idle_threshold: float,
    confidence: float,
    use_clahe: bool,
    raw_zones: dict[str, PolygonPoints],
    reference_res: tuple[int, int],
    slots: tuple,
) -> None:
    metrics_slot, alert_slot, video_slot, table_slot, loading_slot = slots
    loading_slot.info("Status Memuat (Loading State): Menyiapkan decoder video dan bobot YOLO11n...")
    capture = cv2.VideoCapture(str(st.session_state.video_path))
    loading_slot.empty()
    if not capture.isOpened():
        st.session_state.error_message = f"Gagal mendekode video '{st.session_state.video_path}'."
        st.session_state.is_streaming = False
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    scaled_zones = scale_zones_to_frame(raw_zones, reference_res, width, height)
    heatmap_acc = TrafficHeatmapAccumulator(height, width)
    idle_frame_timers: dict[str, int] = {}
    frame_idx = 0
    try:
        while st.session_state.is_streaming and capture.isOpened():
            has_frame, frame = capture.read()
            if not has_frame or frame is None:
                st.session_state.is_streaming = False
                break
            frame_idx += 1
            if frame_skip > 1 and frame_idx % frame_skip != 1:
                st.session_state.skipped_count += 1
                continue
            annotated = process_single_video_frame(
                frame, frame_idx, fps, camera_mode, idle_threshold, confidence, use_clahe, scaled_zones, idle_frame_timers, heatmap_acc
            )
            update_ui_placeholders(camera_mode, annotated, metrics_slot, alert_slot, video_slot, table_slot)
            time.sleep(0.06)
    finally:
        capture.release()


def render_static_snapshot(
    camera_mode: str, metrics_slot, alert_slot, video_slot, table_slot
) -> None:
    if camera_mode == AISLE_MODE:
        metrics_slot.markdown(format_aisle_metrics_html(st.session_state.aisle_summary), unsafe_allow_html=True)
        table_slot.markdown(build_aisle_sectors_markdown(st.session_state.aisle_summary))
    else:
        metrics_slot.markdown(format_table_metrics_html(st.session_state.table_statuses), unsafe_allow_html=True)
        table_slot.markdown(build_table_zones_markdown(st.session_state.table_statuses))

    if st.session_state.last_frame_rgb is not None:
        video_slot.image(
            st.session_state.last_frame_rgb,
            caption=f"Frame Terakhir | Diproses: {st.session_state.processed_count} | Dilewati (skip): {st.session_state.skipped_count}",
            width="stretch",
        )


def main() -> None:
    st.set_page_config(
        page_title="Pemantauan CCTV Perpustakaan Prasmul",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_custom_styles(st.session_state.is_dark_mode)
    render_header()

    camera_mode, frame_skip, idle_threshold, confidence, use_clahe = render_sidebar_controls()
    profile_name = MODE_TO_PROFILE.get(camera_mode, "Mezanin_1")
    try:
        zones, reference_res = load_zones_with_metadata(DEFAULT_ZONES_PATH, profile_name=profile_name)
    except Exception as error:
        st.session_state.error_message = str(error)
        zones, reference_res = {}, (1024, 604)

    render_video_source_buttons(zones)
    if render_empty_or_error_state():
        return

    loading_slot = st.empty()
    metrics_slot = st.empty()
    alert_slot = st.empty()
    col_video, col_table = st.columns([3, 2])
    with col_video:
        st.markdown("## Layar Inspeksi CCTV")
        video_slot = st.empty()
    with col_table:
        st.markdown("## Status & Rincian Zona")
        table_slot = st.empty()

    if st.session_state.is_streaming:
        slots = (metrics_slot, alert_slot, video_slot, table_slot, loading_slot)
        run_streaming_loop(
            camera_mode, frame_skip, idle_threshold, confidence, use_clahe, zones, reference_res, slots
        )
    else:
        render_static_snapshot(
            camera_mode, metrics_slot, alert_slot, video_slot, table_slot
        )


if __name__ == "__main__":
    main()
