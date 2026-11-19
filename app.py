from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import streamlit as st

from src.demo_generator import build_simulated_detections, generate_sample_cctv_video
from src.detector import (
    DEFAULT_ROBOFLOW_MODEL_ID,
    DEFAULT_YOLO_MODEL,
    detect_library_objects,
    detect_library_objects_roboflow,
    load_yolo_model,
    resolve_m2_weights_path,
)
from src.seat_monitor import (
    PolygonPoints,
    TableZoneStatus,
    draw_ignore_zones_overlay,
    draw_seat_monitoring_overlay,
    evaluate_all_table_zones,
    filter_ignored_detections,
    load_aisle_zones_with_metadata,
    load_ignore_zones_with_metadata,
    load_zones_with_metadata,
    scale_zones_to_frame,
)
from src.traffic_heatmap import (
    AisleTrafficSummary,
    TrafficHeatmapAccumulator,
    render_heatmap_overlay,
)
from src.ui_styles import apply_custom_styles


AISLE_MODE = "CCTV01 - Rak 1 (Meja Kiri + Heatmap Rak)"
HYBRID_RAK2_MODE = "CCTV02 - Rak 2 (Seat Occupancy & Abandoned Object)"
TABLE_MODE = "CCTV03 - Mezanin 1 (Seat Occupancy & Abandoned Object)"
ROBOFLOW_BACKEND = "Roboflow RF-DETR NAS v6 (Cloud - Multi-Angle)"
LOCAL_BACKEND = "Lokal YOLO11 Fine-Tuned (best.pt / yolo11s.pt)"
SINGLE_VIEW_MODE = "Monitoring Real-Time (Single Model)"
COMPARE_M0_M2_MODE = "Perbandingan M0 vs M2 (Demo UAS Berdampingan)"
MODE_TO_PROFILE = {
    AISLE_MODE: "Rak_1",
    HYBRID_RAK2_MODE: "Rak_2",
    TABLE_MODE: "Mezanin_1",
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
        "detector_backend": ROBOFLOW_BACKEND,
        "roboflow_api_key": "",
        "last_latency_ms": 0.0,
        "demo_view_mode": SINGLE_VIEW_MODE,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def render_header() -> None:
    col_title, col_theme = st.columns([3, 1])
    with col_title:
        st.markdown("### [LOGO] Pemantauan CCTV Perpustakaan Prasetiya Mulya")
        st.caption(
            "Sistem Pemantauan Okupansi Meja, Barang Tertinggal, dan Trafik Lorong Perpustakaan Berbasis Computer Vision. "
            "Semua metrik dihitung langsung dari frame video dan skala resolusi ROI."
        )
    with col_theme:
        theme_label = (
            "Ubah ke Mode Terang" if st.session_state.is_dark_mode else "Ubah ke Mode Gelap"
        )
        if st.button(theme_label, key="toggle_theme_btn", width="stretch"):
            st.session_state.is_dark_mode = not st.session_state.is_dark_mode
            st.rerun()


def infer_camera_mode_from_filename(filename: str) -> str | None:
    upper = filename.upper()
    if "CCTV01" in upper or "RAK_1" in upper or "RAK1" in upper:
        return AISLE_MODE
    if "CCTV02" in upper or "RAK_2" in upper or "RAK2" in upper:
        return HYBRID_RAK2_MODE
    if "CCTV03" in upper or "MEZANIN" in upper:
        return TABLE_MODE
    return None


def handle_uploaded_video(uploaded_file) -> None:
    if uploaded_file is None:
        return
    if st.session_state.get("last_uploaded_name") == uploaded_file.name:
        return
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.getbuffer())
        st.session_state.video_path = Path(temp_video.name)
        st.session_state.is_simulation_clip = False
        st.session_state.error_message = None
        st.session_state.last_uploaded_name = uploaded_file.name

    inferred_mode = infer_camera_mode_from_filename(uploaded_file.name)
    if inferred_mode and st.session_state.get("camera_mode_select") != inferred_mode:
        st.session_state.pending_camera_mode = inferred_mode
        st.rerun()


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
    pending_mode = st.session_state.pop("pending_camera_mode", None)
    if pending_mode:
        st.session_state.camera_mode_select = pending_mode

    with st.sidebar:
        st.markdown("## Panel Kontrol Kamera")
        st.session_state.demo_view_mode = st.radio(
            "Mode Tampilan Demo UAS",
            options=[SINGLE_VIEW_MODE, COMPARE_M0_M2_MODE],
            index=0,
        )
        camera_mode = st.selectbox(
            "Pilih Kamera CCTV (berdasarkan ID)",
            options=[AISLE_MODE, HYBRID_RAK2_MODE, TABLE_MODE],
            index=2,
            key="camera_mode_select",
        )
        st.session_state.detector_backend = st.selectbox(
            "Model Deteksi (Backend)",
            options=[ROBOFLOW_BACKEND, LOCAL_BACKEND],
            index=0,
        )
        if st.session_state.detector_backend == ROBOFLOW_BACKEND:
            st.session_state.roboflow_api_key = st.text_input(
                "Roboflow API Key (opsional jika sudah di .env)",
                value=st.session_state.roboflow_api_key,
                type="password",
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
                value=0.25,
                step=0.05,
                help="Nilai lebih KECIL (0.20–0.35) lebih sensitif menangkap objek jauh/duduk. Jangan diset tinggi (mis. 0.85) karena deteksi akan terfilter hilang.",
            )
        )
        use_clahe = st.checkbox(
            "Normalisasi Pencahayaan Backlight (CLAHE)",
            value=True,
            help="Menyeimbangkan kontras area gelap di dekat jendela terang sebelum inferensi.",
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
    start_time = time.perf_counter()
    if st.session_state.is_simulation_clip:
        detections = build_simulated_detections(frame_index, camera_mode, scaled_zones)
    elif st.session_state.get("detector_backend") == ROBOFLOW_BACKEND:
        detections = detect_library_objects_roboflow(
            frame,
            model_id=DEFAULT_ROBOFLOW_MODEL_ID,
            api_key=st.session_state.get("roboflow_api_key") or None,
            confidence_threshold=confidence,
            use_clahe=use_clahe,
        )
    else:
        model = get_cached_yolo_model(DEFAULT_YOLO_MODEL)
        detections = detect_library_objects(
            model, frame, confidence_threshold=confidence, use_clahe=use_clahe
        )
    st.session_state.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return detections


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
    lat_ms = float(st.session_state.get("last_latency_ms", 0.0))
    fps_est = (1000.0 / lat_ms) if lat_ms > 0 else 0.0
    video_slot.image(
        rgb_frame,
        caption=(
            f"Stream Aktif | Diproses: {st.session_state.processed_count} | "
            f"Dilewati: {st.session_state.skipped_count} | "
            f"Inferensi: {lat_ms:.1f} ms ({fps_est:.1f} FPS)"
        ),
        width="stretch",
    )
    statuses = st.session_state.table_statuses
    summary = st.session_state.aisle_summary
    if camera_mode != AISLE_MODE:
        metrics_slot.markdown(format_table_metrics_html(statuses), unsafe_allow_html=True)
        table_slot.markdown(build_table_zones_markdown(statuses))
    else:
        metrics_slot.markdown(format_table_metrics_html(statuses), unsafe_allow_html=True)
        combined_md = build_table_zones_markdown(statuses) + "\n\n" + build_aisle_sectors_markdown(summary)
        table_slot.markdown(combined_md)

    abandoned = [item.zone_name for item in statuses if item.is_abandoned]
    if abandoned:
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
    scaled_ignore_zones: dict[str, PolygonPoints] | None = None,
):
    st.session_state.processed_count += 1
    raw_detections = resolve_frame_detections(
        frame, frame_idx, camera_mode, confidence, use_clahe, scaled_zones
    )
    detections = filter_ignored_detections(raw_detections, scaled_ignore_zones)
    annotated = draw_ignore_zones_overlay(frame, scaled_ignore_zones)
    if camera_mode == AISLE_MODE:
        st.session_state.aisle_summary = heatmap_acc.record_persons(
            detections, excluded_zones=scaled_zones
        )
        annotated = render_heatmap_overlay(annotated, heatmap_acc, detections)
    if camera_mode in (TABLE_MODE, AISLE_MODE, HYBRID_RAK2_MODE) and scaled_zones:
        st.session_state.table_statuses = evaluate_all_table_zones(
            scaled_zones,
            detections,
            frame_idx,
            fps,
            idle_threshold,
            idle_frame_timers,
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
    loading_slot.info("Status Memuat (Loading State): Menyiapkan decoder video dan bobot YOLO11s...")
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
    profile_name = MODE_TO_PROFILE.get(camera_mode, "Mezanin_1")
    raw_aisle_zones, aisle_ref = load_aisle_zones_with_metadata(
        DEFAULT_ZONES_PATH, profile_name=profile_name
    )
    scaled_aisle_zones = (
        scale_zones_to_frame(raw_aisle_zones, aisle_ref, width, height) if raw_aisle_zones else {}
    )
    raw_ignore_zones, ignore_ref = load_ignore_zones_with_metadata(
        DEFAULT_ZONES_PATH, profile_name=profile_name
    )
    scaled_ignore_zones = (
        scale_zones_to_frame(raw_ignore_zones, ignore_ref, width, height)
        if raw_ignore_zones
        else {}
    )
    heatmap_acc = TrafficHeatmapAccumulator(height, width, aisle_zones=scaled_aisle_zones)
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
                frame,
                frame_idx,
                fps,
                camera_mode,
                idle_threshold,
                confidence,
                use_clahe,
                scaled_zones,
                idle_frame_timers,
                heatmap_acc,
                scaled_ignore_zones=scaled_ignore_zones,
            )
            update_ui_placeholders(camera_mode, annotated, metrics_slot, alert_slot, video_slot, table_slot)
            time.sleep(0.06)
    finally:
        capture.release()


def render_static_snapshot(
    camera_mode: str, metrics_slot, alert_slot, video_slot, table_slot
) -> None:
    if camera_mode != AISLE_MODE:
        metrics_slot.markdown(format_table_metrics_html(st.session_state.table_statuses), unsafe_allow_html=True)
        table_slot.markdown(build_table_zones_markdown(st.session_state.table_statuses))
    else:
        metrics_slot.markdown(format_table_metrics_html(st.session_state.table_statuses), unsafe_allow_html=True)
        combined_md = (
            build_table_zones_markdown(st.session_state.table_statuses)
            + "\n\n"
            + build_aisle_sectors_markdown(st.session_state.aisle_summary)
        )
        table_slot.markdown(combined_md)

    if st.session_state.last_frame_rgb is not None:
        video_slot.image(
            st.session_state.last_frame_rgb,
            caption=f"Frame Terakhir | Diproses: {st.session_state.processed_count} | Dilewati (skip): {st.session_state.skipped_count}",
            width="stretch",
        )


def render_side_by_side_comparison_view(
    confidence: float,
    use_clahe: bool,
    raw_zones: dict[str, PolygonPoints],
    reference_res: tuple[int, int],
    profile_name: str,
) -> None:
    st.markdown("## Demo Perbandingan Berdampingan: M0 (YOLO COCO) vs M2 (Model Baru)")
    st.caption(
        f"Confidence threshold aktif: **{confidence:.2f}** | Profil ROI: **{profile_name}**. "
        "Unggah 1 gambar dari set demo untuk membandingkan M0 dan M2 secara langsung."
    )
    demo_image = st.file_uploader(
        "Unggah Gambar Set Demo (.jpg / .png) untuk Perbandingan M0 vs M2",
        type=["jpg", "jpeg", "png"],
        key="compare_demo_img",
    )
    if demo_image is None:
        return

    raw_bytes = np.frombuffer(demo_image.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Gagal membaca gambar yang diunggah.")
        return

    height, width = frame.shape[:2]
    scaled_zones = scale_zones_to_frame(raw_zones, reference_res, width, height)
    raw_ignores, ignore_ref = load_ignore_zones_with_metadata(DEFAULT_ZONES_PATH, profile_name)
    scaled_ignores = scale_zones_to_frame(raw_ignores, ignore_ref, width, height) if raw_ignores else {}

    m0_model = get_cached_yolo_model("yolo11s.pt")
    t0 = time.perf_counter()
    m0_dets = filter_ignored_detections(
        detect_library_objects(m0_model, frame, confidence_threshold=confidence, use_clahe=use_clahe),
        scaled_ignores,
    )
    m0_ms = (time.perf_counter() - t0) * 1000.0

    m2_weights = resolve_m2_weights_path()
    if m2_weights is not None:
        m2_model = get_cached_yolo_model(str(m2_weights))
        t1 = time.perf_counter()
        m2_dets = filter_ignored_detections(
            detect_library_objects(m2_model, frame, confidence_threshold=confidence, use_clahe=use_clahe),
            scaled_ignores,
        )
        m2_ms = (time.perf_counter() - t1) * 1000.0
        m2_title = f"### M2 — Model Baru (Fine-Tuned `{m2_weights.name}`)"
    else:
        st.warning(
            "File bobot lokal `best.pt` belum ditemukan di root proyek atau `weights/m2_best.pt`. "
            "Sementara menampilkan Roboflow Cloud sebagai pratinjau; simpan `best.pt` hasil training YOLO untuk perbandingan kecepatan lokal yang setara."
        )
        t1 = time.perf_counter()
        m2_dets = filter_ignored_detections(
            detect_library_objects_roboflow(
                frame,
                model_id=DEFAULT_ROBOFLOW_MODEL_ID,
                api_key=st.session_state.get("roboflow_api_key") or None,
                confidence_threshold=confidence,
                use_clahe=use_clahe,
            ),
            scaled_ignores,
        )
        m2_ms = (time.perf_counter() - t1) * 1000.0
        m2_title = "### M2 — Model Baru (RF-DETR NAS v6 Cloud Fallback)"

    base = draw_ignore_zones_overlay(frame, scaled_ignores)
    m0_statuses = evaluate_all_table_zones(scaled_zones, m0_dets, 1, 15.0, 900.0, {})
    m2_statuses = evaluate_all_table_zones(scaled_zones, m2_dets, 1, 15.0, 900.0, {})
    m0_img = cv2.cvtColor(draw_seat_monitoring_overlay(base, m0_statuses, m0_dets), cv2.COLOR_BGR2RGB)
    m2_img = cv2.cvtColor(draw_seat_monitoring_overlay(base, m2_statuses, m2_dets), cv2.COLOR_BGR2RGB)

    col_m0, col_m2 = st.columns(2)
    with col_m0:
        st.markdown("### M0 — Model Lama (YOLO11s COCO Pretrained)")
        st.image(m0_img, width="stretch")
        st.info(
            f"Deteksi: **{len(m0_dets)} objek** "
            f"(Person: {sum(o.is_person for o in m0_dets)}, Belonging: {sum(o.is_belonging for o in m0_dets)}) | "
            f"Meja Terisi: **{sum(s.is_occupied for s in m0_statuses)}/{len(m0_statuses)}** | "
            f"Kecepatan: **{m0_ms:.1f} ms** ({1000.0 / max(m0_ms, 0.1):.1f} FPS)"
        )
    with col_m2:
        st.markdown(m2_title)
        st.image(m2_img, width="stretch")
        st.success(
            f"Deteksi: **{len(m2_dets)} objek** "
            f"(Person: {sum(o.is_person for o in m2_dets)}, Belonging: {sum(o.is_belonging for o in m2_dets)}) | "
            f"Meja Terisi: **{sum(s.is_occupied for s in m2_statuses)}/{len(m2_statuses)}** | "
            f"Kecepatan: **{m2_ms:.1f} ms** ({1000.0 / max(m2_ms, 0.1):.1f} FPS)"
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

    if st.session_state.get("demo_view_mode") == COMPARE_M0_M2_MODE:
        render_side_by_side_comparison_view(confidence, use_clahe, zones, reference_res, profile_name)
        return

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
