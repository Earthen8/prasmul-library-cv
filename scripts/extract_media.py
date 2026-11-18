import argparse
from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VALID_DAYS = {"Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"}
VALID_CATEGORIES = {"Meja", "Rak"}


def validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found at '{video_path}'. Check the file path and try again."
        )
    if not video_path.is_file():
        raise ValueError(
            f"Path '{video_path}' is not a regular file. Provide a valid video file."
        )


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    validate_video_path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video '{video_path}'. Ensure the file is a supported video format."
        )
    return capture


def calculate_frame_indices(total_frames: int, target_count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video reports 0 total frames. The video header may be corrupted.")
    if target_count <= 0:
        raise ValueError("Target frame count must be greater than 0.")
    if target_count > total_frames:
        raise ValueError(
            f"Requested {target_count} frames, but video only contains {total_frames} frames."
        )
    step = total_frames / target_count
    return [int(index * step) for index in range(target_count)]


def save_single_frame(
    capture: cv2.VideoCapture, frame_index: int, output_file: Path
) -> bool:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    has_frame, frame = capture.read()
    if not has_frame or frame is None:
        return False
    return bool(cv2.imwrite(str(output_file), frame))


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    day_name: str,
    category: str,
    frame_count: int,
    start_number: int = 1,
) -> int:
    if day_name not in VALID_DAYS:
        raise ValueError(f"Invalid day '{day_name}'. Choose from: {sorted(VALID_DAYS)}.")
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Choose from: {sorted(VALID_CATEGORIES)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = open_video_capture(video_path)
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = calculate_frame_indices(total_frames, frame_count)
        saved_count = 0
        for offset, frame_index in enumerate(indices):
            file_number = start_number + offset
            filename = f"{day_name}_{category}_{file_number:02d}.jpg"
            if save_single_frame(capture, frame_index, output_dir / filename):
                saved_count += 1
        return saved_count
    finally:
        capture.release()


def parse_timestamp_to_seconds(timestamp_text: str) -> float:
    parts = timestamp_text.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid timestamp '{timestamp_text}'. Use MM:SS or HH:MM:SS format."
        )
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"Timestamp '{timestamp_text}' contains non-numeric values."
        ) from error

    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def create_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not initialize video writer for '{output_path}'. Check write permissions."
        )
    return writer


def write_frame_range(
    capture: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    start_frame: int,
    end_frame: int,
) -> int:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written_frames = 0
    for _ in range(start_frame, end_frame):
        has_frame, frame = capture.read()
        if not has_frame or frame is None:
            break
        writer.write(frame)
        written_frames += 1
    return written_frames


def trim_video_clip(
    video_path: Path,
    output_path: Path,
    start_timestamp: str,
    duration_seconds: int = 60,
) -> int:
    if duration_seconds <= 0:
        raise ValueError("Clip duration must be greater than 0 seconds.")

    capture = open_video_capture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_seconds = parse_timestamp_to_seconds(start_timestamp)
        start_frame = int(start_seconds * fps)
        if start_frame >= total_frames:
            raise ValueError(
                f"Start time {start_timestamp} exceeds video length ({total_frames / fps:.1f}s)."
            )

        end_frame = min(total_frames, start_frame + int(duration_seconds * fps))
        writer = create_video_writer(output_path, fps, width, height)
        try:
            return write_frame_range(capture, writer, start_frame, end_frame)
        finally:
            writer.release()
    finally:
        capture.release()


def extract_reference_frame(
    video_path: Path,
    output_path: Path,
    timestamp: str = "00:00:00",
) -> Path:
    capture = open_video_capture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        target_frame = int(parse_timestamp_to_seconds(timestamp) * fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_single_frame(capture, target_frame, output_path):
            raise RuntimeError(
                f"Could not extract frame at '{timestamp}' from '{video_path}'."
            )
        return output_path
    finally:
        capture.release()


def score_candidate_frames(
    capture: cv2.VideoCapture,
    step_frames: int,
    total_frames: int,
    model,
    conf: float = 0.25,
) -> list[tuple[int, int]]:
    from src.detector import detect_library_objects

    scored: list[tuple[int, int]] = []
    for frame_idx in range(0, total_frames, step_frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        objs = detect_library_objects(model, frame, confidence_threshold=conf, inference_size=640)
        person_count = sum(1 for obj in objs if obj.is_person)
        scored.append((person_count, frame_idx))
    return scored


def extract_smart_frames(
    video_path: Path,
    output_dir: Path,
    day_name: str,
    category: str,
    frame_count: int = 67,
    start_number: int = 1,
    sample_every_seconds: int = 30,
) -> int:
    from src.detector import load_yolo_model

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = open_video_capture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frames = max(1, int(round(fps * sample_every_seconds)))
        model = load_yolo_model("yolo11n.pt")
        scored = score_candidate_frames(capture, step_frames, total_frames, model)
        scored.sort(key=lambda item: item[0], reverse=True)
        top_indices = sorted(idx for _, idx in scored[:frame_count])

        saved_count = 0
        for offset, frame_idx in enumerate(top_indices):
            filename = f"{day_name}_{category}_{start_number + offset:03d}.jpg"
            if save_single_frame(capture, frame_idx, output_dir / filename):
                saved_count += 1
        return saved_count
    finally:
        capture.release()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract evaluation screenshots and short demo clips from CCTV videos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    frames_parser = subparsers.add_parser("frames", help="Extract evenly spaced frames.")
    frames_parser.add_argument("--input", required=True, type=Path, help="Input video file.")
    frames_parser.add_argument("--output-dir", required=True, type=Path, help="Output folder.")
    frames_parser.add_argument("--day", required=True, choices=sorted(VALID_DAYS))
    frames_parser.add_argument("--category", required=True, choices=sorted(VALID_CATEGORIES))
    frames_parser.add_argument("--count", type=int, default=30, help="Number of frames.")
    frames_parser.add_argument("--start-number", type=int, default=1, help="Starting index.")

    clip_parser = subparsers.add_parser("clip", help="Cut a short demo video clip.")
    clip_parser.add_argument("--input", required=True, type=Path, help="Input video file.")
    clip_parser.add_argument("--output", required=True, type=Path, help="Output MP4 file.")
    clip_parser.add_argument("--start", default="00:00:00", help="Start time (HH:MM:SS).")
    clip_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds.")

    ref_parser = subparsers.add_parser(
        "reference", help="Save a clean empty frame as config/empty_reference.jpg."
    )
    ref_parser.add_argument("--input", required=True, type=Path, help="Input video file.")
    ref_parser.add_argument(
        "--output", type=Path, default=Path("config/empty_reference.jpg"), help="Output JPG path."
    )
    ref_parser.add_argument("--time", default="00:00:00", help="Timestamp (MM:SS or HH:MM:SS).")

    smart_parser = subparsers.add_parser(
        "smart-frames", help="Scan long video and extract top N busiest frames."
    )
    smart_parser.add_argument("--input", required=True, type=Path, help="Input video file.")
    smart_parser.add_argument("--output-dir", required=True, type=Path, help="Output folder.")
    smart_parser.add_argument("--day", required=True, choices=sorted(VALID_DAYS))
    smart_parser.add_argument("--category", required=True, choices=sorted(VALID_CATEGORIES))
    smart_parser.add_argument("--count", type=int, default=67, help="Number of busiest frames.")
    smart_parser.add_argument("--start-number", type=int, default=1, help="Starting index.")
    smart_parser.add_argument(
        "--sample-every", type=int, default=30, help="Scan interval in seconds."
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "frames":
        saved = extract_frames_from_video(
            video_path=args.input,
            output_dir=args.output_dir,
            day_name=args.day,
            category=args.category,
            frame_count=args.count,
            start_number=args.start_number,
        )
        print(f"Saved {saved} screenshots to '{args.output_dir}'.")
    elif args.command == "clip":
        written = trim_video_clip(
            video_path=args.input,
            output_path=args.output,
            start_timestamp=args.start,
            duration_seconds=args.duration,
        )
        print(f"Saved demo clip ({written} frames) to '{args.output}'.")
    elif args.command == "reference":
        saved_path = extract_reference_frame(
            video_path=args.input,
            output_path=args.output,
            timestamp=args.time,
        )
        print(f"Saved empty reference frame to '{saved_path}'.")
    elif args.command == "smart-frames":
        saved = extract_smart_frames(
            video_path=args.input,
            output_dir=args.output_dir,
            day_name=args.day,
            category=args.category,
            frame_count=args.count,
            start_number=args.start_number,
            sample_every_seconds=args.sample_every,
        )
        print(f"Saved {saved} busiest screenshots to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
