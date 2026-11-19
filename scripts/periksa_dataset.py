import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_TOTAL_IMAGES = 1000
MIN_OBJECTS_PER_CLASS = 300
MIN_TEST_IMAGES = 150


def compute_dhash(image_path: Path, hash_size: int = 8) -> str | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if val else "0" for val in diff.flatten())
    return f"{int(bits, 2):016x}"


def compute_md5(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def validate_label_file(label_path: Path, num_classes: int | None = None) -> tuple[list[int], list[str]]:
    if not label_path.exists():
        return [], []
    class_ids: list[int] = []
    errors: list[str] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{line_no} expected 5 columns, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(val) for val in parts[1:])
        except ValueError:
            errors.append(f"{label_path.name}:{line_no} non-numeric label values")
            continue
        if cls_id < 0 or (num_classes is not None and cls_id >= num_classes):
            errors.append(f"{label_path.name}:{line_no} invalid class_id={cls_id}")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"{label_path.name}:{line_no} normalized bbox out of range")
        class_ids.append(cls_id)
    return class_ids, errors


def resolve_split_dirs(dataset_root: Path, split_name: str) -> tuple[Path, Path] | None:
    candidates = [split_name, "val"] if split_name == "valid" else [split_name]
    for candidate in candidates:
        img_dir = dataset_root / candidate / "images"
        lbl_dir = dataset_root / candidate / "labels"
        if img_dir.exists():
            return img_dir, lbl_dir
    return None


def inspect_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    num_classes: int | None = None,
) -> dict:
    image_files = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    class_counts: Counter[int] = Counter()
    errors: list[str] = []
    md5_map: dict[str, str] = {}
    dhash_map: dict[str, str] = {}
    background_count = 0

    for img_path in image_files:
        dhash = compute_dhash(img_path)
        if dhash is None:
            errors.append(f"Corrupted image: {img_path.name}")
            continue
        md5_map[img_path.name] = compute_md5(img_path)
        dhash_map[img_path.name] = dhash

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        cls_ids, lbl_errors = validate_label_file(lbl_path, num_classes=num_classes)
        errors.extend(lbl_errors)
        if not cls_ids:
            background_count += 1
        class_counts.update(cls_ids)

    return {
        "split": split_name,
        "image_count": len(image_files),
        "background_count": background_count,
        "class_counts": dict(class_counts),
        "errors": errors,
        "md5_map": md5_map,
        "dhash_map": dhash_map,
    }


def detect_cross_split_leakage(split_reports: list[dict]) -> dict[str, list[tuple[str, str]]]:
    exact_leaks: list[tuple[str, str]] = []
    near_leaks: list[tuple[str, str]] = []
    for idx_a in range(len(split_reports)):
        for idx_b in range(idx_a + 1, len(split_reports)):
            rep_a, rep_b = split_reports[idx_a], split_reports[idx_b]
            md5_b = {val: name for name, val in rep_b["md5_map"].items()}
            dhash_b = {val: name for name, val in rep_b["dhash_map"].items()}
            for name_a, hash_val in rep_a["md5_map"].items():
                if hash_val in md5_b:
                    exact_leaks.append((f"{rep_a['split']}/{name_a}", f"{rep_b['split']}/{md5_b[hash_val]}"))
            for name_a, dhash_val in rep_a["dhash_map"].items():
                if dhash_val in dhash_b:
                    near_leaks.append((f"{rep_a['split']}/{name_a}", f"{rep_b['split']}/{dhash_b[dhash_val]}"))
    return {"exact_duplicates": exact_leaks, "near_duplicates": near_leaks}


def print_dataset_summary(split_reports: list[dict], leakage: dict) -> None:
    total_images = sum(rep["image_count"] for rep in split_reports)
    total_bg = sum(rep["background_count"] for rep in split_reports)
    total_errors = sum(len(rep["errors"]) for rep in split_reports)
    total_classes: Counter[int] = Counter()
    for rep in split_reports:
        total_classes.update(rep["class_counts"])

    print("=" * 68)
    print("LAPORAN PEMERIKSAAN DATASET (periksa_dataset.py)")
    print("=" * 68)
    for rep in split_reports:
        print(
            f"Split [{rep['split']:5s}] : {rep['image_count']:4d} gambar | "
            f"background: {rep['background_count']:3d} | kelas: {rep['class_counts']} | "
            f"error label: {len(rep['errors'])}"
        )
    bg_ratio = (total_bg / total_images * 100.0) if total_images else 0.0
    print("-" * 68)
    print(f"Total gambar      : {total_images} (target UAS >= {MIN_TOTAL_IMAGES})")
    print(f"Gambar background : {total_bg} ({bg_ratio:.1f}%, saran UAS 5-10%)")
    print(f"Total objek/kelas : {dict(total_classes)} (target UAS >= {MIN_OBJECTS_PER_CLASS}/kelas)")
    print(f"Total error label : {total_errors}")
    print(f"Kebocoran exact   : {len(leakage['exact_duplicates'])} pasang")
    print(f"Kebocoran near    : {len(leakage['near_duplicates'])} pasang")
    print("=" * 68)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify YOLO dataset integrity, class counts, and train/val/test leakage (UAS requirement)."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Root folder containing train/valid/test.")
    parser.add_argument("--num-classes", type=int, default=None, help="Expected number of classes.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{args.dataset_dir}' not found.")

    reports: list[dict] = []
    for split_name in ("train", "valid", "test"):
        resolved = resolve_split_dirs(args.dataset_dir, split_name)
        if resolved is not None:
            reports.append(inspect_split(split_name, resolved[0], resolved[1], args.num_classes))

    if not reports:
        reports.append(inspect_split("all", args.dataset_dir, args.dataset_dir, args.num_classes))

    leakage = detect_cross_split_leakage(reports)
    print_dataset_summary(reports, leakage)


if __name__ == "__main__":
    main()
