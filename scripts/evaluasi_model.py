import argparse
import csv
from pathlib import Path
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (
    DEFAULT_ROBOFLOW_MODEL_ID,
    DetectedObject,
    detect_library_objects,
    detect_library_objects_roboflow,
    load_yolo_model,
    resolve_m2_weights_path,
)

CLASS_NAMES = {0: "person", 1: "belonging"}
IOU_THRESHOLDS = [round(0.50 + 0.05 * step, 2) for step in range(10)]


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def load_ground_truth_boxes(
    label_path: Path, img_width: int, img_height: int
) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(val) for val in parts[1:])
        x1 = int(round((cx - w / 2.0) * img_width))
        y1 = int(round((cy - h / 2.0) * img_height))
        x2 = int(round((cx + w / 2.0) * img_width))
        y2 = int(round((cy + h / 2.0) * img_height))
        boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def map_detection_to_eval_class(obj: DetectedObject) -> int:
    return 0 if obj.is_person else 1


def compute_ap_at_iou(
    predictions: list[tuple[str, float, tuple[int, int, int, int]]],
    gt_by_image: dict[str, list[tuple[int, int, int, int]]],
    iou_threshold: float,
) -> tuple[float, float, float]:
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return (0.0, 0.0, 0.0)
    if not predictions:
        return (0.0, 0.0, 0.0)

    sorted_preds = sorted(predictions, key=lambda item: item[1], reverse=True)
    matched: dict[str, set[int]] = {img: set() for img in gt_by_image}
    tp = np.zeros(len(sorted_preds), dtype=np.float64)
    fp = np.zeros(len(sorted_preds), dtype=np.float64)

    for idx, (img_name, _conf, pred_box) in enumerate(sorted_preds):
        gt_boxes = gt_by_image.get(img_name, [])
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched[img_name]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[idx] = 1.0
            matched[img_name].add(best_gt_idx)
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return (float(precisions[-1]), float(recalls[-1]), ap)


def evaluate_predictions_dataset(
    preds_by_class: dict[int, list[tuple[str, float, tuple[int, int, int, int]]]],
    gt_by_class: dict[int, dict[str, list[tuple[int, int, int, int]]]],
) -> dict[str, float]:
    active_classes = [cls_id for cls_id, gt_map in gt_by_class.items() if sum(len(v) for v in gt_map.values()) > 0]
    if not active_classes:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}

    p50_list, r50_list, ap50_list, ap50_95_list = [], [], [], []
    metrics: dict[str, float] = {}
    for cls_id in active_classes:
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        p50, r50, ap50 = compute_ap_at_iou(preds_by_class.get(cls_id, []), gt_by_class[cls_id], 0.50)
        aps = [compute_ap_at_iou(preds_by_class.get(cls_id, []), gt_by_class[cls_id], thr)[2] for thr in IOU_THRESHOLDS]
        ap_95 = float(np.mean(aps))
        p50_list.append(p50)
        r50_list.append(r50)
        ap50_list.append(ap50)
        ap50_95_list.append(ap_95)
        metrics[f"{cls_name}_precision"] = round(p50, 4)
        metrics[f"{cls_name}_recall"] = round(r50, 4)
        metrics[f"{cls_name}_mAP50"] = round(ap50, 4)
        metrics[f"{cls_name}_mAP50_95"] = round(ap_95, 4)

    metrics["precision"] = round(float(np.mean(p50_list)), 4)
    metrics["recall"] = round(float(np.mean(r50_list)), 4)
    metrics["mAP50"] = round(float(np.mean(ap50_list)), 4)
    metrics["mAP50_95"] = round(float(np.mean(ap50_95_list)), 4)
    return metrics


def run_model_benchmark(
    model_tag: str,
    model_spec: str,
    images_dir: Path,
    labels_dir: Path,
    conf: float = 0.25,
    imgsz: int = 960,
) -> dict[str, str | float]:
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files:
        raise ValueError(f"No images found in '{images_dir}'.")

    is_roboflow = model_spec.startswith("roboflow:")
    rf_model_id = model_spec.split(":", 1)[1] if is_roboflow else ""
    yolo_model = None if is_roboflow else load_yolo_model(model_spec)

    preds_by_class: dict[int, list[tuple[str, float, tuple[int, int, int, int]]]] = {0: [], 1: []}
    gt_by_class: dict[int, dict[str, list[tuple[int, int, int, int]]]] = {0: {}, 1: {}}
    latencies_ms: list[float] = []

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for cls_id, gt_box in load_ground_truth_boxes(labels_dir / f"{img_path.stem}.txt", w, h):
            gt_by_class.setdefault(cls_id, {}).setdefault(img_path.name, []).append(gt_box)

        start = time.perf_counter()
        if is_roboflow:
            dets = detect_library_objects_roboflow(frame, model_id=rf_model_id, confidence_threshold=conf)
        else:
            dets = detect_library_objects(yolo_model, frame, confidence_threshold=conf, inference_size=imgsz)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        for obj in dets:
            eval_cls = map_detection_to_eval_class(obj)
            preds_by_class.setdefault(eval_cls, []).append((img_path.name, obj.confidence, obj.bbox))

    metrics = evaluate_predictions_dataset(preds_by_class, gt_by_class)
    avg_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
    row: dict[str, str | float] = {
        "model": model_tag,
        "spec": model_spec,
        "images": len(image_files),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "avg_latency_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
    }
    for cls_name in CLASS_NAMES.values():
        for metric_suffix in ("precision", "recall", "mAP50", "mAP50_95"):
            key = f"{cls_name}_{metric_suffix}"
            row[key] = metrics.get(key, 0.0)
    return row


def run_ultralytics_val_benchmark(
    model_tag: str,
    model_weights: str,
    data_yaml: Path,
    split: str = "test",
    conf: float = 0.25,
    imgsz: int = 960,
) -> dict[str, str | float]:
    model = load_yolo_model(model_weights)
    slug = model_tag.lower().replace(" ", "_").replace("(", "").replace(")", "")
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        plots=True,
        project="results/val",
        name=slug,
        verbose=False,
    )
    box = val_results.box
    speed = getattr(val_results, "speed", {})
    infer_ms = float(speed.get("inference", 0.0))
    fps = (1000.0 / infer_ms) if infer_ms > 0 else 0.0
    row: dict[str, str | float] = {
        "model": model_tag,
        "spec": model_weights,
        "images": int(getattr(val_results, "seen", 0) or 0),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "avg_latency_ms": round(infer_ms, 2),
        "fps": round(fps, 2),
    }
    ap_indices = list(getattr(box, "ap_class_index", []))
    for idx, cls_id in enumerate(ap_indices):
        cls_name = CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")
        p, r, ap50, ap = box.class_result(idx)
        row[f"{cls_name}_precision"] = round(float(p), 4)
        row[f"{cls_name}_recall"] = round(float(r), 4)
        row[f"{cls_name}_mAP50"] = round(float(ap50), 4)
        row[f"{cls_name}_mAP50_95"] = round(float(ap), 4)
    return row


def print_delta_summary(rows: list[dict[str, str | float]]) -> None:
    if len(rows) < 2:
        return
    m0, m2 = rows[0], rows[-1]
    print("-" * 72)
    print(f"Perbandingan Utama: {m0['model']} -> {m2['model']}")
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        v0, v2 = float(m0[metric]), float(m2[metric])
        abs_diff = v2 - v0
        rel_pct = ((abs_diff / v0) * 100.0) if v0 > 0 else 0.0
        print(f"  {metric:10s}: {v0:.4f} -> {v2:.4f} (Δ={abs_diff:+.4f} | {rel_pct:+.1f}%)")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M0, M1, and M2 on a labeled test set using Ultralytics model.val() or custom dir."
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to YOLO data.yaml (uses official model.val()).")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split for model.val().")
    parser.add_argument("--images-dir", type=Path, default=None, help="Test images dir (if --data is not set).")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Test labels dir (if --data is not set).")
    parser.add_argument("--m0", default="yolo11s.pt", help="M0 pretrained COCO weights path.")
    parser.add_argument("--m1", default=None, help="Optional M1 baseline fine-tuned weights path.")
    default_m2 = str(resolve_m2_weights_path() or "best.pt")
    parser.add_argument("--m2", default=default_m2, help="M2 final fine-tuned YOLO weights path (e.g. best.pt).")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--output-csv", type=Path, default=Path("results/perbandingan_m0_m1_m2.csv"))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        pass

    specs = [("M0 (Pretrained)", args.m0)]
    if args.m1:
        specs.append(("M1 (Baseline FT)", args.m1))
    specs.append(("M2 (Final)", args.m2))

    if args.data is not None:
        if not args.data.exists():
            raise FileNotFoundError(f"data.yaml not found at '{args.data}'.")
        rows = [
            run_ultralytics_val_benchmark(
                tag, spec, args.data, split=args.split, conf=args.conf, imgsz=args.imgsz
            )
            for tag, spec in specs
        ]
    else:
        if args.images_dir is None or args.labels_dir is None:
            raise ValueError("Provide either '--data path/to/data.yaml' OR both '--images-dir' and '--labels-dir'.")
        rows = [
            run_model_benchmark(tag, spec, args.images_dir, args.labels_dir, conf=args.conf, imgsz=args.imgsz)
            for tag, spec in specs
        ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']:18s} | P={row['precision']:.4f} R={row['recall']:.4f} "
            f"mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} | "
            f"{row['avg_latency_ms']:.1f} ms/img ({row['fps']:.1f} FPS)"
        )
    print_delta_summary(rows)
    print(f"Saved comparison table to '{args.output_csv}'.")


if __name__ == "__main__":
    main()
