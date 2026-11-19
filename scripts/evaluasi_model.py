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
    return {
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M0, M1, and M2 on a labeled test set (Precision, Recall, mAP50, mAP50-95, ms/img, FPS)."
    )
    parser.add_argument("--images-dir", required=True, type=Path, help="Test images directory.")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Test YOLO .txt labels directory.")
    parser.add_argument("--m0", default="yolo11s.pt", help="M0 pretrained COCO weights path.")
    parser.add_argument("--m1", default=None, help="Optional M1 baseline fine-tuned weights path.")
    parser.add_argument("--m2", default=f"roboflow:{DEFAULT_ROBOFLOW_MODEL_ID}", help="M2 final model path or roboflow:<id>.")
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

    rows = [
        run_model_benchmark(tag, spec, args.images_dir, args.labels_dir, conf=args.conf, imgsz=args.imgsz)
        for tag, spec in specs
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']:18s} | P={row['precision']:.4f} R={row['recall']:.4f} "
            f"mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} | "
            f"{row['avg_latency_ms']:.1f} ms/img ({row['fps']:.1f} FPS)"
        )
    print(f"Saved comparison table to '{args.output_csv}'.")


if __name__ == "__main__":
    main()
