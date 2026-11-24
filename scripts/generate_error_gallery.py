import argparse
import csv
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluasi_model import compute_iou, load_ground_truth_boxes, map_detection_to_eval_class
from src.detector import detect_library_objects, load_yolo_model


CLASS_NAMES = {0: "person", 1: "belonging"}


def classify_image_errors(
    gt_boxes: list[tuple[int, tuple[int, int, int, int]]],
    preds: list[tuple[int, float, tuple[int, int, int, int]]],
    iou_match_thr: float = 0.50,
    iou_loose_thr: float = 0.15,
) -> dict:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_count = 0
    misclassified: list[tuple[int, int, tuple[int, int, int, int]]] = []
    loose_boxes: list[tuple[int, float, tuple[int, int, int, int]]] = []
    false_positives: list[tuple[int, tuple[int, int, int, int]]] = []

    for pred_idx, (pred_cls, _conf, pred_box) in enumerate(preds):
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx

        if best_gt_idx < 0 or best_iou < iou_loose_thr:
            false_positives.append((pred_cls, pred_box))
            continue

        gt_cls = gt_boxes[best_gt_idx][0]
        if best_iou >= iou_match_thr and pred_cls == gt_cls:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            tp_count += 1
        elif best_iou >= iou_match_thr and pred_cls != gt_cls:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            misclassified.append((gt_cls, pred_cls, pred_box))
        else:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            loose_boxes.append((pred_cls, best_iou, pred_box))

    missed_gt = [
        gt_boxes[idx] for idx in range(len(gt_boxes)) if idx not in matched_gt
    ]
    return {
        "tp": tp_count,
        "missed": missed_gt,
        "false_positives": false_positives,
        "loose": loose_boxes,
        "misclassified": misclassified,
    }


def pick_primary_error_category(err: dict) -> str | None:
    if err["misclassified"]:
        return "salah_kelas"
    if err["loose"]:
        return "kotak_tidak_pas"
    if len(err["missed"]) >= max(1, len(err["false_positives"])):
        return "objek_terlewat"
    if err["false_positives"]:
        return "deteksi_palsu"
    return None


def draw_error_visualization(
    frame: np.ndarray,
    gt_boxes: list[tuple[int, tuple[int, int, int, int]]],
    err: dict,
    header_text: str,
) -> np.ndarray:
    canvas = frame.copy()
    for gt_cls, (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 190, 60), 2)
        cv2.putText(
            canvas, f"GT:{CLASS_NAMES.get(gt_cls, gt_cls)}", (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 230, 60), 1, cv2.LINE_AA
        )

    for _cls, (x1, y1, x2, y2) in err["false_positives"]:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 235), 2)
        cv2.putText(canvas, "FP", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 235), 1)

    for _cls, iou, (x1, y1, x2, y2) in err["loose"]:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(canvas, f"IoU:{iou:.2f}", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    for gt_cls, pred_cls, (x1, y1, x2, y2) in err["misclassified"]:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 40, 220), 2)
        cv2.putText(canvas, f"{gt_cls}->{pred_cls}", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 40, 220), 1)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(canvas, header_text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    return canvas


def collect_galleries(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    m0_weights: str,
    m2_weights: str,
    conf: float = 0.25,
    imgsz: int = 960,
    max_m0: int = 16,
    max_m2: int = 24,
    max_compare: int = 12,
) -> list[dict[str, str | int]]:
    m0_model = load_yolo_model(m0_weights)
    m2_model = load_yolo_model(m2_weights)
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    rows: list[dict[str, str | int]] = []
    saved_m0, saved_m2, saved_cmp = 0, 0, 0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt_boxes = load_ground_truth_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
        if not gt_boxes:
            continue

        m0_dets = detect_library_objects(m0_model, frame, confidence_threshold=conf, inference_size=imgsz)
        m2_dets = detect_library_objects(m2_model, frame, confidence_threshold=conf, inference_size=imgsz)
        m0_preds = [(map_detection_to_eval_class(o), o.confidence, o.bbox) for o in m0_dets]
        m2_preds = [(map_detection_to_eval_class(o), o.confidence, o.bbox) for o in m2_dets]

        m0_err = classify_image_errors(gt_boxes, m0_preds)
        m2_err = classify_image_errors(gt_boxes, m2_preds)

        m0_cat = pick_primary_error_category(m0_err)
        if m0_cat and saved_m0 < max_m0:
            dest = output_dir / "m0_failures" / m0_cat / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            header = f"M0 Kegagalan ({m0_cat}) | TP={m0_err['tp']}/{len(gt_boxes)} Missed={len(m0_err['missed'])} FP={len(m0_err['false_positives'])}"
            cv2.imwrite(str(dest), draw_error_visualization(frame, gt_boxes, m0_err, header))
            saved_m0 += 1
            rows.append({"gallery": "m0_failures", "category": m0_cat, "file": img_path.name, "tp": m0_err["tp"], "gt": len(gt_boxes), "missed": len(m0_err["missed"]), "fp": len(m0_err["false_positives"])})

        m2_cat = pick_primary_error_category(m2_err)
        if m2_cat and saved_m2 < max_m2:
            dest = output_dir / "m2_errors" / m2_cat / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            header = f"M2 Kesalahan ({m2_cat}) | TP={m2_err['tp']}/{len(gt_boxes)} Missed={len(m2_err['missed'])} FP={len(m2_err['false_positives'])}"
            cv2.imwrite(str(dest), draw_error_visualization(frame, gt_boxes, m2_err, header))
            saved_m2 += 1
            rows.append({"gallery": "m2_errors", "category": m2_cat, "file": img_path.name, "tp": m2_err["tp"], "gt": len(gt_boxes), "missed": len(m2_err["missed"]), "fp": len(m2_err["false_positives"])})

        if m2_err["tp"] >= m0_err["tp"] + 4 and saved_cmp < max_compare:
            left = draw_error_visualization(frame, gt_boxes, m0_err, f"M0 (Pretrained): TP={m0_err['tp']}/{len(gt_boxes)}")
            right = draw_error_visualization(frame, gt_boxes, m2_err, f"M2 (Fine-Tuned): TP={m2_err['tp']}/{len(gt_boxes)}")
            combined = np.hstack([left, right])
            dest = output_dir / "m0_fail_m2_success" / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), combined)
            saved_cmp += 1
            rows.append({"gallery": "m0_fail_m2_success", "category": "improvement", "file": img_path.name, "tp": m2_err["tp"], "gt": len(gt_boxes), "missed": len(m2_err["missed"]), "fp": len(m2_err["false_positives"])})

        if saved_m0 >= max_m0 and saved_m2 >= max_m2 and saved_cmp >= max_compare:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M0 failure and M2 error galleries for UAS report.")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/test/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/test/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/galleries"))
    parser.add_argument("--m0", default="yolo11s.pt")
    parser.add_argument("--m2", default="best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    rows = collect_galleries(args.images_dir, args.labels_dir, args.output_dir, args.m0, args.m2, args.conf, args.imgsz)
    csv_path = args.output_dir / "gallery_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Generated {len(rows)} gallery entries in '{args.output_dir}'. Summary saved to '{csv_path}'.")


if __name__ == "__main__":
    main()
