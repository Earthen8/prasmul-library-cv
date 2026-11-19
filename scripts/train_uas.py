import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENT_CONFIGS = {
    "m1": {
        "base_model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 100,
        "patience": 30,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "notes": "Baseline M1 (yolo11s, imgsz=640, default augmentation)",
    },
    "iter1_aug": {
        "base_model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 100,
        "patience": 30,
        "hsv_v": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "notes": "Iterasi 1: Augmentasi pencahayaan (hsv_v=0.5) untuk silau jendela",
    },
    "iter2_imgsz": {
        "base_model": "yolo11s.pt",
        "imgsz": 960,
        "epochs": 100,
        "patience": 30,
        "hsv_v": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "notes": "Iterasi 2: Tingkatkan resolusi inferensi/training imgsz=960 untuk objek jauh",
    },
    "iter3_m2": {
        "base_model": "yolo11m.pt",
        "imgsz": 960,
        "epochs": 100,
        "patience": 30,
        "hsv_v": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "notes": "Iterasi 3 (Kandidat M2): Kapasitas model yolo11m.pt + imgsz=960",
    },
}


def run_single_experiment(stage: str, data_yaml: Path, seed: int, batch: int) -> dict[str, str | float]:
    from ultralytics import YOLO

    cfg = EXPERIMENT_CONFIGS[stage]
    run_name = f"{stage}_seed{seed}"
    model = YOLO(cfg["base_model"])
    results = model.train(
        data=str(data_yaml),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch=batch,
        seed=seed,
        hsv_v=cfg["hsv_v"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        project="runs/uas",
        name=run_name,
    )
    box = results.box
    return {
        "experiment": run_name,
        "base_model": cfg["base_model"],
        "imgsz": cfg["imgsz"],
        "seed": seed,
        "val_precision": round(float(box.mp), 4),
        "val_recall": round(float(box.mr), 4),
        "val_mAP50": round(float(box.map50), 4),
        "val_mAP50_95": round(float(box.map), 4),
        "weights_path": str(Path("runs/uas") / run_name / "weights" / "best.pt"),
        "notes": cfg["notes"],
    }


def append_experiment_log(rows: list[dict[str, str | float]], log_csv: Path) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible M1 baseline and 3 iterative experiments toward M2 (UAS Step 2B)."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to exported YOLO data.yaml.")
    parser.add_argument(
        "--stage",
        choices=[*EXPERIMENT_CONFIGS.keys(), "all"],
        default="m1",
        help="Experiment stage to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42], help="Random seeds (min 2).")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--log-csv", type=Path, default=Path("results/tabel_percobaan_uas.csv"))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found at '{args.data}'. Export YOLO dataset from Roboflow first.")

    stages = list(EXPERIMENT_CONFIGS.keys()) if args.stage == "all" else [args.stage]
    rows = [
        run_single_experiment(stage, args.data, seed, args.batch)
        for stage in stages
        for seed in args.seeds
    ]
    append_experiment_log(rows, args.log_csv)
    print(f"Saved {len(rows)} experiment results to '{args.log_csv}'.")


if __name__ == "__main__":
    main()
