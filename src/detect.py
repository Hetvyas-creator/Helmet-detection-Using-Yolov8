import os
import argparse
import random
import cv2
from ultralytics import YOLO
from utils import ensure_dir, bundle_light_inference


def run_inference(weights, source, n_samples=10):
    model = YOLO(weights)

    preds = model.predict(
        source=source,
        save=True,
        project="runs/detect",
        name="exp",
        imgsz=640,
        exist_ok=False,
        conf=0.25
    )

    run_dir = preds[0].save_dir
    run_name = os.path.basename(run_dir)

    overlay_dir = os.path.join(run_dir, "pred_overlays")
    ensure_dir(overlay_dir)

    sampled_preds = random.sample(preds, min(n_samples, len(preds)))
    for pred in sampled_preds:
        img_overlay = pred.plot()
        out_name = os.path.splitext(os.path.basename(pred.path))[0] + "_overlay.jpg"
        cv2.imwrite(os.path.join(overlay_dir, out_name), img_overlay)

    print(f"{len(sampled_preds)} overlays saved to {overlay_dir}")

    # 🔴 define training results dir manually
    train_results_dir = "train/helmet_yolov8/exp"   # change if different

    bundle_light_inference(weights, overlay_dir, train_results_dir, run_name, output_dir="Light_Bundles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="model/yolov11/best.pt", help="Path to weights")
    parser.add_argument("--source", type=str, default="data/images/test", help="Folder with test images")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of overlays to save")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        n_samples=args.n_samples
    )

