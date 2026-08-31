import os
import sys
import argparse
import time
import json
import random
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add project root to sys.path to allow src imports
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import load_inference_model, predict_image, resolve_device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Inference Testing")
    parser.add_argument("--samples-per-class", type=int, default=3, help="Number of samples per class to test")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    parser.add_argument("--output-dir", type=str, default="reports/frontend_test", help="Output directory for reports")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--specific-images", type=str, nargs="+", default=None, help="List of specific image paths to test")
    return parser.parse_args()

def select_samples(split_file, split_name, samples_per_class, seed):
    df = pd.read_csv(split_file)
    df_split = df[df['split'] == split_name]
    
    selected_rows = []
    
    # Sort to ensure reproducibility before sampling
    class_groups = df_split.groupby('label')
    
    for class_name, group in class_groups:
        group_sorted = group.sort_values(by='filepath')
        n_samples = min(samples_per_class, len(group_sorted))
        if n_samples < samples_per_class:
            print(f"Warning: Class {class_name} has only {n_samples} samples in split {split_name} (requested {samples_per_class}).")
        
        # Sample using pandas with seed
        sampled = group_sorted.sample(n=n_samples, random_state=seed)
        selected_rows.append(sampled)
        
    if not selected_rows:
        return pd.DataFrame()
        
    return pd.concat(selected_rows)

def create_gallery(results_df, output_dir):
    # results_df has columns: image_path_relative, ground_truth, predicted_class, confidence_score, top1_correct, status
    # We only plot successfully processed images
    valid_df = results_df[results_df['status'] == 'Success']
    if len(valid_df) == 0:
        return
        
    images_per_page = 15
    num_pages = math.ceil(len(valid_df) / images_per_page)
    
    for page in range(num_pages):
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, len(valid_df))
        page_df = valid_df.iloc[start_idx:end_idx]
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(page_df.iterrows()):
            ax = axes[i]
            img_path = row['image_path_relative']
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except:
                ax.text(0.5, 0.5, 'Image Load Error', ha='center', va='center')
                
            color = 'green' if row['top1_correct'] else 'red'
            title = f"GT: {row['ground_truth']}\nPred: {row['predicted_class']}\nConf: {row['confidence_score']*100:.1f}%"
            ax.set_title(title, color=color, fontsize=9)
            ax.axis('off')
            
        for i in range(len(page_df), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        page_suffix = f"_{page+1:02d}" if num_pages > 1 else ""
        gallery_path = os.path.join(output_dir, f"prediction_gallery{page_suffix}.png")
        plt.savefig(gallery_path)
        plt.close(fig)

def create_confidence_summary(results_df, output_dir):
    valid_df = results_df[results_df['status'] == 'Success']
    if len(valid_df) == 0:
        return
        
    correct_confs = valid_df[valid_df['top1_correct'] == True]['confidence_score'].values
    incorrect_confs = valid_df[valid_df['top1_correct'] == False]['confidence_score'].values
    
    if len(correct_confs) > 0 and len(incorrect_confs) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(correct_confs, bins=10, alpha=0.5, label='Correct', color='green')
        plt.hist(incorrect_confs, bins=10, alpha=0.5, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Score Distribution: Correct vs Incorrect')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "confidence_summary.png"))
        plt.close()
    else:
        print("Note: Skipping confidence_summary.png as either all predictions were correct or all were incorrect.")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Resolving device: {args.device}")
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    print("Loading model and labels...")
    model, idx_to_class, cfg = load_inference_model(device=device)
    
    # Assertions on model/label loading
    assert len(idx_to_class) == 15, f"Expected 15 classes, got {len(idx_to_class)}"
    assert min(idx_to_class.keys()) == 0 and max(idx_to_class.keys()) == 14, "Class indices must cover 0 through 14"
    assert not model.training, "Model must be in evaluation mode"
    
    # Warm-up inference
    print("Running warm-up inference...")
    dummy_img = Image.new('RGB', (224, 224))
    _ = predict_image(model, idx_to_class, dummy_img, device)
    
    images_to_test = []
    
    if args.specific_images:
        for img_path in args.specific_images:
            # Try to resolve ground truth from path if it contains PlantVillage
            ground_truth = "unknown"
            if "PlantVillage" in img_path:
                parts = Path(img_path).parts
                if len(parts) >= 2:
                    ground_truth = parts[-2].replace("_", " ") # Assuming parent dir is class name
            images_to_test.append({"image_path": img_path, "class_name": ground_truth, "split": "specific"})
    else:
        split_csv = Path("data/splits/split.csv")
        if not split_csv.exists():
            raise FileNotFoundError(f"split.csv not found at {split_csv}")
            
        df_selected = select_samples(split_csv, args.split, args.samples_per_class, args.seed)
        
        for _, row in df_selected.iterrows():
            images_to_test.append({
                "image_path": row['filepath'],
                "class_name": row['label'].replace("_", " "),
                "split": args.split
            })
            
    if not images_to_test:
        print("No images selected for testing.")
        return
        
    print(f"Testing {len(images_to_test)} images...")
    
    results = []
    total_processed = 0
    total_failed = 0
    
    for i, item in enumerate(images_to_test):
        img_path = item["image_path"]
        ground_truth = item["class_name"]
        split = item["split"]
        
        result_row = {
            "sample_number": i + 1,
            "split": split,
            "image_path_relative": img_path,
            "ground_truth": ground_truth,
            "predicted_class": "",
            "top1_correct": False,
            "top3_contains_ground_truth": False,
            "confidence_score": 0.0,
            "top2_class": "",
            "top2_score": 0.0,
            "top3_class": "",
            "top3_score": 0.0,
            "inference_time_ms": 0.0,
            "device": str(device),
            "status": "Failed",
            "error_message": ""
        }
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            predicted_class, confidence, top_classes, top_probs = predict_image(
                model, idx_to_class, img, device
            )
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            
            # Structural assertions inside prediction
            assert len(top_classes) == 3, "Top-classes must be 3"
            assert len(top_probs) == 3, "Top-probs must be 3"
            assert top_probs[0] >= top_probs[1] >= top_probs[2], "Scores not sorted descending"
            assert 0.99 <= sum(top_probs) <= 1.01 or sum(top_probs) <= 1.0, f"Scores sum to {sum(top_probs)}, expects ~1.0"
            assert np.isfinite(top_probs).all(), "Non-finite probability scores"
            assert predicted_class == top_classes[0], "Predicted class doesn't match top-1"
            assert confidence == top_probs[0], "Confidence doesn't match top-1 score"
            
            top1_correct = (predicted_class == ground_truth) if ground_truth != "unknown" else False
            top3_contains = (ground_truth in top_classes) if ground_truth != "unknown" else False
            
            result_row.update({
                "predicted_class": predicted_class,
                "top1_correct": top1_correct,
                "top3_contains_ground_truth": top3_contains,
                "confidence_score": confidence,
                "top2_class": top_classes[1],
                "top2_score": top_probs[1],
                "top3_class": top_classes[2],
                "top3_score": top_probs[2],
                "inference_time_ms": inference_time_ms,
                "status": "Success"
            })
            total_processed += 1
            
        except Exception as e:
            result_row["error_message"] = str(e)
            total_failed += 1
            print(f"Error processing {img_path}: {e}")
            
        results.append(result_row)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "predictions.csv", index=False)
    
    valid_results = df_results[df_results["status"] == "Success"]
    known_gt_results = valid_results[valid_results["ground_truth"] != "unknown"]
    
    top1_correct_count = int(known_gt_results["top1_correct"].sum())
    top3_correct_count = int(known_gt_results["top3_contains_ground_truth"].sum())
    total_known = len(known_gt_results)
    
    sample_top1_acc = top1_correct_count / total_known if total_known > 0 else 0
    sample_top3_acc = top3_correct_count / total_known if total_known > 0 else 0
    mean_conf = float(valid_results["confidence_score"].mean()) if len(valid_results) > 0 else 0
    mean_time = float(valid_results["inference_time_ms"].mean()) if len(valid_results) > 0 else 0
    median_time = float(valid_results["inference_time_ms"].median()) if len(valid_results) > 0 else 0
    
    summary = {
        "model_name": cfg["model"].get("display_name", cfg["model"]["name"]),
        "checkpoint_filename": "vit_b16_best.pt",
        "split_tested": args.split if not args.specific_images else "specific",
        "seed": args.seed,
        "requested_samples_per_class": args.samples_per_class,
        "total_images_selected": len(images_to_test),
        "total_successfully_processed": total_processed,
        "total_failed": total_failed,
        "number_of_classes_represented": len(known_gt_results["ground_truth"].unique()),
        "top1_correct_count": top1_correct_count,
        "top3_correct_count": top3_correct_count,
        "sample_top1_accuracy": sample_top1_acc,
        "sample_top3_accuracy": sample_top3_acc,
        "mean_confidence_score": mean_conf,
        "mean_inference_time_ms": mean_time,
        "median_inference_time_ms": median_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "test_timestamp": datetime.now().isoformat(),
        "disclaimer": "This is a frontend functional sample test and not the official evaluation"
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # Class summary
    if len(known_gt_results) > 0:
        class_groups = known_gt_results.groupby("ground_truth")
        class_summaries = []
        for cls_name, group in class_groups:
            n_images = len(group)
            t1_correct = int(group["top1_correct"].sum())
            t3_correct = int(group["top3_contains_ground_truth"].sum())
            mean_c = float(group["confidence_score"].mean())
            
            class_summaries.append({
                "class_name": cls_name,
                "images_tested": n_images,
                "top1_correct": t1_correct,
                "top3_correct": t3_correct,
                "sample_top1_accuracy": t1_correct / n_images,
                "sample_top3_accuracy": t3_correct / n_images,
                "mean_confidence": mean_c
            })
            
        df_class_summary = pd.DataFrame(class_summaries)
        df_class_summary.to_csv(output_dir / "class_summary.csv", index=False)
        
    # Visualizations
    create_gallery(df_results, output_dir)
    create_confidence_summary(df_results, output_dir)
    
    print("\n=== Automated Frontend Sample-Test Summary ===")
    print(f"Total Selected: {len(images_to_test)}")
    print(f"Processed: {total_processed}")
    print(f"Failed: {total_failed}")
    if total_known > 0:
        print(f"Sample Top-1 Accuracy: {sample_top1_acc*100:.2f}%")
        print(f"Sample Top-3 Accuracy: {sample_top3_acc*100:.2f}%")
    print(f"Mean Inference Time: {mean_time:.2f} ms")
    print("==============================================")
    print(f"Reports saved to: {output_dir}")

if __name__ == "__main__":
    main()
