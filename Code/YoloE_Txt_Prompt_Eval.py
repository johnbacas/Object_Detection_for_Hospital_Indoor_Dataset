import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environment
import matplotlib.pyplot as plt
import supervision as sv
from supervision.metrics.mean_average_precision import MeanAveragePrecision as SupervisionMAP
import json
from ultralytics import YOLOE
from typing import List, Dict
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision as TorchmetricsMAP

def load_ground_truth_annotations(txt_path: str, image_width: int, image_height: int, classes: List[str]) -> sv.Detections:
    """
    Load ground truth annotations from txt file and return as supervision Detections object.
    
    Args:
        txt_path (str): Path to txt annotation file
        image_width (int): Width of the image
        image_height (int): Height of the image
        classes (list): List of class names
    
    Returns:
        sv.Detections: Ground truth detections
    """
    if not os.path.exists(txt_path):
        return sv.Detections.empty()
    
    with open(txt_path, 'r') as f:
        annotations = f.readlines()
    
    if not annotations:
        return sv.Detections.empty()
    
    xyxy = []
    class_ids = []
    confidences = []
    
    for annotation in annotations:
        parts = annotation.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:5])
            
            # Convert normalized coordinates to pixel coordinates
            xmin = int((center_x - width / 2) * image_width)
            ymin = int((center_y - height / 2) * image_height)
            xmax = int((center_x + width / 2) * image_width)
            ymax = int((center_y + height / 2) * image_height)
            
            xyxy.append([xmin, ymin, xmax, ymax])
            class_ids.append(class_id)
            confidences.append(1.0)  # Ground truth has confidence 1.0
    
    if not xyxy:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(xyxy),
        class_id=np.array(class_ids),
        confidence=np.array(confidences)
    )

def calculate_metrics_per_class(gt_detections: sv.Detections, pred_detections: sv.Detections, 
                               classes: List[str], iou_threshold: float = 0.5) -> Dict:
    """
    Calculate precision, recall, and AP per class using supervision library.
    
    Args:
        gt_detections: Ground truth detections
        pred_detections: Predicted detections
        classes: List of class names
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dict: Metrics per class
    """
    class_metrics = {}
    
    for class_id, class_name in enumerate(classes):
        # Filter detections for current class
        gt_class_mask = gt_detections.class_id == class_id
        pred_class_mask = pred_detections.class_id == class_id
        
        gt_class = gt_detections[gt_class_mask]
        pred_class = pred_detections[pred_class_mask]
        
        if len(gt_class) == 0 and len(pred_class) == 0:
            class_metrics[class_name] = {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0
            }
            continue
        
        if len(gt_class) == 0:
            class_metrics[class_name] = {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'tp': 0,
                'fp': len(pred_class),
                'fn': 0
            }
            continue
        
        if len(pred_class) == 0:
            class_metrics[class_name] = {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_class)
            }
            continue
        
        # Use supervision's IoU calculation for simpler metrics
        try:
            if len(pred_class) == 0:
                tp, fp, fn = 0, 0, len(gt_class)
            elif len(gt_class) == 0:
                tp, fp, fn = 0, len(pred_class), 0
            else:
                # Use supervision's box_iou for IoU calculation
                iou_matrix = sv.box_iou_batch(gt_class.xyxy, pred_class.xyxy)
                
                # Simple matching: for each prediction, find best GT match
                tp = 0
                matched_gt = set()
                
                # Sort predictions by confidence (descending)
                sorted_indices = np.argsort(-pred_class.confidence)
                
                for pred_idx in sorted_indices:
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx in range(len(gt_class)):
                        if gt_idx not in matched_gt:
                            iou = iou_matrix[gt_idx, pred_idx]
                            if iou > best_iou and iou >= iou_threshold:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                
                fp = len(pred_class) - tp
                fn = len(gt_class) - tp
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Calculate AP using simple approximation
            ap = precision * recall if precision > 0 and recall > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {e}")
            tp, fp, fn = 0, len(pred_class), len(gt_class)
            precision = recall = ap = 0.0
        
        class_metrics[class_name] = {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return class_metrics

def main():
    # Define class names
    classes = ['door_handle', 'chair', 'exam_table', 'countertop', 'waste_bin', 'utility_cart', 'computer',
               'keyboard', 'mouse', 'telephone', 'handle', 'tv', 'table', 'surgical_light', 'bedside_monitor',
               'dispenser', 'faucet', 'hospital_bed', 'medical_drawer', 'bedrail', 'foot_board', 'sofa',
               'overbed_table', 'sink', 'breathing_tube', 'curtain', 'bedside_table', 'electrosurgical_unit',
               'staff', 'operating_bed', 'patient', 'ventilator', 'iv_pole', 'infusion_pump',
               'surgical_instrument', 'visitor', 'toilet', 'sequential_compression', 'syringe_pump',
               'toilet_handle', 'hallway_assist_bar', 'xray_machine', 'xray_bed', 'light_switch',
               'restroom_assist_bar', 'panda_baby_warmer', 'press_to_open', 'push_latch', 'elevator_panel',
               'drawer', 'counter', 'cabinet', 'medical_waste_container', 'person', 'wheel_chair', 'incubator']
    classes = [s.replace('_', ' ').lower() for s in classes]
    
    # Define directories
    test_images_dir = 'dataset/test/images'
    test_labels_dir = 'dataset/test/labels'
    
    # Check if directories exist
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        return
    if not os.path.exists(test_labels_dir):
        print(f"Error: Test labels directory not found: {test_labels_dir}")
        return
    
    # Load YOLO-E model
    print("Loading YOLO-E model...")
    model = YOLOE("yoloe-11l-seg.pt")
    img_size = 640+640  # Define image size for prediction
    
    # Get all test images
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_images = sorted(test_images)
    #test_images = test_images[:20]  # Limit to first 20 images for evaluation
    
    print(f"Found {len(test_images)} test images")
    
    # Store all ground truth and prediction detections
    all_gt_detections = []
    all_pred_detections = []
    all_class_metrics_50 = []
    all_class_metrics_95 = []
    
    map_metric = SupervisionMAP()
    map_metric_torch = TorchmetricsMAP(class_metrics=True)
    map_metric.reset()

    # Process each test image
    for i, image_name in enumerate(test_images):
        #if i % 50 == 0:
        #    print(f"Processing image {i+1}/{len(test_images)}: {image_name}")
        
        # Load image
        image_path = os.path.join(test_images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        image_height, image_width, _ = image.shape
        
        # Load ground truth annotations using supervision
        label_name = image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(test_labels_dir, label_name)
        
        # GT Detection for Supervision MAP
        gt_detections = load_ground_truth_annotations(label_path, image_width, image_height, classes)
        print(f"GT Detections:{gt_detections.class_id}")


        # GT Detection for TorchMetrics MAP
        target = [
            dict(
                boxes=tensor(gt_detections.xyxy),
                labels=tensor(gt_detections.class_id),
            )
        ]

        if len(gt_detections) == 0:
            continue
        
        # Get unique classes for text prompting
        unique_class_ids = np.unique(gt_detections.class_id)
        unique_classes = [classes[class_id] for class_id in unique_class_ids if class_id < len(classes)]
        #print(f"Unique classes in GT: {unique_classes}")

        if not unique_classes:
            continue
        
        # Set classes for YOLO-E text prompting
        model.set_classes(unique_classes, model.get_text_pe(unique_classes))
        
        # Run prediction
        results = model.predict(image, imgsz=img_size, verbose=False)
        prediction = results[0]
        
        # Convert predictions to supervision Detections
        pred_detections = sv.Detections.from_ultralytics(prediction)
        #print(f"Pred Detections:\n{pred_detections.xyxy}\n{pred_detections.class_id}\n{pred_detections.confidence}")

        
        # Map prediction class indices back to original class space
        if len(pred_detections) > 0:
            # Create mapping from model output indices to original class indices
            mapped_class_ids = []
            for pred_class_idx in pred_detections.class_id:
                if pred_class_idx < len(unique_classes):
                    original_class_name = unique_classes[pred_class_idx]
                    original_class_id = classes.index(original_class_name)
                    mapped_class_ids.append(original_class_id)
                else:
                    mapped_class_ids.append(-1)  # Invalid class
            
            # Filter out invalid predictions
            valid_mask = np.array(mapped_class_ids) >= 0
            if np.any(valid_mask):
                pred_detections = pred_detections[valid_mask]
                pred_detections.class_id = np.array(mapped_class_ids)[valid_mask]  # Predictions for Supervision MAP
        print(f"Mapped Pred Class Id: {pred_detections.class_id}")

        pred = [   # Predictions for TorchMetrics MAP
            dict(
                boxes=tensor(pred_detections.xyxy),
                scores=tensor(pred_detections.confidence),
                labels=tensor(pred_detections.class_id),
            )
        ]

        # Store detections for global metrics
        all_gt_detections.append(gt_detections)
        all_pred_detections.append(pred_detections)
        
        # Calculate per-image metrics for mAP@0.5
        img_metrics_50 = calculate_metrics_per_class(gt_detections, pred_detections, classes, iou_threshold=0.5)
        total_tp = sum(m['tp'] for m in img_metrics_50.values())
        total_fp = sum(m['fp'] for m in img_metrics_50.values())
        total_fn = sum(m['fn'] for m in img_metrics_50.values())
        print(f"{image_name} - TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        all_class_metrics_50.append(img_metrics_50)
        #for key, value in img_metrics_50.items():
        #    print(f"{key:23} {value}")
        
        # Calculate per-image metrics for mAP@0.5:0.95 (average over multiple IoU thresholds)
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        img_metrics_95 = []
        for iou_thresh in iou_thresholds:
            metrics = calculate_metrics_per_class(gt_detections, pred_detections, classes, iou_threshold=iou_thresh)
            img_metrics_95.append(metrics)
        all_class_metrics_95.append(img_metrics_95)

        # Update global mAP metrics (Supervision and TorchMetrics)
        map_metric.update(pred_detections, gt_detections)
        map_metric_torch.update(pred, target)


    print("Calculating final mAP scores...")
    map_result = map_metric.compute() # Compute Supervision mAP
    map_result_torch = map_metric_torch.compute() # Compute TorchMetrics mAP

    print('SuperVision mAP50-95:', map_result.map50_95)
    print('Torchmetrics mAP50-95:', map_result_torch['map'])
    print('Supervision mAP Scores per IOU:', map_result.mAP_scores)
    print('Torchmetrics mAP50:', map_result_torch['map_50'])
    print('SuperVision mAP75:', map_result.map75)
    print('Torchmetrics mAP75:', map_result_torch['map_75'])
    print('Torchmetrics mAP Scores per class:', map_result_torch['map_per_class'].tolist())

    mAPscores=[]
    print()
    for i, object_ap in enumerate(map_result.ap_per_class):
        mAPscores.append([classes[i],float(object_ap[0]),float(object_ap[5]),float(sum(object_ap)/len(object_ap))])
        #scores.append(object_ap[0])
    print(f"SuperVision mAP50, 75, 50-95 for every class {mAPscores}")
    print()
    model_name = os.path.splitext(model.model_name)[0]
    # Save detailed results
    results_dict = {
        'model_used': model_name,
        'total_images_processed': len(test_images),
        'Supervision_mAP_50': map_result.map50,
        'Torchmetrics_mAP_50': map_result_torch['map_50'].item(),
        'Supervision_mAP_75': map_result.map75,
        'Torchmetrics_mAP_75': map_result_torch['map_75'].item(),
        'Supervision_mAP_50_95': map_result.map50_95,
        'Torchmetrics_mAP_50_95': map_result_torch['map'].item(),
        'Supervision_mAP50_75_50-95_scores_per_class': mAPscores,
        'Torchmetrics_mAP50-95_scores_per_class': map_result_torch['map_per_class'].tolist(),
        'Supervision_library_version': sv.__version__ if hasattr(sv, '__version__') else 'unknown'
    }
    
    # Save results to JSON
    output_file = f'{model_name}_{img_size}_txt_prompt_eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\nEvaluation completed using Supervision and TorchMetrics libraries!")

if __name__ == "__main__":
    main()
