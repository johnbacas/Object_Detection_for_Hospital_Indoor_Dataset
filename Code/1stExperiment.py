from utils import get_class_to_images_mapping, get_random_images_for_class, get_indexed_images_for_class

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import supervision as sv
import json
import time
from datetime import datetime
from supervision.metrics.mean_average_precision import MeanAveragePrecision as SupervisionMAP
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


if __name__ == "__main__":
    # Generate class to images mapping
    train_class_mapping = get_class_to_images_mapping('dataset/train')
    test_class_mapping = get_class_to_images_mapping('dataset/test')
    
    # Configuration
    class_to_search = 55  # Class ID to search for
    number_of_experiments = 3  # Number of experiments to run
    number_of_images = 15  # Number of reference images
    OUTPUT_DIR = "Results_1st_15_random_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # Select all images of the class in test set
    test_images, test_boxes, _, _ = get_indexed_images_for_class(
        class_to_search,
        list(range(len(test_class_mapping.get(class_to_search, [])))), 
        test_class_mapping
    )
    
    for exp_num in range(number_of_experiments):
        # Select reference images from train set
        reference_images, reference_bboxes, _, _ = get_random_images_for_class(
            class_to_search, 
            number_of_images,
            train_class_mapping, 
            'dataset/train'
        )
        
        indexes = [train_class_mapping[class_to_search].index(img) for img in reference_images]
        print(f"Selected reference image indexes: {indexes}")

        print(f"Class: {class_to_search}")
        print(f"Reference images: {reference_images}")
        print(f"Number of Test images: {len(test_images)}")
        
        # Initialize model
        model = YOLOE("yoloe-11l-seg.pt")
        model.model.model[-1].nc = 1
        model.model.names = ['obj0']
        vps_predictor = YOLOEVPSegPredictor()
        vps_predictor.setup_model(model=model.model)
        
        # Extract embeddings from reference images
        reference_image_paths = ['dataset/train/images/' + img for img in reference_images]
        embeddings_list = []
        for i, img_path in enumerate(reference_image_paths):
            img = cv2.imread(img_path)
            np_boxes = np.array(reference_bboxes[i])
            np_classes = np.array(list(range(len(np_boxes))))
            visual_prompts = dict(bboxes=np_boxes, cls=np_classes)
            vps_predictor.set_prompts(visual_prompts)
            embeddings = vps_predictor.get_vpe(img)
            embeddings_list.append(embeddings)
        
        # Concatenate and average embeddings
        final_embedding = torch.cat(embeddings_list, dim=1)
        mean_emb = final_embedding.mean(dim=1, keepdim=True)
        mean_emb_norm = F.normalize(mean_emb, p=2, dim=2, eps=1e-12)
        
        # Set the embeddings on the model
        model.set_classes(model.model.names, mean_emb_norm)
        
        # Prepare test images
        test_image_paths = ['dataset/test/images/' + img for img in test_images]
        
        # Initialize metrics
        map_metric = SupervisionMAP()
        map_metric.reset()
        
        # Process in batches
        batch = 30
        for j in range(0, len(test_image_paths), batch):
            batch_images = test_image_paths[j:j+batch]
            results = model.predict(batch_images, conf=0.25)
            
            predictions_list = []
            for i, pred in enumerate(results):
                pred_detections = sv.Detections.from_ultralytics(pred)
                pred_detections.class_id = np.array([0]*len(pred_detections.class_id))
                predictions_list.append(pred_detections)
            
            gt_list = []
            batch_test = test_boxes[j:j+batch]
            for i, bboxes in enumerate(batch_test):
                gt_detections = sv.Detections(
                    xyxy=np.array(bboxes),
                    class_id=np.array([0]*len(bboxes))
                )
                gt_list.append(gt_detections)
            
            # Update metrics
            map_metric.update(predictions_list, gt_list)
        
        # Compute final metrics
        map_result = map_metric.compute()
        
        # Extract mAP values
        mAP50 = float(map_result.map50)
        mAP75 = float(map_result.map75)
        mAP50_95 = float(map_result.map50_95)
        
        print(f"\nmAP50: {mAP50:.4f}")
        print(f"mAP75: {mAP75:.4f}")
        print(f"mAP50-95: {mAP50_95:.4f}")
        
        # Prepare results dictionary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dict = {
            #'timestamp': datetime.now().isoformat(),
            'timestamp': timestamp,
            'class_to_search': class_to_search,
            'mAP50': mAP50,
            'mAP75': mAP75,
            'mAP50_95': mAP50_95,
            'reference_images': reference_images,
            'reference_image_indexes': indexes,
            'reference_image_bboxes': [len(bboxes) for bboxes in reference_bboxes],
            'final_embedding': mean_emb_norm.squeeze().cpu().numpy().tolist()
        }
        #'''
        # Save results
        output_filename = f'{OUTPUT_DIR}/experiment_class_{class_to_search}_{timestamp}.json'
        #os.makedirs('Results_2', exist_ok=True)
        with open(output_filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_filename}")
        print('-'*70)
        #'''
        time.sleep(2)