from utils import get_class_to_images_mapping, get_random_images_for_class, get_indexed_images_for_class , annotate_images, annotate_ref_images

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
from sklearn.cluster import kmeans_plusplus

def extract_embeddings(vps_predictor, image_path, bboxes):
    img = cv2.imread(image_path)
    np_boxes = np.array(bboxes)
    #np_classes = np.array([0] * len(np_boxes)) # Dummy class ID
    np_classes = np.array(list(range(len(np_boxes)))) # Dummy class IDs
    visual_prompts = dict(bboxes=np_boxes, cls=np_classes)
    vps_predictor.set_prompts(visual_prompts)
    embeddings = vps_predictor.get_vpe(img)
    #print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings

def sample_diverse_embeddings(
    predictor,
    image_path_list,
    bboxes_list,
    N,
    min_distance,
    max_distance, 
    extract_embedding_fn,
    verbose=True
):
    """
    Sample N diverse embeddings using k-means++ inspired probabilistic acceptance
    Assumes extract_embedding_fn returns torch tensors
    
    Algorithm:
    1. Randomly select first image
    2. For each subsequent image:  
       - Extract embedding
       - Calculate max distance to already selected embeddings using torch.cdist
       - Accept with probability based on distance (min_distance=0%, max_distance=100%)
    3. Continue until N embeddings collected or images exhausted
    
    Args:  
        image_path_list: list of image paths
        N: number of embeddings to collect
        min_distance: distance threshold for 0% acceptance probability
        max_distance: distance threshold for 100% acceptance probability
        extract_embedding_fn: function that takes image_path and returns torch. Tensor embedding
        verbose: whether to print progress (default: True)
        
    Returns:
        embedding_list:  list of selected embeddings (torch.Tensor)
        selected_paths: list of selected image paths
        stats: dictionary with sampling statistics
    """
    
    # Make a copy to avoid modifying original list
    remaining_images = image_path_list.copy()
    remaining_boxes = bboxes_list.copy()
    
    embedding_tensor = None
    selected_paths = []
    
    # Statistics tracking
    total_attempts = 0
    accepted_count = 0
    rejection_count = 0
    acceptance_probabilities = []
    distances_seen = []
    
    if verbose:
        print(f"K-means++ Diversity Sampling (torch.cdist)")
        print("="*70)
        print(f"Target centroids (N): {N}")
        print(f"Total images:  {len(remaining_images)}")
        print(f"Distance range: [{min_distance:.4f}, {max_distance:.4f}]")
        print("="*70)
    
    # Step 1: Select first image randomly
    first_idx = np.random.randint(0, len(remaining_images))
    #print(f"First index: {first_idx}")
    first_path = remaining_images.pop(first_idx)
    #print(f"First path: {first_path}")
    first_boxes = remaining_boxes.pop(first_idx)
    #print(f"First boxes: {first_boxes}")
    if verbose:
        print(f"\n[Iteration 1/{N}]")
        print(f"  Randomly selected: {first_path}")
    
    first_embedding = extract_embedding_fn(predictor, first_path, first_boxes)
    print(f"  Extracted embedding shape: {first_embedding.shape}") # (1, N, 512 )
    embeddings_tensor = first_embedding.squeeze(0)
    print(f"  Embeddings tensor shape: {embeddings_tensor.shape}")  # (N, 512)
    selected_paths.append(first_path)
    
    accepted_count += 1
    total_attempts += 1
    
    if verbose:
        print(f"  ✓ Accepted (first selection)")
        print(f"  Embeddings collected: {embeddings_tensor.shape[0]}/{N}")
    
    # Step 2: Iteratively sample with probabilistic acceptance
    while len(selected_paths) < N and len(remaining_images) > 0:
        iteration = len(selected_paths) + 1
        total_attempts += 1
        
        if verbose:
            print(f"\n[Attempt {total_attempts}, Target {iteration}/{N}]")
        
        #print(f"  Remaining images: {len(remaining_images)}")
        #print(f"  Boxes list length: {len(remaining_boxes)}")

        # Take next image (random selection)
        candidate_idx = np.random.randint(0, len(remaining_images))
        candidate_path = remaining_images.pop(candidate_idx)
        candidate_boxes = remaining_boxes.pop(candidate_idx)
        
        if verbose:
            print(f"  Candidate: {candidate_path}")
        
        # Extract embedding (torch. Tensor)
        candidate_embedding = extract_embedding_fn(predictor, candidate_path, candidate_boxes)
        candidate_embedding = candidate_embedding.squeeze(0)  # (N, 512)
        # Calculate distances to all embeddings in embedding_list using torch.cdist
        distances = torch.cdist(candidate_embedding, embeddings_tensor)
        #print(f"  Distances shape: {distances.shape}")
        #print(f"  Distances: {distances}")
        distances = distances.flatten()  # shape [num_selected]
        # Find minimum distance
        #max_dist = distances.max().item()
        min_dist = distances.min().item()
        distances_seen.append(min_dist)
        
        if verbose:
            print(f"  Min distance to selected: {min_dist:.4f}")
        
        # Calculate acceptance probability
        if min_dist >= max_distance:
            acceptance_prob = 1.0
        elif min_dist <= min_distance:
            acceptance_prob = 0.0
        else:
            acceptance_prob = (min_dist - min_distance) / (max_distance - min_distance)
            #acceptance_prob = (max_distance - max_dist) / (max_distance - min_distance)
        
        acceptance_probabilities.append(acceptance_prob)
        
        if verbose:
            print(f"  Acceptance probability: {acceptance_prob:.2%}")
        
        # Accept or reject based on probability
        if np.random.random() < acceptance_prob:
            embeddings_tensor = torch.cat((embeddings_tensor, candidate_embedding), dim=0)
            selected_paths.append(candidate_path)
            accepted_count += 1
            if verbose:
                print(f"  ✓ ACCEPTED")
                print(f"  Embeddings collected: {embeddings_tensor.shape[0]}/{N}")
        else:
            rejection_count += 1
            if verbose:
                print(f"  ✗ REJECTED")
                print(f"  Remaining images: {len(remaining_images)}")
    
    # Final statistics
    if verbose:
        print(f"\n{'='*70}")
        print(f"SAMPLING COMPLETE")
        print(f"{'='*70}")
        print(f"Embeddings collected: {embeddings_tensor.shape[0]}/{N}")
        print(f"Total attempts: {total_attempts}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {rejection_count}")
        print(f"Acceptance rate: {accepted_count/total_attempts:.2%}")
        
        if embeddings_tensor.shape[0] < N:
            print(f"\n⚠ Warning: Only collected {embeddings_tensor.shape[0]} embeddings (target was {N})")
            print(f"  Reason: Ran out of images (processed all {len(image_path_list)} images)")
        
    
    stats = {
        'Min_distance': min_distance,
        'Max_distance': max_distance,
        'total_attempts': total_attempts,
        'accepted_count': accepted_count,
        'rejection_count': rejection_count,
        'acceptance_rate': accepted_count / total_attempts if total_attempts > 0 else 0,
        'acceptance_probabilities': acceptance_probabilities,
        'distances_seen': distances_seen,
        'selected_paths': selected_paths,
        'N_target': N,
        'N_collected': len(selected_paths),
        'embeddings_collected': embeddings_tensor.shape[0]
    }
    
    return embeddings_tensor, selected_paths, stats


if __name__ == "__main__":
    # Generate class to images mapping
    # It's a dictionary where keys are class IDs and values are
    # lists of image names containing those classes

    #class_mapping = get_class_to_images_mapping()
    train_class_mapping = get_class_to_images_mapping('dataset/train')
    test_class_mapping = get_class_to_images_mapping('dataset/test')


    #class_to_search = 0  # Example class ID to search for
    number_of_experiments = 3  # Number of experiments to run
    OUTPUT_DIR = "Results_2nd_Dist_8_14_k_15"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #random_imgs = 30  # Number of random images for embeddings
    centroids = 15
    max_dist=1.4
    min_dist=0.8
    
    for class_to_search in range(56):  # Loop over class IDs
        number_of_train_images = len(train_class_mapping[class_to_search])
        number_of_test_images = len(test_class_mapping.get(class_to_search, [])) # Select all images of the class in test set
        
        test_images, test_boxes, _, _ = get_random_images_for_class(
            class_to_search,
            number_of_test_images,
            test_class_mapping
        )
        test_image_paths = ['dataset/test/images/' + img for img in test_images]

        reference_images, reference_bboxes, _, _ = get_random_images_for_class(
            class_to_search,
            number_of_train_images,
            train_class_mapping,
            'dataset/train'
        )
        reference_image_paths = ['dataset/train/images/' + img for img in reference_images]
            
        model = YOLOE("yoloe-11l-seg.pt")
        model.model.model[-1].nc = 1
        model.model.names = ['obj0']
        vps_predictor = YOLOEVPSegPredictor()
        vps_predictor.setup_model(model=model.model)
            
        for exp_num in range(number_of_experiments):
            start_time = time.time()
            final_embedding, selected_images, stats = sample_diverse_embeddings(
                vps_predictor,
                reference_image_paths,
                reference_bboxes,
                centroids,
                min_distance=min_dist,
                max_distance=max_dist,
                extract_embedding_fn=extract_embeddings,
                verbose=False
            )    
            stop_time = time.time()
            embeddings_time = stop_time - start_time
            print(f"Time taken to compute embeddings: {embeddings_time:.2f} seconds")
            
            for img_path in selected_images:
                print(f"Selected image: {os.path.basename(img_path)}")
            print(f"Final embedding shape: {final_embedding.shape}")
            
            mean_embedding = final_embedding.mean(dim=0, keepdim=True)  # shape [1, 512]
            mean_norm_embedding = F.normalize(mean_embedding, p=2, dim=1, eps=1e-12)  
            mean_emb_norm = mean_norm_embedding.unsqueeze(0)  # shape [1, 1, 512]

            model.set_classes(model.model.names, mean_emb_norm)

            map_metric = SupervisionMAP()
            map_metric.reset()

            batch = 30
            for j in range(0, len(test_image_paths), batch):
                batch_images = test_image_paths[j:j+batch]
                results = model.predict(batch_images, conf=0.25)
                #for i, r in enumerate(results):
                #    r.plot(save=True, filename=f"Outputs/output_{i}.jpg")

                predictions_list = []
                for i, pred in enumerate(results):
                    pred_detections = sv.Detections.from_ultralytics(pred)
                    pred_detections.class_id = np.array([class_to_search]*len(pred_detections.class_id))
                    #print(f"Prediction results for image {i}:", pred_detections.xyxy, pred_detections.class_id)
                    predictions_list.append(pred_detections)

                gt_list = []
                batch_test = test_boxes[j:j+batch]
                for i, bboxes in enumerate(batch_test):
                    gt_detections = sv.Detections(
                        xyxy = np.array(bboxes),
                        class_id = np.array([class_to_search]*len(bboxes))
                    )
                    #print(f"Ground truth for image {i}:", gt_detections.xyxy, gt_detections.class_id)
                    gt_list.append(gt_detections)

                    # Annotate and save images
                #annotate_images(test_images[j:j+batch], 'dataset/test/', gt_list, predictions_list)
                map_result = map_metric.update(predictions_list, gt_list)
            
            #annotate_ref_images(reference_images, 'dataset/train/', reference_bboxes)
            #print('Reference Images:')
            #for img, bboxes in (zip(reference_images, reference_bboxes)):
            #    print(f'{os.path.basename(img)} with {len(bboxes)} box(es)')

            map_result = map_metric.compute()
            print(f"mAP50_95 Result: {map_result.map50_95:.5f}")
            print(f"mAP50: {map_result.map50:.5f}")
            
            # Collect essential experiment data
            experiment_results = {
                "timestamp": datetime.now().isoformat(),
                "class_to_search": class_to_search,
                "number_of_reference_images": len(selected_images),
                "number_of_test_images": number_of_test_images,
                "mAP50": float(map_result.map50),
                "mAP75": float(map_result.map75),
                "mAP50_95": float(map_result.map50_95),
                "embeddings_time_seconds": embeddings_time,
                "reference_images": [os.path.basename(path) for path in selected_images],
                "total_reference_image_bboxes": final_embedding.shape[0],
                "final_embedding": mean_emb_norm.squeeze().tolist()  # Convert tensor to list
            }

            #print(mean_emb_norm.squeeze().tolist())
            #'''
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"2ndexperiment_class_{class_to_search}_{timestamp_str}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save to JSON file
            try:
                with open(filepath, 'w') as f:
                    json.dump([experiment_results, stats], f, indent=2)
                print(f"Experiment results saved to: {filepath}")
            except Exception as e:
                print(f"Error saving experiment results: {e}")
            print('-'*70)
            #'''
            time.sleep(2) # time between experiments
        time.sleep(5) # time between classes