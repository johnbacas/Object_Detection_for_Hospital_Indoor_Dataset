import os
import cv2
import numpy as np
import supervision as sv
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def get_class_to_images_mapping(hospital_data_path: str) -> Dict[int, List[str]]:
    """
    Returns a dictionary with class IDs as keys and lists of image names containing those classes as values.
    
    Args:
        hospital_data_path: Path to the Hospital_Scene_Data-main folder
        
    Returns:
        Dictionary mapping class_id -> list of image names
    """
    # Initialize the mapping dictionary
    class_to_images = defaultdict(list)
    
    # Path to the txt labels folder
    labels_path = os.path.join(hospital_data_path, "labels")
    images_path = os.path.join(hospital_data_path, "images")
    
    # Check if paths exist
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels folder not found: {labels_path}")
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images folder not found: {images_path}")
    
    # Get all label files
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    print(f"Processing {len(label_files)} label files...")
    
    for label_file in label_files:
        # Extract image name (remove .txt extension)
        image_name = label_file.replace('.txt', '')
        
        # Check if corresponding image exists
        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_exists = False
        for ext in possible_extensions:
            if os.path.exists(os.path.join(images_path, image_name + ext)):
                image_exists = True
                image_name += ext  # Store the full image name with extension
                break
        
        if not image_exists:
            print(f"Warning: No corresponding image found for {label_file}")
            continue
        
        # Read the label file
        label_path = os.path.join(labels_path, label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Parse each line to extract class IDs
            classes_in_image = set()
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO format: class_id x y w h
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
            
            # Add this image to all classes found in it
            for class_id in classes_in_image:
                class_to_images[class_id].append(image_name)
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    # Convert defaultdict to regular dict and sort image lists
    result = {}
    for class_id, image_list in class_to_images.items():
        result[class_id] = sorted(image_list)
    
    #print(f"Class distribution:")
    #for class_id in sorted(result.keys()):
    #    print(f"  Class {class_id}: {len(result[class_id])} images")
    
    return result


def get_bboxes_for_images(image_list: List[str], class_id: int, dataset_path: str = "dataset/test") -> List[List[List[int]]]:
    """
    Extract bounding boxes for a list of images for a specific class.
    
    Args:
        image_list: List of image names
        class_id: The class ID to filter bounding boxes for
        dataset_path: Path to the dataset folder
        
    Returns:
        List of lists containing bounding boxes for each image in (xmin, ymin, xmax, ymax) format
    """
    labels_path = os.path.join(dataset_path, "labels")
    images_path = os.path.join(dataset_path, "images")
    bounding_boxes_list = []
    
    for image_name in image_list:
        # Get the corresponding label file name (remove image extension, add .txt)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_path, label_name)
        img_path = os.path.join(images_path, image_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            bounding_boxes_list.append([])
            continue
        img_height, img_width = image.shape[:2]
        
        image_bboxes = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse each line to extract bounding boxes
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) >= 5:  # YOLO format: class_id x y w h
                            bbox_class_id = int(parts[0])
                            if bbox_class_id == class_id:
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                xmin = int((x_center - width / 2) * img_width)
                                ymin = int((y_center - height / 2) * img_height)
                                xmax = int((x_center + width / 2) * img_width)
                                ymax = int((y_center + height / 2) * img_height)
                            
                                # Store bounding boxes in (xmin, ymin, xmax, ymax) format
                                image_bboxes.append([xmin, ymin, xmax, ymax])
                            
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
        else:
            print(f"Warning: Label file not found for {image_name}")
        
        bounding_boxes_list.append(image_bboxes)
    
    return bounding_boxes_list


def get_random_images_for_class(class_id: int, count: int, class_mapping: Dict[int, List[str]], dataset_path: str = "dataset/test") -> Tuple[List[str], List[List[List[int]]], List[str], List[List[List[int]]]]:
    """
    Returns random images for a given class ID along with their ground truth bounding boxes,
    plus the remaining images and their bounding boxes.
    
    Args:
        class_id: The class ID to get images for
        count: Number of random images to return
        class_mapping: Dictionary mapping class_id -> list of image names
        dataset_path: Path to the dataset folder (default: "dataset/test")
        
    Returns:
        Tuple of (selected_images, selected_bboxes, remaining_images, remaining_bboxes)
        - selected_images: List of selected random images
        - selected_bboxes: List of lists containing bounding boxes for selected images in (xmin, ymin, xmax, ymax) format
        - remaining_images: List of image names for remaining images
        - remaining_bboxes: List of lists containing bounding boxes for remaining images in (xmin, ymin, xmax, ymax) format
    """
    if class_id not in class_mapping:
        print(f"Warning: Class ID {class_id} not found in class mapping")
        return [], [], [], []
    
    available_images = class_mapping[class_id]
    
    if len(available_images) == 0:
        print(f"Warning: No images found for class ID {class_id}")
        return [], [], [], []
    
    # If we have fewer images than requested, return all available
    if len(available_images) <= count:
        print(f"Note: Class {class_id} has only {len(available_images)} images, returning all")
        selected_images = available_images.copy()
        remaining_images = []
    else:
        # Return random sample of requested count
        selected_images = random.sample(available_images, count)
        remaining_images = [img for img in available_images if img not in selected_images]
    
    # Get bounding boxes for selected and remaining images using the separate function
    selected_bboxes = get_bboxes_for_images(selected_images, class_id, dataset_path)
    remaining_bboxes = get_bboxes_for_images(remaining_images, class_id, dataset_path)
    
    return selected_images, selected_bboxes, remaining_images, remaining_bboxes


def get_indexed_images_for_class(class_id: int, indexes: List[int], class_mapping: Dict[int, List[str]], dataset_path: str = "dataset/test") -> Tuple[List[str], List[List[List[int]]], List[str], List[List[List[int]]]]:
    """
    Returns specific images by indexes for a given class ID along with their ground truth bounding boxes,
    plus the remaining images and their bounding boxes.
    
    Args:
        class_id: The class ID to get images for
        indexes: List of indexes to select from the available images for this class
        class_mapping: Dictionary mapping class_id -> list of image names
        dataset_path: Path to the dataset folder (default: "dataset/test")
        
    Returns:
        Tuple of (indexed_images, indexed_bboxes, remaining_images, remaining_bboxes)
        - indexed_images: List of image filenames at the specified indexes
        - indexed_bboxes: List of lists containing bounding boxes for indexed images in (xmin, ymin, xmax, ymax) format
        - remaining_images: List of image filenames for remaining images
        - remaining_bboxes: List of lists containing bounding boxes for remaining images in (xmin, ymin, xmax, ymax) format
    """
    if class_id not in class_mapping:
        print(f"Warning: Class ID {class_id} not found in class mapping")
        return [], [], [], []
    
    available_images = class_mapping[class_id]
    
    if len(available_images) == 0:
        print(f"Warning: No images found for class ID {class_id}")
        return [], [], [], []
    
    # Validate indexes and select images
    indexed_images = []
    valid_indexes = []
    
    for idx in indexes:
        if 0 <= idx < len(available_images):
            indexed_images.append(available_images[idx])
            valid_indexes.append(idx)
        else:
            print(f"Warning: Index {idx} is out of range for class {class_id} (available: 0-{len(available_images)-1})")
    
    if len(indexed_images) == 0:
        print(f"Warning: No valid indexes provided for class ID {class_id}")
        return [], [], [], []
    
    # Get remaining images (exclude the indexed ones)
    remaining_images = [img for i, img in enumerate(available_images) if i not in valid_indexes]
    
    # Get bounding boxes for indexed and remaining images using the separate function
    indexed_bboxes = get_bboxes_for_images(indexed_images, class_id, dataset_path)
    remaining_bboxes = get_bboxes_for_images(remaining_images, class_id, dataset_path)
    
    return indexed_images, indexed_bboxes, remaining_images, remaining_bboxes


def annotate_images(image_names: List[str], dataset_path: str, gt_list: List[sv.Detections], pred_list: List[sv.Detections]):
    """
    Annotate images using supervision box annotator with ground truth and predictions.
    
    Args:
        image_names: List of image names (filenames)
        dataset_path: Path to the dataset folder
        gt_list: List of sv.Detections objects for ground truth (green boxes)
        pred_list: List of sv.Detections objects for predictions (red boxes)
    
    Saves annotated images to 'Annotated Images' folder with:
    - Green boxes for ground truth
    - Red boxes for predictions
    """
    
    # Create output directory
    output_dir = "Annotated Images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize supervision box annotators with different colors
    green_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    red_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
    
    images_path = os.path.join(dataset_path, "images")
    
    print(f"Annotating {len(image_names)} images...")
    
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(images_path, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        annotated_image = image.copy()
        
        # Add green GT boxes if available
        if i < len(gt_list) and len(gt_list[i].xyxy) > 0:
            gt_detections = gt_list[i]
            annotated_image = green_annotator.annotate(scene=annotated_image, detections=gt_detections)
        
        # Add red prediction boxes if available
        if i < len(pred_list) and len(pred_list[i].xyxy) > 0:
            pred_detections = pred_list[i]
            annotated_image = red_annotator.annotate(scene=annotated_image, detections=pred_detections)
        
        # Save annotated image
        if len(pred_list[i].xyxy) > 0:
            output_path = os.path.join(output_dir, f"pred_{image_name}")
        else:
            output_path = os.path.join(output_dir, f"no_pred_{image_name}")
        cv2.imwrite(output_path, annotated_image)
        #print(f"Saved: {output_path}")
    
    #print(f"All annotated images saved to '{output_dir}' folder")


def annotate_ref_images(image_names: List[str], dataset_path: str, bboxes_list: List[List[List[int]]]):
    """
    Annotate reference images using supervision box annotator with blue color.
    
    Args:
        image_names: List of image names (filenames)
        dataset_path: Path to the dataset folder
        bboxes_list: List of lists containing bounding boxes for each image in (xmin, ymin, xmax, ymax) format
    
    Saves annotated images to 'Annotated Images' folder with:
    - Blue boxes for the bounding boxes
    """
    
    # Create output directory
    output_dir = "Annotated Images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize supervision box annotator with blue color
    blue_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
    
    images_path = os.path.join(dataset_path, "images")
    
    print(f"Annotating {len(image_names)} reference imag(es)")
    
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(images_path, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        annotated_image = image.copy()
        
        # Add blue boxes if available
        if i < len(bboxes_list) and len(bboxes_list[i]) > 0:
            # Convert bboxes to supervision format
            xyxy_array = np.array(bboxes_list[i])
            # Create dummy class IDs (all zeros)
            class_ids = np.array([0] * len(bboxes_list[i]))
            detections = sv.Detections(xyxy=xyxy_array, class_id=class_ids)
            
            annotated_image = blue_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"ref_{image_name}")
        cv2.imwrite(output_path, annotated_image)
        #print(f"Saved: {output_path}")
    
    print(f"All reference images saved to '{output_dir}' folder")
