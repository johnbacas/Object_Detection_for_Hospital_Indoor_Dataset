import os
import cv2
import torch
import supervision as sv
import numpy as np
import glob
import re


from supervision.metrics import MeanAveragePrecision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

classes = ['door_handle', 'chair', 'exam_table', 'countertop', 'waste_bin', 'utility_cart', 'computer',\
           'keyboard', 'mouse', 'telephone', 'handle', 'tv', 'table', 'surgical_light', 'bedside_monitor',\
           'dispenser', 'faucet', 'hospital_bed', 'medical_drawer', 'bedrail', 'foot_board', 'sofa',\
           'overbed_table', 'sink', 'breathing_tube', 'curtain', 'bedside_table', 'electrosurgical_unit',\
           'staff', 'operating_bed', 'patient', 'ventilator', 'iv_pole', 'infusion_pump',\
           'surgical_instrument', 'visitor', 'toilet', 'sequential_compression', 'syringe_pump',\
           'toilet_handle', 'hallway_assist_bar', 'xray_machine', 'xray_bed', 'light_switch',\
           'restroom_assist_bar', 'panda_baby_warmer', 'press_to_open', 'push_latch', 'elevator_panel',\
           'drawer', 'counter', 'cabinet', 'medical_waste_container', 'person', 'wheel_chair', 'incubator']
classes = [s.replace('_', ' ').lower() for s in classes]
#print(classes)


# Check if CUDA is available and set the device accordingly
#device = 0 if torch.cuda.is_available() else 'cpu'
#print(f"Using device: {device}")
# build grounding dino from huggingface
gd_model = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(gd_model)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model)#.to(device)

# Create a dictionary mapping class IDs to class names and vice versa
class_names_dict = {i: name for i, name in enumerate(classes)}
print(class_names_dict)
class_idxs_dict = {v: k for k, v in class_names_dict.items()}
print(class_idxs_dict)

# Dataset
class HospitalSceneDataset(Dataset):
    @staticmethod
    def extract_number(filename):  # extract number form filename eg. from 'image_23.jpg' extract integer 23
        match = re.search(r'image_(\d+)\.(jpg|png)$', filename, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    def __init__(self, image_dir, label_dir, transform=None): #, target_size=(1280, 720)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        # Find image files
        self.image_files = []
        self.extensions = ['*.jpg', '*.png']
        for extension in self.extensions:
            self.image_files.extend(glob.glob(os.path.join(image_dir, extension)))

        self.image_files = sorted(self.image_files, key=self.extract_number)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file_path = self.image_files[idx]
        # Extract the base filename (e.g., image_270.jpg)
        img_filename = os.path.basename(img_file_path)

        # Construct the label file path based on the corrected structure
        label_file_name = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file_name)

        # Load image and convert to RGB
        img_bgr = cv2.imread(img_file_path)
        if img_bgr is None:
            print(f"Warning: Could not load image {img_file_path}")
            return img_file_path, label_path, None, [], []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h,w = img_rgb.shape[:2]

        boxes = []
        labels = []

        try:
            with open(label_path, 'r') as f:
                annotations = f.readlines()

            for annotation in annotations:
                parts = annotation.split()
                labels.append(class_names_dict[int(parts[0])])
                xcycwh = list(map(float, parts[1:]))
                boxes.append(xcycwh)
            boxes = np.array(boxes)
            #print(boxes.shape)
            boxes = sv.xcycwh_to_xyxy(xcycwh=boxes)
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            # boxes = sv.denormalize_boxes(boxes, (w, h)) #it gives wrong results

        except FileNotFoundError:
            print(f"Warning: Label file not found for {label_path}")
            # Return image and empty lists if label file is missing
            return img_file_path, label_path, img_rgb, [], []


        # Apply other transformations if provided
        if self.transform:
            img_rgb = self.transform(img_rgb)
            # Note: For this subtask, we are not implementing bounding box transformation.
            # The original bounding boxes are returned. In a real scenario, you would
            # need to transform bounding boxes along with the image.

        return img_file_path, label_path, img_rgb, boxes, labels

def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    img_paths = []
    lbl_paths = []
    for img_path, lbl_path, image, image_boxes, image_labels in batch:
        if image is not None: # Handle cases where image loading failed
            images.append(image)
            boxes.append(torch.tensor(image_boxes, dtype=torch.float32))
            labels.append(image_labels) # Keep labels as list of strings for now
            img_paths.append(img_path)
            lbl_paths.append(lbl_path)

    # You might need to adjust how images are stacked based on your model's input requirements
    # For now, we return a list of images and lists of tensors for boxes and labels
    return img_paths, lbl_paths, images, boxes, labels

image_dir = 'dataset/test/images'
label_dir = 'dataset/test/labels'

# Instantiate the dataset
hospital_dataset = HospitalSceneDataset(image_dir=image_dir, label_dir=label_dir) #, target_size=(1280, 720)) # Specify target size

# Instantiate a DataLoader
batch_size = 4
data_loader = DataLoader(hospital_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Dataset size: {len(hospital_dataset)}")
print(f"DataLoader batch size: {data_loader.batch_size}")

# prepare text prompt. Put dots between classes
def create_caption_from_classes(gt_labels):
    captions = []
    for label_list in gt_labels:
        classes = list(set(label_list))  # prompt with unique labels
        caption = ". ".join(classes) # create a string where classes are seperated by dot
        caption += '.'  # add dot at the end
        captions.append(caption)
    #print(captions)
    return captions


#G.Dino Predict
def gdino_predict(images, texts_prompt, box_threshold, text_threshold):
    inputs = processor(images=images, text=texts_prompt, return_tensors="pt", padding=True, truncation=True)#.to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(k.shape[0], k.shape[1]) for k in images],
    )

    pred_labels = [result['text_labels'] for result in results]
    pred_boxes = [result['boxes'] for result in results]
    pred_scores = [result['scores'] for result in results]

    return pred_labels, pred_boxes, pred_scores


# Remove predicted labels, bounding boxes and scores for labels that are not in ground truth labels
def filter_predictions(ground_truth_labels, predicted_labels, predicted_boxes, predicted_scores):
    pred_labels = []
    pred_boxes = []
    pred_scores = []
    for i, labels in enumerate(predicted_labels):
        l=[]
        b=[]
        s=[]
        for j, label in enumerate(labels):
            if label in ground_truth_labels[i]:
                l.append(label)
                b.append(predicted_boxes[i][j])
                s.append(predicted_scores[i][j])
        pred_labels.append(l)
        if len(b) > 0:
          pred_boxes.append(torch.stack(b))
          pred_scores.append(torch.stack(s))
        else:
          pred_boxes.append(torch.empty(0, 4)) # Append empty tensor for empty boxes
          pred_scores.append(torch.empty(0)) # Append empty tensor for empty scores
    return pred_labels, pred_boxes, pred_scores


# find true positive from predicted
def find_tp(gt_detection: sv.Detections, pred_detection: sv.Detections, iou_threshold=0.5):
    classes = set(gt_detection.class_id.tolist())
    tp_info = []
    tp = 0
    for cls in classes:
        gt_mask = gt_detection.class_id == cls
        pred_mask = pred_detection.class_id == cls
        gt_boxes = gt_detection.xyxy[gt_mask]
        pred_boxes = pred_detection.xyxy[pred_mask]
        pred_scores = pred_detection.confidence[pred_mask]

        # Sort predictions by confidence descending
        sort_idx = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        # IoU matrix using supervision's box_iou
        iou_matrix = sv.box_iou_batch(pred_boxes, gt_boxes)
        #print(iou_matrix)

        # Initialize arrays for true positives (tp), false positives (fp
        assigned_gt = set()
        for pred_idx, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in assigned_gt:
                    continue
                iou = iou_matrix[pred_idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                tp+=1
                assigned_gt.add(best_gt_idx)
                tp_info.append([cls, pred_box, pred_score, best_iou])
    return tp, tp_info


# create detections from labels, bboxes and scores (scores needed only for creating detections from predictions).
def create_detections(labels, boxes, scores=None):
    detections = []
    if scores is None:
        for label, box in zip(labels, boxes):
            xyxy = np.array(box.cpu()) # Move to CPU
            class_ids = np.array([class_idxs_dict.get(name, 100) for name in label])
            detections.append(sv.Detections(xyxy=xyxy, class_id=class_ids))
    else:
        for label, box, score in zip(labels, boxes, scores):
            xyxy = np.array(box.cpu()) # Move to CPU
            class_ids = np.array([class_idxs_dict.get(name, 100) for name in label])
            confidences=np.array(score.cpu()) # Move to CPU
            detections.append(sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confidences))
    return detections

# Calculate mAP for all images in dataset
# True Positives are given for IoU thres of 50%

map_metric = MeanAveragePrecision()
map_metric.reset()
batch_n=1
classes_detected = set()
gt_classes_so_far = set()
#new_labels = []

for batch in data_loader:
    img_paths, lbl_paths, images, gt_boxes, gt_labels = batch

    print(f"Batch {batch_n}")
    print(img_paths)
    print(lbl_paths)

    for labels in gt_labels:
        for label in labels:
    #        if label not in classes:
    #            print(f"Warning: Label '{label}' from image {img_paths} not found in original class_names.")
    #            name_to_index[label] = len(classes) # if a label present in xml file is not in the dataset's gt, add it.
    #            index_to_name[len(classes)] = label
    #            classes.append(label)
    #            new_labels.append(label)
            gt_classes_so_far.add(label)


    # G.Dino makes prediction
    texts_prompt = create_caption_from_classes(gt_labels)
    pred_labels, pred_boxes, pred_scores = gdino_predict(images, texts_prompt, 0.25, 0.25)

    # Filter results
    f_pred_labels, f_pred_boxes, f_pred_scores = filter_predictions(gt_labels, pred_labels, pred_boxes, pred_scores)

    # create detections
    gt_detections = create_detections(gt_labels, gt_boxes)
    pred_detections = create_detections(f_pred_labels, f_pred_boxes, f_pred_scores)

    # Find True Positive bboxes and update mAP
    for gt_detection, pred_detection in zip(gt_detections, pred_detections):
        # map_result = map_metric.update(pred_detection, gt_detection) # update returns self, no need to assign
        map_metric.update(pred_detection, gt_detection) # Call update method
        tp, tp_info = find_tp(gt_detection, pred_detection, 0.5)
        print('TP / GT : ', tp, '/', len(gt_detection.xyxy), '=', tp/len(gt_detection.xyxy))
        for info in tp_info:
            classes_detected.add(info[0])

    sort_gt = sorted([class_idxs_dict[label] for label in list(gt_classes_so_far)])
    sort_pred = sorted(list(classes_detected))
    batch_n += 1

    map_result = map_metric.compute() # Compute mAP at the end

print('mAP50-95:', map_result.map50_95)
print('mAP Scores per IoU:', map_result.mAP_scores)

scores=[]
for i, object_ap in enumerate(map_result.ap_per_class):
    scores.append(sum(object_ap)/len(object_ap))  # Average mAP
sorted_scores_with_index = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print('Objects with their mAP50-95 in descending order\n')
for index, object_ap in sorted_scores_with_index:
    print(f"{index:2} {class_names_dict.get(index, 'unknown'):23}: {object_ap:.3f}")

print('AP50, AP75, AP50-95 for each class')
for i, (object_ap_all_ious, class_mAP50_95) in enumerate(zip(map_result.ap_per_class, scores)):
    print(f"{i} {class_names_dict.get(i, 'unknown')}:\t {object_ap_all_ious[0]:} \t {object_ap_all_ious[5]:} \t {class_mAP50_95:}")