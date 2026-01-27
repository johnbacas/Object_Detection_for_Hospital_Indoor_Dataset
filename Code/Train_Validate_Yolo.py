# Define the paths for train, validation, and test data
import os
import yaml
import ultralytics
import numpy as np

from ultralytics import YOLO

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

# Create a dictionary mapping class IDs to class names and vice versa
class_names_dict = {i: name for i, name in enumerate(classes)}
#print(class_names_dict)
class_idxs_dict = {v: k for k, v in class_names_dict.items()}
#print(class_idxs_dict)

dataset_dir = 'dataset'
train_img_dir = os.path.join(dataset_dir, 'train', 'images')
val_img_dir = os.path.join(dataset_dir, 'val', 'images')
test_img_dir = os.path.join(dataset_dir, 'test', 'images')


# Create the YAML data structure
yolo_config = {
    'path': '../dataset', # Dataset root directory
    'train': 'train/images',  # Train images relative to 'path'
    'val': 'val/images',      # Val images relative to 'path'
    'test': 'test/images',    # Test images relative to 'path'
    'nc': len(classes),       # Number of classes
    'names': classes          # Class names
}

# Define the output path for the YAML file
yaml_file_path = 'hospital_dataset.yaml'

# Write the YAML data to a file
with open(yaml_file_path, 'w') as f:
    yaml.dump(yolo_config, f, default_flow_style=None)

print(f"YOLO configuration file '{yaml_file_path}' created successfully.")

ultralytics.checks()

model=YOLO('yolo11m.pt')
#print(model.model.state_dict)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
results = model.train(data='./hospital_dataset.yaml', epochs=100, patience=5, batch=0.7, imgsz=640, plots=True, device=0)

print('--------------------- Training completed -------------------------')

model = YOLO("./runs/detect/train5/weights/best.pt")  # load best trained model
#print(model.model.state_dict)

# Validate the model with test set
metrics = model.val(data='./hospital_dataset.yaml', imgsz=640, batch=12, split='test', plots=True, device=0)

print('mAP50-95:', metrics.box.map)
print('mAP Scores per IoU:', np.mean(metrics.box.all_ap, axis=0))

scores=[]
for i, object_ap in enumerate(metrics.box.maps):
    scores.append(object_ap)
sorted_scores_with_index = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print('Objects with their mAP50-95 in descending order\n')
for index, object_ap in sorted_scores_with_index:
    print(f"{index:2} {class_names_dict.get(index, 'unknown'):23}: {object_ap:.3f}")

print('--------------------- Validation to test set completed -------------------------')