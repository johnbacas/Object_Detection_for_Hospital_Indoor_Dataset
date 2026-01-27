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
class_idxs_dict = {v: k for k, v in class_names_dict.items()}

model = YOLO("./runs/detect/train6/weights/best.pt")  # load besttrained model
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