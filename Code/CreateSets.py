import os
import xml.etree.ElementTree as ET
import shutil
import random

from collections import defaultdict

# Convert annotations from xml to txt yolo format

def convert_annotation(xml_path, output_dir, class_list):
    """
    Converts an XML annotation file to YOLO TXT format.

    Args:
        xml_path (str): Path to the XML annotation file.
        output_dir (str): Directory to save the output TXT file.
        class_list (list): List of class names.
    """
    in_file = open(xml_path)
    out_file = open(os.path.join(output_dir, os.path.basename(xml_path).replace('.xml', '.txt')), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text.replace('_', ' ').lower()
        print(cls, end=', ')
        if cls not in class_list or int(difficult) == 1:
            continue
        cls_id = class_list.index(cls)
        xmlbox = obj.find('bndbox')
         # Get coordinates
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)

        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        out_file.write(str(cls_id) + " " + " ".join([str(x_center), str(y_center), str(width), str(height)]) + '\n')
    print('')
    in_file.close()
    out_file.close()

# Classes of our dataset

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

xml_dir = 'Hospital_Scene_Data-main/labels' # Path to your XML files
output_dir = 'Hospital_Scene_Data-main/txt' # Output directory for txt yolo format files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert all XML files in the directory to yolo compatible txt files
#for xml_file in os.listdir(xml_dir):
#    print(xml_file, end=': ')
#    if xml_file.endswith('.xml'):
#        convert_annotation(os.path.join(xml_dir, xml_file), output_dir, classes)

print("Conversion complete!")

# Create train, validate and test sets
# Each class must be present in every set

# Configurable parameters
IMG_DIR = 'Hospital_Scene_Data-main/images'
LBL_DIR = 'Hospital_Scene_Data-main/txt'
OUT_DIR = 'dataset'
SPLITS = ['train', 'val', 'test']
RATIOS = [0.7, 0.1, 0.2]  # Train, Val, Test

# 1. Gather image <-> classes mapping
image_to_classes = dict()
class_to_images = defaultdict(set)

for img_file in os.listdir(IMG_DIR):
    if img_file.lower().endswith(('.jpg', '.png')):
        label_file = os.path.join(LBL_DIR, img_file.rsplit('.', 1)[0] + '.txt')
        class_ids = set()
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        cls_id = line.split()[0]  # YOLO format: class_id x_center y_center width height
                        class_ids.add(cls_id)
        if classes:
            image_to_classes[img_file] = class_ids
            for cls in class_ids:
                class_to_images[cls].add(img_file)

all_images = list(image_to_classes.keys())
#random.seed(42)
random.shuffle(all_images)

# 2. Ensure each class appears in each split
split_images = {split: set() for split in SPLITS}
for cls, img_list in class_to_images.items():
    img_list = list(img_list)
    random.shuffle(img_list)
    # Assign one image per split for this class
    for i, split in enumerate(SPLITS):
        if i < len(img_list):
            split_images[split].add(img_list[i])

# 3. Fill splits by ratio, without duplication
already_selected = set()
for split in SPLITS:
    already_selected |= split_images[split]

remaining = [img for img in all_images if img not in already_selected]
n_total = len(all_images)
n_split = [int(r * n_total) for r in RATIOS[:-1]]
n_split.append(n_total - sum(n_split))  # Last split gets what's left

for i, split in enumerate(SPLITS):
    needed = n_split[i] - len(split_images[split])
    add_imgs = remaining[:needed]
    split_images[split].update(add_imgs)
    remaining = remaining[needed:]

# 4. Copy files
for split in SPLITS:
    img_out = os.path.join(OUT_DIR, split, 'images')
    lbl_out = os.path.join(OUT_DIR, split, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_file in split_images[split]:
        shutil.copy(os.path.join(IMG_DIR, img_file), os.path.join(img_out, img_file))
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_lbl_path = os.path.join(LBL_DIR, label_file)
        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, os.path.join(lbl_out, label_file))

print("Splitting complete. Every class is present in train, val, and test splits.")