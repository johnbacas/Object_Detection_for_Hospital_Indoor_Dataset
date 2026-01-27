import os
from collections import defaultdict

SPLITS = ['train', 'val', 'test']
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

# Check number of images in each set (train, val, test)
dataset_dir = 'dataset'
image_extensions = ['.jpg', '.jpeg', '.png']

for split in SPLITS:
    image_folder = os.path.join(dataset_dir, split, 'images')
    if os.path.exists(image_folder):
        count = 0
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                count += 1
        print(f"Number of images in {split} folder: {count}")
    else:
        print(f"{split} image folder not found at {image_folder}")

# check how many classes are in each set (train, val, test)
for split in SPLITS:
    label_folder = os.path.join(dataset_dir, split, 'labels')
    class_counts = defaultdict(int)
    if os.path.exists(label_folder):
        for filename in os.listdir(label_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(label_folder, filename), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            if class_id < len(classes):
                                class_name = classes[class_id]
                                class_counts[class_name] += 1
                            else:
                                print(f"Warning: Class ID {class_id} out of range in file {filename}")

        print(f"\nClass counts in {split} folder:")
        for i, (class_name, count) in enumerate(class_counts.items()):
            print(f"{i+1}) {class_name}: {count}")
    else:
        print(f"\n{split} label folder not found at {label_folder}")