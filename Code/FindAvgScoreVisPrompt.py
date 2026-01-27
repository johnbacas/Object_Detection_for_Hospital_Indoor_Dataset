import os
import json
from collections import defaultdict
import numpy as np

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

RESULTS_DIR = "Results_1st_15_random_images"  # Update if your folder name is different

# 1. List all JSON files in Results folder
json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
json_files.sort()
# 2. Extract information from each JSON file
# Structure: {class_id: ([mAP50s], [mAP75s], [mAP50_95s])}
class_scores = defaultdict(lambda: ([], [], []))

for file in json_files:
    file_path = os.path.join(RESULTS_DIR, file)
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            data = data[0]  # Access the first element if json structure is a list
        class_id = data.get("class_to_search")
        #ref_img = data.get("reference_images")
        mAP50 = data.get("mAP50")
        mAP75 = data.get("mAP75")
        mAP50_95 = data.get("mAP50_95")
        #print(f"Class ID: {class_id}, Ref Image: {ref_img}, mAP50_95: {mAP50_95}")
        if class_id is not None:
            class_scores[class_id][0].append(mAP50)
            class_scores[class_id][1].append(mAP75)
            class_scores[class_id][2].append(mAP50_95)

# 4. Compute average for each class and each metric
class_averages = {}
for class_id, (mAP50s, mAP75s, mAP50_95s) in class_scores.items():
    avg50 = float(np.mean(mAP50s) if mAP50s else None)
    avg75 = float(np.mean(mAP75s) if mAP75s else None)
    avg5095 = float(np.mean(mAP50_95s) if mAP50_95s else None)
    class_averages[class_id] = (avg50, avg75, avg5095)

# 5. Compute overall averages across all classes
all_mAP50 = [avg[0] for avg in class_averages.values() if avg[0] is not None]
all_mAP75 = [avg[1] for avg in class_averages.values() if avg[1] is not None]
all_mAP5095 = [avg[2] for avg in class_averages.values() if avg[2] is not None]

overall_avg50 = np.mean(all_mAP50) if all_mAP50 else None
overall_avg75 = np.mean(all_mAP75) if all_mAP75 else None
overall_avg5095 = np.mean(all_mAP5095) if all_mAP5095 else None

print("Per-class averages (class_id: (mAP50, mAP75, mAP50_95)):")
for class_id, avgs in sorted(class_averages.items()):
    print(f"{class_id}: {avgs}")

print("\nOverall averages:")
print(f"mAP50: {overall_avg50}")
print(f"mAP75: {overall_avg75}")
print(f"mAP50_95: {overall_avg5095}")

# Save results to 'final_results.txt'
output_path = RESULTS_DIR + "/final_results_new.txt"
with open(output_path, "w") as f:
    f.write("Per-class averages (class_id: (mAP50, mAP75, mAP50_95)):\n")
    for class_id, avgs in sorted(class_averages.items()):
        f.write(f"{class_id} {classes[class_id]}: {avgs[0]}     {avgs[1]}     {avgs[2]}\n")
    f.write("\nOverall averages:\n")
    f.write(f"mAP50: {overall_avg50}\n")
    f.write(f"mAP75: {overall_avg75}\n")
    f.write(f"mAP50_95: {overall_avg5095}\n")