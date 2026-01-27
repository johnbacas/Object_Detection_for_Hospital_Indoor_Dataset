from utils import get_class_to_images_mapping
import json
import os

# Create mappings
train_class_mapping = get_class_to_images_mapping('dataset/train')
test_class_mapping = get_class_to_images_mapping('dataset/test')

folder_name = 'Class_mappings'
# Create output directory if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Save train mapping
train_output_path = os.path.join(folder_name, 'train_class_mapping.json')
with open(train_output_path, 'w') as f:
    # Convert dict keys to strings since JSON requires string keys
    train_mapping_str_keys = {str(k): v for k, v in train_class_mapping.items()}
    json.dump(train_mapping_str_keys, f, indent=2)
print(f"Train class mapping saved to: {train_output_path}")

# Save test mapping
test_output_path = os.path.join(folder_name, 'test_class_mapping.json')
with open(test_output_path, 'w') as f:
    # Convert dict keys to strings since JSON requires string keys
    test_mapping_str_keys = {str(k): v for k, v in test_class_mapping.items()}
    json.dump(test_mapping_str_keys, f, indent=2)
print(f"Test class mapping saved to: {test_output_path}")

# Print summary
print(f"\nTrain mapping: {len(train_class_mapping)} classes")
print(f"Test mapping: {len(test_class_mapping)} classes")
