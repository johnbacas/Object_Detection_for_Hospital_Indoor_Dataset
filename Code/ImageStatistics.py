from utils import get_class_to_images_mapping
import os
import cv2
from collections import defaultdict

if __name__ == "__main__":
    dataset_path = "dataset/train"
    class_to_images = get_class_to_images_mapping(dataset_path)
    
    for class_id in range(56):
        print(f"Class: {class_id}, First 15 images: {class_to_images[class_id][:15]}")
        print()

    folder_path = "Hospital_Scene_Data-main/images"
    size_count = defaultdict(int)
    sizes = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                size_count[(w, h)] += 1
                sizes.append((w, h))
    #for size, count in size_count.items():
    #    print(f"Size: {size}, Count: {count}")
    # Find the 10 most common sizes
    if size_count:
        top10 = sorted(size_count.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most common image sizes:")
        for size, count in top10:
            print(f"Size: {size}, Count: {count}")
    if sizes:
        min_size = min(sizes, key=lambda s: s[0]*s[1])
        max_size = max(sizes, key=lambda s: s[0]*s[1])
        print(f"Smallest size: {min_size}")
        print(f"Largest size: {max_size}")
    else:
        print("No images found.")