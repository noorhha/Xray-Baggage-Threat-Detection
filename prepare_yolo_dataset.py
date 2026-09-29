import os
import cv2
import random
import numpy as np

DATASET_DIR = "TrainData"
OUTPUT_DIR = "yolo_dataset"

# Folder names to YOLO class IDs
FOLDER_TO_CLASS = {
    "GUN": 0,
    "knife": 1,
    "shuriken": 2
}

# Mask pixel values to YOLO class IDs
MASK_VALUE_TO_CLASS = {
    1: 0,  # gun
    2: 1,  # knife
    3: 2   # shuriken
}

CLASS_NAMES = {
    "GUN": "gun",
    "knife": "knife",
    "shuriken": "shuriken",
    "safe": "safe"
}

random.seed(42)


def create_folders():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def bbox_from_binary_mask(mask_binary, img_w, img_h):
    ys, xs = np.where(mask_binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(np.min(xs))
    y1 = int(np.min(ys))
    x2 = int(np.max(xs))
    y2 = int(np.max(ys))

    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    x_center = (x1 + box_w / 2) / img_w
    y_center = (y1 + box_h / 2) / img_h
    norm_w = box_w / img_w
    norm_h = box_h / img_h

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    if norm_w <= 0 or norm_h <= 0:
        return None

    return x_center, y_center, norm_w, norm_h


def get_boxes_from_mask(mask, folder_class_name, img_w, img_h):
    """
    Returns YOLO label rows:
    class_id x_center y_center width height

    Handles:
    - Binary masks: [0, 1]
    - Class-coded masks: [0, 1, 2, 3]
    """

    boxes = []

    unique_values = np.unique(mask)
    nonzero_values = [int(v) for v in unique_values if int(v) != 0]

    # Case 1: class-coded mask, e.g. [0,1,2,3] or [0,1,2]
    # This means one image may contain multiple objects.
    if any(v > 1 for v in nonzero_values):
        for mask_value, class_id in MASK_VALUE_TO_CLASS.items():
            mask_binary = (mask == mask_value).astype(np.uint8)

            bbox = bbox_from_binary_mask(mask_binary, img_w, img_h)

            if bbox is None:
                continue

            x_center, y_center, norm_w, norm_h = bbox
            boxes.append((class_id, x_center, y_center, norm_w, norm_h))

    # Case 2: binary mask, e.g. [0,1]
    # Use the folder name as the class.
    else:
        if folder_class_name not in FOLDER_TO_CLASS:
            return boxes

        class_id = FOLDER_TO_CLASS[folder_class_name]

        mask_binary = (mask > 0).astype(np.uint8)

        bbox = bbox_from_binary_mask(mask_binary, img_w, img_h)

        if bbox is not None:
            x_center, y_center, norm_w, norm_h = bbox
            boxes.append((class_id, x_center, y_center, norm_w, norm_h))

    return boxes


def collect_items():
    all_items = []

    # Threat folders
    for class_name in ["GUN", "knife", "shuriken"]:
        image_folder = os.path.join(DATASET_DIR, class_name)
        mask_folder = os.path.join(DATASET_DIR, "annotations", class_name)

        if not os.path.exists(image_folder):
            print("Missing image folder:", image_folder)
            continue

        if not os.path.exists(mask_folder):
            print("Missing mask folder:", mask_folder)
            continue

        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(image_folder, filename)
                mask_path = os.path.join(mask_folder, filename)

                if os.path.exists(mask_path):
                    all_items.append({
                        "type": "threat",
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "class_name": class_name,
                        "filename": filename
                    })
                else:
                    print("Missing mask for:", image_path)

    # Safe folder
    safe_folder = os.path.join(DATASET_DIR, "safe")

    if os.path.exists(safe_folder):
        for filename in os.listdir(safe_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(safe_folder, filename)

                all_items.append({
                    "type": "safe",
                    "image_path": image_path,
                    "mask_path": None,
                    "class_name": "safe",
                    "filename": filename
                })
    else:
        print("Missing safe folder:", safe_folder)

    return all_items


def process_items(items, split):
    written = 0
    skipped = 0
    total_boxes = 0

    for item in items:
        image = cv2.imread(item["image_path"])

        if image is None:
            skipped += 1
            print("Could not read image:", item["image_path"])
            continue

        img_h, img_w = image.shape[:2]

        base_name = os.path.splitext(item["filename"])[0]
        class_prefix = CLASS_NAMES[item["class_name"]]

        # Unique filename to avoid overwriting same names from different folders
        new_base_name = f"{class_prefix}_{base_name}"
        new_image_name = new_base_name + ".png"
        new_label_name = new_base_name + ".txt"

        output_image_path = os.path.join(OUTPUT_DIR, "images", split, new_image_name)
        output_label_path = os.path.join(OUTPUT_DIR, "labels", split, new_label_name)

        cv2.imwrite(output_image_path, image)

        # Safe image = empty label file
        if item["type"] == "safe":
            open(output_label_path, "w").close()
            written += 1
            continue

        mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            skipped += 1
            print("Could not read mask:", item["mask_path"])
            continue

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )

        boxes = get_boxes_from_mask(
            mask=mask,
            folder_class_name=item["class_name"],
            img_w=img_w,
            img_h=img_h
        )

        if len(boxes) == 0:
            skipped += 1
            print("No valid object box found for:", item["mask_path"])
            continue

        with open(output_label_path, "w") as f:
            for class_id, x_center, y_center, norm_w, norm_h in boxes:
                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{norm_w:.6f} "
                    f"{norm_h:.6f}\n"
                )

        total_boxes += len(boxes)
        written += 1

    return written, skipped, total_boxes


def write_yaml():
    yaml_text = """path: yolo_dataset
train: images/train
val: images/val

names:
  0: gun
  1: knife
  2: shuriken
"""

    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_text)


def main():
    create_folders()

    all_items = collect_items()
    print("Total collected images:", len(all_items))

    random.shuffle(all_items)

    split_index = int(0.8 * len(all_items))
    train_items = all_items[:split_index]
    val_items = all_items[split_index:]

    train_written, train_skipped, train_boxes = process_items(train_items, "train")
    val_written, val_skipped, val_boxes = process_items(val_items, "val")

    write_yaml()

    print("\nYOLO dataset created successfully.")
    print("Total images:", len(all_items))
    print("Training images written:", train_written)
    print("Validation images written:", val_written)
    print("Training skipped:", train_skipped)
    print("Validation skipped:", val_skipped)
    print("Training boxes:", train_boxes)
    print("Validation boxes:", val_boxes)

    print("\nLogic used:")
    print("Safe images -> empty label file")
    print("Binary masks [0,1] -> folder class")
    print("Class-coded masks [0,1,2,3] -> separate boxes for gun/knife/shuriken")


if __name__ == "__main__":
    main()