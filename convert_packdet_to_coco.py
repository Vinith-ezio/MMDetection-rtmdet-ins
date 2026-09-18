from pathlib import Path
import json
import math
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_ROOT = Path(r"E:\MMdetection\PackDet\data")

IMAGE_ROOT = DATA_ROOT / "images"
LABEL_ROOT = DATA_ROOT / "labels"
OUTPUT_ROOT = DATA_ROOT / "annotations"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Dataset classes
# --------------------------------------------------

CLASSES = [
    "plastic_bottle",
    "glass_bottle",
    "can",
    "jar",
    "carton_beverage",
    "tube",
]


# --------------------------------------------------
# Polygon area calculation
# --------------------------------------------------

def polygon_area(points):
    """
    Calculate polygon area using the shoelace formula.

    points format:
    [(x1, y1), (x2, y2), ...]
    """

    area = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


# --------------------------------------------------
# Convert one dataset split
# --------------------------------------------------

def convert_split(split_name):
    image_dir = IMAGE_ROOT / split_name
    label_dir = LABEL_ROOT / split_name
    output_json = OUTPUT_ROOT / f"instances_{split_name.lower()}.json"

    coco = {
        "info": {
            "description": f"PackDet {split_name} dataset"
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    for class_id, class_name in enumerate(CLASSES):
        coco["categories"].append({
            "id": class_id + 1,
            "name": class_name,
            "supercategory": "object"
        })

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        [
            path for path in image_dir.rglob("*")
            if path.suffix.lower() in image_extensions
        ]
    )

    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        relative_image_path = image_path.relative_to(DATA_ROOT)

        with Image.open(image_path) as image:
            width, height = image.size

        coco["images"].append({
            "id": image_id,
            "file_name": str(relative_image_path).replace("\\", "/"),
            "width": width,
            "height": height
        })

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Skipping image without label: {image_path.name}")
            coco["images"].pop()
            continue

        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            class_id = int(values[0])
            coordinates = list(map(float, values[1:]))

            if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                print(
                    f"Skipping invalid polygon: "
                    f"{label_path}, line {line_number}"
                )
                continue

            points = []

            for index in range(0, len(coordinates), 2):
                x_normalized = coordinates[index]
                y_normalized = coordinates[index + 1]

                x_pixel = x_normalized * width
                y_pixel = y_normalized * height

                points.append((x_pixel, y_pixel))

            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]

            x_min = max(0.0, min(x_values))
            y_min = max(0.0, min(y_values))
            x_max = min(float(width), max(x_values))
            y_max = min(float(height), max(y_values))

            bbox_width = max(0.0, x_max - x_min)
            bbox_height = max(0.0, y_max - y_min)

            segmentation = []

            for x, y in points:
                segmentation.extend([x, y])

            area = polygon_area(points)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "segmentation": [segmentation],
                "area": area,
                "bbox": [
                    x_min,
                    y_min,
                    bbox_width,
                    bbox_height
                ],
                "iscrowd": 0
            })

            annotation_id += 1

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2)

    print(f"\nCreated: {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    convert_split("Train")
    convert_split("Validation")