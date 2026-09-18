from pathlib import Path

LABEL_DIR = Path("PackDet/data/labels/Train")

total_files = 0
total_objects = 0
invalid_lines = 0

for label_file in LABEL_DIR.glob("*.txt"):
    total_files += 1

    with open(label_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if not values:
                continue

            # One class ID plus at least 3 coordinate pairs
            if len(values) < 7:
                print(
                    f"Invalid annotation: "
                    f"{label_file.name}, line {line_number}"
                )
                invalid_lines += 1
                continue

            # After class ID, coordinates must come in x/y pairs
            coordinate_count = len(values) - 1

            if coordinate_count % 2 != 0:
                print(
                    f"Odd number of coordinates: "
                    f"{label_file.name}, line {line_number}"
                )
                invalid_lines += 1
                continue

            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]

            if not 0 <= class_id <= 5:
                print(
                    f"Invalid class ID {class_id}: "
                    f"{label_file.name}, line {line_number}"
                )
                invalid_lines += 1

            if any(value < 0 or value > 1 for value in coordinates):
                print(
                    f"Coordinate outside 0-1 range: "
                    f"{label_file.name}, line {line_number}"
                )
                invalid_lines += 1

            total_objects += 1

print()
print("Total label files:", total_files)
print("Total polygon objects:", total_objects)
print("Invalid annotation lines:", invalid_lines)