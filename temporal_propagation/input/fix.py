import os

root_dir = "./zurich/annotations"

for root, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".json") and "leftImg8bit" in file:
            old_path = os.path.join(root, file)

            new_file = file.replace("leftImg8bit", "gtFine_polygons")
            new_path = os.path.join(root, new_file)

            os.rename(old_path, new_path)
            print(f"{file} -> {new_file}")

print("Done.")